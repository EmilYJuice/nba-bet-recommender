#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import math

# -----------------------------
# Artifact loading
# -----------------------------
def _load_artifacts(art_dir):
    """
    Loads Hybrid model artifacts from artifact dir.
    Expects:
      - user_factors.npy
      - item_factors.npy
      - hybrid_artifacts.json   (contains user_to_idx, item_to_idx, idx_to_item, team_classes)
      - xgb_model.json          (XGBoost model)
    Optionally:
      - train_meta.json         (half_life_days, cap_value, ref_date)
    """
    p = Path(art_dir)

    user_f = np.load(p / "user_factors.npy")
    item_f = np.load(p / "item_factors.npy")

    with open(p / "hybrid_artifacts.json") as f:
        maps = json.load(f)

    meta = None
    meta_path = p / "train_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Load XGBoost model
    xgb_model = None
    xgb_path = p / "xgb_model.json"
    if xgb_path.exists():
        try:
            xgb_model = xgb.Booster()
            xgb_model.load_model(str(xgb_path))
        except Exception as e:
            print(f"Warning: Could not load XGBoost model: {e}")
            xgb_model = None

    return user_f, item_f, maps, meta, xgb_model


# -----------------------------
# Popularity fallback
# -----------------------------
def _build_popularity(train_csv, half_life_days, cap_value, ref_date=None):
    df = pd.read_csv(train_csv, dtype={"mask_id":"string","event_description":"string"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # compute cap if not provided
    if cap_value is None:
        cap_value = float(df["amount_sum"].quantile(0.99))
    df["amt_capped"] = df["amount_sum"].clip(upper=cap_value)

    ref = pd.to_datetime(ref_date) if ref_date else df["date"].max()
    days_since = (ref - df["date"]).dt.days.clip(lower=0)
    decay = np.power(0.5, days_since / float(half_life_days))
    df["w"] = df["amt_capped"] * decay

    pop = (df.groupby("event_description", as_index=False)["w"]
             .sum()
             .rename(columns={"w":"pop_weight"})
             .set_index("event_description")["pop_weight"])
    return pop


# -----------------------------
# Candidates (per date)
# -----------------------------
def _candidates_for_date(sched_df, date_str):
    """Extract neutral game pairs for a given date from vendor schedule."""
    sub = sched_df.query("date == @date_str").copy()
    if sub.empty:
        return []
    # Use the "Game" column which has "TeamA & TeamB" format
    return (sub["Game"]
              .dropna()
              .drop_duplicates()
              .tolist())

def _expand_game_orientations(game_neutral):
    """
    Convert 'TeamA & TeamB' to both orientations:
    - 'TeamA @ TeamB'  
    - 'TeamB @ TeamA'
    Returns tuple: (orientation1, orientation2)
    """
    if " & " not in game_neutral:
        return (game_neutral, game_neutral)  # fallback if format unexpected
    
    teams = [team.strip() for team in game_neutral.split(" & ")]
    if len(teams) != 2:
        return (game_neutral, game_neutral)  # fallback
    
    team_a, team_b = teams
    return (f"{team_a} @ {team_b}", f"{team_b} @ {team_a}")

def _neutral_display_format(game_neutral):
    """Keep 'TeamA & TeamB' format for display"""
    return game_neutral

def _generate_pair_key(game_neutral):
    """Convert 'TeamA & TeamB' to pair_key format 'TeamA | TeamB'"""
    if " & " not in game_neutral:
        return game_neutral  # fallback if format unexpected
    return game_neutral.replace(" & ", " | ")

def _extract_team_features(event_description):
    """Extract team names from event description (matches 02_new_model_approach.py)."""
    if pd.isna(event_description):
        return None, None, None
    
    desc = str(event_description).strip()
    
    # Handle @ format: "Team A @ Team B" -> away=A, home=B
    if ' @ ' in desc:
        parts = desc.split(' @ ')
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip(), 'away'
    
    # Handle vs format: "Team A vs. Team B" -> home=A, away=B  
    if ' vs. ' in desc or ' vs ' in desc:
        desc = desc.replace(' vs. ', ' vs ')
        parts = desc.split(' vs ')
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip(), 'home'
    
    return None, None, None

# Cache for expensive calculations
_team_strength_cache = None
_user_features_cache = None

def _build_hybrid_features(user_id, game_description, game_date, maps, train_df, ref_date):
    """Build features for hybrid model prediction (matches 02_new_model_approach.py)."""
    import pandas as pd
    global _team_strength_cache, _user_features_cache
    
    team1, team2, venue = _extract_team_features(game_description)
    
    if team1 is None or team2 is None:
        return None
    
    # Calculate team strength from training data (cached)
    if _team_strength_cache is None and train_df is not None:
        team_strength = {}
        for _, row in train_df.iterrows():
            t1, t2, _ = _extract_team_features(row['event_description'])
            if t1: team_strength[t1] = team_strength.get(t1, 0) + row['amount_sum']
            if t2: team_strength[t2] = team_strength.get(t2, 0) + row['amount_sum']
        _team_strength_cache = {team: np.log1p(score) for team, score in team_strength.items()}
    
    # Calculate user features from training data (cached)
    if _user_features_cache is None and train_df is not None:
        user_stats = train_df.groupby('mask_id').agg(
            avg_bet_amount=('amount_sum', 'mean'),
            total_bets=('mask_id', 'size')
        ).to_dict('index')
        _user_features_cache = user_stats
    
    # Use cached values or empty defaults
    team_strength = _team_strength_cache or {}
    user_features = _user_features_cache or {}
    
    # Temporal features - use provided game_date
    days_since_ref = (pd.to_datetime(game_date) - ref_date).days
    game_dt = pd.to_datetime(game_date)
    
    # Team encoders (simplified - use index in team_classes)
    team_classes = maps.get('team_classes', {})
    team1_list = team_classes.get('team1', [])
    team2_list = team_classes.get('team2', [])
    
    team1_enc = team1_list.index(team1) if team1 in team1_list else 0
    team2_enc = team2_list.index(team2) if team2 in team2_list else 0
    
    venue_enc = 1 if venue == 'home' else 0
    
    # Playoff round logic
    playoff_round = 1
    if game_dt.month == 5 and game_dt.day > 15:
        playoff_round = 3 # Conference Finals
    elif game_dt.month == 5:
        playoff_round = 2 # Second Round
    elif game_dt.month == 6:
        playoff_round = 4 # NBA Finals
    
    # Convert to pair_key for pair_pop calculation
    pair_key = None
    if team1 and team2:
        pair_key = " | ".join(sorted([team1, team2]))
    
    features = [
        0.0,  # als_score - placeholder, will be filled in later
        0.0,  # pair_pop - would need pair popularity data
        0.0,  # item_pop - would need item popularity data  
        team1_enc,  # away_id (team1 encoded as away)
        team2_enc,  # home_id (team2 encoded as home)
        team_strength.get(team1, 0),  # away_strength
        team_strength.get(team2, 0),  # home_strength
        days_since_ref,  # days_since_ref
        game_dt.dayofweek,  # dow
        game_dt.month,  # month
        user_features.get(user_id, {}).get('total_bets', 0),  # user_total_bets
        user_features.get(user_id, {}).get('avg_bet_amount', 0) * user_features.get(user_id, {}).get('total_bets', 0)  # user_total_amt (approx)
    ]
    
    return features


# -----------------------------
# Scoring (Hybrid ALS + XGBoost)
# -----------------------------
def _score_one_user_hybrid(uid, candidates, user_f, item_f, user_to_idx, item_to_idx, 
                          maps, xgb_model, train_df, ref_date, game_date, pop):
    """
    Score neutral game candidates using hybrid ALS+XGBoost model.
    candidates: list of neutral games like "TeamA & TeamB"
    Returns: list of (neutral_game, best_score) sorted by score desc
    """
    # Cold user → popularity only 
    if uid not in user_to_idx:
        scores = []
        for game_neutral in candidates:
            orient1, orient2 = _expand_game_orientations(game_neutral)
            score1 = float(pop.get(orient1, 0.0))
            score2 = float(pop.get(orient2, 0.0))
            best_score = max(score1, score2)
            scores.append((game_neutral, best_score))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    # For known users, use hybrid model
    uidx = user_to_idx[uid]
    uvec = user_f[uidx]

    scores = []
    for game_neutral in candidates:
        # Get both orientations
        orient1, orient2 = _expand_game_orientations(game_neutral)
        
        # Try to score using hybrid model for both orientations
        hybrid_scores = []
        
        for orient in [orient1, orient2]:
            # Get ALS score
            als_score = 0.0
            if orient in item_to_idx:
                iidx = item_to_idx[orient]
                ivec = item_f[iidx]
                als_score = float(ivec @ uvec)
            else:
                # Use popularity for cold items
                als_score = float(pop.get(orient, 0.0))
            
            # Use XGBoost model if available
            if xgb_model is not None:
                try:
                    # Build features for this game
                    features = _build_hybrid_features(uid, orient, game_date, maps, train_df, ref_date)
                    if features is not None:
                        # Set ALS score in features (index 0)
                        features[0] = als_score
                        
                        # Create DMatrix with feature names (matching 02_new_model_approach.py)
                        feature_names = [
                            'als_score', 'pair_pop', 'item_pop',
                            'away_id', 'home_id', 'away_strength', 'home_strength',
                            'days_since_ref', 'dow', 'month',
                            'user_total_bets', 'user_total_amt'
                        ]
                        dmatrix = xgb.DMatrix(np.array([features]), feature_names=feature_names)
                        xgb_score = float(xgb_model.predict(dmatrix)[0])
                        hybrid_scores.append(xgb_score)
                    else:
                        # Fallback to ALS score if feature building fails
                        hybrid_scores.append(als_score)
                except Exception as e:
                    print(f"Warning: XGBoost prediction failed for {orient}: {e}")
                    # Fallback to ALS score
                    hybrid_scores.append(als_score)
            else:
                # No XGBoost model, use ALS score only
                hybrid_scores.append(als_score)
        
        # Take the best score from both orientations
        best_score = max(hybrid_scores) if hybrid_scores else 0.0
        scores.append((game_neutral, best_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# Backward compatibility: keep old function name but use hybrid scoring
def _score_one_user(uid, candidates, user_f, item_f, user_to_idx, item_to_idx, idx_to_item, pop):
    """Legacy function - redirects to hybrid scoring with minimal features."""
    # For backward compatibility, we'll use a simplified version
    # This is mainly for cases where we don't have full hybrid artifacts
    
    scores = []
    if uid not in user_to_idx:
        # Cold user fallback
        for game_neutral in candidates:
            orient1, orient2 = _expand_game_orientations(game_neutral)
            score1 = float(pop.get(orient1, 0.0))
            score2 = float(pop.get(orient2, 0.0))
            best_score = max(score1, score2)
            scores.append((game_neutral, best_score))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    uidx = user_to_idx[uid]
    uvec = user_f[uidx]

    for game_neutral in candidates:
        orient1, orient2 = _expand_game_orientations(game_neutral)
        
        # Score both orientations with ALS only
        scores_orientations = []
        for orient in [orient1, orient2]:
            if orient in item_to_idx:
                iidx = item_to_idx[orient]
                ivec = item_f[iidx]
                als_score = float(ivec @ uvec)
                scores_orientations.append(als_score)
            else:
                pop_score = float(pop.get(orient, 0.0))
                scores_orientations.append(pop_score)
        
        best_score = max(scores_orientations) if scores_orientations else 0.0
        scores.append((game_neutral, best_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate Top-K picks per date (one row per user-date).")
    ap.add_argument("--schedule", default="data/raw/playoff_schedule.csv",
                    help="Vendor schedule CSV (has Date, Game with 'TeamA & TeamB' format).")
    ap.add_argument("--artifacts", default="artifacts/hybrid",
                    help="Artifacts dir (hybrid).")
    ap.add_argument("--train", default="data/processed/df_train_rolled.csv",
                    help="Rolled train CSV (for popularity fallback if meta missing).")
    ap.add_argument("--outdir", default="reports/daily",
                    help="Output root for daily picks.")
    ap.add_argument("--topk", type=int, default=3, help="K for Top-K.")
    ap.add_argument("--include-validation-users", default="",
                    help="Optional df_validation_rolled.csv to include cold users.")
    args = ap.parse_args()

    # Load artifacts
    user_f, item_f, maps, meta, xgb_model = _load_artifacts(args.artifacts)

    # id maps
    user_to_idx = {str(k): int(v) for k, v in maps["user_to_idx"].items()}
    item_to_idx = {str(k): int(v) for k, v in maps["item_to_idx"].items()}
    # idx_to_item keys may be strings ("0","1",...), normalize to str(int(k))
    idx_to_item = {str(int(k)): v for k, v in maps["idx_to_item"].items()}

    # Popularity meta (fallback defaults if train_meta.json wasn’t saved)
    DEFAULT_HALF_LIFE = 30
    half_life = int(meta["half_life_days"]) if (meta and "half_life_days" in meta) else DEFAULT_HALF_LIFE
    cap_value = float(meta["cap_value"]) if (meta and "cap_value" in meta) else None
    ref_date  = meta.get("ref_date") if meta else None
    if ref_date is None:
        # use max date in the train file if not provided
        _tmp = pd.read_csv(args.train)
        ref_date = pd.to_datetime(_tmp["date"], errors="coerce").max()

    pop = _build_popularity(args.train,
                            half_life_days=half_life,
                            cap_value=cap_value,
                            ref_date=ref_date)

    # Load training data for feature building (needed for hybrid model)
    train_df = pd.read_csv(args.train, dtype={"mask_id": "string"})
    
    # Convert ref_date to pandas datetime
    if isinstance(ref_date, str):
        ref_date = pd.to_datetime(ref_date)

    # Load schedule + dates  
    sched = pd.read_csv(args.schedule, dtype={"Date":"string", "Game":"string"})
    sched["date"] = pd.to_datetime(sched["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    dates = sorted(sched["date"].dropna().unique().tolist())

    # Users: training users ∪ optional validation users
    users = set(user_to_idx.keys())
    if args.include_validation_users:
        val = pd.read_csv(args.include_validation_users, dtype={"mask_id":"string"})
        users.update(val["mask_id"].dropna().astype(str).unique().tolist())
    users = sorted(users)

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for ds in dates:
        cands = _candidates_for_date(sched, ds)
        if not cands:
            continue

        rows = []
        for uid in users:
            # Use hybrid scoring if XGBoost model is available
            if xgb_model is not None:
                scores = _score_one_user_hybrid(uid, cands, user_f, item_f,
                                               user_to_idx, item_to_idx, maps, 
                                               xgb_model, train_df, ref_date, ds, pop)
            else:
                # Fallback to ALS-only scoring
                scores = _score_one_user(uid, cands, user_f, item_f,
                                        user_to_idx, item_to_idx, idx_to_item, pop)
            top_list = []
            for r, (neutral_game, sc) in enumerate(scores[:args.topk], start=1):
                # Keep original format for display
                display_game = _neutral_display_format(neutral_game)
                pair_key = _generate_pair_key(neutral_game)
                top_list.append({
                    f"game_{r}": display_game, 
                    "score": float(sc),
                    "event_description": display_game,  # for compatibility with dashboard
                    "pair_key": pair_key  # for pair-key evaluation
                })
            rows.append({"mask_id": uid, "date": ds, "top_3": json.dumps(top_list, ensure_ascii=False)})

        out_dir = out_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "picks.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved picks → {out_path} ({len(rows)} rows)")
        total_rows += len(rows)

    print(f"Done. Total user-date rows written: {total_rows}")


if __name__ == "__main__":
    main()
