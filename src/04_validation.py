import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PICKS_ROOT = "reports/daily"
VAL_CSV    = "data/processed/df_validation_rolled.csv"
PRESEASON_SCHEDULE = "data/raw/playoff_schedule.csv"
ACTUAL_SCHEDULE = "data/processed/playoff_schedule_actual_enriched.csv"
K = 3  # Top-K 

# ---------- helpers ----------
AT_RE  = re.compile(r"@", flags=re.IGNORECASE)
VS_RE  = re.compile(r"\bvs\.?\b", flags=re.IGNORECASE)

def pair_key(s: str):
    if pd.isna(s): return None
    t = str(s).strip()
    if not t: return None
    if AT_RE.search(t):
        a, b = [x.strip() for x in AT_RE.split(t, maxsplit=1)]
    elif VS_RE.search(t):
        a, b = [x.strip() for x in VS_RE.split(t, maxsplit=1)]
    else:
        parts = re.split(r"\s*(?:&|and)\s*", t, flags=re.IGNORECASE)
        if len(parts) != 2: return None
        a, b = parts[0].strip(), parts[1].strip()
    return " | ".join(sorted([a, b]))

def load_picks_json(root: str) -> pd.DataFrame:
    rows = []
    for day_dir in sorted(Path(root).glob("*")):
        f = day_dir / "picks.csv"
        if not f.exists(): continue
        df = pd.read_csv(f, dtype={"mask_id":"string"})
        if "top_3" not in df.columns:
            # allow generic name top_k, fallback
            col = next((c for c in df.columns if c.startswith("top_")), None)
            if not col: continue
            df = df.rename(columns={col: "top_3"})
        # parse JSON list and explode to rows with rank/event_description
        for _, r in df.iterrows():
            try:
                items = json.loads(r["top_3"])
            except Exception:
                items = []
            # preserve order → rank = position
            for idx, it in enumerate(items[:K], start=1):
                # allow either {"game_1": "...", "score": x} or {"event_description": "...", "score": x}
                ev = it.get(f"game_{idx}") or it.get("event_description")
                # Extract pair_key directly from picks (new format) or compute it (legacy format)
                pair_key_val = it.get("pair_key")
                if not pair_key_val and ev:
                    pair_key_val = pair_key(ev)  # fallback to computing it
                
                rows.append({"mask_id": str(r["mask_id"]),
                             "date": r["date"],
                             "rank": idx,
                             "event_description": ev,
                             "score": it.get("score", np.nan),
                             "pair_key": pair_key_val})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out

def load_validation(val_csv: str) -> pd.DataFrame:
    val = pd.read_csv(val_csv, dtype={"mask_id":"string"})
    val["date"] = pd.to_datetime(val["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    keep = ["mask_id","date","event_description","amount_sum"]
    return val[[c for c in keep if c in val.columns]]

def dcg_at_k(rels):
    # rels: list/array of relevance values in rank order (length ≤ K)
    rels = np.asarray(rels, dtype=float)
    denom = np.log2(np.arange(2, len(rels)+2))
    return float(np.sum(rels / denom)) if rels.size else 0.0

def per_user_day_metrics(df_group, validation_data=None):
    # binary relevance and $ relevance in rank order
    rel_bin = (df_group.sort_values("rank")["amount_sum"] > 0).astype(int).tolist()
    rel_amt = df_group.sort_values("rank")["amount_sum"].tolist()
    hits = int(any(x > 0 for x in rel_amt))
    precision = sum(rel_bin) / K
    
    # recall needs ground-truth count that day
    # Use provided validation_data or fall back to global val
    uid, d = df_group["mask_id"].iloc[0], df_group["date"].iloc[0]
    if validation_data is not None:
        gt = validation_data[(validation_data["mask_id"] == uid) & (validation_data["date"] == d)]
    else:
        gt = val[(val["mask_id"] == uid) & (val["date"] == d)]
    gt_pos = int((gt["amount_sum"] > 0).sum())
    recall = (sum(rel_bin) / gt_pos) if gt_pos > 0 else np.nan
    wcap = sum(rel_amt)

    # nDCG (binary)
    dcg = dcg_at_k(rel_bin)
    ideal_bin = sorted([1]*gt_pos + [0]*max(0, K-gt_pos), reverse=True)[:K]
    idcg = dcg_at_k(ideal_bin)
    ndcg = (dcg / idcg) if idcg > 0 else np.nan

    # nDCG weighted by $ (amount)
    # ideal amounts: top-K amounts in ground truth
    ideal_amt = sorted(gt["amount_sum"].tolist(), reverse=True)[:K] if gt_pos > 0 else []
    idcg_amt = dcg_at_k(ideal_amt)
    ndcg_amt = (dcg_at_k(rel_amt) / idcg_amt) if idcg_amt > 0 else np.nan

    return pd.Series({
        "hit@k": hits,
        "precision@k": precision,
        "recall@k": recall,
        "w$@k": wcap,
        "nDCG@k": ndcg,
        "nDCG$@k": ndcg_amt
    })

def apply_metrics_with_validation(validation_data):
    """Create a wrapper function for per_user_day_metrics with specific validation data"""
    def wrapper(df_group):
        return per_user_day_metrics(df_group, validation_data)
    return wrapper

def simulate_two_week_periods(preseason_csv, actual_csv, validation_csv):
    """
    Simulate 2-week betting periods using pre-season schedule vs actual games.
    Returns list of 2-week periods with schedule drift analysis.
    """
    preseason = pd.read_csv(preseason_csv)
    actual = pd.read_csv(actual_csv)
    validation = pd.read_csv(validation_csv, dtype={"mask_id": "string"})
    
    # Parse dates
    preseason['date'] = pd.to_datetime(preseason['Date'], format='%d-%m-%Y', errors='coerce')
    actual['date'] = pd.to_datetime(actual['date'], errors='coerce')
    validation['date'] = pd.to_datetime(validation['date'], errors='coerce')
    
    # Get date range
    start_date = validation['date'].min()
    end_date = validation['date'].max()
    
    periods = []
    current_date = start_date
    
    while current_date + timedelta(days=14) <= end_date:
        period_end = current_date + timedelta(days=13)  # 14-day period
        
        # Games in this 2-week period
        preseason_games = preseason[
            (preseason['date'] >= current_date) & 
            (preseason['date'] <= period_end)
        ].copy()
        
        actual_games = actual[
            (actual['date'] >= current_date) & 
            (actual['date'] <= period_end)
        ].copy()
        
        validation_bets = validation[
            (validation['date'] >= current_date) & 
            (validation['date'] <= period_end)
        ].copy()
        
        if len(validation_bets) > 0:  # Only include periods with actual betting activity
            periods.append({
                'start_date': current_date,
                'end_date': period_end,
                'preseason_games': preseason_games,
                'actual_games': actual_games,
                'validation_bets': validation_bets
            })
        
        current_date += timedelta(days=7)  # Move by 1 week for overlapping periods
    
    return periods

def analyze_schedule_drift(preseason_games, actual_games):
    """
    Analyze schedule drift by comparing scheduled vs actually played games.
    Games in pre-season = total rows (scheduled slots)
    Games actually played = rows with non-empty event_description (games that happened)
    """
    # Count all scheduled game slots (rows) in preseason schedule
    total_preseason_games = len(preseason_games)
    
    # Count actually played games (rows with non-empty event_description)
    actual_played_games = 0
    for _, row in preseason_games.iterrows():
        if pd.notna(row['event_description']) and row['event_description'].strip():
            actual_played_games += 1
    
    # Games cancelled = scheduled but not played (empty event_description)
    games_cancelled = total_preseason_games - actual_played_games
    
    drift_metrics = {
        'total_preseason_games': total_preseason_games,
        'total_actual_games': actual_played_games,
        'games_in_both': actual_played_games,  # Games that were scheduled and played
        'games_cancelled_or_moved': games_cancelled,
        'schedule_accuracy': actual_played_games / total_preseason_games if total_preseason_games > 0 else 0,
        'cancelled_games': [],
        'added_games': []
    }
    
    return drift_metrics

# ---------- load and simulate ----------
print("=== NBA PLAYOFF BETTING SIMULATION ===")
print("Simulating 2-week betting periods...")

periods = simulate_two_week_periods(PRESEASON_SCHEDULE, ACTUAL_SCHEDULE, VAL_CSV)
print(f"Generated {len(periods)} two-week simulation periods")

picks_exploded = load_picks_json(PICKS_ROOT)
val = load_validation(VAL_CSV)

print(f"Picks rows (exploded): {len(picks_exploded):,}")
print(f"Validation rows: {len(val):,}")

# ---------- SIMULATION-BASED VALIDATION ----------
# Analyze each 2-week period separately
period_results = []
all_drift_metrics = []

for i, period in enumerate(periods):
    print(f"\n=== PERIOD {i+1}: {period['start_date'].strftime('%Y-%m-%d')} to {period['end_date'].strftime('%Y-%m-%d')} ===")
    
    # Analyze schedule drift for this period
    drift = analyze_schedule_drift(period['preseason_games'], period['actual_games'])
    all_drift_metrics.append(drift)
    
    print(f"Schedule Accuracy: {drift['schedule_accuracy']:.1%}")
    print(f"Games in pre-season: {drift['total_preseason_games']}")
    print(f"Games actually played: {drift['total_actual_games']}")
    print(f"Games cancelled/moved: {drift['games_cancelled_or_moved']}")
    
    # Get picks for this period
    period_start_str = period['start_date'].strftime('%Y-%m-%d')
    period_end_str = period['end_date'].strftime('%Y-%m-%d')
    
    period_picks = picks_exploded[
        (picks_exploded['date'] >= period_start_str) & 
        (picks_exploded['date'] <= period_end_str) &
        (picks_exploded['rank'] <= K)
    ].copy()
    
    period_validation = period['validation_bets'].copy()
    period_validation['date'] = period_validation['date'].dt.strftime('%Y-%m-%d')
    
    if len(period_picks) == 0 or len(period_validation) == 0:
        print("No picks or validation data for this period")
        continue
    
    # Evaluate recommendations vs actual betting using pair_key for fairness
    period_validation_pair = period_validation.copy()
    period_validation_pair["pair_key"] = period_validation_pair["event_description"].apply(pair_key)
    
    merged_period = period_picks.merge(period_validation_pair, on=["mask_id","date","pair_key"], how="left", suffixes=("_pick","_val"))
    merged_period["amount_sum"] = merged_period["amount_sum"].fillna(0.0)
    
    # Calculate metrics for this period
    if len(merged_period) > 0:
        period_metrics = merged_period.groupby(["mask_id","date"], dropna=False).apply(apply_metrics_with_validation(period_validation_pair)).reset_index()
        
        period_summary = {
            'period': i+1,
            'start_date': period_start_str,
            'end_date': period_end_str,
            'hit_rate': period_metrics["hit@k"].mean(),
            'precision': period_metrics["precision@k"].mean(),
            'recall': period_metrics["recall@k"].mean(skipna=True),
            'dollar_capture': period_metrics["w$@k"].mean(),
            'ndcg': period_metrics["nDCG@k"].mean(skipna=True),
            'users_evaluated': len(period_metrics),
            'total_picks': len(period_picks),
            'total_bets': len(period_validation),
            'schedule_accuracy': drift['schedule_accuracy']
        }
        
        period_results.append(period_summary)
        
        print(f"Hit@{K}: {period_summary['hit_rate']:.3f}")
        print(f"Precision@{K}: {period_summary['precision']:.3f}")
        print(f"Users evaluated: {period_summary['users_evaluated']}")

# ---------- OVERALL SIMULATION RESULTS ----------
print(f"\n=== OVERALL SIMULATION RESULTS ===")
if period_results:
    overall_hit_rate = np.mean([p['hit_rate'] for p in period_results])
    overall_precision = np.mean([p['precision'] for p in period_results])
    overall_schedule_accuracy = np.mean([p['schedule_accuracy'] for p in period_results])
    
    print(f"Average Hit@{K} across periods: {overall_hit_rate:.3f}")
    print(f"Average Precision@{K} across periods: {overall_precision:.3f}")
    print(f"Average Schedule Accuracy: {overall_schedule_accuracy:.1%}")
    print(f"Total periods evaluated: {len(period_results)}")

# ---------- PRIMARY EVALUATION: Pair-Key Based (Fair) ----------
print(f"\n=== PRIMARY EVALUATION: Pair-Key Based (Full Dataset) ===")
pk = picks_exploded[picks_exploded["rank"] <= K].copy()

# Prepare validation data with pair_key
val_pair = val.copy()
val_pair["pair_key"] = val_pair["event_description"].apply(pair_key)

# Use pair_key for fair evaluation (orientation-agnostic)
merged_pair = pk.merge(val_pair, on=["mask_id","date","pair_key"], how="left", suffixes=("_pick","_val"))
merged_pair["amount_sum"] = merged_pair["amount_sum"].fillna(0.0)
by_pd_pair = merged_pair.groupby(["mask_id","date"], dropna=False).apply(apply_metrics_with_validation(val_pair)).reset_index()

macro_pair = {
    f"Hit@{K}": by_pd_pair["hit@k"].mean(),
    f"Precision@{K}": by_pd_pair["precision@k"].mean(),
    f"Recall@{K}": by_pd_pair["recall@k"].mean(skipna=True),
    f"w$@{K}": by_pd_pair["w$@k"].mean(),
    f"nDCG@{K}": by_pd_pair["nDCG@k"].mean(skipna=True),
    f"nDCG$@{K}": by_pd_pair["nDCG$@k"].mean(skipna=True),
}
print("Macro (pair_key - PRIMARY):", macro_pair)

# ---------- STRICT EVALUATION (for comparison) ----------
print(f"\n=== STRICT EVALUATION (Exact Match - for comparison) ===")
merged_strict = pk.merge(val, on=["mask_id","date","event_description"], how="left")
merged_strict["amount_sum"] = merged_strict["amount_sum"].fillna(0.0)
by_pd_strict = merged_strict.groupby(["mask_id","date"], dropna=False).apply(apply_metrics_with_validation(val)).reset_index()

macro_strict = {
    f"Hit@{K}": by_pd_strict["hit@k"].mean(),
    f"Precision@{K}": by_pd_strict["precision@k"].mean(),
    f"Recall@{K}": by_pd_strict["recall@k"].mean(skipna=True),
    f"w$@{K}": by_pd_strict["w$@k"].mean(),
    f"nDCG@{K}": by_pd_strict["nDCG@k"].mean(skipna=True),
    f"nDCG$@{K}": by_pd_strict["nDCG$@k"].mean(skipna=True),
}
print("Macro (strict - COMPARISON):", macro_strict)

# ---------- save results ----------
Path("reports/metrics").mkdir(parents=True, exist_ok=True)
by_pd_strict.to_csv("reports/metrics/metrics_k3_strict.csv", index=False, encoding="utf-8")
by_pd_pair.to_csv("reports/metrics/metrics_k3_pairkey.csv", index=False, encoding="utf-8")

# Save simulation results
if period_results:
    pd.DataFrame(period_results).to_csv("reports/metrics/simulation_periods.csv", index=False, encoding="utf-8")
    print(f"\nSaved simulation results to reports/metrics/simulation_periods.csv")

# Save schedule drift analysis
if all_drift_metrics:
    drift_df = pd.DataFrame(all_drift_metrics)
    drift_df.to_csv("reports/metrics/schedule_drift_analysis.csv", index=False, encoding="utf-8")
    print(f"Saved schedule drift analysis to reports/metrics/schedule_drift_analysis.csv")

print(f"\n=== SIMULATION COMPLETE ===")
