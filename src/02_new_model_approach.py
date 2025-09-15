"""
Hybrid ALS + XGBoost trainer (improved)
- ALS learns orientation-specific preferences ("A @ B" vs "B @ A")
- Builds pair-level vectors for neutral "A & B" days
- Uses recency-decayed amounts for both ALS and feature engineering
- XGBoost ranks with ALS score + pair popularity + temporal/team/user features
Artifacts written to: --artifacts dir
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import implicit
import xgboost as xgb

# --------------------------
# Text parsing & normalization
# --------------------------

import re
AT_RE  = re.compile(r"\s*@\s*", flags=re.IGNORECASE)
VS_RE  = re.compile(r"\s*vs\.?\s*", flags=re.IGNORECASE)
AMP_RE = re.compile(r"\s*&\s*|\s*and\s*", flags=re.IGNORECASE)

TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "L.A. Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
    "OKC": "Oklahoma City Thunder",
    "OKC Thunder": "Oklahoma City Thunder",
    "Portland Trail Blazer": "Portland Trail Blazers",
    "Portland Trailblazers": "Portland Trail Blazers",
    "GS Warriors": "Golden State Warriors",
    "G.S. Warriors": "Golden State Warriors",
    "NY Knicks": "New York Knicks",
    "N.Y. Knicks": "New York Knicks",
    "Timberwolves": "Minnesota Timberwolves",
}

def _norm_team(name: str) -> str:
    n = str(name).strip()
    return TEAM_ALIASES.get(n, n)

def extract_teams(text: str) -> Tuple[str, str, str]:
    """
    Returns (away, home, sep) where sep in {'@','vs','&',None}.
    'A @ B' -> (A,B,'@')
    'A vs B' -> (B,A,'vs') [canonical away@home is B@A]
    'A & B' -> (None,None,'&')  # unknown orientation
    """
    if not isinstance(text, str):
        return None, None, None
    s = text.strip()
    if not s:
        return None, None, None

    if AT_RE.search(s):
        a, b = [t.strip() for t in AT_RE.split(s, maxsplit=1)]
        return _norm_team(a), _norm_team(b), "@"
    if VS_RE.search(s):
        a, b = [t.strip() for t in VS_RE.split(s, maxsplit=1)]
        # "A vs B": home=A, away=B → away@home = B@A
        return _norm_team(b), _norm_team(a), "vs"
    if AMP_RE.search(s):
        a, b = [t.strip() for t in AMP_RE.split(s, maxsplit=1)]
        return _norm_team(a), _norm_team(b), "&"
    return None, None, None

def to_pair_key(text: str) -> str:
    a, h, sep = extract_teams(text)
    if sep == "&":
        # unknown orientation; still have two names
        if not a or not h:
            return None
        return " | ".join(sorted([_norm_team(a), _norm_team(h)]))
    if a and h:
        return " | ".join(sorted([_norm_team(a), _norm_team(h)]))
    return None

def to_event_at(text: str) -> str:
    """Return canonical 'away @ home' if possible, else None."""
    a, h, sep = extract_teams(text)
    if a and h:
        if sep == "@":
            return f"{a} @ {h}"
        if sep == "vs":  # B vs A → away@home = B @ A already handled in extract_teams
            return f"{a} @ {h}"
    return None


# --------------------------
# Core Model
# --------------------------

class HybridNBARecommender:
    def __init__(self, als_factors=32, als_reg=0.08, als_iters=20):
        self.als_factors = als_factors
        self.als_reg = als_reg
        self.als_iters = als_iters

        self.als_model = None
        self.user_to_idx: Dict[str,int] = {}
        self.item_to_idx: Dict[str,int] = {}
        self.idx_to_item: Dict[int,str] = {}

        # Pair-level (neutral) vectors
        self.pair_key_to_idx: Dict[str,int] = {}
        self.pair_vectors: np.ndarray | None = None

        # XGB
        self.xgb_model: xgb.XGBRegressor | None = None

        # Meta for inference
        self.meta = {}

    # ---------- weighting helpers ----------
    @staticmethod
    def _add_recency_weight(df: pd.DataFrame, half_life_days: int, cap_quantile: float) -> Tuple[pd.DataFrame, pd.Timestamp, float]:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cap = df["amount_sum"].quantile(cap_quantile)
        df["amt_capped"] = df["amount_sum"].clip(upper=cap)
        ref = df["date"].max()
        days_since = (ref - df["date"]).dt.days.clip(lower=0)
        decay = np.power(0.5, days_since / float(half_life_days))
        df["weight"] = df["amt_capped"] * decay
        return df, ref, float(cap)

    # ---------- ALS ----------
    def train_als(self, train_df: pd.DataFrame, half_life_days=30, cap_quantile=0.99):
        print("=== Stage 1: ALS (orientation-specific items) ===")
        dfw, ref, cap_value = self._add_recency_weight(train_df, half_life_days, cap_quantile)

        # Use canonical "away @ home" if available
        ev_at = train_df["event_description"].astype(str).apply(to_event_at)
        dfw = dfw.assign(event_at=ev_at)
        dfw["event_used"] = dfw["event_at"].fillna(train_df["event_description"])
        dfw = dfw.dropna(subset=["mask_id","event_used","weight"])

        interactions = (dfw.groupby(["mask_id","event_used"], as_index=False)["weight"]
                          .sum().rename(columns={"weight":"w"}))

        users_cat = interactions["mask_id"].astype("category")
        items_cat = interactions["event_used"].astype("category")
        u_idx = users_cat.cat.codes.to_numpy(np.int32)
        i_idx = items_cat.cat.codes.to_numpy(np.int32)
        w     = interactions["w"].to_numpy(np.float32)

        mat = coo_matrix((w, (u_idx, i_idx)),
                         shape=(len(users_cat.cat.categories), len(items_cat.cat.categories))).tocsr()

        self.als_model = implicit.als.AlternatingLeastSquares(
            factors=self.als_factors,
            regularization=self.als_reg,
            iterations=self.als_iters
        )
        self.als_model.fit(mat)

        self.user_to_idx = {str(u): int(i) for i, u in enumerate(users_cat.cat.categories)}
        self.item_to_idx = {str(it): int(i) for i, it in enumerate(items_cat.cat.categories)}
        self.idx_to_item = {int(i): str(it) for i, it in enumerate(items_cat.cat.categories)}

        # Save meta
        self.meta.update({
            "ref_date": str(ref.date()),
            "half_life_days": int(half_life_days),
            "cap_value": cap_value
        })
        print(f"ALS trained: {len(self.user_to_idx)} users, {len(self.item_to_idx)} items")

    # ---------- Pair vectors ----------
    def build_pair_vectors(self):
        """
        Build one vector per pair_key by averaging the two orientation vectors (A@B and B@A) if both exist.
        If only one exists, use it as-is.
        """
        print("=== Building pair-level vectors for neutral 'A & B' ===")
        item_vecs = self.als_model.item_factors
        buckets: Dict[str, List[np.ndarray]] = {}

        for text, idx in self.item_to_idx.items():
            pk = to_pair_key(text)
            if not pk:
                continue
            buckets.setdefault(pk, []).append(item_vecs[idx])

        keys = sorted(buckets.keys())
        vecs = []
        for pk in keys:
            arr = np.vstack(buckets[pk])
            if arr.shape[0] == 1:
                vecs.append(arr[0])
            else:
                vecs.append(arr.mean(axis=0))  # average; you could try np.max for max-pool
        self.pair_key_to_idx = {k:i for i,k in enumerate(keys)}
        self.pair_vectors = np.vstack(vecs) if vecs else np.zeros((0, self.als_model.factors), dtype=np.float32)

        print(f"Pair vectors built: {len(self.pair_key_to_idx)}")

    # ---------- Popularity ----------
    @staticmethod
    def build_popularity(train_df: pd.DataFrame, half_life_days: int, cap_value: float, ref_date: str | None):
        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amt_capped"] = df["amount_sum"].clip(upper=cap_value)
        ref = pd.to_datetime(ref_date) if ref_date else df["date"].max()
        days_since = (ref - df["date"]).dt.days.clip(lower=0)
        decay = np.power(0.5, days_since / float(half_life_days))
        df["w"] = df["amt_capped"] * decay

        # Popularity for orientation items
        pop_item = (df.groupby("event_description", as_index=False)["w"].sum()
                      .rename(columns={"w":"pop"}).set_index("event_description")["pop"])

        # Pair popularity = sum of both orientations
        df["pair_key"] = df["event_description"].astype(str).apply(to_pair_key)
        pop_pair = (df.dropna(subset=["pair_key"]).groupby("pair_key", as_index=False)["w"].sum()
                      .rename(columns={"w":"pop"}).set_index("pair_key")["pop"])

        return pop_item, pop_pair

    # ---------- XGB ----------
    def _feature_frame(self, df: pd.DataFrame, ref_date: pd.Timestamp,
                       pop_item, pop_pair) -> pd.DataFrame:
        """Build per (mask_id,event,date) training rows with engineered features."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["event_at"] = df["event_description"].astype(str).apply(to_event_at)
        df["event_used"] = df["event_at"].fillna(df["event_description"])
        df["pair_key"] = df["event_description"].astype(str).apply(to_pair_key)

        # ALS score (dot with item vec)
        def als_score_row(row):
            uid = str(row["mask_id"])
            it  = row["event_used"]
            if uid in self.user_to_idx and it in self.item_to_idx:
                uvec = self.als_model.user_factors[self.user_to_idx[uid]]
                ivec = self.als_model.item_factors[self.item_to_idx[it]]
                return float(ivec @ uvec)
            return 0.0

        df["als_score"] = df.apply(als_score_row, axis=1)

        # Pair popularity & item popularity
        df["pair_pop"] = df["pair_key"].map(pop_pair).fillna(0.0)
        df["item_pop"] = df["event_used"].map(pop_item).fillna(0.0)

        # Temporal
        days_since_ref = (df["date"] - ref_date).dt.days
        df["days_since_ref"] = days_since_ref.clip(lower=-365, upper=365).fillna(0).astype(int)
        df["dow"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        # Teams (encoded as categorical integers via vocabulary)
        def teams_for_row(row):
            a, h, sep = extract_teams(row["event_used"])
            return a, h

        tmp = df["event_used"].apply(lambda s: pd.Series(teams_for_row(pd.Series({"event_used": s}))))
        tmp.columns = ["away_team","home_team"]
        df = pd.concat([df, tmp], axis=1)

        # Map teams to ids from observed vocabulary
        teams = pd.unique(pd.concat([df["away_team"], df["home_team"]]).dropna())
        team_to_id = {t:i+1 for i,t in enumerate(sorted(teams))}  # 0 reserved for unknown
        df["away_id"] = df["away_team"].map(team_to_id).fillna(0).astype(int)
        df["home_id"] = df["home_team"].map(team_to_id).fillna(0).astype(int)

        # Simple team strength from decayed totals on training set (re-use popularity by team)
        # Build from df itself (decayed). Count team exposures.
        team_strength = {}
        for _, r in df.dropna(subset=["away_team","home_team"]).iterrows():
            amt = float(r.get("amount_sum", 0.0))
            a = r["away_team"]; h = r["home_team"]
            team_strength[a] = team_strength.get(a, 0.0) + amt
            team_strength[h] = team_strength.get(h, 0.0) + amt
        df["away_strength"] = df["away_team"].map(team_strength).fillna(0.0)
        df["home_strength"] = df["home_team"].map(team_strength).fillna(0.0)

        # User features (decayed)
        # Aggregate decayed amount per user and bet count
        decayed_user = (df.groupby("mask_id", as_index=False)
                          .agg(user_total_bets=("mask_id","size"),
                               user_total_amt=("amount_sum","sum")))
        u_feat = dict(zip(decayed_user["mask_id"], zip(decayed_user["user_total_bets"],
                                                       decayed_user["user_total_amt"])))
        df["user_total_bets"] = df["mask_id"].map(lambda u: u_feat.get(u, (0,0.0))[0]).fillna(0)
        df["user_total_amt"]  = df["mask_id"].map(lambda u: u_feat.get(u, (0,0.0))[1]).fillna(0.0)

        # Label
        y = df["amount_sum"].astype(float)

        feature_cols = [
            "als_score","pair_pop","item_pop",
            "away_id","home_id","away_strength","home_strength",
            "days_since_ref","dow","month",
            "user_total_bets","user_total_amt",
        ]
        X = df[feature_cols].fillna(0.0)

        return X, y, feature_cols, team_to_id

    def train_xgb(self, train_df: pd.DataFrame):
        print("=== Stage 2: XGBoost ranking/regression ===")

        ref_date = pd.to_datetime(self.meta["ref_date"])
        pop_item, pop_pair = self.build_popularity(train_df,
                                                   self.meta["half_life_days"],
                                                   self.meta["cap_value"],
                                                   self.meta["ref_date"])

        # Time-based split (last 20% dates as validation)
        dfd = train_df.copy()
        dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
        min_d, max_d = dfd["date"].min(), dfd["date"].max()
        cutoff = min_d + (max_d - min_d) * 0.8
        train_split = dfd[dfd["date"] <= cutoff].copy()
        valid_split = dfd[dfd["date"] >  cutoff].copy()

        Xtr, ytr, feature_cols, team_vocab = self._feature_frame(train_split, ref_date, pop_item, pop_pair)
        Xva, yva, _, _                    = self._feature_frame(valid_split, ref_date, pop_item, pop_pair)

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=50
        )

        self.xgb_model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False
        )

        best_it = int(self.xgb_model.get_booster().best_iteration or 0)
        print(f"XGB best_iteration: {best_it}, Val RMSE: {self.xgb_model.get_booster().best_score:.4f}")

        # keep meta
        self.meta.update({
            "xgb_feature_cols": feature_cols,
            "team_vocab": team_vocab,
        })

    # ---------- Save ----------
    def save(self, outdir: str):
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        # ALS factors
        np.save(out / "user_factors.npy", self.als_model.user_factors)
        np.save(out / "item_factors.npy", self.als_model.item_factors)

        # Pair vectors
        if self.pair_vectors is not None:
            np.save(out / "pair_vectors.npy", self.pair_vectors)
        with open(out / "pair_maps.json", "w") as f:
            json.dump({"pair_key_to_idx": self.pair_key_to_idx}, f)

        # ID maps
        with open(out / "id_maps.json", "w") as f:
            json.dump({
                "user_to_idx": self.user_to_idx,
                "item_to_idx": self.item_to_idx,
                "idx_to_item": {str(k): v for k,v in self.idx_to_item.items()}
            }, f)

        # XGB
        if self.xgb_model:
            self.xgb_model.save_model(str(out / "xgb_model.json"))

        # Meta
        with open(out / "train_meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

        print(f"Saved artifacts to: {out.resolve()}")


# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Hybrid ALS + XGBoost NBA Recommender (improved)")
    ap.add_argument("--train", default="data/processed/df_train_rolled.csv")
    ap.add_argument("--artifacts", default="artifacts/hybrid")
    ap.add_argument("--half-life-days", type=int, default=30)
    ap.add_argument("--cap-quantile", type=float, default=0.99)
    ap.add_argument("--factors", type=int, default=32)
    ap.add_argument("--reg", type=float, default=0.08)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.train, dtype={"mask_id":"string", "event_description":"string"})
    # Normalize text a bit upfront (helps item identity)
    df["event_description"] = df["event_description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    print(f"Training rows: {len(df):,} | users={df['mask_id'].nunique():,} | items(raw)={df['event_description'].nunique():,}")

    model = HybridNBARecommender(
        als_factors=args.factors,
        als_reg=args.reg,
        als_iters=args.iters
    )

    # Train ALS
    model.train_als(df, half_life_days=args.half_life_days,
                       cap_quantile=args.cap_quantile)

    # Build pair-level vectors for neutral schedules
    model.build_pair_vectors()

    # Train XGB on engineered features
    model.train_xgb(df)

    # Save
    model.save(args.artifacts)


if __name__ == "__main__":
    main()
