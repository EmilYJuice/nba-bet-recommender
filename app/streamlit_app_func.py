# ------------------------------------------------------------
# Backend Functions
# ------------------------------------------------------------
import json, ast
import os
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from datetime import timedelta


# Load environment variables from .env file
load_dotenv()

REPO = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "metrics_strict": REPO / "reports/metrics/metrics_k3_strict.csv",
    "metrics_pair":   REPO / "reports/metrics/metrics_k3_pairkey.csv",
    "picks_root":     REPO / "reports/daily",
    "sim_vendor":     REPO / "reports/metrics/schedule_drift_daily.csv",
    "sim_actual":     REPO / "reports/metrics/simulation_periods.csv",
    "sim_drift":      REPO / "reports/metrics/schedule_drift_analysis.csv",
    "user_history":   REPO / "data/processed/df_train_rolled.csv",
    "idmaps": [
        REPO / "artifacts/hybrid/hybrid_artifacts.json",
        REPO / "artifacts/als/id_maps.json",
    ],
}

# ----------------- Data Loading & Utils --------------------------
@st.cache_data(show_spinner=False)
def read_csv_cached(path: Path, dtype=None) -> Optional[pd.DataFrame]:
    """Read CSV file with caching."""
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def read_json_cached(path: Path) -> Optional[dict]:
    """Read JSON file with caching."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def ensure_datestr(s: pd.Series) -> pd.Series:
    """Convert series to date string format."""
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")

def safe_mean(s: Optional[pd.Series]) -> float:
    """Safely calculate mean, handling empty series."""
    if s is None or s.empty: 
        return float("nan")
    return float(s.mean())

def infer_k_value(df: Optional[pd.DataFrame], fallback=3) -> int:
    """Infer k value from dataframe columns."""
    if df is None: 
        return fallback
    for c in df.columns:
        if "@k" in c.lower():
            try: 
                return int(c.lower().split("@")[-1].split("_")[0])
            except: 
                pass
    return fallback

def parse_top_json(x):
    """Parse top recommendations JSON string."""
    if pd.isna(x): 
        return []
    try:
        v = json.loads(x)
        return v if isinstance(v, list) else []
    except Exception:
        try:
            v = ast.literal_eval(str(x))
            return v if isinstance(v, list) else []
        except Exception:
            return []

@st.cache_data(show_spinner=False)
def load_idmap_users(paths) -> Optional[set]:
    """Load user ID mappings from artifacts."""
    for p in paths:
        if Path(p).exists():
            d = read_json_cached(Path(p))
            if d and "user_to_idx" in d:
                return set(map(str, d["user_to_idx"].keys()))
    return None

def load_all_data(metrics_path, pair_path, user_history_path, sim_vendor, sim_actual, sim_drift):
    """Load all required data files."""
    # Main metrics
    metrics = read_csv_cached(Path(metrics_path), dtype={"mask_id":"string"})
    pair = read_csv_cached(Path(pair_path), dtype={"mask_id":"string"}) if Path(pair_path).exists() else None
    user_betting_history = read_csv_cached(Path(user_history_path), dtype={"mask_id":"string"}) if Path(user_history_path).exists() else None

    # Ensure date formatting
    if metrics is not None and "date" in metrics.columns:
        metrics["date"] = ensure_datestr(metrics["date"])
    if pair is not None and "date" in pair.columns:
        pair["date"] = ensure_datestr(pair["date"])
    if user_betting_history is not None and "date" in user_betting_history.columns:
        user_betting_history["date"] = ensure_datestr(user_betting_history["date"])

    # Simulation data
    sim_v = read_csv_cached(Path(sim_vendor), dtype={"mask_id":"string"})
    sim_a = read_csv_cached(Path(sim_actual), dtype={"mask_id":"string"})
    sim_d = read_csv_cached(Path(sim_drift))

    # Warm/cold user mapping
    warm_user_set = load_idmap_users(DEFAULTS["idmaps"])

    return {
        'metrics': metrics,
        'pair': pair,
        'user_betting_history': user_betting_history,
        'sim_v': sim_v,
        'sim_a': sim_a,
        'sim_d': sim_d,
        'warm_user_set': warm_user_set
    }

# --------------- KPI Calculations ----------------------
def calculate_kpi(df: Optional[pd.DataFrame], col: str, fmt="{:.3f}", missing="—"):
    """Calculate KPI metrics for dashboard."""
    if df is None or col not in (df.columns if hasattr(df, "columns") else []):
        return missing
    v = safe_mean(df[col])
    return missing if pd.isna(v) else fmt.format(v)

# --------------- User Profile & Analysis ----------------------
def derive_user_profile(u_hist: pd.DataFrame) -> Dict[str, str]:
    """Generate lightweight profile heuristics for captioning."""
    prof = {}
    if u_hist is None or u_hist.empty:
        return prof
    
    # Venue bias via string contains
    ed = u_hist.get("event_description", pd.Series([], dtype=str)).astype(str).str.lower()
    prof["likes_home"] = "yes" if (ed.str.contains(" vs ").mean() > 0.55) else "no"  # vs ≈ home-left
    
    # Day-of-week preference
    if "date" in u_hist.columns:
        dow = pd.to_datetime(u_hist["date"], errors="coerce").dt.dayofweek.value_counts(normalize=True)
        if not dow.empty:
            fav = int(dow.idxmax())
            prof["fav_dow"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][fav]
    return prof

def get_user_data(mask_id: str, metrics: pd.DataFrame, user_betting_history: pd.DataFrame):
    """Get user-specific data for display."""
    # Get user's actual betting history from training data
    user_history = user_betting_history[user_betting_history["mask_id"]==mask_id].copy() if user_betting_history is not None else pd.DataFrame()
    
    # Get user's recommendation performance from metrics
    user_metrics = metrics[metrics["mask_id"]==mask_id].copy() if metrics is not None else pd.DataFrame()
    
    return user_history, user_metrics

def get_daily_performance_data(metrics: pd.DataFrame):
    """Process daily performance metrics."""
    if metrics is None or "date" not in metrics.columns:
        return None
    
    m = metrics.copy()
    by_date = (m.groupby("date", as_index=False)
            .agg(hit=("hit@k","mean"),
                 precision=("precision@k","mean"),
                 recall=("recall@k","mean"),
                 wcap=("w$@k","mean"),
                 ndcg=("nDCG@k","mean"),
                 ndcg_money=("nDCG$@k","mean"),
                 n_user_days=("mask_id","nunique"))
            .sort_values("date"))
    return by_date

def get_picks_data(picks_root: str, date_str: str, mask_id: str):
    """Get picks data for specific date and user."""
    root = Path(picks_root)
    if not root.exists():
        return None, []
    
    f = root / date_str / "picks.csv"
    if f.exists():
        dfp = pd.read_csv(f, dtype={"mask_id":"string"})
        row = dfp.loc[dfp["mask_id"]==mask_id]
        if not row.empty:
            items = parse_top_json(row["top_3"].iloc[0]) if "top_3" in row.columns else []
            return dfp, items
    return None, []

# --------------- AI Integration ----------------------
def generate_ai_recommendation(mask_id: str, top_picks: List[dict], user_history: pd.DataFrame, api_key: str = None) -> str:
    """Generate personalized betting recommendation using OpenAI GPT-4o"""
    print(f"[DEBUG] AI function called for user: {mask_id}")
    
    final_api_key = (
        api_key or 
        os.getenv('OPENAI_API_KEY') or 
        st.secrets.get('OPENAI_API_KEY', None)
    )
    
    print(f"[DEBUG] API key available: {'Yes' if final_api_key else 'No'}")
    if final_api_key:
        print(f"[DEBUG] API key starts with: {final_api_key[:10]}...")
    
    if not final_api_key:
        print("[DEBUG] No API key found - returning error message")
        return "AI recommendations unavailable - Please enter it in the sidebar."
    
    try:
        print(f"[DEBUG] Processing user history - rows: {len(user_history)}")
        print(f"[DEBUG] User history columns: {list(user_history.columns) if not user_history.empty else 'No data'}")
        
        # Prepare user betting history summary
        history_summary = "No betting history available"
        if not user_history.empty:
            total_bets = len(user_history)
            print(f"[DEBUG] User has {total_bets} betting records")
            
            # Game-specific wagering patterns
            game_betting_info = []
            if 'event_description' in user_history.columns and 'amount_sum' in user_history.columns:
                # Group by game and sum wagers (using amount_sum from df_train_rolled)
                game_wagers = (user_history.groupby('event_description')['amount_sum']
                              .agg(['sum', 'count', 'mean'])
                              .sort_values('sum', ascending=False)
                              .head(5))
                
                for game, stats in game_wagers.iterrows():
                    game_betting_info.append(
                        f"  • {game}: ${stats['sum']:.2f} across {int(stats['count'])} bets (avg: ${stats['mean']:.2f})"
                    )
            elif 'event_description' in user_history.columns:
                # Just show frequency if no wager amounts available
                game_counts = user_history['event_description'].value_counts().head(5)
                for game, count in game_counts.items():
                    game_betting_info.append(f"  • {game}: {count} bets")
            
            games_text = "\n".join(game_betting_info) if game_betting_info else "  • No specific game data available"
            
            history_summary = f"""
            - Total betting instances: {total_bets}
            - Top games by wagering activity:{games_text}
            """
        
        # Prepare today's recommendations
        print(f"[DEBUG] Processing {len(top_picks)} top picks")
        picks_text = ""
        if top_picks:
            picks_list = []
            for i, pick in enumerate(top_picks[:3]):
                # Extract game name from game_X key or event_description
                game_name = pick.get('event_description') or next((v for k,v in pick.items() if k.startswith("game_")), "Unknown game")
                confidence = pick.get('score', 'N/A')
                picks_list.append(f"{i+1}. {game_name} (Confidence: {confidence})")
            picks_text = "\n".join(picks_list)
            print(f"[DEBUG] Picks text prepared: {picks_text[:100]}...")
        else:
            picks_text = "No recommendations available for today"
            print("[DEBUG] No picks available for today")
        
        # Create the prompt
        prompt = f"""
        You are an expert NBA betting advisor. Based on a player's game-specific betting history and today's recommendations, create an encouraging and personalized message to motivate them to consider today's picks.

        Player ID: {mask_id}

        Player's Game-Specific Betting History:
        {history_summary}

        Today's AI Recommendations:
        {picks_text}

        Guidelines:
        1. Be encouraging and positive, can use emoji.
        2. Reference their specific game preferences and betting patterns from history
        3. Connect today's recommendations to similar teams/matchups they've bet on before
        4. Highlight why today's picks might appeal based on their historical game choices
        5. Keep it conversational and engaging (maximum 2 sentences)
        6. Don't guarantee wins, but build excitement about the opportunity
        7. If they have history with similar teams in today's picks, mention that connection

        Generate a personalized recommendation message:
        """

        # Call OpenAI API
        print("[DEBUG] Calling OpenAI API...")
        client = openai.OpenAI(api_key=final_api_key)
        
        print(f"[DEBUG] Prompt length: {len(prompt)} characters")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a friendly NBA betting advisor who creates personalized, encouraging recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        print(f"[DEBUG] AI response received - length: {len(ai_response)} characters")
        print(f"[DEBUG] AI response preview: {ai_response[:100]}...")
        
        return ai_response
        
    except Exception as e:
        print(f"[DEBUG] ERROR in AI function: {str(e)}")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        return f"AI recommendation temporarily unavailable: {str(e)}"

def generate_ai_caption(mask_id: str, top3: List[dict], profile: Dict[str,str]) -> str:
    """Generate deterministic, deploy-safe caption as fallback."""
    if not top3:
        return "No personalized picks available for this user on the selected date."
    
    bits = []
    if profile.get("likes_home") == "yes":
        bits.append("you've shown a tilt toward home matchups")
    if "fav_dow" in profile:
        bits.append(f"you're often active around {profile['fav_dow']}s")
    reason = "; ".join(bits) if bits else "your recent betting history"
    
    names = []
    for i, it in enumerate(top3, start=1):
        # Extract game name from game_X key or event_description
        name = it.get("event_description") or next((v for k,v in it.items() if k.startswith("game_")), None)
        if name: names.append(name)
    rec_line = ", ".join(names[:3])
    return f"Based on {reason}, consider: {rec_line}."

# --------------- Simulation Analysis ----------------------

def get_simulation_chart_data(sim_v: pd.DataFrame, sim_a: pd.DataFrame):
    if sim_v is not None and sim_a is not None and "date" in sim_v.columns and "date" in sim_a.columns:
        v = sim_v.groupby("date")["hit@k"].mean() if "hit@k" in sim_v.columns else pd.Series(dtype=float)
        a = sim_a.groupby("date")["hit@k"].mean() if "hit@k" in sim_a.columns else pd.Series(dtype=float)
        chart_df = pd.concat([v.rename("Vendor"), a.rename("Actual")], axis=1).fillna(0.0)
        return chart_df if not chart_df.empty else None
    return None

# ================ AI ANALYSIS FUNCTIONS ================

def generate_ai_schedule_trend_analysis(sim_data: pd.DataFrame, selected_period_data: dict, schedule_stats: dict, api_key: str) -> str:
    if not api_key or sim_data is None or sim_data.empty:
        return "AI analysis unavailable - API key required"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Calculate comprehensive trend statistics
        hit_rate_trend = sim_data['hit_rate'].diff().mean() if 'hit_rate' in sim_data.columns else 0
        precision_trend = sim_data['precision'].diff().mean() if 'precision' in sim_data.columns else 0
        schedule_accuracy_trend = sim_data['schedule_accuracy'].diff().mean() if 'schedule_accuracy' in sim_data.columns else 0
        
        # Performance degradation over time
        early_periods = sim_data.head(3) if len(sim_data) >= 6 else sim_data.head(len(sim_data)//2)
        later_periods = sim_data.tail(3) if len(sim_data) >= 6 else sim_data.tail(len(sim_data)//2)
        
        early_hit_rate = early_periods['hit_rate'].mean() if 'hit_rate' in early_periods.columns else 0
        later_hit_rate = later_periods['hit_rate'].mean() if 'hit_rate' in later_periods.columns else 0
        hit_rate_decline = early_hit_rate - later_hit_rate
        
        early_schedule_acc = early_periods['schedule_accuracy'].mean() if 'schedule_accuracy' in early_periods.columns else 1.0
        later_schedule_acc = later_periods['schedule_accuracy'].mean() if 'schedule_accuracy' in later_periods.columns else 1.0
        schedule_degradation = early_schedule_acc - later_schedule_acc
        
        # Current period context
        games_cancelled = schedule_stats.get('games_cancelled', 0)
        total_scheduled = schedule_stats.get('total_scheduled', 1)
        cancellation_rate = games_cancelled / total_scheduled if total_scheduled > 0 else 0
        current_period = selected_period_data.get('period', 'Unknown')
        
        # Correlation analysis
        correlation = sim_data[['hit_rate', 'schedule_accuracy']].corr().iloc[0, 1] if len(sim_data) > 3 else 0
        
        best_period = sim_data.loc[sim_data['hit_rate'].idxmax()] if 'hit_rate' in sim_data.columns else None
        worst_period = sim_data.loc[sim_data['hit_rate'].idxmin()] if 'hit_rate' in sim_data.columns else None
        
        # Format best and worst period strings safely
        best_period_str = f"Period {best_period['period']} (Hit Rate: {best_period['hit_rate']:.3f})" if best_period is not None else "N/A"
        worst_period_str = f"Period {worst_period['period']} (Hit Rate: {worst_period['hit_rate']:.3f})" if worst_period is not None else "N/A"
        
        prompt = f"""
        You are analyzing the relationship between NBA schedule disruptions and betting recommendation performance degradation over time.
        
        PERFORMANCE DEGRADATION ANALYSIS:
        - Early Periods (1-3) Hit Rate: {early_hit_rate:.3f}
        - Later Periods Hit Rate: {later_hit_rate:.3f}
        - Performance Decline: {hit_rate_decline:+.3f} ({hit_rate_decline/early_hit_rate*100:+.1f}% change)
        
        SCHEDULE DISRUPTION TRENDS:
        - Early Period Schedule Accuracy: {early_schedule_acc:.3f}
        - Later Period Schedule Accuracy: {later_schedule_acc:.3f}
        - Schedule Degradation: {schedule_degradation:+.3f}
        - Hit Rate vs Schedule Correlation: {correlation:.3f}
        
        CURRENT PERIOD CONTEXT (Period {current_period}):
        - Games Scheduled: {total_scheduled}
        - Games Cancelled: {games_cancelled} ({cancellation_rate:.1%})
        - Current Hit Rate: {selected_period_data.get('hit_rate', 0):.3f}
        - Current Schedule Accuracy: {selected_period_data.get('schedule_accuracy', 0):.3f}
        
        OVERALL TRENDS:
        - Hit Rate Trend: {hit_rate_trend:+.4f} per period
        - Schedule Accuracy Trend: {schedule_accuracy_trend:+.4f} per period
        - Best Performance: {best_period_str}
        - Worst Performance: {worst_period_str}
        
        Provide comprehensive insights explaining:
        1. WHY hit rates decline in later simulation periods
        2. HOW schedule disruptions directly impact recommendation accuracy
        3. The CAUSE-AND-EFFECT relationship between schedule drift and performance degradation
        4. BUSINESS implications and mitigation strategies
        5. Whether the current period follows expected patterns
        
        Focus on actionable insights about the temporal degradation pattern. Be specific about the schedule-performance relationship.
        Please give your response in good bullet points format.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Schedule-trend analysis error: {str(e)}"

def compute_date_specific_diagnostics(user_history: pd.DataFrame, 
                                     selected_date: str,
                                     top_picks: list,
                                     user_metrics: pd.DataFrame = None,
                                     recent_days: int = 28) -> dict:
    """
    Builds concrete facts for AI analysis for a specific selected date:
      - User's favorite teams (lifetime and recent)
      - What user actually bet on the selected date
      - Which recommended games user actually bet on (if any)
      - Hit/miss status for the selected date
      - Reasons for miss (if applicable)
    Expects user_history columns: ['date','event_description','amount_sum']
    """
    hist = user_history.copy()
    if "date" in hist.columns:
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    else:
        raise ValueError("user_history must contain a 'date' column")

    selected_date_dt = pd.to_datetime(selected_date, errors="coerce")

    # --- team extraction helper ---
    import re
    AT_RE = re.compile(r"@", re.I)
    VS_RE = re.compile(r"\bvs\.?\b", re.I)
    def teams_from_event(s):
        if pd.isna(s): return None, None
        t = str(s).strip()
        if AT_RE.search(t):
            a, b = [x.strip() for x in AT_RE.split(t, 1)]
            return a, b
        if VS_RE.search(t):
            a, b = [x.strip() for x in VS_RE.split(t, 1)]
            return a, b
        parts = re.split(r"\s*(?:&|and)\s*", t)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return None, None

    # explode to team-level rows (so we can rank favorites)
    tmp = []
    for _, r in hist.iterrows():
        a, b = teams_from_event(r.get("event_description"))
        amt = float(r.get("amount_sum", 0.0))
        d = r.get("date")
        if a: tmp.append({"date": d, "team": a, "amount_sum": amt})
        if b: tmp.append({"date": d, "team": b, "amount_sum": amt})
    team_hist = pd.DataFrame(tmp)

    # lifetime favorites by $, top 5
    top_all = (team_hist.groupby("team", as_index=False)["amount_sum"]
               .sum().sort_values("amount_sum", ascending=False).head(5))
    top_all_list = top_all.to_dict("records")

    # Recent favorites (last N days before selected date)
    recent_cut = selected_date_dt - timedelta(days=recent_days)
    recent = team_hist[team_hist["date"] >= recent_cut]
    top_recent = (recent.groupby("team", as_index=False)["amount_sum"]
                  .sum().sort_values("amount_sum", ascending=False).head(5))
    top_recent_list = top_recent.to_dict("records")

    # Get user's actual bets on the selected date
    date_bets = hist[hist["date"] == selected_date_dt]
    actual_games_on_date = date_bets["event_description"].dropna().tolist() if not date_bets.empty else []
    total_bet_amount_on_date = date_bets["amount_sum"].sum() if not date_bets.empty else 0

    # Extract recommended game names from top_picks
    recommended_games = []
    for pick in (top_picks or []):
        game_name = pick.get('event_description') or next((v for k,v in pick.items() if k.startswith("game_")), "Unknown game")
        recommended_games.append(game_name)

    # Check which recommended games user actually bet on
    games_overlap = []
    for rec_game in recommended_games:
        for actual_game in actual_games_on_date:
            if rec_game == actual_game:  # Exact match
                games_overlap.append(rec_game)

    # Get hit/miss status for this date
    hit_status = None
    if user_metrics is not None and not user_metrics.empty:
        date_metrics = user_metrics[user_metrics["date"] == selected_date]
        if not date_metrics.empty:
            hit_status = int(date_metrics["hit@k"].iloc[0])

    # Check favorite team involvement
    favs = set([t["team"] for t in top_all_list])
    fav_teams_in_actual_bets = []
    fav_teams_in_recommendations = []
    
    for game in actual_games_on_date:
        for fav_team in favs:
            if fav_team in str(game):
                fav_teams_in_actual_bets.append(fav_team)
    
    for game in recommended_games:
        for fav_team in favs:
            if fav_team in str(game):
                fav_teams_in_recommendations.append(fav_team)

    return {
        "selected_date": selected_date,
        "hit_status": hit_status,  # 1 = hit, 0 = miss, None = no data
        "top_teams_all_time": top_all_list,
        "top_teams_recent": top_recent_list,
        "recommended_games": recommended_games,
        "actual_games_bet": actual_games_on_date,
        "games_user_bet_on": games_overlap,  # Games that were both recommended AND bet on
        "total_bet_amount": total_bet_amount_on_date,
        "favorite_teams_in_actual_bets": sorted(set(fav_teams_in_actual_bets)),
        "favorite_teams_in_recommendations": sorted(set(fav_teams_in_recommendations)),
        "had_betting_activity": len(actual_games_on_date) > 0,
        "recommendation_alignment": len(games_overlap) / len(recommended_games) if recommended_games else 0
    }


def generate_date_specific_ai_analysis(
    user_id: str,
    selected_date: str,
    diagnostics: dict,               # output of compute_date_specific_diagnostics(...)
    api_key: str,
    k: int = 3
) -> dict:
    """
    Returns a markdown string with date-specific miss analysis.
    Only runs if hit_status = 0 (miss) for the selected date.
    Provides specific reasons why recommendations failed on that exact date.
    Includes schedule shift/cancellation analysis.
    """
    if not api_key:
        return {"error": "AI analysis unavailable - API key required"}

    # Check if this date was a miss - only analyze misses
    hit_status = diagnostics.get("hit_status")
    if hit_status is None:
        return {"error": "No recommendation data available for this date"}
    if hit_status == 1:
        return {"info": "Recommendations were successful on this date - no analysis needed"}

    # Get date-specific data from diagnostics
    top_teams_all = diagnostics.get("top_teams_all_time", [])
    top_teams_recent = diagnostics.get("top_teams_recent", [])
    recommended_games = diagnostics.get("recommended_games", [])
    actual_games_bet = diagnostics.get("actual_games_bet", [])
    games_user_bet_on = diagnostics.get("games_user_bet_on", [])
    fav_teams_in_actual = diagnostics.get("favorite_teams_in_actual_bets", [])
    fav_teams_in_recs = diagnostics.get("favorite_teams_in_recommendations", [])
    had_betting_activity = diagnostics.get("had_betting_activity", False)
    total_bet_amount = diagnostics.get("total_bet_amount", 0)
    
    # Calculate user preference diversity  
    total_lifetime_bets = sum([t.get("amount_sum", 0) for t in top_teams_all])
    top_team_concentration = top_teams_all[0].get("amount_sum", 0) / total_lifetime_bets if total_lifetime_bets > 0 and top_teams_all else 0

    # Load schedule drift data for the selected date
    schedule_info = {}
    try:
        schedule_drift_path = DEFAULTS["sim_vendor"]  # This points to schedule_drift_daily.csv
        schedule_df = read_csv_cached(Path(schedule_drift_path))
        if schedule_df is not None and "date" in schedule_df.columns:
            date_schedule = schedule_df[schedule_df["date"] == selected_date]
            if not date_schedule.empty:
                row = date_schedule.iloc[0]
                schedule_info = {
                    "vendor_candidates": int(row.get("vendor_candidates", 0)),
                    "actual_candidates": int(row.get("actual_candidates", 0)),
                    "strict_overlap": int(row.get("strict_overlap", 0)),
                    "vendor_only": int(row.get("vendor_only", 0)),
                    "actual_only": int(row.get("actual_only", 0)),
                    "strict_overlap_rate": float(row.get("strict_overlap_rate", 0)),
                    "schedule_drift_detected": bool(row.get("strict_overlap_rate", 1.0) < 0.8),  # Less than 80% overlap indicates drift
                    "games_cancelled_or_moved": int(row.get("vendor_only", 0)),  # Games in vendor schedule but not played
                    "unscheduled_games_played": int(row.get("actual_only", 0))   # Games played but not in vendor schedule
                }
    except Exception as e:
        print(f"Warning: Could not load schedule drift data: {e}")
        schedule_info = {
            "schedule_drift_detected": bool(False),
            "games_cancelled_or_moved": int(0),
            "unscheduled_games_played": int(0),
            "strict_overlap_rate": float(1.0)
        }

    context = {
        "user_id": str(user_id),
        "selected_date": selected_date,
        "k": int(k),
        "hit_status": hit_status,  # This is 0 (miss)
        "recommended_games": recommended_games,
        "actual_games_bet": actual_games_bet,
        "games_overlap": games_user_bet_on,  # Games both recommended AND bet on
        "had_betting_activity": had_betting_activity,
        "total_bet_amount_on_date": total_bet_amount,
        "favorite_teams_in_actual_bets": fav_teams_in_actual,
        "favorite_teams_in_recommendations": fav_teams_in_recs,
        "top_teams_lifetime": top_teams_all[:3],  # User's top 3 favorite teams
        "top_teams_recent": top_teams_recent[:3],
        "top_team_concentration": round(top_team_concentration, 3),  # How focused user's preferences are
        "schedule_analysis": schedule_info,  # Schedule drift and cancellation data
        "miss_analysis": {
            "no_overlap": bool(len(games_user_bet_on) == 0),  # User bet on completely different games
            "no_betting_activity": bool(not had_betting_activity),  # User didn't bet at all
            "favorite_teams_misaligned": bool(len(set(fav_teams_in_actual) - set(fav_teams_in_recs)) > 0),
            "schedule_issues_detected": bool(schedule_info.get("schedule_drift_detected", False))  # Games cancelled/moved
        }
    }

    # Format schedule analysis for prompt
    schedule_summary = ""
    if schedule_info:
        overlap_rate = schedule_info.get("strict_overlap_rate", 1.0)
        cancelled_games = schedule_info.get("games_cancelled_or_moved", 0)
        added_games = schedule_info.get("unscheduled_games_played", 0)
        
        if schedule_info.get("schedule_drift_detected", False):
            schedule_summary = f"⚠️ SCHEDULE DRIFT DETECTED: Only {overlap_rate:.1%} of expected games were played. {cancelled_games} recommended games may have been cancelled/postponed, and {added_games} unscheduled games were played."
        else:
            schedule_summary = f"✅ Schedule stable: {overlap_rate:.1%} of expected games played as scheduled."

    prompt = (
        f"You are analyzing why NBA betting recommendations failed for user {user_id} on {selected_date}.\n\n"
        f"The user's Top-{k} recommendations were: {', '.join(recommended_games)}\n"
        f"The user actually bet on: {', '.join(actual_games_bet) if actual_games_bet else 'No betting activity'}\n\n"
        f"SCHEDULE STATUS: {schedule_summary}\n\n"
        "ANALYSIS TASK:\n"
        "Explain in plain English why our recommendations missed on this specific date. Be concrete and factual.\n"
        "IMPORTANT: Check if any recommended games were cancelled/postponed or if schedule changes affected availability.\n\n"
        f"FORMAT YOUR RESPONSE EXACTLY AS:\n"
        f"### Miss Analysis for {selected_date}\n\n"
        "**What Happened:**\n"
        f"- Recommended games: {', '.join(recommended_games)}\n"
        f"- User actually bet on: {', '.join(actual_games_bet) if actual_games_bet else 'Nothing (no betting activity)'}\n"
        f"- Games overlap: {', '.join(games_user_bet_on) if games_user_bet_on else 'None'}\n"
        f"- Schedule status: {schedule_summary}\n\n"
        "**Why We Missed:**\n"
        "- [Check first: Were any recommended games cancelled, postponed, or moved? Reference schedule_analysis data]\n"
        "- [Specific reason 2: User preference misalignment, team favoritism, etc.]\n"
        "- [Specific reason 3: Betting pattern changes, timing issues, etc.]\n\n"
        "**User Preference Insights:**\n"
        f"- User's top favorite teams: {', '.join([t['team'] for t in top_teams_all[:3]]) if top_teams_all else 'No clear favorites'}\n"
        f"- Preference concentration: {top_team_concentration:.1%} (higher = more focused on specific teams)\n"
        "- Recent betting pattern: [analysis based on recent vs lifetime preferences]\n\n"
        f"CONTEXT:\n{json.dumps(context, ensure_ascii=False)}"
    )

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more consistent, factual responses
            max_tokens=800    # More tokens for detailed analysis
        )
        text = resp.choices[0].message.content.strip()
        return {"analysis_text": text}
    except Exception as e:
        return {"error": f"Root cause analysis error: {e}"}

def generate_ai_root_cause_analysis_old(user_id: str, user_metrics: pd.DataFrame, user_history: pd.DataFrame, 
                                   missed_picks: List[dict], period_data: dict, api_key: str) -> dict:
    """Generate AI-powered root cause analysis for recommendation misses with date-level detail."""
    if not api_key:
        return {"error": "AI analysis unavailable - API key required"}
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Compute miss stats
        total_recommendations = len(user_metrics) if not user_metrics.empty else 0
        hits = user_metrics['hit@k'].sum() if 'hit@k' in user_metrics.columns else 0
        misses = total_recommendations - hits
        miss_rate = misses / total_recommendations if total_recommendations > 0 else 0
        
        # Find all dates where hit@k = 0
        miss_dates = user_metrics.loc[user_metrics['hit@k'] == 0, 'date'].dropna().unique().tolist()
        miss_dates_str = ", ".join(miss_dates) if miss_dates else "None"
        
        # Historical profile
        total_historical_bets = len(user_history)
        unique_games_bet = user_history['event_description'].nunique() if not user_history.empty else 0
        avg_bet_amount = user_history['amount_sum'].mean() if 'amount_sum' in user_history.columns else 0
        schedule_accuracy = period_data.get('schedule_accuracy', 1.0)
        
        # Prompt in plain English
        prompt = f"""
        We are analyzing recommendation misses for user {user_id}.

        Write your answer in **plain English with bullet points**. 
        Do not include any introduction or summary like "Certainly!" or "Here’s an explanation." 
        Start immediately with the section headings and bullet points.

        Summary:
        - Miss Rate: {miss_rate:.1%} ({misses}/{total_recommendations})
        - Missed Dates (hit@k = 0): {miss_dates_str}
        - Betting History: {total_historical_bets} total bets on {unique_games_bet} games
        - Avg Bet: ${avg_bet_amount:.2f}
        - Schedule Accuracy during period: {schedule_accuracy:.1%}

        Required sections:
        - Likely Causes of Misses
        - Suggestions for Improvement

        Within each section, please explain in plain English, using bullet points:
        1. What likely caused these misses (e.g., games cancelled, model ranked user’s favorite teams too low)?
        3. Suggest simple, actionable improvements (e.g., boost favorite teams, adjust model features).
        Keep the explanation clear, short, and easy to understand.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3
        )
        
        return {"analysis_text": response.choices[0].message.content.strip()}
    
    except Exception as e:
        return {"error": f"Root cause analysis error: {str(e)}"}
