import os, json, re, uuid
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

def normalize_game_description(game_desc: str) -> str:
    """
    Normalize game descriptions to enable proper comparison between recommendations and bets.
    Recommendations: "Team A & Team B"
    Actual bets: "Team A @ Team B" or "Team B @ Team A"
    
    Returns a normalized string with teams in alphabetical order.
    """
    if not game_desc:
        return ""
    
    # Replace @ with & and split on &
    normalized = game_desc.replace(" @ ", " & ")
    
    # Split teams and sort alphabetically for consistent comparison
    if " & " in normalized:
        teams = [team.strip() for team in normalized.split(" & ")]
        teams.sort()  # Alphabetical order ensures "Warriors & Lakers" == "Lakers & Warriors"
        return " & ".join(teams)
    
    return normalized.strip()

def analyze_schedule_issues(date: str, recommended_games: list) -> dict:
    """
    Analyze schedule-related issues for a specific date that might affect recommendations.
    
    Args:
        date: Date in YYYY-MM-DD format
        recommended_games: List of recommended game descriptions
        
    Returns:
        Dict with schedule analysis including cancellations and over-scheduling
    """
    try:
        # Load vendor schedule
        vendor_schedule = pd.read_csv("data/raw/playoff_schedule.csv")
        
        # Convert date format: YYYY-MM-DD to DD-MM-YYYY for vendor schedule lookup
        date_parts = date.split('-')
        vendor_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
        
        # Get games scheduled for this date
        date_games = vendor_schedule[vendor_schedule['Date'] == vendor_date]
        
        analysis = {
            "date": date,
            "vendor_date_format": vendor_date,
            "total_scheduled": len(date_games),
            "cancelled_games": [],
            "over_scheduled_games": [],
            "playable_games": [],
            "schedule_issues": []
        }
        
        if date_games.empty:
            analysis["schedule_issues"].append("No games found in vendor schedule for this date")
            return analysis
        
        for _, game in date_games.iterrows():
            game_name = game['Game']
            event_desc = game['event_description']
            game_count = game['game_count']
            
            # Check if over-scheduled (game_count >= 8) - always check this first
            if game_count >= 8:
                analysis["over_scheduled_games"].append({
                    "game": game_name,
                    "game_count": game_count,
                    "reason": f"Over-scheduled: Game #{game_count} exceeds NBA playoff max of 7 games"
                })
                analysis["schedule_issues"].append(f"⚠️ {game_name}: Over-scheduled (game #{game_count}/7 max)")
                
                # Also check if it was cancelled/shifted
                if pd.isna(event_desc) or event_desc == '':
                    analysis["cancelled_games"].append({
                        "game": game_name,
                        "game_count": game_count,
                        "reason": "Game cancelled or shifted AND over-scheduled"
                    })
                    analysis["schedule_issues"].append(f"🚫 {game_name}: Also cancelled/shifted")
            
            # Check if game was cancelled/shifted (null event_description) but not over-scheduled
            elif pd.isna(event_desc) or event_desc == '':
                analysis["cancelled_games"].append({
                    "game": game_name,
                    "game_count": game_count,
                    "reason": "Game cancelled or shifted (no event_description in actual schedule)"
                })
                analysis["schedule_issues"].append(f"🚫 {game_name}: Cancelled/shifted (game #{game_count})")
                analysis["over_scheduled_games"].append({
                    "game": game_name,
                    "game_count": game_count,
                    "reason": f"Over-scheduled: Game #{game_count} exceeds NBA playoff max of 7 games"
                })
                analysis["schedule_issues"].append(f"⚠️ {game_name}: Over-scheduled (game #{game_count}/7 max)")
            
            else:
                analysis["playable_games"].append({
                    "game": game_name,
                    "event_description": event_desc,
                    "game_count": game_count
                })
        
        # Check if any recommended games were affected
        analysis["affected_recommendations"] = []
        if recommended_games:
            for rec_game in recommended_games:
                normalized_rec = normalize_game_description(rec_game)
                issues_found = []
                
                # Check against over-scheduled games first (more critical)
                for over_sched in analysis["over_scheduled_games"]:
                    normalized_over = normalize_game_description(over_sched["game"])
                    if normalized_rec == normalized_over:
                        issues_found.append({
                            "recommended_game": rec_game,
                            "issue": "over_scheduled", 
                            "details": over_sched
                        })
                
                # Check against cancelled games
                for cancelled in analysis["cancelled_games"]:
                    normalized_cancelled = normalize_game_description(cancelled["game"])
                    if normalized_rec == normalized_cancelled:
                        issues_found.append({
                            "recommended_game": rec_game,
                            "issue": "cancelled",
                            "details": cancelled
                        })
                
                # Add all issues found for this game
                analysis["affected_recommendations"].extend(issues_found)
        
        return analysis
        
    except Exception as e:
        return {
            "error": f"Failed to analyze schedule issues: {str(e)}",
            "date": date,
            "schedule_issues": [f"Error loading schedule data: {str(e)}"]
        }

def get_available_dates():
    """Get list of available dates for daily picks (cached)"""
    daily_picks_dir = Path("reports/daily")
    if not daily_picks_dir.exists():
        return []
    
    dates = []
    for date_folder in daily_picks_dir.iterdir():
        if date_folder.is_dir() and (date_folder / "picks.csv").exists():
            dates.append(date_folder.name)
    
    return sorted(dates)

from openai import OpenAI 

@st.cache_data
def load_chatbot_context_data():
    """Load core context data (excluding daily picks which are loaded on-demand)"""
    ctx = {}
    
    # Metrics and simulation data (small files, load immediately)
    p_metrics = Path("reports/metrics/metrics_k3_pairkey.csv")
    sim_periods = Path("reports/metrics/simulation_periods.csv")
    
    if p_metrics.exists():
        ctx["pair_metrics"] = pd.read_csv(p_metrics, dtype={"mask_id":"string"})
    else:
        ctx["pair_metrics"] = pd.DataFrame()

    if sim_periods.exists():
        ctx["simulation_periods"] = pd.read_csv(sim_periods)
    else:
        ctx["simulation_periods"] = pd.DataFrame()
    
    # User betting history (load once, used frequently)
    validation_data = Path("data/processed/df_validation_rolled.csv")
    if validation_data.exists():
        ctx["user_betting_history"] = pd.read_csv(validation_data, dtype={"mask_id":"string"})
    else:
        ctx["user_betting_history"] = pd.DataFrame()
    
    # Daily picks: DON'T load all files here, use lazy loading instead
    ctx["daily_picks_dir"] = Path("reports/daily")
    ctx["daily_picks"] = {}  # Will be populated on-demand
    
    return ctx

@st.cache_data
def load_daily_picks_for_date(date_str: str) -> pd.DataFrame:
    """Lazy-load daily picks for a specific date only when needed"""
    daily_picks_dir = Path("reports/daily")
    picks_file = daily_picks_dir / date_str / "picks.csv"
    
    if picks_file.exists():
        try:
            return pd.read_csv(picks_file, dtype={"mask_id":"string"})
        except Exception as e:
            st.warning(f"Failed to load picks for {date_str}: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()

@st.cache_data  
def get_available_pick_dates() -> list:
    """Get list of available dates for daily picks (cached)"""
    daily_picks_dir = Path("reports/daily")
    if not daily_picks_dir.exists():
        return []
    
    dates = []
    for date_folder in daily_picks_dir.iterdir():
        if date_folder.is_dir() and (date_folder / "picks.csv").exists():
            dates.append(date_folder.name)
    
    return sorted(dates)

def analyze_user_accuracy(mask_id: str, date_filter: str | None, context_data: dict) -> dict:
    df = context_data.get("pair_metrics", pd.DataFrame())
    if df.empty:
        return {"error": "No pair-key metrics found."}

    sub = df[df["mask_id"] == str(mask_id)]
    if date_filter:
        sub = sub[sub["date"] == date_filter]

    if sub.empty:
        return {"error": f"No rows for user {mask_id}"}

    out = {
        "mask_id": str(mask_id),
        "total_days": int(sub["date"].nunique()),
        "hit_days": int(sub[sub["hit@k"] == 1]["date"].nunique()),
        "avg_hit_rate": float(sub["hit@k"].mean()),
        "avg_precision": float(sub["precision@k"].mean()),
        "avg_recall": float(sub["recall@k"].mean(skipna=True)),
        "avg_ndcg": float(sub["nDCG@k"].mean(skipna=True)),
        "avg_ndcg_dollar": float(sub["nDCG$@k"].mean(skipna=True)),
        "total_dollar_capture": float(sub["w$@k"].sum()),
        "miss_dates": sub.loc[sub["hit@k"] == 0, "date"].dropna().unique().tolist(),
    }
    return out


def get_daily_recommendations_and_bets(mask_id: str, date: str, context_data: dict) -> dict:
    """Get detailed recommendations and actual betting activity for a specific user and date."""
    result = {
        "mask_id": mask_id,
        "date": date,
        "recommendations": [],
        "actual_bets": [],
        "matches": [],
        "summary": {}
    }
    
    try:
        # Use optimized loading with agent-level caching
        # Note: This requires the agent instance, but we're in a standalone function
        # For now, use the Streamlit-cached version directly
        picks_df = load_daily_picks_for_date(date)
        
        if not picks_df.empty:
            user_picks = picks_df[picks_df["mask_id"] == mask_id]
            
            if not user_picks.empty:
                import json
                top_3_raw = user_picks.iloc[0]["top_3"]
                # Parse the JSON string
                recommendations = json.loads(top_3_raw)
                result["recommendations"] = recommendations
        
        # Get actual betting activity for this date
        betting_history = context_data.get("user_betting_history", pd.DataFrame())
        if not betting_history.empty:
            user_bets = betting_history[
                (betting_history["mask_id"] == mask_id) & 
                (betting_history["date"] == date)
            ]
            
            if not user_bets.empty:
                result["actual_bets"] = user_bets.to_dict('records')
        
        # Find matches between recommendations and actual bets using normalized game descriptions
        if result["recommendations"] and result["actual_bets"]:
            # Normalize recommendation games: "Team A & Team B" -> sorted format
            rec_games = set()
            for rec in result["recommendations"]:
                game_desc = rec.get("event_description", "") or rec.get("game_1", "")
                normalized_game = normalize_game_description(game_desc)
                if normalized_game:
                    rec_games.add(normalized_game)
            
            # Normalize actual bet games: "Team A @ Team B" -> sorted format  
            bet_games = set()
            for bet in result["actual_bets"]:
                normalized_game = normalize_game_description(bet["event_description"])
                if normalized_game:
                    bet_games.add(normalized_game)
            
            # Find matches between normalized game descriptions
            matches = rec_games.intersection(bet_games)
            result["matches"] = list(matches)
            
            # Add debug information for transparency
            result["debug"] = result.get("debug", {})
            result["debug"]["normalized_recommendations"] = list(rec_games)
            result["debug"]["normalized_bets"] = list(bet_games) 
            result["debug"]["original_recommendations"] = [
                rec.get("event_description", "") or rec.get("game_1", "") 
                for rec in result["recommendations"]
            ]
            result["debug"]["original_bets"] = [bet["event_description"] for bet in result["actual_bets"]]
            
            # Summary statistics
            result["summary"] = {
                "total_recommendations": len(result["recommendations"]),
                "total_bets": len(result["actual_bets"]),
                "matches_found": len(matches),
                "hit_rate": len(matches) / len(result["recommendations"]) if result["recommendations"] else 0,
                "precision": len(matches) / len(result["actual_bets"]) if result["actual_bets"] else 0,
                "total_bet_amount": sum(bet["amount_sum"] for bet in result["actual_bets"])
            }
        
        # Analyze schedule issues that might affect recommendations and betting
        if result["recommendations"]:
            recommended_game_names = [
                rec.get("event_description", "") or rec.get("game_1", "") 
                for rec in result["recommendations"]
            ]
            schedule_analysis = analyze_schedule_issues(date, recommended_game_names)
            result["schedule_analysis"] = schedule_analysis
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to analyze recommendations and bets: {str(e)}"}

def get_user_betting_pattern(mask_id: str, context_data: dict, days_back: int = 30) -> dict:
    """Get user's betting patterns and preferences over time."""
    result = {
        "mask_id": mask_id,
        "betting_summary": {},
        "favorite_games": [],
        "betting_frequency": {},
        "recent_activity": []
    }
    
    try:
        betting_history = context_data.get("user_betting_history", pd.DataFrame())
        if betting_history.empty:
            return {"error": "No betting history data available"}
        
        user_bets = betting_history[betting_history["mask_id"] == mask_id].copy()
        if user_bets.empty:
            return {"error": f"No betting history found for user {mask_id}"}
        
        # Convert date to datetime for analysis
        user_bets['date'] = pd.to_datetime(user_bets['date'])
        
        # Recent activity (last N days)
        recent_cutoff = user_bets['date'].max() - pd.Timedelta(days=days_back)
        recent_bets = user_bets[user_bets['date'] >= recent_cutoff]
        
        # Summary statistics
        result["betting_summary"] = {
            "total_bets": len(user_bets),
            "total_amount": user_bets["amount_sum"].sum(),
            "average_bet": user_bets["amount_sum"].mean(),
            "date_range": f"{user_bets['date'].min().strftime('%Y-%m-%d')} to {user_bets['date'].max().strftime('%Y-%m-%d')}",
            "unique_games": user_bets["event_description"].nunique(),
            "betting_days": user_bets['date'].nunique()
        }
        
        # Favorite games (most frequently bet on)
        game_counts = user_bets["event_description"].value_counts().head(10)
        result["favorite_games"] = [
            {"game": game, "bet_count": count, "total_amount": user_bets[user_bets["event_description"]==game]["amount_sum"].sum()}
            for game, count in game_counts.items()
        ]
        
        # Betting frequency by date
        daily_activity = user_bets.groupby('date').agg({
            'amount_sum': ['sum', 'count'],
            'event_description': 'nunique'
        }).round(2)
        
        # Recent activity details
        result["recent_activity"] = recent_bets.sort_values('date', ascending=False).head(20).to_dict('records')
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to analyze betting pattern: {str(e)}"}

def get_recommendation_quality_insights(mask_id: str, context_data: dict) -> dict:
    """Tiny scaffold: list miss dates + simple streaks."""
    df = context_data.get("pair_metrics", pd.DataFrame())
    if df.empty:
        return {"error": "No pair-key metrics found."}
    sub = df[df["mask_id"] == str(mask_id)].sort_values("date")
    miss = sub[sub["hit@k"] == 0]["date"].tolist()
    # detect simple consecutive miss streaks
    streaks = []
    cur = []
    for d in miss:
        if not cur:
            cur = [d]
        else:
            prev = pd.to_datetime(cur[-1])
            curd = pd.to_datetime(d)
            if (curd - prev).days == 1:
                cur.append(d)
            else:
                if len(cur) > 1: streaks.append(cur)
                cur = [d]
    if len(cur) > 1: streaks.append(cur)

    return {"miss_dates": miss, "miss_streaks": streaks}


def analyze_period_performance(period_num: int | None = None, context_data: dict = None) -> dict:
    sim = (context_data or {}).get("simulation_periods", pd.DataFrame())
    if sim.empty:
        # fallback: summarize across all user-days if sim not available
        pm = (context_data or {}).get("pair_metrics", pd.DataFrame())
        if pm.empty:
            return {"error": "No simulation or pair-key metrics available."}
        return {
            "total_periods": 0,
            "hit_rate": float(pm["hit@k"].mean()),
            "users_evaluated": int(pm[["mask_id","date"]].drop_duplicates().shape[0]),
            "schedule_accuracy": np.nan,
            "avg_hit_rate": float(pm["hit@k"].mean()),
            "best_period": "(n/a)",
            "worst_period": "(n/a)",
        }

    if period_num is None:
        # Overall across periods
        return {
            "total_periods": int(len(sim)),
            "avg_hit_rate": float(sim["hit_rate"].mean()),
            "best_period": int(sim.loc[sim["hit_rate"].idxmax(), "period"]) if not sim.empty else None,
            "worst_period": int(sim.loc[sim["hit_rate"].idxmin(), "period"]) if not sim.empty else None,
            "users_evaluated": int(sim["users_evaluated"].sum()) if "users_evaluated" in sim else None,
            "schedule_accuracy": float(sim["schedule_accuracy"].mean()) if "schedule_accuracy" in sim else None,
        }

    row = sim[sim["period"] == int(period_num)]
    if row.empty:
        return {"error": f"Period {period_num} not found."}
    r = row.iloc[0].to_dict()
    return {
        "period": int(r.get("period")),
        "hit_rate": float(r.get("hit_rate", np.nan)),
        "users_evaluated": int(r.get("users_evaluated", 0)) if r.get("users_evaluated") is not None else None,
        "schedule_accuracy": float(r.get("schedule_accuracy", np.nan)) if r.get("schedule_accuracy") is not None else None,
        "start_date": r.get("start_date"),
        "end_date": r.get("end_date"),
    }

def display_analysis_data(analysis: dict):
    """Render compact metric cards for known analysis types."""
    if not analysis or "data" not in analysis:
        return
    typ = analysis.get("type")
    data = analysis["data"]
    if "error" in data:
        st.warning(data["error"])
        return

    if typ == "user_analysis":
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Hit Rate", f"{data['avg_hit_rate']:.1%}")
        with c2: st.metric("Precision", f"{data['avg_precision']:.1%}")
        with c3: st.metric("Active Days", f"{data['hit_days']}/{data['total_days']}")
        with c4: st.metric("$ Captured", f"${data['total_dollar_capture']:.2f}")
        if "miss_dates" in data and data["miss_dates"]:
            with st.expander("Miss dates"):
                st.write(", ".join(sorted(set(map(str, data["miss_dates"])))))

    elif typ == "period_analysis":
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Hit@3", f"{data.get('hit_rate', np.nan):.1%}")
        with c2: st.metric("Schedule Acc.", f"{data.get('schedule_accuracy', np.nan):.1%}" if data.get('schedule_accuracy') is not None else "n/a")
        with c3: st.metric("Users", data.get('users_evaluated', "n/a"))

    elif typ == "overall_analysis":
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Periods", data['total_periods'])
        with c2: st.metric("Avg Hit Rate", f"{data['avg_hit_rate']:.1%}")
        with c3: st.metric("Best Period", data['best_period'])
        with c4: st.metric("Worst Period", data['worst_period'])
    
    elif typ == "daily_analysis":
        st.subheader(f"📅 Daily Analysis: {data.get('date')} - User {data.get('mask_id')}")
        
        # Debug information (show if there are issues)
        if data.get("debug"):
            debug_info = data["debug"]
            if not debug_info.get("user_found", True) or not debug_info.get("picks_file_loaded", True):
                with st.expander("🔍 Debug Information", expanded=True):
                    st.json(debug_info)
        
        # Recommendations section
        if data.get("recommendations"):
            st.write("🎯 **Recommendations (Top 3):**")
            for i, rec in enumerate(data["recommendations"], 1):
                game = rec.get("event_description", rec.get(f"game_{i}", "Unknown"))
                score = rec.get("score", 0)
                st.write(f"{i}. {game} (Score: {score:.2f})")
        else:
            st.write("🎯 **Recommendations:** None found for this date")
            
            # Show debug info if recommendations are missing
            if data.get("debug"):
                with st.expander("🔍 Debug Information", expanded=False):
                    st.json(data["debug"])
        
        # Actual bets section
        if data.get("actual_bets"):
            st.write("💰 **Actual Bets Placed:**")
            for bet in data["actual_bets"]:
                game = bet.get("event_description", "Unknown")
                amount = bet.get("amount_sum", 0)
                tickets = bet.get("tickets_n", 0)
                st.write(f"• {game}: ${amount:.2f} ({tickets} tickets)")
        else:
            st.write("💰 **Actual Bets:** No bets placed this date")
        
        # Matches section
        if data.get("matches"):
            st.success(f"✅ **Matches Found:** {len(data['matches'])} games")
            for match in data["matches"]:
                st.write(f"🎯 {match}")
        else:
            st.warning("❌ **No matches** between recommendations and actual bets")
        
        # Summary metrics
        if data.get("summary"):
            summary = data["summary"]
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Hit Rate", f"{summary.get('hit_rate', 0):.1%}")
            with c2: st.metric("Total Bet", f"${summary.get('total_bet_amount', 0):.2f}")
            with c3: st.metric("Recommendations", summary.get('total_recommendations', 0))
            with c4: st.metric("Actual Bets", summary.get('total_bets', 0))
    
    elif typ == "betting_pattern":
        st.subheader(f"📊 Betting Pattern Analysis: User {data.get('mask_id')}")
        
        if data.get("betting_summary"):
            summary = data["betting_summary"]
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total Bets", summary.get('total_bets', 0))
            with c2: st.metric("Total Amount", f"${summary.get('total_amount', 0):.2f}")
            with c3: st.metric("Avg Bet", f"${summary.get('average_bet', 0):.2f}")
            with c4: st.metric("Unique Games", summary.get('unique_games', 0))
            
            st.write(f"📅 **Active Period:** {summary.get('date_range', 'Unknown')}")
            st.write(f"🗓️ **Betting Days:** {summary.get('betting_days', 0)}")
        
        # Favorite games
        if data.get("favorite_games"):
            st.write("🏆 **Favorite Games (Most Bet On):**")
            for fav in data["favorite_games"][:5]:  # Top 5
                game = fav.get("game", "Unknown")
                count = fav.get("bet_count", 0)
                amount = fav.get("total_amount", 0)
                st.write(f"• {game}: {count} bets, ${amount:.2f} total")
        
        # Recent activity
        if data.get("recent_activity"):
            with st.expander("📈 Recent Betting Activity", expanded=False):
                recent_df = pd.DataFrame(data["recent_activity"])
                if not recent_df.empty:
                    st.dataframe(recent_df[["date", "event_description", "amount_sum", "tickets_n"]].head(10))

class NBAAnalysisAgent:
    """Conversational AI agent for NBA betting analysis with persistent memory."""
    def __init__(self, context_data: dict):
        self.context_data = context_data
        self.conversation_memory = []
        self.user_preferences = {}
        self.session_id = str(uuid.uuid4())[:8]
        self.analysis_cache = {}
        self._daily_picks_cache = {}  # Internal cache for loaded daily picks
        self._cache_stats = {"hits": 0, "misses": 0}  # Performance tracking

    def add_to_memory(self, role: str, content: str, metadata: dict = None):
        self.conversation_memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        if len(self.conversation_memory) > 20:
            self.conversation_memory = self.conversation_memory[-20:]

    def _build_context_summary(self) -> str:
        summary = ""
        sim = self.context_data.get("simulation_periods", pd.DataFrame())
        if not sim.empty:
            summary += f"- {len(sim)} simulation periods analyzed\n"
            summary += f"- Average Hit@3: {sim['hit_rate'].mean():.1%}\n"
            summary += f"- Best period performance: {sim['hit_rate'].max():.1%}\n"
        pm = self.context_data.get("pair_metrics", pd.DataFrame())
        if not pm.empty:
            total_users = pm["mask_id"].nunique()
            summary += f"- {total_users} users in validation dataset\n"
            summary += f"- Average user hit rate: {pm['hit@k'].mean():.1%}\n"
        return summary

    def get_system_prompt(self) -> str:
        return f"""You are an AI analyst for an NBA betting recommender system with comprehensive data access. 
                Answer using the structured analysis data provided in user messages. Be conversational, detailed, and insightful.

                🎯 **Your Capabilities:**
                - Analyze daily recommendations vs actual betting behavior
                - Compare what games were recommended to what users actually bet on
                - Identify user betting patterns and preferences
                - Explain recommendation accuracy and system performance
                - Provide insights into why recommendations succeed or fail

                📊 **Available Data Types:**
                - Daily recommendations (Top 3 games per user per date with scores)
                - User betting history (actual games bet on, amounts, dates)
                - Performance metrics (hit rates, precision, schedule accuracy)
                - Simulation period analysis across different time windows

                🔍 **Analysis Types You Can Perform:**
                1. **Daily Analysis**: "What games were recommended to user X on date Y? Did they bet on them?"
                2. **Betting Patterns**: "What are user X's favorite games and betting habits?"
                3. **Performance Analysis**: "How accurate were recommendations for user X overall?"
                4. **Period Comparisons**: "How did period 3 perform compared to others?"

                **IMPORTANT: Smart Game Matching System**
                The system uses intelligent matching that handles format differences:
                - Recommendations: "Team A & Team B" format
                - Actual Bets: "Team A @ Team B" or "Team B @ Team A" format
                - Matching Logic: Normalizes both formats and ignores team order
                - Example: "Lakers & Warriors" matches "Warriors @ Lakers" ✅

                Key metrics to reference:
                - Hit@3: Did ≥1 of top-3 recommendations match actual bets that day? (uses smart matching)
                - Precision@3: What fraction of top-3 recommendations were actually bet on? (uses smart matching)
                - Schedule Accuracy: How well did actual games match pre-season schedule?
                - Dollar Capture: Total betting volume captured by recommendations

                **When analyzing matches:**
                - Focus on the normalized matching results in the data
                - If games overlap (same teams), that IS a match regardless of format/order
                - Explain that "Team A & Team B" and "Team B @ Team A" are the same game

                **CRITICAL: Schedule Issue Analysis**
                Always check for schedule-related issues that explain low betting activity:
                
                1. **Cancelled/Shifted Games**: When event_description is null/empty, the game was cancelled or shifted
                   - Players can't bet on games that don't happen
                   - Explain: "This game was scheduled but cancelled/postponed, so no betting occurred"
                
                2. **Over-Scheduled Games**: When game_count ≥ 8, the vendor over-scheduled beyond NBA rules
                   - NBA playoff series have max 7 games between two teams
                   - Game count 8+ indicates scheduling error
                   - Explain: "This was game #8+ which violates NBA playoff rules (max 7 games)"
                   - Note: "Recommendations should be disabled for over-scheduled games due to data quality issues"
                
                3. **Schedule Impact on Betting**: 
                   - If recommended games had schedule issues, explain why users couldn't/shouldn't bet
                   - Don't penalize recommendation accuracy for games that couldn't be played
                   - Focus on playable games when calculating meaningful hit rates

                **When schedule issues are detected, always explain:**
                - Which specific games were affected and why
                - How this impacts the user's betting behavior (couldn't bet vs chose not to bet)
                - Whether recommendations should have been made for problematic games

                Context Summary:
                {self._build_context_summary()}

                **Instructions:**
                - Use specific data from the analysis results provided
                - Be detailed when explaining what games were recommended vs bet on
                - Provide actionable insights about recommendation quality
                - Reference actual dollar amounts, game names, and dates when available
                - Maintain conversational context from previous messages"""

    def process_user_message(self, message: str) -> dict:
        self.add_to_memory("user", message)
        analysis_result = self._check_for_analysis_request(message)
        ai_response = self._generate_contextual_response(message, analysis_result)
        self.add_to_memory("assistant", ai_response, {"analysis": analysis_result})
        return {
            "response": ai_response,
            "analysis_data": analysis_result,
            "conversation_id": self.session_id,
            "memory_length": len(self.conversation_memory)
        }

    def _check_for_analysis_request(self, message: str) -> dict:
        msg = message.lower()
        # Enhanced regex for mask_id (numbers/letters)
        user_match = re.search(r'(?:user|player|mask[_ ]?id)\s+([A-Za-z0-9_-]+)', msg)
        
        # Enhanced date matching for multiple formats
        date_str = None
        # Format 1: YYYY-MM-DD (2025-04-26)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', message)
        if date_match:
            date_str = date_match.group(1)
        else:
            # Format 2: Apr 26 2025, April 26 2025
            month_day_year = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})[,\s]+(\d{4})', msg)
            if month_day_year:
                month_abbr = month_day_year.group(1)
                day = int(month_day_year.group(2))
                year = int(month_day_year.group(3))
                
                month_map = {
                    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08', 
                    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                }
                
                if month_abbr in month_map:
                    date_str = f"{year}-{month_map[month_abbr]}-{day:02d}"
        
        period_match = re.search(r'period\s+(\d+)', msg)

        # Check for detailed daily analysis (recommendations + betting activity)
        if (user_match and date_str and 
            any(k in msg for k in ["recommend", "game", "bet", "activity", "play", "wager"])):
            mask_id = user_match.group(1)
            data = self._get_daily_recommendations_and_bets_optimized(mask_id, date_str)
            return {"type": "daily_analysis", "data": data}
        
        # Check for user betting pattern analysis
        if (user_match and 
            any(k in msg for k in ["pattern", "behavior", "prefer", "favorite", "history", "betting"])):
            mask_id = user_match.group(1)
            data = get_user_betting_pattern(mask_id, self.context_data)
            return {"type": "betting_pattern", "data": data}

        # Standard user performance analysis
        if user_match and any(k in msg for k in ["accuracy","performance","analy","how","miss","hit"]):
            mask_id = user_match.group(1)
            data = analyze_user_accuracy(mask_id, date_str, self.context_data)
            if "error" not in data:
                data["quality_insights"] = get_recommendation_quality_insights(mask_id, self.context_data)
            return {"type":"user_analysis","data":data}

        if period_match:
            pnum = int(period_match.group(1))
            data = analyze_period_performance(pnum, self.context_data)
            return {"type":"period_analysis","data":data}

        if any(k in msg for k in ["overall","summary","all periods","aggregate"]):
            data = analyze_period_performance(None, self.context_data)
            return {"type":"overall_analysis","data":data}

        return {}

    def _generate_contextual_response(self, message: str, analysis_result: dict) -> str:
        # Build compact context for the model: the “analysis results” are your ground truth
        user_prompt = f"Question: {message}\n\n"
        if analysis_result and analysis_result.get("data"):
            user_prompt += "Structured analysis (ground truth):\n"
            user_prompt += json.dumps(analysis_result["data"], ensure_ascii=False, default=str)

        try:
            # Try multiple methods to get API key
            api_key = None
            
            # Method 1: Environment variable (works everywhere)
            api_key = os.getenv("OPENAI_API_KEY")
            
            # Method 2: Streamlit secrets (only if available)
            if not api_key:
                try:
                    api_key = st.secrets.get("OPENAI_API_KEY")
                except:
                    pass  # Secrets not configured, that's okay
            
            if not api_key:
                return "🔑 OpenAI API key not found. Please set OPENAI_API_KEY environment variable or configure Streamlit secrets."
            
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.2,           # crisp, factual
                max_tokens=600,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error generating response: {e}"

    def get_conversation_summary(self) -> str:
        if not self.conversation_memory:
            return "No conversation history yet."
        user_msgs = [m for m in self.conversation_memory if m["role"]=="user"]
        tags = []
        for m in user_msgs:
            t = m["content"].lower()
            if "user" in t or "mask" in t: tags.append("User analysis")
            if "period" in t: tags.append("Period analysis")
            if "overall" in t or "summary" in t: tags.append("Overall performance")
        return f"We've discussed: {', '.join(sorted(set(tags)))} ({len(self.conversation_memory)} messages)"
    
    def get_cache_stats(self) -> dict:
        """Get performance statistics about data loading and caching"""
        return {
            "daily_picks_loaded": len(self._daily_picks_cache),
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "hit_rate": self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"]) if (self._cache_stats["hits"] + self._cache_stats["misses"]) > 0 else 0,
            "available_dates": len(get_available_pick_dates()),
            "memory_usage_mb": sum(df.memory_usage(deep=True).sum() for df in self._daily_picks_cache.values()) / 1024 / 1024
        }
    
    def get_daily_picks_optimized(self, date: str) -> pd.DataFrame:
        """Get daily picks with intelligent caching"""
        # Check agent's internal cache first
        if date in self._daily_picks_cache:
            self._cache_stats["hits"] += 1
            return self._daily_picks_cache[date]
        
        # Load from disk (this itself is cached by Streamlit)
        self._cache_stats["misses"] += 1
        picks_df = load_daily_picks_for_date(date)
        
        # Cache in agent memory (limit cache size to prevent memory bloat)
        if len(self._daily_picks_cache) < 10:  # Keep only 10 most recent dates
            self._daily_picks_cache[date] = picks_df
        else:
            # Remove oldest cached date
            oldest_date = min(self._daily_picks_cache.keys())
            del self._daily_picks_cache[oldest_date]
            self._daily_picks_cache[date] = picks_df
            
        return picks_df
    
    def _get_daily_recommendations_and_bets_optimized(self, mask_id: str, date: str) -> dict:
        """Optimized version using agent's internal caching"""
        result = {
            "mask_id": mask_id,
            "date": date,
            "recommendations": [],
            "actual_bets": [],
            "matches": [],
            "summary": {},
            "debug": {}  # Add debug info
        }
        
        try:
            # Use optimized daily picks loading
            picks_df = self.get_daily_picks_optimized(date)
            
            result["debug"]["picks_file_loaded"] = not picks_df.empty
            result["debug"]["total_records"] = len(picks_df) if not picks_df.empty else 0
            
            if not picks_df.empty:
                # Debug: Check if user exists
                user_picks = picks_df[picks_df["mask_id"] == mask_id]
                result["debug"]["user_found"] = not user_picks.empty
                result["debug"]["available_users"] = picks_df["mask_id"].unique()[:10].tolist()  # First 10 users
                
                if not user_picks.empty:
                    import json
                    top_3_raw = user_picks.iloc[0]["top_3"]
                    result["debug"]["raw_json"] = top_3_raw[:200] + "..." if len(top_3_raw) > 200 else top_3_raw
                    
                    try:
                        recommendations = json.loads(top_3_raw)
                        result["recommendations"] = recommendations
                        result["debug"]["json_parsed"] = True
                    except Exception as e:
                        result["debug"]["json_parse_error"] = str(e)
                        return result
                else:
                    result["debug"]["message"] = f"User {mask_id} not found in picks for {date}"
            
            # Get actual betting activity (already cached in context_data)
            betting_history = self.context_data.get("user_betting_history", pd.DataFrame())
            if not betting_history.empty:
                user_bets = betting_history[
                    (betting_history["mask_id"] == mask_id) & 
                    (betting_history["date"] == date)
                ]
                
                if not user_bets.empty:
                    result["actual_bets"] = user_bets.to_dict('records')
            
            # Find matches between recommendations and actual bets using normalized game descriptions
            if result["recommendations"] and result["actual_bets"]:
                # Normalize recommendation games: "Team A & Team B" -> sorted format
                rec_games = set()
                for rec in result["recommendations"]:
                    game_desc = rec.get("event_description", "") or rec.get("game_1", "")
                    normalized_game = normalize_game_description(game_desc)
                    if normalized_game:
                        rec_games.add(normalized_game)
                
                # Normalize actual bet games: "Team A @ Team B" -> sorted format  
                bet_games = set()
                for bet in result["actual_bets"]:
                    normalized_game = normalize_game_description(bet["event_description"])
                    if normalized_game:
                        bet_games.add(normalized_game)
                
                # Find matches between normalized game descriptions
                matches = rec_games.intersection(bet_games)
                result["matches"] = list(matches)
                
                # Add debug information for transparency
                result["debug"]["normalized_recommendations"] = list(rec_games)
                result["debug"]["normalized_bets"] = list(bet_games) 
                result["debug"]["original_recommendations"] = [
                    rec.get("event_description", "") or rec.get("game_1", "") 
                    for rec in result["recommendations"]
                ]
                result["debug"]["original_bets"] = [bet["event_description"] for bet in result["actual_bets"]]
                
                # Summary statistics
                result["summary"] = {
                    "total_recommendations": len(result["recommendations"]),
                    "total_bets": len(result["actual_bets"]),
                    "matches_found": len(matches),
                    "hit_rate": len(matches) / len(result["recommendations"]) if result["recommendations"] else 0,
                    "precision": len(matches) / len(result["actual_bets"]) if result["actual_bets"] else 0,
                    "total_bet_amount": sum(bet["amount_sum"] for bet in result["actual_bets"])
                }
            
            # Analyze schedule issues that might affect recommendations and betting
            if result["recommendations"]:
                recommended_game_names = [
                    rec.get("event_description", "") or rec.get("game_1", "") 
                    for rec in result["recommendations"]
                ]
                schedule_analysis = analyze_schedule_issues(date, recommended_game_names)
                result["schedule_analysis"] = schedule_analysis
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to analyze recommendations and bets: {str(e)}"}
def get_analysis_agent(context_data) -> NBAAnalysisAgent:
    # simple singleton in session
    if "analysis_agent" not in st.session_state:
        st.session_state.analysis_agent = NBAAnalysisAgent(context_data)
    return st.session_state.analysis_agent

def reset_agent_conversation():
    if "analysis_agent" in st.session_state:
        st.session_state.analysis_agent.conversation_memory = []
        st.session_state.analysis_agent.session_id = str(uuid.uuid4())[:8]
