# streamlit_app.py
# ------------------------------------------------------------
# NBA Betting Recommender — Streamlit Dashboard Frontend
# Clean UI implementation importing functions from streamlit_app_func.py
# ------------------------------------------------------------
import os
from pathlib import Path
import json, ast
from typing import Optional, Dict, List
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
import openai
import os

# Load environment variables from .env file
load_dotenv()

# Import all backend functions
from streamlit_app_func import (
    DEFAULTS, load_all_data, calculate_kpi, infer_k_value, 
    get_daily_performance_data, get_user_data, derive_user_profile,
    get_picks_data, 
    generate_ai_recommendation, generate_ai_caption, 
    get_simulation_chart_data, generate_ai_schedule_trend_analysis,
    compute_date_specific_diagnostics, 
    generate_date_specific_ai_analysis
)

from ai_agent import (
    load_chatbot_context_data,
    NBAAnalysisAgent,
    get_analysis_agent,
    reset_agent_conversation,
    display_analysis_data,
)


st.set_page_config(
    page_title="NBA Recommender — Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏀"
)

# ================ CUSTOM CSS STYLING ================
st.markdown("""
<style>
    /* Main background gradient */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 1rem;
    }
    
    /* Custom card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
    }
    
    /* Enhanced metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.2rem;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Enhanced Tab Button styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-radius: 0;
        padding: 0.5rem 0;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 20px !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        margin: 0 0.2rem !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s ease !important;
        min-width: auto !important;
    }
    
    /* Individual tab colors */
    .stTabs [data-baseweb="tab"]:nth-child(1) {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53) !important;
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(2) {
        background: linear-gradient(135deg, #4ECDC4, #44A08D) !important;
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(3) {
        background: linear-gradient(135deg, #45B7D1, #2980B9) !important;
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(4) {
        background: linear-gradient(135deg, #9B59B6, #8E44AD) !important;
    }
    
    .stTabs [aria-selected="true"] {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25) !important;
        border: 2px solid rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50, #34495e);
    }
    
    /* Enhanced containers */
    .user-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #4ECDC4;
    }
    
    /* Pick cards */
    .pick-card {
        background: linear-gradient(145deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        margin: 0.5rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Info/success boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# --------------- Defaults (repo-relative) -------------------
REPO = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "metrics_strict": REPO / "reports/metrics/metrics_k3_pairkey.csv",  # Now using pair-key as primary
    "metrics_pair":   REPO / "reports/metrics/metrics_k3_strict.csv",   # Strict as comparison
    "picks_root":     REPO / "reports/daily",
    "sim_vendor":     REPO / "reports/metrics/simulation_periods.csv",
    "sim_actual":     REPO / "reports/metrics/simulation_periods.csv",
    "sim_drift":      REPO / "reports/metrics/schedule_drift_analysis.csv",
    "user_history":   REPO / "data/processed/df_train_rolled.csv",
    "idmaps": [
        REPO / "artifacts/hybrid/hybrid_artifacts.json",
        REPO / "artifacts/als/id_maps.json",
    ],
}

st.set_page_config(page_title="NBA Recommender — Dashboard", layout="wide")

# ----------------- Utils & Loaders --------------------------
@st.cache_data(show_spinner=False)
def _read_csv(path: Path, dtype=None) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def _ensure_datestr(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")

def _safe_mean(s: Optional[pd.Series]) -> float:
    if s is None or s.empty: return float("nan")
    return float(s.mean())

def _infer_k(df: Optional[pd.DataFrame], fallback=3) -> int:
    if df is None: return fallback
    for c in df.columns:
        if "@k" in c.lower():
            try: return int(c.lower().split("@")[-1].split("_")[0])
            except: pass
    return fallback

def _parse_top_json(x):
    if pd.isna(x): return []
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
def _load_idmap_users(paths) -> Optional[set]:
    for p in paths:
        if Path(p).exists():
            d = _read_json(Path(p))
            if d and "user_to_idx" in d:
                return set(map(str, d["user_to_idx"].keys()))
    return None

# --------------- OpenAI GPT Integration ----------------------
def _generate_ai_recommendation(mask_id: str, top_picks: List[dict], user_history: pd.DataFrame, api_key: str = None) -> str:
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
        1. Be encouraging and positive
        2. Reference their specific game preferences and betting patterns from history
        3. Connect today's recommendations to similar teams/matchups they've bet on before
        4. Highlight why today's picks might appeal based on their historical game choices
        5. Keep it conversational and engaging (maximum 2-3 sentences)
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

# --------------- “AI” Caption (rule-based stub) -------------
def _derive_user_profile(u_hist: pd.DataFrame) -> Dict[str, str]:
    """Lightweight profile heuristics for captioning."""
    prof = {}
    if u_hist is None or u_hist.empty:
        return prof
    # venue bias via string contains
    ed = u_hist.get("event_description", pd.Series([], dtype=str)).astype(str).str.lower()
    prof["likes_home"] = "yes" if (ed.str.contains(" vs ").mean() > 0.55) else "no"  # vs ≈ home-left
    # day-of-week preference
    if "date" in u_hist.columns:
        dow = pd.to_datetime(u_hist["date"], errors="coerce").dt.dayofweek.value_counts(normalize=True)
        if not dow.empty:
            fav = int(dow.idxmax())
            prof["fav_dow"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][fav]
    return prof

def _ai_caption(mask_id: str, top3: List[dict], profile: Dict[str,str]) -> str:
    """Deterministic, deploy-safe caption; replace with LLM if desired."""
    if not top3:
        return "No personalized picks available for this user on the selected date."
    bits = []
    if profile.get("likes_home") == "yes":
        bits.append("you’ve shown a tilt toward home matchups")
    if "fav_dow" in profile:
        bits.append(f"you’re often active around {profile['fav_dow']}s")
    reason = "; ".join(bits) if bits else "your recent betting history"
    names = []
    for i, it in enumerate(top3, start=1):
        # Extract game name from game_X key or event_description
        name = it.get("event_description") or next((v for k,v in it.items() if k.startswith("game_")), None)
        if name: names.append(name)
    rec_line = ", ".join(names[:3])
    return f"Based on {reason}, consider: {rec_line}."

# ---------------- Sidebar (paths & filters) -----------------
st.sidebar.header("Paths & Filters")
metrics_path = st.sidebar.text_input("Primary metrics CSV (pair-key)", str(DEFAULTS["metrics_strict"]))
pair_path    = st.sidebar.text_input("Strict metrics CSV (comparison)", str(DEFAULTS["metrics_pair"]))
picks_root   = st.sidebar.text_input("Picks root (date folders)", str(DEFAULTS["picks_root"]))
sim_vendor   = st.sidebar.text_input("Sim Vendor metrics CSV", str(DEFAULTS["sim_vendor"]))
sim_actual   = st.sidebar.text_input("Sim Actual metrics CSV", str(DEFAULTS["sim_actual"]))
sim_drift    = st.sidebar.text_input("Sim Drift CSV", str(DEFAULTS["sim_drift"]))
user_history_path = st.sidebar.text_input("User History CSV", str(DEFAULTS["user_history"]))

st.sidebar.markdown("---")
k = st.sidebar.selectbox("K", [3,5,10], index=0)
min_days = st.sidebar.number_input("Min user-days for user stats", min_value=1, value=3, step=1)
show_pair = st.sidebar.checkbox("Show pair-key diagnostic", value=True)
st.sidebar.button("Refresh caches", on_click=lambda: st.cache_data.clear())

# ---------------- Load data from backend -----------------------
data = load_all_data(metrics_path, pair_path, user_history_path, sim_vendor, sim_actual, sim_drift)
metrics = data['metrics']
pair = data['pair']
user_betting_history = data['user_betting_history']
sim_v = data['sim_v']
sim_a = data['sim_a']
sim_d = data['sim_d']
warm_user_set = data['warm_user_set']

K_infer = infer_k_value(metrics, fallback=k)

# ---------------- Header KPIs -------------------------------
st.markdown('''
<h1 style="text-align: center; font-size: 3rem; font-weight: bold; margin-bottom: 2rem;">
    🏀 <span class="main-title">NBA Betting Recommender Dashboard</span>
</h1>
''', unsafe_allow_html=True)
st.markdown("---")

# Enhanced KPI section with better spacing and visual appeal
st.markdown("### 📊 **Performance Metrics Overview**")
st.markdown("")

krow = st.columns(6)
with krow[0]: st.metric(f"Success Rate", calculate_kpi(metrics, "hit@k"))
with krow[1]: st.metric(f"Accuracy", calculate_kpi(metrics, "precision@k"))
with krow[2]: st.metric(f"Coverage Rate", calculate_kpi(metrics, "recall@k"))
with krow[3]: st.metric(f"Dollar Capture", calculate_kpi(metrics, "w$@k", fmt="{:.2f}"))
with krow[4]: st.metric(f"Ranking Quality", calculate_kpi(metrics, "nDCG@k"))
with krow[5]: st.metric(f"Dollar Ranking Quality", calculate_kpi(metrics, "nDCG$@k"))

# ---------------- Enhanced Tabs with Colored Buttons -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Daily Performance", 
    "👤 Player Analysis", 
    "🧪 Validation & Simulation",
    "🤖 AI Assistant"  
])

# ---- Daily Performance Tab ----
with tab1:
    st.subheader("Daily Performance (pair-key)")
    daily_data = get_daily_performance_data(metrics)
    if daily_data is None:
        st.info("Pair-key metrics CSV with per-(mask_id,date) rows not found.")
    else:
        c1, c2 = st.columns([2,1])
        with c1:
            st.line_chart(daily_data.set_index("date")[["hit","precision"]])
        with c2:
            st.dataframe(daily_data, use_container_width=True, height=420)

# ---- Player Analysis Tab ----
with tab2:
    st.subheader("👤 Player Explorer")
    
    # Get available users from metrics data
    available_users = []
    if metrics is not None:
        available_users = sorted(metrics["mask_id"].dropna().unique().tolist())
    
    if not available_users:
        st.info("No users found in the metrics data.")
    else:
        # User selection dropdown
        mask_id = st.selectbox("Select Player (Mask ID)", options=available_users, key="user_selector")
        
        # Get user data from backend
        user_history, user_metrics = get_user_data(mask_id, metrics, user_betting_history)
        
        # Display user metrics with enhanced styling
        st.markdown("---")
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="user-section">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">📈 Recommendation Performance</h3>
            </div>
            """, unsafe_allow_html=True)
            if not user_metrics.empty:
                user_metrics["date_dt"] = pd.to_datetime(user_metrics["date"], errors="coerce")
                cols = st.columns(2)
                with cols[0]: st.metric("Days observed", f"{user_metrics['date'].nunique()}")
                with cols[1]: st.metric("Avg Success Rate (Hit@K)", f"{user_metrics['hit@k'].mean():.3f}")
                cols = st.columns(2)
                with cols[0]: st.metric("Avg Accuracy (Precision@K)", f"{user_metrics['precision@k'].mean():.3f}")
                with cols[1]: st.metric("Avg Dollar Capture (w$@K)", f"{user_metrics['w$@k'].mean():.2f}" if "w$@k" in user_metrics.columns else "—")
                # Update chart with user-friendly column names
                chart_data = user_metrics.set_index("date_dt")[["hit@k","precision@k"]].copy()
                chart_data.columns = ["Success Rate", "Accuracy"]
                st.line_chart(chart_data)
            else:
                st.info("No recommendation performance data available")
        
        with col2:
            st.markdown("""
            <div class="user-section">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">🎯 Historical Betting Patterns</h3>
            </div>
            """, unsafe_allow_html=True)
            if not user_history.empty:
                total_bets = len(user_history)
                unique_games = user_history['event_description'].nunique() if 'event_description' in user_history.columns else 0
                active_days = user_history['date'].nunique() if 'date' in user_history.columns else 0
                date_range = f"{user_history['date'].min()} to {user_history['date'].max()}" if 'date' in user_history.columns else "Unknown"
                avg_amount = user_history['amount_sum'].mean() if 'amount_sum' in user_history.columns else 0
                
                cols = st.columns(2)
                with cols[0]: st.metric("Total Bets", f"{total_bets:,}")
                with cols[1]: st.metric("Total Active Days", f"{active_days:,}")
                cols = st.columns(2)
                with cols[0]: st.metric("Unique Games", f"{unique_games:,}")
                with cols[1]: st.metric("Avg Bet Amount", f"${avg_amount:.2f}")
                st.caption(f"Period: {date_range}")
                
                # Show top games
                if 'event_description' in user_history.columns and 'amount_sum' in user_history.columns:
                    top_games = (user_history.groupby('event_description')['amount_sum']
                               .agg(['sum', 'count'])
                               .sort_values('sum', ascending=False)
                               .head(5))
                    st.markdown("**Top 5 Games by Total Wagered:**")
                    for game, stats in top_games.iterrows():
                        st.caption(f"• {game}: ${stats['sum']:.2f} ({int(stats['count'])} bets)")
            else:
                st.info("No historical betting data available")

        # Historical bets detailed table
        if not user_history.empty:
            with st.expander("📋 Detailed Betting History"):
                display_cols = ['date', 'event_description', 'amount_sum', 'tickets_n'] if all(col in user_history.columns for col in ['date', 'event_description', 'amount_sum', 'tickets_n']) else user_history.columns
                st.dataframe(user_history[display_cols].head(100), use_container_width=True)

        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #2c3e50; font-size: 2.5rem;">🎯 Today's Recommended Games to Bet</h2>
        </div>
        """, unsafe_allow_html=True)
        # st.markdown("**Today’s Top-3**")
        root = Path(DEFAULTS["picks_root"])
        if root.exists():
            dates = sorted([p.name for p in root.iterdir() if p.is_dir()])
            if dates:
                ds = st.selectbox("Pick a date", options=dates, key="user_pick_date")
                f = root / ds / "picks.csv"
                if f.exists():
                    dfp = pd.read_csv(f, dtype={"mask_id":"string"})
                    row = dfp.loc[dfp["mask_id"]==mask_id]
                    if row.empty:
                        st.info(f"No picks for {mask_id} on {ds}.")
                    else:
                        items = get_picks_data(str(DEFAULTS["picks_root"]), ds, mask_id)[1]
                        
                        # Check vendor schedule for over-scheduled games (game_count >= 8)
                        over_scheduled_detected = False
                        try:
                            playoff_schedule = pd.read_csv(REPO / "data/raw/playoff_schedule.csv")
                            # Convert vendor schedule date format to match ds (YYYY-MM-DD)
                            playoff_schedule['date_formatted'] = pd.to_datetime(playoff_schedule['Date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
                            
                            # Check each recommended game for over-scheduling
                            for item in items:
                                game_name = item.get("event_description") or next((v for k,v in item.items() if k.startswith("game_")), None)
                                if game_name:
                                    # Check if this game on this date has game_count >= 8 in vendor schedule
                                    vendor_check = playoff_schedule[
                                        (playoff_schedule['date_formatted'] == ds) & 
                                        (playoff_schedule['Game'] == game_name)
                                    ]
                                    if not vendor_check.empty and vendor_check['game_count'].iloc[0] >= 8:
                                        over_scheduled_detected = True
                                        break
                        except Exception as e:
                            print(f"Warning: Could not load vendor schedule for over-scheduling check: {e}")
                        
                        if over_scheduled_detected:
                            # Show warning instead of recommendation cards
                            st.warning("⚠️ **No recommended/betting games available for today due to over-scheduled games in vendor schedule.**")
                            st.info("Some games show 8+ games in a playoff series, which violates NBA rules. Recommendations are disabled for data quality.")
                        else:
                            # Enhanced picks cards with beautiful styling
                            st.markdown("<br>", unsafe_allow_html=True)
                            c = st.columns(3, gap="large")
                            for i in range(min(3, len(items))):
                                # Extract game name from game_X key or event_description
                                name = items[i].get("event_description") or next((v for k,v in items[i].items() if k.startswith("game_")), None)
                                score = items[i].get("score", np.nan)
                                with c[i]:
                                    # Create custom styled pick card
                                    score_display = f"{score:.4f}" if pd.notna(score) else "—"
                                    rank_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
                                    st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(145deg, {rank_colors[i]}, {rank_colors[i]}dd);
                                        color: white;
                                        padding: 2rem 1rem;
                                        border-radius: 20px;
                                        text-align: center;
                                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                                        margin: 0.5rem 0;
                                        border: 3px solid rgba(255, 255, 255, 0.3);
                                    ">
                                        <h1 style="font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">#{i+1}</h1>
                                        <h3 style="margin: 1rem 0; font-weight: bold; line-height: 1.3;">{name}</h3>
                                        <div style="
                                            background: rgba(255, 255, 255, 0.2);
                                            padding: 0.5rem;
                                            border-radius: 10px;
                                            margin-top: 1rem;
                                        ">
                                            <strong>Score: {score_display}</strong>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                            # AI recommendation section
                            env_api_key = os.getenv('OPENAI_API_KEY')
                            
                            if env_api_key:
                                api_key = env_api_key
                            else:
                                api_key = st.sidebar.text_input("OpenAI API Key (required)", type="password", help="No API key found in .env file. Please enter manually.")
                                if not api_key:
                                    st.warning("⚠️ No OpenAI API key found. Please add OPENAI_API_KEY to your .env file or enter it manually in the sidebar.")
                        
                            if api_key:
                                ai_recommendation = generate_ai_recommendation(mask_id, items, user_history, api_key)
                                # ai_recommendation = "place holder"
                                
                                # Enhanced AI Recommendation Card
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                                    color: #2c3e50;
                                    padding: 2rem;
                                    border-radius: 20px;
                                    margin: 1.5rem 0;
                                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
                                    border: 2px solid rgba(52, 152, 219, 0.2);
                                ">
                                    <div style="
                                        background: rgba(52, 152, 219, 0.08);
                                        padding: 1.5rem;
                                        border-radius: 15px;
                                        font-size: 1.1rem;
                                        line-height: 1.6;
                                        border-left: 4px solid #3498db;
                                        color: #2c3e50;
                                    ">
                                        {ai_recommendation}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Fallback to rule-based caption
                                profile = derive_user_profile(user_metrics if not user_metrics.empty else pd.DataFrame())
                                st.info(generate_ai_caption(mask_id, items, profile))
                
                            # ================ AI ROOT-CAUSE ANALYSIS FOR MISSES ================
                            if api_key:
                                st.markdown("---")
                                
                                # Calculate missed recommendations
                                if not user_metrics.empty:
                                    hit_rate = user_metrics['hit@k'].mean()
                                    miss_rate = 1 - hit_rate
                                    
                                    if miss_rate > 0.1:  # Only show if miss rate > 10%
                                        with st.spinner("🤖 AI analyzing recommendation misses..."):
                                            # Get missed picks (simplified - in practice you'd have more detailed data)
                                            missed_picks = []
                                            if sim_v is not None and 'period' in sim_v.columns:
                                                # Use first period as context
                                                period_context = sim_v.iloc[0].to_dict()
                                            else:
                                                period_context = {'schedule_accuracy': 1.0, 'hit_rate': hit_rate}
                                            
                                            # Get date-specific diagnostics for the selected date
                                            diagnostics = compute_date_specific_diagnostics(
                                                user_history, ds, items, user_metrics, recent_days=28
                                            )
                                            
                                            # Only run AI analysis if this date was a miss (hit@k = 0)
                                            if diagnostics.get("hit_status") == 0:
                                                root_cause_analysis = generate_date_specific_ai_analysis(
                                                    mask_id, ds, diagnostics, api_key
                                                )
                                            else:
                                                root_cause_analysis = {"info": "At least one recommendations were successful on this date"}
                                        
                                        if "error" not in root_cause_analysis:
                                            # Display AI analysis results
                                            if "analysis_text" in root_cause_analysis:
                                                # Fallback text format - Expandable dropdown
                                                with st.expander(f"🔍 Miss Analysis for User {mask_id}", expanded=False):
                                                    st.markdown(root_cause_analysis['analysis_text'], unsafe_allow_html=True)
                                            else:
                                                with st.expander(f"🔍 Miss Analysis for User {mask_id}", expanded=False):
                                                    st.markdown(root_cause_analysis, unsafe_allow_html=True)
                                        else:
                                            st.error(f"❌ {root_cause_analysis['error']}")
                                    else:
                                        st.success(f"✅ **Excellent Performance!** User has {hit_rate:.1%} hit rate - recommendations are well-aligned with betting behavior")
                                else:
                                    st.info("📊 No performance data available for root-cause analysis")
                        
                else:
                    st.warning(f"Missing picks.csv at {f}")
        else:
            st.info("Picks root not found on disk.")

# ---- Validation & Simulation Tab ----
with tab3:
    # Enhanced Schedule Drift Analysis by Simulation Period
    st.subheader("**📈 Schedule Drift Analysis by Simulation Period**")
    
    if sim_v is None and sim_a is None and sim_d is None:
        st.info("No simulation files found (optional).")
    else:
        # cols = st.columns(2)
        # if sim_v is not None:
        #     with cols[0]:
        #         st.markdown("**Vendor Schedule (strict)**")
        #         st.metric(f"Success Rate (Hit@{K_infer})", calculate_kpi(sim_v, "hit@k"))
        #         st.metric(f"Accuracy (Precision@{K_infer})", calculate_kpi(sim_v, "precision@k"))
        # if sim_a is not None:
        #     with cols[1]:
        #         st.markdown("**Actual Schedule (strict)**")
        #         st.metric(f"Success Rate (Hit@{K_infer})", calculate_kpi(sim_a, "hit@k"))
        #         st.metric(f"Accuracy (Precision@{K_infer})", calculate_kpi(sim_a, "precision@k"))

        # Simulation chart
        chart_data = get_simulation_chart_data(sim_v, sim_a)
        if chart_data is not None:
            st.line_chart(chart_data)
        
        # Overall trend visualization first
        if sim_v is not None and 'period' in sim_v.columns:
            st.markdown("**Performance Trends Across All Periods**")
            
            # Create trend chart data
            trend_data = sim_v[['period', 'hit_rate', 'precision', 'schedule_accuracy']].copy()
            trend_data = trend_data.set_index('period')
            trend_data.columns = ['Hit Rate', 'Precision', 'Schedule Accuracy']
            st.line_chart(trend_data)
        
        # Load simulation periods for dropdown
        if sim_v is not None and 'period' in sim_v.columns:
            periods = sorted(sim_v['period'].unique())
            period_options = []
            for period in periods:
                period_data = sim_v[sim_v['period'] == period].iloc[0]
                start_date = period_data.get('start_date', 'Unknown')
                end_date = period_data.get('end_date', 'Unknown')
                period_options.append(f"Period {period}: {start_date} to {end_date}")
            
            # Dropdown for simulation period selection
            selected_period_str = st.selectbox(
                "**Select 2-week simulation period:**", 
                options=period_options,
                key="drift_period_selector"
            )
            
            # Extract period number from selection
            selected_period = int(selected_period_str.split(":")[0].split(" ")[1])
            
            # Get data for selected period
            period_data = sim_v[sim_v['period'] == selected_period].iloc[0]
            
            # Display period metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Hit Rate", f"{period_data.get('hit_rate', 0):.3f}")
            with col2:
                st.metric("Precision", f"{period_data.get('precision', 0):.3f}")
            with col3:
                st.metric("Schedule Accuracy", f"{period_data.get('schedule_accuracy', 0):.3f}")
            with col4:
                st.metric("Total Bets", f"{period_data.get('total_bets', 0):,}")
            
            # # Visualization of key metrics for this period
            # st.markdown("**📊 Performance Metrics Visualization**")
            
            # # Create comparison chart data
            # metrics_data = {
            #     'Metric': ['Hit Rate', 'Precision', 'Schedule Accuracy', 'Dollar Capture', 'NDCG'],
            #     'Value': [
            #         period_data.get('hit_rate', 0),
            #         period_data.get('precision', 0), 
            #         period_data.get('schedule_accuracy', 0),
            #         period_data.get('dollar_capture', 0),
            #         period_data.get('ndcg', 0)
            #     ]
            # }
            # metrics_chart_df = pd.DataFrame(metrics_data)
            # st.bar_chart(metrics_chart_df.set_index('Metric')['Value'])
            
            # Additional insights
            st.markdown("**🔍 Period Insights**")
            
            # Calculate some insights
            users_evaluated = period_data.get('users_evaluated', 0)
            total_picks = period_data.get('total_picks', 0)
            avg_picks_per_user = total_picks / users_evaluated if users_evaluated > 0 else 0
            
            insight_col1, insight_col2 = st.columns(2)
            with insight_col1:
                st.info(f"👥 **{users_evaluated:,}** recommendations made")
                st.info(f"🎯 **{avg_picks_per_user:.1f}** average picks per recommendation")
            with insight_col2:
                accuracy = period_data.get('schedule_accuracy', 0)
                if accuracy >= 0.8:
                    st.success(f"✅ High schedule accuracy ({accuracy:.1%})")
                elif accuracy >= 0.6:
                    st.warning(f"⚠️ Medium schedule accuracy ({accuracy:.1%})")
                else:
                    st.error(f"❌ Low schedule accuracy ({accuracy:.1%})")
                    
                hit_rate = period_data.get('hit_rate', 0)
                if hit_rate >= 0.15:
                    st.success(f"🎯 Strong hit rate ({hit_rate:.1%})")
                elif hit_rate >= 0.10:
                    st.info(f"📈 Moderate hit rate ({hit_rate:.1%})")
                else:
                    st.warning(f"📉 Lower hit rate ({hit_rate:.1%})")
        
        # Show period-specific schedule comparison
        if sim_v is not None and 'period' in sim_v.columns:
            st.markdown("**📋 Schedule Comparison for Selected Period**")
            
            try:
                # Load schedule data
                vendor_schedule = pd.read_csv(REPO / "data/raw/playoff_schedule.csv")
                actual_schedule = pd.read_csv(REPO / "data/processed/playoff_schedule_actual_enriched.csv")
                
                # Convert dates and filter for selected period
                period_start = period_data.get('start_date', '')
                period_end = period_data.get('end_date', '')
                
                if period_start and period_end:
                    # Filter vendor schedule for period
                    vendor_schedule['date'] = pd.to_datetime(vendor_schedule['Date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
                    period_vendor = vendor_schedule[
                        (vendor_schedule['date'] >= period_start) & 
                        (vendor_schedule['date'] <= period_end)
                    ].copy()
                    
                    # Filter actual schedule for period  
                    actual_schedule['date'] = pd.to_datetime(actual_schedule['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    period_actual = actual_schedule[
                        (actual_schedule['date'] >= period_start) & 
                        (actual_schedule['date'] <= period_end)
                    ].copy()
                    
                    if len(period_vendor) > 0:
                        # Create comparison dataframe
                        comparison_data = []
                        
                        for _, vendor_row in period_vendor.iterrows():
                            vendor_game = vendor_row['Game']
                            vendor_date = vendor_row['date']
                            
                            # Find matching actual game
                            actual_match = period_actual[
                                (period_actual['Game'] == vendor_game) & 
                                (period_actual['date'] == vendor_date)
                            ]
                            
                            if len(actual_match) > 0:
                                actual_event = actual_match.iloc[0]['event_description']
                                status = "✅ Played as scheduled"
                                status_color = "success"
                            else:
                                actual_event = "❌ Game not played"
                                status = "❌ Cancelled/Postponed"
                                status_color = "error"
                            
                            comparison_data.append({
                                'Date': vendor_date,
                                'Vendor Schedule (Game)': vendor_game,
                                'Actual Game Played': actual_event,
                                'Status': status
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Display summary stats
                        total_scheduled = len(comparison_df)
                        games_played = len(comparison_df[comparison_df['Status'].str.contains('✅')])
                        games_cancelled = total_scheduled - games_played
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Games Scheduled", total_scheduled)
                        with col2:
                            st.metric("Games Played", games_played, delta=f"{games_played/total_scheduled:.1%}" if total_scheduled > 0 else "0%")
                        with col3:
                            st.metric("Games Cancelled", games_cancelled, delta=f"-{games_cancelled/total_scheduled:.1%}" if total_scheduled > 0 else "0%")
                        
                        # Display detailed comparison table
                        st.dataframe(comparison_df, use_container_width=True, height=400)
                        
                        # # Highlight cancelled games if any
                        # if games_cancelled > 0:
                        #     cancelled_games = comparison_df[comparison_df['Status'].str.contains('❌')]
                        #     st.warning(f"**{games_cancelled} games were cancelled/postponed during this period:**")
                        #     for _, game in cancelled_games.iterrows():
                        #         st.error(f"🚫 **{game['Date']}**: {game['Vendor Schedule (Game)']} - Not played as scheduled")
                    else:
                        st.info(f"No games scheduled for period {period_start} to {period_end}")
                else:
                    st.warning("Period dates not available for schedule comparison")
                    
            except Exception as e:
                st.error(f"Error loading schedule data: {str(e)}")
                # Fallback to original table if available
                if sim_d is not None:
                    st.markdown("**📋 Overall Schedule Drift Summary (Fallback)**")
                    keep = [c for c in ["total_preseason_games","total_actual_games","games_in_both",
                                        "games_cancelled_or_moved","schedule_accuracy","cancelled_games","added_games"]
                            if c in (sim_d.columns if hasattr(sim_d, "columns") else [])]
                    st.dataframe(sim_d[keep] if keep else sim_d, use_container_width=True, height=300)
        
        # ================ AI-POWERED INSIGHTS SECTION ================
        st.markdown("---")
        st.markdown("## 🤖 **AI-Powered Schedule & Trend Analysis**")
        
        # Check for API key
        env_api_key = os.getenv('OPENAI_API_KEY')
        if env_api_key:
            api_key = env_api_key
        else:
            api_key = st.sidebar.text_input("OpenAI API Key (for AI insights)", type="password", help="Required for AI-powered analysis features")
        
        if not api_key:
            st.warning("⚠️ Enter OpenAI API Key in sidebar to enable AI-powered insights")
        elif sim_v is not None and len(sim_v) > 3 and 'selected_period' in locals():
            # Initialize session state for AI analysis
            if 'ai_analysis_result' not in st.session_state:
                st.session_state.ai_analysis_result = None
            if 'ai_analysis_running' not in st.session_state:
                st.session_state.ai_analysis_running = False
            
            # Calculate schedule statistics for AI
            schedule_stats = {
                'total_scheduled': total_scheduled if 'total_scheduled' in locals() else 0,
                'games_cancelled': games_cancelled if 'games_cancelled' in locals() else 0
            }
            
            # Show button to trigger AI analysis or status
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.session_state.ai_analysis_result is None and not st.session_state.ai_analysis_running:
                    st.info("**AI Analysis Ready** - Click 'Generate AI Insights' to analyze schedule impact and performance trends")
                elif st.session_state.ai_analysis_running:
                    st.info("🔄 **AI Analysis Running** - Please wait while analyzing performance trends...")
                elif st.session_state.ai_analysis_result:
                    st.success("✅ **AI Analysis Complete** - Results displayed below")
            
            with col2:
                # Show Generate button only if no results yet
                if st.session_state.ai_analysis_result is None:
                    if st.button("🤖 Generate AI Insights", disabled=st.session_state.ai_analysis_running):
                        st.session_state.ai_analysis_running = True
                        st.rerun()
                else:
                    # Show Regenerate button when results are available
                    if st.button("🔄 Regenerate Analysis"):
                        st.session_state.ai_analysis_result = None
                        st.session_state.ai_analysis_running = True
                        st.rerun()
            
            # Run AI analysis if triggered
            if st.session_state.ai_analysis_running and st.session_state.ai_analysis_result is None:
                try:
                    schedule_trend_analysis = generate_ai_schedule_trend_analysis(
                        sim_v, period_data.to_dict(), schedule_stats, api_key
                    )
                    st.session_state.ai_analysis_result = schedule_trend_analysis
                    st.session_state.ai_analysis_running = False
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ AI Analysis Error: {str(e)}")
                    st.session_state.ai_analysis_running = False
            
            # Display AI results if available
            if st.session_state.ai_analysis_result:
                # Combined Analysis Card
                st.markdown(f"""
                <div style="
                    background: transparent;
                    color: #2c3e50;
                    padding: 2rem;
                    border-radius: 20px;
                    margin: 1rem 0;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    border: 2px solid #e9ecef;
                ">
                    <div style="
                        background: rgba(248, 249, 250, 0.8);
                        padding: 1.5rem;
                        border-radius: 15px;
                        font-size: 1.1rem;
                        line-height: 1.6;
                    ">
                        {st.session_state.ai_analysis_result}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Need at least 4 simulation periods and selected period for comprehensive schedule-trend analysis")

# ---- AI Assistant chat bot Tab ----
with tab4:
    st.header("🤖 AI Analysis Agent")
    st.markdown("*Chat about simulation accuracy, user performance, and recommendation quality.*")

    with st.spinner("Loading analysis data..."):
        chatbot_context = load_chatbot_context_data()

    if chatbot_context and (not chatbot_context["pair_metrics"].empty or not chatbot_context["simulation_periods"].empty):
        agent = get_analysis_agent(chatbot_context)

        c1, c2, c3, c4 = st.columns([2,1,1,1])
        with c1:
            st.info(f"🧠 **Agent Memory:** {agent.get_conversation_summary()}")
        with c2:
            if st.button("🔄 Refresh Data"):
                st.session_state.analysis_agent.context_data = load_chatbot_context_data()
                st.success("Data refreshed")
                st.rerun()
        with c3:
            if st.button("🧹 Clear Memory"):
                reset_agent_conversation()
                st.success("Memory cleared")
                st.rerun()
        with c4:
            st.metric("Session", agent.session_id)

        st.divider()

        # Show convo
        for msg in agent.conversation_memory:
            with st.chat_message("user" if msg["role"]=="user" else "assistant"):
                st.markdown(msg["content"])
                analysis = msg.get("metadata", {}).get("analysis")
                if analysis and analysis.get("data") and "error" not in analysis["data"]:
                    with st.expander("📊 Analysis Data", expanded=False):
                        display_analysis_data(analysis)

        # Starters
        if not agent.conversation_memory:
            st.subheader("💡 Try one:")
            cols = st.columns(2)
            starters = [
                "Give me an overall performance summary.",
                "Analyze user 100245’s accuracy.",
                "Which simulation period performed best and why?",
                "How does schedule accuracy relate to hit rate?"
            ]
            for i, q in enumerate(starters):
                with cols[i%2]:
                    if st.button(f"💬 {q}", key=f"starter_{i}"):
                        out = agent.process_user_message(q)
                        st.rerun()

        # Input
        if prompt := st.chat_input("Ask about accuracy, misses, users, or periods…"):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    out = agent.process_user_message(prompt)
                    st.markdown(out["response"])
                    if out["analysis_data"] and out["analysis_data"].get("data") and "error" not in out["analysis_data"]["data"]:
                        with st.expander("📊 Analysis Data", expanded=False):
                            display_analysis_data(out["analysis_data"])
            st.rerun()
    else:
        st.error("No validation data found. Generate metrics first.")
