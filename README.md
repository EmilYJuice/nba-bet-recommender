# NBA Bet Recommender — Final Case Study Repo

## Run locally
###  Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables (Optional - for AI recommendations):**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key:
   # OPENAI_API_KEY=sk-proj-your-key-here
   ```

4. **Run the pipeline:**
   ```bash
   python -m src.nba_reco.pipeline.train
   python -m src.nba_reco.pipeline.validate
   python -m src.nba_reco.pipeline.daily_run --date 2025-04-21
   ```

5. **Launch dashboard:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Access to the dashboard only
https://nba-bet-recommender-321836100678.us-central1.run.app

## Features

- **AI-Powered Recommendations**: GPT-4o generates personalized betting advice (requires OpenAI API key)
- **Validation Dashboard**: 2-week simulation analysis with schedule drift detection
- **User Explorer**: Deep dive into individual betting patterns and preferences
- **Performance Metrics**: Hit rates, precision, recall, and dollar capture across different periods
