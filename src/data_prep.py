"""
NBA Betting Data Preparation and Quality Checks
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CELL 1: Load Data and Basic Setup
# =============================================================================

def load_and_basic_info():
    """Load data and show basic information"""
    print("=" * 60)
    print("NBA BETTING DATA PREPARATION - BASIC INFO")
    print("=" * 60)
    
    # Load the data
    df = pd.read_excel('../data/raw/df_train.xlsx')
    
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


# =============================================================================
# CELL 3: Row Count and Missing Values
# =============================================================================

def row_count_and_missing_check(df):
    """Check row count and missing values"""
    print("\n" + "=" * 60)
    print("ROW COUNT & MISSING VALUES")
    print("=" * 60)
    
    expected_rows = 216852
    actual_rows = len(df)
    
    print(f"Expected rows: {expected_rows:,}")
    print(f"Actual rows: {actual_rows:,}")
    
    if actual_rows == expected_rows:
        print("✅ Row count matches expectation")
    else:
        diff = actual_rows - expected_rows
        print(f"❌ Row count mismatch: {diff:+,} rows")
    
    print("\nMissing Values by Column:")
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)
    
    for col in df.columns:
        missing_count = missing_summary[col]
        missing_percent = missing_pct[col]
        status = "✅" if missing_count == 0 else "⚠️"
        print(f"  {status} {col}: {missing_count:,} ({missing_percent}%)")
    

# =============================================================================
# CELL 4: Duplicate Detection
# =============================================================================

def duplicate_detection(df):
    """Check for exact and logical duplicates"""
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTION")
    print("=" * 60)
    
    # Exact duplicates
    exact_dupes = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_dupes:,}")
    
    if exact_dupes > 0:
        print("❌ Found exact duplicates")
        print("Sample duplicates:")
        print(df[df.duplicated(keep=False)].head())
    else:
        print("✅ No exact duplicates found")
    
    # Logical duplicates (same mask_id, betdate, event_description)
    if all(col in df.columns for col in ['mask_id', 'betdate', 'event_description']):
        logical_dupes = df.duplicated(subset=['mask_id', 'betdate', 'event_description']).sum()
        print(f"\nLogical duplicates (same mask_id + betdate + event_description): {logical_dupes:,}")
        
        if logical_dupes > 0:
            print("⚠️ Found logical duplicates")
            # Show examples
            dupe_mask = df.duplicated(subset=['mask_id', 'betdate', 'event_description'], keep=False)
            print("Sample logical duplicates:")
            print(df[dupe_mask][['mask_id', 'betdate', 'event_description', 'wager_amount']].head(10))
        else:
            print("✅ No logical duplicates found")
    


# =============================================================================
# CELL 5: mask_id Sanity Checks
# =============================================================================

def mask_id_sanity_check(df):
    """Comprehensive mask_id validation"""
    print("\n" + "=" * 60)
    print("MASK_ID SANITY CHECKS")
    print("=" * 60)
    
    if 'mask_id' not in df.columns:
        print("❌ mask_id column not found")
        return df
    
    mask_id_col = df['mask_id']
    
    # Type check
    print(f"Data type: {mask_id_col.dtype}")
    
    # Null/empty check
    null_count = mask_id_col.isnull().sum()
    empty_count = (mask_id_col == '').sum() if mask_id_col.dtype == 'object' else 0
    
    print(f"Null values: {null_count:,}")
    print(f"Empty strings: {empty_count:,}")
    
    if null_count == 0 and empty_count == 0:
        print("✅ No null or empty mask_ids")
    else:
        print("❌ Found null or empty mask_ids")
    
    # Cardinality
    unique_players = mask_id_col.nunique()
    print(f"\nUnique players (cardinality): {unique_players:,}")
    
    # Records per player statistics
    records_per_player = mask_id_col.value_counts()
    
    print(f"\nRecords per player statistics:")
    print(f"  Min: {records_per_player.min():,}")
    print(f"  Median: {records_per_player.median():.0f}")
    print(f"  95th percentile: {records_per_player.quantile(0.95):.0f}")
    print(f"  Max: {records_per_player.max():,}")
    
    # Singletons (players with only 1 record)
    singletons = (records_per_player == 1).sum()
    print(f"  Singletons (1 record): {singletons:,} ({singletons/unique_players*100:.1f}%)")
    
    # Whales (top 1% of players by record count)
    whale_threshold = records_per_player.quantile(0.99)
    whales = (records_per_player >= whale_threshold).sum()
    print(f"  Whales (top 1%): {whales:,} players with ≥{whale_threshold:.0f} records")
    
    # Non-conforming IDs check
    if mask_id_col.dtype == 'object':
        # Check for spaces, weird characters
        has_spaces = mask_id_col.str.contains(' ', na=False).sum()
        has_special_chars = mask_id_col.str.contains(r'[^a-zA-Z0-9_-]', na=False).sum()
        
        print(f"\nID format issues:")
        print(f"  Contains spaces: {has_spaces:,}")
        print(f"  Contains special chars: {has_special_chars:,}")
        
        if has_spaces > 0 or has_special_chars > 0:
            print("⚠️ Non-conforming mask_ids found")
            # Show examples
            problematic = mask_id_col[
                mask_id_col.str.contains(' ', na=False) | 
                mask_id_col.str.contains(r'[^a-zA-Z0-9_-]', na=False)
            ].unique()[:10]
            print("Examples:", list(problematic))
        else:
            print("✅ All mask_ids have clean format")
    

# =============================================================================
# CELL 6: betdate Sanity Checks
# =============================================================================

def betdate_sanity_check(df):
    """Comprehensive betdate validation and parsing"""
    print("\n" + "=" * 60)
    print("BETDATE SANITY CHECKS")
    print("=" * 60)
    
    if 'betdate' not in df.columns:
        print("❌ betdate column not found")
        return df
    
    betdate_col = df['betdate'].copy()
    
    print(f"Original data type: {betdate_col.dtype}")
    print(f"Sample values:")
    print(betdate_col.head(10).tolist())
    
    # Try robust parsing with multiple formats
    parsed_dates = []
    parse_failures = 0
    
    for idx, date_val in enumerate(betdate_col):
        if pd.isna(date_val):
            parsed_dates.append(pd.NaT)
            continue
            
        try:
            # Try pandas parsing first
            if isinstance(date_val, str):
                parsed_date = pd.to_datetime(date_val, infer_datetime_format=True)
            else:
                parsed_date = pd.to_datetime(date_val)
            
            # Strip timezone and keep only date
            if parsed_date.tz is not None:
                parsed_date = parsed_date.tz_localize(None)
            
            parsed_dates.append(parsed_date.date())
            
        except:
            parsed_dates.append(pd.NaT)
            parse_failures += 1
    
    df['parsed_betdate'] = parsed_dates
    
    print(f"\nParsing results:")
    print(f"  Successfully parsed: {len(df) - parse_failures:,}")
    print(f"  Parse failures: {parse_failures:,} ({parse_failures/len(df)*100:.2f}%)")
    
    if parse_failures > 0:
        print("❌ Date parsing issues found")
        failed_mask = pd.isna(df['parsed_betdate']) & pd.notna(df['betdate'])
        if failed_mask.any():
            print("Examples of unparseable dates:")
            print(df[failed_mask]['betdate'].head(10).tolist())
    else:
        print("✅ All dates parsed successfully")
    
    # Date range analysis
    valid_dates = df['parsed_betdate'].dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        
        print(f"\nDate range:")
        print(f"  Min date: {min_date}")
        print(f"  Max date: {max_date}")
        print(f"  Span: {(max_date - min_date).days} days")
        
        # Check for future dates
        today = datetime.now().date()
        future_dates = (valid_dates > today).sum()
        if future_dates > 0:
            print(f"⚠️  Future dates found: {future_dates:,}")
        else:
            print("✅ No future dates")
        
        # Check for pre-season anomalies (before October for NBA)
        pre_season = valid_dates[pd.to_datetime(valid_dates).dt.month < 10]
        if len(pre_season) > 0:
            print(f"ℹ️  Pre-regular-season dates: {len(pre_season):,}")
        
        # Daily distribution
        daily_counts = pd.Series(valid_dates).value_counts().sort_index()
        print(f"\nDaily bet distribution:")
        print(f"  Days with bets: {len(daily_counts):,}")
        print(f"  Avg bets per day: {daily_counts.mean():.1f}")
        print(f"  Max bets in a day: {daily_counts.max():,}")
        
        # Check for gaps (days with 0 bets)
        date_range = pd.date_range(min_date, max_date, freq='D')
        missing_days = len(date_range) - len(daily_counts)
        if missing_days > 0:
            print(f"⚠️  Days with no bets: {missing_days:,}")
    
    # Check for duplicates at (player, date) level
    if 'mask_id' in df.columns:
        player_date_dupes = df.groupby(['mask_id', 'parsed_betdate']).size()
        multi_bets_same_day = (player_date_dupes > 1).sum()
        print(f"\nPlayer-date duplicates:")
        print(f"  Player-date combinations with multiple bets: {multi_bets_same_day:,}")
        if multi_bets_same_day > 0:
            print("ℹ️  Players betting multiple times same day detected")
    
