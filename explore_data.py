#!/usr/bin/env python3
"""
Soccer Data Exploration Script
Explores the Kaggle European Soccer Database to understand the structure and content.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def connect_to_database(db_path="data/database.sqlite"):
    """Connect to the SQLite database"""
    conn = sqlite3.connect(db_path)
    return conn

def explore_database_structure(conn):
    """Explore the structure of the database"""
    print("=== DATABASE STRUCTURE ===")
    
    # Get all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Number of tables: {len(tables)}")
    print("Tables:", [table[0] for table in tables])
    
    # For each table, show structure and sample size
    for table_name in [table[0] for table in tables]:
        print(f"\n--- Table: {table_name} ---")
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"Columns: {len(columns)}")
        for col in columns[:10]:  # Show first 10 columns
            print(f"  {col[1]} ({col[2]})")
        if len(columns) > 10:
            print(f"  ... and {len(columns) - 10} more columns")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"Rows: {row_count:,}")

def load_key_tables(conn):
    """Load the key tables into pandas DataFrames"""
    print("\n=== LOADING KEY TABLES ===")
    
    # Load matches table
    matches_df = pd.read_sql_query("SELECT * FROM Match", conn)
    print(f"Matches loaded: {len(matches_df):,} rows")
    
    # Load teams table
    teams_df = pd.read_sql_query("SELECT * FROM Team", conn)
    print(f"Teams loaded: {len(teams_df):,} rows")
    
    # Load leagues table
    leagues_df = pd.read_sql_query("SELECT * FROM League", conn)
    print(f"Leagues loaded: {len(leagues_df):,} rows")
    
    # Load players table
    players_df = pd.read_sql_query("SELECT * FROM Player", conn)
    print(f"Players loaded: {len(players_df):,} rows")
    
    return matches_df, teams_df, leagues_df, players_df

def analyze_matches(matches_df, teams_df, leagues_df):
    """Analyze the matches data"""
    print("\n=== MATCHES ANALYSIS ===")
    
    # Basic info
    print(f"Match data shape: {matches_df.shape}")
    print(f"Date range: {matches_df['date'].min()} to {matches_df['date'].max()}")
    
    # Check for missing values in key columns
    key_columns = ['home_team_goal', 'away_team_goal', 'home_team_api_id', 'away_team_api_id']
    print("\nMissing values in key columns:")
    for col in key_columns:
        if col in matches_df.columns:
            missing = matches_df[col].isnull().sum()
            print(f"  {col}: {missing} ({missing/len(matches_df)*100:.1f}%)")
    
    # Goal statistics
    if 'home_team_goal' in matches_df.columns and 'away_team_goal' in matches_df.columns:
        print(f"\nGoal Statistics:")
        print(f"Average home goals: {matches_df['home_team_goal'].mean():.2f}")
        print(f"Average away goals: {matches_df['away_team_goal'].mean():.2f}")
        print(f"Total goals per match: {(matches_df['home_team_goal'] + matches_df['away_team_goal']).mean():.2f}")
    
    # Leagues and seasons
    if 'league_id' in matches_df.columns:
        print(f"\nLeagues in dataset: {matches_df['league_id'].nunique()}")
        
    if 'season' in matches_df.columns:
        print(f"Seasons: {sorted(matches_df['season'].unique())}")
    
    # Sample of matches data
    print(f"\nSample matches:")
    sample_cols = ['date', 'home_team_goal', 'away_team_goal']
    if all(col in matches_df.columns for col in sample_cols):
        print(matches_df[sample_cols].head())

def analyze_teams(teams_df):
    """Analyze the teams data"""
    print("\n=== TEAMS ANALYSIS ===")
    
    print(f"Teams data shape: {teams_df.shape}")
    print(f"Unique teams: {teams_df['team_api_id'].nunique()}")
    
    # Sample teams
    print(f"\nSample teams:")
    display_cols = ['team_long_name', 'team_short_name']
    available_cols = [col for col in display_cols if col in teams_df.columns]
    if available_cols:
        print(teams_df[available_cols].head(10))

def create_basic_visualizations(matches_df):
    """Create basic visualizations of the data"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Goal distribution
    if 'home_team_goal' in matches_df.columns and 'away_team_goal' in matches_df.columns:
        # Home vs Away goals
        axes[0, 0].hist([matches_df['home_team_goal'], matches_df['away_team_goal']], 
                       bins=range(0, 8), alpha=0.7, label=['Home', 'Away'])
        axes[0, 0].set_title('Goal Distribution')
        axes[0, 0].set_xlabel('Goals')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Total goals per match
        total_goals = matches_df['home_team_goal'] + matches_df['away_team_goal']
        axes[0, 1].hist(total_goals, bins=range(0, 12), alpha=0.7)
        axes[0, 1].set_title('Total Goals per Match')
        axes[0, 1].set_xlabel('Total Goals')
        axes[0, 1].set_ylabel('Frequency')
    
    # Matches over time
    if 'date' in matches_df.columns:
        # Convert date to datetime if it's not already
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        matches_by_month = matches_df.groupby(matches_df['date'].dt.to_period('M')).size()
        axes[1, 0].plot(matches_by_month.index.astype(str), matches_by_month.values)
        axes[1, 0].set_title('Matches Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Number of Matches')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Home advantage
    if 'home_team_goal' in matches_df.columns and 'away_team_goal' in matches_df.columns:
        home_wins = (matches_df['home_team_goal'] > matches_df['away_team_goal']).sum()
        away_wins = (matches_df['home_team_goal'] < matches_df['away_team_goal']).sum()
        draws = (matches_df['home_team_goal'] == matches_df['away_team_goal']).sum()
        
        results = ['Home Win', 'Draw', 'Away Win']
        counts = [home_wins, draws, away_wins]
        
        axes[1, 1].pie(counts, labels=results, autopct='%1.1f%%')
        axes[1, 1].set_title('Match Results Distribution')
    
    plt.tight_layout()
    plt.savefig('soccer_data_exploration.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'soccer_data_exploration.png'")

def main():
    """Main exploration function"""
    print("Starting Soccer Data Exploration...")
    print("=" * 50)
    
    # Connect to database
    try:
        conn = connect_to_database()
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    try:
        # Explore database structure
        explore_database_structure(conn)
        
        # Load key tables
        matches_df, teams_df, leagues_df, players_df = load_key_tables(conn)
        
        # Analyze the data
        analyze_matches(matches_df, teams_df, leagues_df)
        analyze_teams(teams_df)
        
        # Create visualizations
        create_basic_visualizations(matches_df)
        
        print("\n" + "=" * 50)
        print("Data exploration completed successfully!")
        print("Next steps:")
        print("1. Review the generated visualizations")
        print("2. Identify features for team rating model")
        print("3. Plan data preprocessing pipeline")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()