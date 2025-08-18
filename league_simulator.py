#!/usr/bin/env python3
"""
League Table Simulation and Prediction

This script provides two types of league analysis:

1. EXPECTED POINTS TABLE: 
   - Retroactive analysis of matches already played
   - Uses expected goals from Poisson model to calculate what points teams "should have" earned
   - Compares actual vs expected performance

2. PREDICTED FINAL TABLE:
   - Forward prediction of remaining fixtures in the season
   - Simulates remaining matches to predict final league positions
   - Provides confidence intervals and uncertainty analysis
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from match_simulator import MatchSimulator
from collections import defaultdict

class LeagueSimulator:
    """
    Complete league season simulator with expected points and predictions
    """
    
    def __init__(self, model_path="models/soccer_rating_model.pkl"):
        """Initialize with trained match prediction model"""
        self.match_simulator = MatchSimulator(model_path)
        self.db_path = "data/database.sqlite"
        
    def get_available_leagues_and_seasons(self):
        """Get all available leagues and seasons in the database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT DISTINCT 
            l.name as league_name,
            m.season,
            COUNT(m.id) as match_count,
            MIN(m.date) as season_start,
            MAX(m.date) as season_end
        FROM Match m
        JOIN League l ON m.league_id = l.id
        GROUP BY l.name, m.season
        ORDER BY l.name, m.season
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_league_matches(self, league_name, season):
        """Get all matches for a specific league and season"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            m.id as match_id,
            m.date,
            m.home_team_api_id,
            m.away_team_api_id,
            m.home_team_goal,
            m.away_team_goal,
            ht.team_long_name as home_team_name,
            at.team_long_name as away_team_name,
            l.name as league_name,
            m.season
        FROM Match m
        JOIN Team ht ON m.home_team_api_id = ht.team_api_id
        JOIN Team at ON m.away_team_api_id = at.team_api_id
        JOIN League l ON m.league_id = l.id
        WHERE l.name = ? AND m.season = ?
        ORDER BY m.date
        """
        
        df = pd.read_sql_query(query, conn, params=[league_name, season])
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def calculate_actual_standings(self, matches_df, cutoff_date=None):
        """Calculate actual league table based on real match results"""
        if cutoff_date:
            cutoff_date = pd.to_datetime(cutoff_date)
            played_matches = matches_df[matches_df['date'] <= cutoff_date].copy()
        else:
            played_matches = matches_df.copy()
        
        # Initialize team stats
        teams = set(played_matches['home_team_api_id'].unique()) | set(played_matches['away_team_api_id'].unique())
        
        standings = {}
        for team_id in teams:
            # Get team name
            team_name = self._get_team_name(played_matches, team_id)
            
            standings[team_id] = {
                'team_name': team_name,
                'matches_played': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'goal_difference': 0,
                'points': 0
            }
        
        # Process each match
        for _, match in played_matches.iterrows():
            home_id = match['home_team_api_id']
            away_id = match['away_team_api_id']
            home_goals = match['home_team_goal']
            away_goals = match['away_team_goal']
            
            # Update match counts
            standings[home_id]['matches_played'] += 1
            standings[away_id]['matches_played'] += 1
            
            # Update goals
            standings[home_id]['goals_for'] += home_goals
            standings[home_id]['goals_against'] += away_goals
            standings[away_id]['goals_for'] += away_goals
            standings[away_id]['goals_against'] += home_goals
            
            # Update results and points
            if home_goals > away_goals:  # Home win
                standings[home_id]['wins'] += 1
                standings[home_id]['points'] += 3
                standings[away_id]['losses'] += 1
            elif home_goals < away_goals:  # Away win
                standings[away_id]['wins'] += 1
                standings[away_id]['points'] += 3
                standings[home_id]['losses'] += 1
            else:  # Draw
                standings[home_id]['draws'] += 1
                standings[home_id]['points'] += 1
                standings[away_id]['draws'] += 1
                standings[away_id]['points'] += 1
        
        # Calculate goal differences
        for team_id in standings:
            standings[team_id]['goal_difference'] = (
                standings[team_id]['goals_for'] - standings[team_id]['goals_against']
            )
        
        # Convert to DataFrame and sort
        standings_df = pd.DataFrame.from_dict(standings, orient='index')
        standings_df = standings_df.sort_values(
            ['points', 'goal_difference', 'goals_for'], 
            ascending=[False, False, False]
        ).reset_index()
        standings_df.rename(columns={'index': 'team_id'}, inplace=True)
        standings_df['position'] = range(1, len(standings_df) + 1)
        
        return standings_df
    
    def calculate_expected_points_table(self, matches_df, cutoff_date=None):
        """
        Calculate expected points table based on Poisson model predictions
        for matches that have already been played
        """
        if cutoff_date:
            cutoff_date = pd.to_datetime(cutoff_date)
            played_matches = matches_df[matches_df['date'] <= cutoff_date].copy()
        else:
            played_matches = matches_df.copy()
        
        print(f"Calculating expected points for {len(played_matches)} matches...")
        
        # Initialize team stats
        teams = set(played_matches['home_team_api_id'].unique()) | set(played_matches['away_team_api_id'].unique())
        
        expected_standings = {}
        for team_id in teams:
            team_name = self._get_team_name(played_matches, team_id)
            
            expected_standings[team_id] = {
                'team_name': team_name,
                'matches_played': 0,
                'expected_wins': 0.0,
                'expected_draws': 0.0,
                'expected_losses': 0.0,
                'expected_goals_for': 0.0,
                'expected_goals_against': 0.0,
                'expected_goal_difference': 0.0,
                'expected_points': 0.0,
                'actual_points': 0,
                'points_difference': 0.0
            }
        
        # Process each match with expected outcomes
        for i, match in played_matches.iterrows():
            if i % 50 == 0:
                print(f"  Processed {i}/{len(played_matches)} matches")
                
            home_id = match['home_team_api_id']
            away_id = match['away_team_api_id']
            actual_home_goals = match['home_team_goal']
            actual_away_goals = match['away_team_goal']
            
            try:
                # Get expected outcome from model
                prediction = self.match_simulator.simulate_match(home_id, away_id)
                
                expected_home_goals = prediction['expected_goals']['home']
                expected_away_goals = prediction['expected_goals']['away']
                match_probs = prediction['match_results']
                
                # Update match counts
                expected_standings[home_id]['matches_played'] += 1
                expected_standings[away_id]['matches_played'] += 1
                
                # Update expected goals
                expected_standings[home_id]['expected_goals_for'] += expected_home_goals
                expected_standings[home_id]['expected_goals_against'] += expected_away_goals
                expected_standings[away_id]['expected_goals_for'] += expected_away_goals
                expected_standings[away_id]['expected_goals_against'] += expected_home_goals
                
                # Update expected results and points
                expected_standings[home_id]['expected_wins'] += match_probs['home_win']
                expected_standings[home_id]['expected_draws'] += match_probs['draw']
                expected_standings[home_id]['expected_losses'] += match_probs['away_win']
                expected_standings[home_id]['expected_points'] += (
                    3 * match_probs['home_win'] + 1 * match_probs['draw']
                )
                
                expected_standings[away_id]['expected_wins'] += match_probs['away_win']
                expected_standings[away_id]['expected_draws'] += match_probs['draw']
                expected_standings[away_id]['expected_losses'] += match_probs['home_win']
                expected_standings[away_id]['expected_points'] += (
                    3 * match_probs['away_win'] + 1 * match_probs['draw']
                )
                
                # Update actual points for comparison
                if actual_home_goals > actual_away_goals:  # Home win
                    expected_standings[home_id]['actual_points'] += 3
                elif actual_home_goals < actual_away_goals:  # Away win
                    expected_standings[away_id]['actual_points'] += 3
                else:  # Draw
                    expected_standings[home_id]['actual_points'] += 1
                    expected_standings[away_id]['actual_points'] += 1
                    
            except Exception as e:
                print(f"Error processing match {home_id} vs {away_id}: {e}")
                # Skip this match if prediction fails
                continue
        
        # Calculate expected goal differences and points differences
        for team_id in expected_standings:
            expected_standings[team_id]['expected_goal_difference'] = (
                expected_standings[team_id]['expected_goals_for'] - 
                expected_standings[team_id]['expected_goals_against']
            )
            expected_standings[team_id]['points_difference'] = (
                expected_standings[team_id]['expected_points'] - 
                expected_standings[team_id]['actual_points']
            )
        
        # Convert to DataFrame and sort by expected points
        expected_df = pd.DataFrame.from_dict(expected_standings, orient='index')
        expected_df = expected_df.sort_values(
            ['expected_points', 'expected_goal_difference', 'expected_goals_for'], 
            ascending=[False, False, False]
        ).reset_index()
        expected_df.rename(columns={'index': 'team_id'}, inplace=True)
        expected_df['expected_position'] = range(1, len(expected_df) + 1)
        
        return expected_df
    
    def predict_final_table(self, league_name, season, cutoff_date, num_simulations=1000):
        """
        Predict final league table by simulating remaining fixtures
        """
        print(f"Predicting final table for {league_name} {season} from {cutoff_date}")
        
        # Get all matches for the league/season
        matches_df = self.get_league_matches(league_name, season)
        
        if matches_df.empty:
            raise ValueError(f"No matches found for {league_name} {season}")
        
        # Calculate current standings
        current_standings = self.calculate_actual_standings(matches_df, cutoff_date)
        
        # Get remaining fixtures
        cutoff_date = pd.to_datetime(cutoff_date)
        remaining_fixtures = matches_df[matches_df['date'] > cutoff_date].copy()
        
        print(f"Current standings calculated ({len(current_standings)} teams)")
        print(f"Remaining fixtures: {len(remaining_fixtures)} matches")
        
        if remaining_fixtures.empty:
            print("Season is complete!")
            return current_standings, None
        
        # Simulate remaining season
        expected_additional_points = self._simulate_remaining_fixtures(
            remaining_fixtures, num_simulations
        )
        
        # Combine current points with expected additional points
        final_predictions = current_standings.copy()
        
        for i, row in final_predictions.iterrows():
            team_id = row['team_id']
            current_points = row['points']
            
            if team_id in expected_additional_points:
                stats = expected_additional_points[team_id]
                final_predictions.at[i, 'additional_points'] = stats['mean']
                final_predictions.at[i, 'predicted_final_points'] = current_points + stats['mean']
                final_predictions.at[i, 'points_std'] = stats['std']
                final_predictions.at[i, 'points_p25'] = current_points + stats['p25']
                final_predictions.at[i, 'points_p75'] = current_points + stats['p75']
            else:
                # No remaining matches for this team
                final_predictions.at[i, 'additional_points'] = 0
                final_predictions.at[i, 'predicted_final_points'] = current_points
                final_predictions.at[i, 'points_std'] = 0
                final_predictions.at[i, 'points_p25'] = current_points
                final_predictions.at[i, 'points_p75'] = current_points
        
        # Sort by predicted final points
        final_predictions = final_predictions.sort_values(
            ['predicted_final_points', 'goal_difference', 'goals_for'], 
            ascending=[False, False, False]
        ).reset_index(drop=True)
        final_predictions['predicted_position'] = range(1, len(final_predictions) + 1)
        
        return final_predictions, remaining_fixtures
    
    def _simulate_remaining_fixtures(self, remaining_fixtures, num_simulations):
        """Simulate remaining fixtures to get expected additional points"""
        teams = set(remaining_fixtures['home_team_api_id'].unique()) | set(remaining_fixtures['away_team_api_id'].unique())
        
        print(f"Simulating {len(remaining_fixtures)} remaining matches with {num_simulations} simulations...")
        
        all_simulations = defaultdict(list)
        
        for sim in range(num_simulations):
            if sim % 100 == 0:
                print(f"  Simulation {sim}/{num_simulations}")
            
            simulation_points = {team: 0 for team in teams}
            
            for _, match in remaining_fixtures.iterrows():
                home_id = match['home_team_api_id']
                away_id = match['away_team_api_id']
                
                try:
                    prediction = self.match_simulator.simulate_match(home_id, away_id)
                    results = prediction['match_results']
                    
                    # Sample outcome based on probabilities
                    rand = np.random.random()
                    
                    if rand < results['home_win']:
                        simulation_points[home_id] += 3
                    elif rand < results['home_win'] + results['draw']:
                        simulation_points[home_id] += 1
                        simulation_points[away_id] += 1
                    else:
                        simulation_points[away_id] += 3
                        
                except Exception as e:
                    # Default to draw if prediction fails
                    simulation_points[home_id] += 1
                    simulation_points[away_id] += 1
            
            for team in teams:
                all_simulations[team].append(simulation_points[team])
        
        # Calculate statistics
        expected_points = {}
        for team in teams:
            points_array = np.array(all_simulations[team])
            expected_points[team] = {
                'mean': np.mean(points_array),
                'std': np.std(points_array),
                'p25': np.percentile(points_array, 25),
                'p75': np.percentile(points_array, 75)
            }
        
        return expected_points
    
    def _get_team_name(self, matches_df, team_id):
        """Helper to get team name from team ID"""
        home_match = matches_df[matches_df['home_team_api_id'] == team_id]
        if not home_match.empty:
            return home_match['home_team_name'].iloc[0]
        
        away_match = matches_df[matches_df['away_team_api_id'] == team_id]
        if not away_match.empty:
            return away_match['away_team_name'].iloc[0]
        
        return f"Team_{team_id}"
    
    def print_expected_points_table(self, expected_df):
        """Print formatted expected points table"""
        print("=" * 110)
        print("EXPECTED POINTS TABLE (Based on Match Predictions)")
        print("=" * 110)
        
        print(f"{'Pos':<3} {'Team':<25} {'MP':<3} {'xPts':<5} {'Pts':<4} {'Diff':<5} {'xGF':<5} {'xGA':<5} {'xGD':<5}")
        print("-" * 110)
        
        for _, row in expected_df.iterrows():
            pos = row['expected_position']
            team = row['team_name'][:23]
            mp = int(row['matches_played'])
            xpts = row['expected_points']
            pts = int(row['actual_points'])
            diff = row['points_difference']
            xgf = row['expected_goals_for']
            xga = row['expected_goals_against']
            xgd = row['expected_goal_difference']
            
            print(f"{pos:<3} {team:<25} {mp:<3} {xpts:<5.1f} {pts:<4} {diff:<+5.1f} {xgf:<5.1f} {xga:<5.1f} {xgd:<+5.1f}")
        
        print("-" * 110)
        print("xPts=Expected Points, Pts=Actual Points, Diff=xPts-Pts")
        print("xGF=Expected Goals For, xGA=Expected Goals Against, xGD=Expected Goal Difference")
    
    def print_predicted_table(self, predictions_df, cutoff_date):
        """Print formatted predicted final table"""
        print("=" * 100)
        print(f"PREDICTED FINAL LEAGUE TABLE (from {cutoff_date})")
        print("=" * 100)
        
        print(f"{'Pos':<3} {'Team':<25} {'MP':<3} {'Pts':<4} {'Pred':<5} {'Add':<4} {'Range':<8} {'GD':<4}")
        print("-" * 100)
        
        for _, row in predictions_df.iterrows():
            pos = row['predicted_position']
            team = row['team_name'][:23]
            mp = row['matches_played']
            current_pts = int(row['points'])
            pred_pts = row['predicted_final_points']
            add_pts = row['additional_points']
            points_range = f"{row['points_p25']:.0f}-{row['points_p75']:.0f}"
            gd = row['goal_difference']
            
            print(f"{pos:<3} {team:<25} {mp:<3} {current_pts:<4} {pred_pts:<5.1f} {add_pts:<4.1f} {points_range:<8} {gd:<+4.0f}")
        
        print("-" * 100)
        print("Pred=Predicted Final Points, Add=Additional Points Expected")
        print("Range=25th-75th percentile of final points")

def main():
    """Main function for interactive league simulation"""
    print("League Simulator - Expected Points & Predictions")
    print("=" * 60)
    
    try:
        simulator = LeagueSimulator()
    except:
        print("Could not initialize simulator. Please ensure model is trained.")
        return
    
    # Show available leagues and seasons
    leagues_df = simulator.get_available_leagues_and_seasons()
    print("\nAvailable leagues and seasons:")
    print(leagues_df.to_string(index=False))
    
    while True:
        try:
            print("\n" + "=" * 60)
            print("1. Expected Points Table (retroactive analysis)")
            print("2. Predicted Final Table (forward prediction)")
            print("3. Both analyses")
            print("q. Quit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice.lower() == 'q':
                break
            
            if choice in ['1', '2', '3']:
                league = input("Enter league name: ").strip()
                season = input("Enter season: ").strip()
                cutoff = input("Enter cutoff date (YYYY-MM-DD): ").strip()
                
                matches_df = simulator.get_league_matches(league, season)
                if matches_df.empty:
                    print("No matches found for that league/season")
                    continue
                
                if choice in ['1', '3']:
                    print("\nCalculating Expected Points Table...")
                    expected_df = simulator.calculate_expected_points_table(matches_df, cutoff)
                    simulator.print_expected_points_table(expected_df)
                
                if choice in ['2', '3']:
                    print("\nPredicting Final Table...")
                    predictions_df, remaining = simulator.predict_final_table(
                        league, season, cutoff
                    )
                    if predictions_df is not None:
                        simulator.print_predicted_table(predictions_df, cutoff)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()