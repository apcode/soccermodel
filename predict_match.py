#!/usr/bin/env python3
"""
Match Prediction Script

This script demonstrates how to load a trained soccer rating model 
and use it to predict match outcomes.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from team_rating_model import SoccerRatingModel

def predict_match_probabilities(lambda_home, lambda_away, max_goals=5):
    """
    Calculate match outcome probabilities given expected goals
    """
    # Calculate probabilities for different scorelines
    prob_grid = np.zeros((max_goals + 1, max_goals + 1))
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_home = poisson.pmf(home_goals, lambda_home)
            prob_away = poisson.pmf(away_goals, lambda_away)
            prob_grid[home_goals, away_goals] = prob_home * prob_away
    
    # Calculate match outcome probabilities
    home_win_prob = np.sum(prob_grid[np.triu_indices(max_goals + 1, k=1)])
    draw_prob = np.sum(np.diag(prob_grid))
    away_win_prob = np.sum(prob_grid[np.tril_indices(max_goals + 1, k=-1)])
    
    return home_win_prob, draw_prob, away_win_prob, prob_grid

def find_team_by_name(df, team_name):
    """
    Find team API ID by searching team names
    """
    # Search in home team names
    home_matches = df[df['home_team_name'].str.contains(team_name, case=False, na=False)]
    if not home_matches.empty:
        return home_matches.iloc[0]['home_team_api_id'], home_matches.iloc[0]['home_team_name']
    
    # Search in away team names
    away_matches = df[df['away_team_name'].str.contains(team_name, case=False, na=False)]
    if not away_matches.empty:
        return away_matches.iloc[0]['away_team_api_id'], away_matches.iloc[0]['away_team_name']
    
    return None, None

def main():
    """Main prediction demonstration"""
    print("Soccer Match Prediction")
    print("=" * 40)
    
    # Load the trained model
    try:
        model = SoccerRatingModel()
        model.load_model()
    except FileNotFoundError:
        print("No trained model found! Please run team_rating_model.py first to train a model.")
        return
    
    # Load match data to get team names
    df = model.load_and_prepare_data()
    
    # Get team ratings
    ratings = model.get_team_ratings(df)
    
    print(f"\nLoaded model with {len(ratings)} teams")
    print(f"Home advantage: {model.home_advantage:.3f}")
    
    # Show top teams
    print("\nTop 10 teams by overall strength:")
    print(ratings.head(10)[['team_name', 'attack_rating', 'defense_rating', 'overall_strength']])
    
    # Example predictions
    print("\n" + "=" * 40)
    print("EXAMPLE PREDICTIONS")
    print("=" * 40)
    
    # Get some interesting matchups
    top_teams = ratings.head(5)
    
    for i in range(min(3, len(top_teams) - 1)):
        home_team_id = top_teams.iloc[i]['team_api_id']
        away_team_id = top_teams.iloc[i + 1]['team_api_id']
        
        home_team_name = top_teams.iloc[i]['team_name']
        away_team_name = top_teams.iloc[i + 1]['team_name']
        
        # Predict match
        lambda_home, lambda_away = model.predict_goals(home_team_id, away_team_id)
        
        # Calculate probabilities
        home_win_prob, draw_prob, away_win_prob, prob_grid = predict_match_probabilities(
            lambda_home, lambda_away
        )
        
        print(f"\n{home_team_name} vs {away_team_name}")
        print(f"Expected goals: {lambda_home:.2f} - {lambda_away:.2f}")
        print(f"Win probabilities: Home {home_win_prob:.1%} | Draw {draw_prob:.1%} | Away {away_win_prob:.1%}")
        
        # Most likely scoreline
        most_likely_idx = np.unravel_index(prob_grid.argmax(), prob_grid.shape)
        most_likely_prob = prob_grid[most_likely_idx]
        print(f"Most likely score: {most_likely_idx[0]}-{most_likely_idx[1]} ({most_likely_prob:.1%})")
    
    # Interactive prediction
    print("\n" + "=" * 40)
    print("INTERACTIVE PREDICTION")
    print("=" * 40)
    print("Enter team names to predict a match (or 'quit' to exit)")
    print("Available teams:")
    print(", ".join(ratings['team_name'].head(10).values))
    print("...")
    
    while True:
        try:
            home_input = input("\nHome team (or 'quit'): ").strip()
            if home_input.lower() == 'quit':
                break
                
            away_input = input("Away team: ").strip()
            
            # Find teams
            home_id, home_name = find_team_by_name(df, home_input)
            away_id, away_name = find_team_by_name(df, away_input)
            
            if home_id is None:
                print(f"Could not find team matching '{home_input}'")
                continue
                
            if away_id is None:
                print(f"Could not find team matching '{away_input}'")
                continue
            
            # Predict match
            lambda_home, lambda_away = model.predict_goals(home_id, away_id)
            home_win_prob, draw_prob, away_win_prob, prob_grid = predict_match_probabilities(
                lambda_home, lambda_away
            )
            
            print(f"\n{home_name} vs {away_name}")
            print(f"Expected goals: {lambda_home:.2f} - {lambda_away:.2f}")
            print(f"Win probabilities: Home {home_win_prob:.1%} | Draw {draw_prob:.1%} | Away {away_win_prob:.1%}")
            
            # Show most likely scorelines
            flat_probs = prob_grid.flatten()
            sorted_indices = np.argsort(flat_probs)[::-1]
            
            print("Most likely scorelines:")
            for i in range(min(5, len(sorted_indices))):
                idx = sorted_indices[i]
                home_goals, away_goals = np.unravel_index(idx, prob_grid.shape)
                prob = flat_probs[idx]
                print(f"  {home_goals}-{away_goals}: {prob:.1%}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()