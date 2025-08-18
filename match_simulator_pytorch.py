#!/usr/bin/env python3
"""
Match Simulation Engine using PyTorch Model

This script uses the trained PyTorch Poisson model to simulate match outcomes.
Compatible with both CPU and GPU trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from team_rating_model_pytorch import SoccerRatingModelPyTorch
import sqlite3

class MatchSimulatorPyTorch:
    """
    Soccer match simulator using PyTorch Poisson goal model
    """
    
    def __init__(self, model_path="models/soccer_rating_model_pytorch.pkl"):
        """Initialize simulator with trained PyTorch model"""
        self.model = SoccerRatingModelPyTorch()
        try:
            self.model.load_model(model_path)
            print(f"PyTorch model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"PyTorch model not found at {model_path}")
            print("Please run team_rating_model_pytorch.py first to train a model")
            raise
        
        # Load team data for lookups
        self.team_data = self.model.load_and_prepare_data()
        
    def simulate_match(self, home_team, away_team, max_goals=5, num_simulations=10000):
        """
        Simulate a match between two teams using PyTorch model
        """
        # Get team IDs
        home_id, home_name = self._get_team_info(home_team)
        away_id, away_name = self._get_team_info(away_team)
        
        if home_id is None or away_id is None:
            raise ValueError("Could not find one or both teams")
        
        # Get expected goals from PyTorch model
        lambda_home, lambda_away = self.model.predict_goals(home_id, away_id)
        
        print(f"\nSimulating: {home_name} vs {away_name}")
        print(f"Expected goals: {lambda_home:.3f} - {lambda_away:.3f}")
        
        # Calculate exact probabilities using Poisson PMF
        prob_matrix = self._calculate_probability_matrix(lambda_home, lambda_away, max_goals)
        
        # Calculate match result probabilities
        results = self._calculate_match_results(prob_matrix)
        
        # Get most likely scorelines
        scorelines = self._get_scoreline_probabilities(prob_matrix, max_goals)
        
        # Run Monte Carlo simulation for validation
        mc_results = self._monte_carlo_simulation(lambda_home, lambda_away, num_simulations)
        
        return {
            'home_team': home_name,
            'away_team': away_name,
            'home_id': home_id,
            'away_id': away_id,
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away
            },
            'match_results': results,
            'scorelines': scorelines,
            'probability_matrix': prob_matrix,
            'monte_carlo': mc_results
        }
    
    def _get_team_info(self, team_identifier):
        """Get team ID and name from identifier"""
        if isinstance(team_identifier, (int, np.integer)):
            # Assume it's a team ID
            team_match = self.team_data[
                (self.team_data['home_team_api_id'] == team_identifier) |
                (self.team_data['away_team_api_id'] == team_identifier)
            ]
            if not team_match.empty:
                if team_identifier in team_match['home_team_api_id'].values:
                    return team_identifier, team_match[team_match['home_team_api_id'] == team_identifier]['home_team_name'].iloc[0]
                else:
                    return team_identifier, team_match[team_match['away_team_api_id'] == team_identifier]['away_team_name'].iloc[0]
        else:
            # Assume it's a team name - search for it
            home_matches = self.team_data[
                self.team_data['home_team_name'].str.contains(str(team_identifier), case=False, na=False)
            ]
            if not home_matches.empty:
                return home_matches.iloc[0]['home_team_api_id'], home_matches.iloc[0]['home_team_name']
                
            away_matches = self.team_data[
                self.team_data['away_team_name'].str.contains(str(team_identifier), case=False, na=False)
            ]
            if not away_matches.empty:
                return away_matches.iloc[0]['away_team_api_id'], away_matches.iloc[0]['away_team_name']
        
        return None, None
    
    def _calculate_probability_matrix(self, lambda_home, lambda_away, max_goals):
        """Calculate probability matrix for all scorelines"""
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob_home = poisson.pmf(home_goals, lambda_home)
                prob_away = poisson.pmf(away_goals, lambda_away)
                prob_matrix[home_goals, away_goals] = prob_home * prob_away
        
        return prob_matrix
    
    def _calculate_match_results(self, prob_matrix):
        """Calculate overall match result probabilities"""
        max_goals = prob_matrix.shape[0] - 1
        
        # Home win: home_goals > away_goals
        home_win = 0
        for h in range(max_goals + 1):
            for a in range(h):  # a < h
                home_win += prob_matrix[h, a]
        
        # Away win: away_goals > home_goals  
        away_win = 0
        for h in range(max_goals + 1):
            for a in range(h + 1, max_goals + 1):  # a > h
                away_win += prob_matrix[h, a]
        
        # Draw: home_goals = away_goals
        draw = 0
        for i in range(max_goals + 1):
            draw += prob_matrix[i, i]
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win
        }
    
    def _get_scoreline_probabilities(self, prob_matrix, max_goals):
        """Get scoreline probabilities sorted by likelihood"""
        scorelines = []
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = prob_matrix[home_goals, away_goals]
                scorelines.append({
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'probability': prob,
                    'scoreline': f"{home_goals}-{away_goals}"
                })
        
        # Sort by probability
        scorelines.sort(key=lambda x: x['probability'], reverse=True)
        
        return scorelines
    
    def _monte_carlo_simulation(self, lambda_home, lambda_away, num_simulations):
        """Run Monte Carlo simulation for validation"""
        home_goals = np.random.poisson(lambda_home, num_simulations)
        away_goals = np.random.poisson(lambda_away, num_simulations)
        
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        return {
            'home_win': home_wins / num_simulations,
            'draw': draws / num_simulations,
            'away_win': away_wins / num_simulations,
            'avg_home_goals': np.mean(home_goals),
            'avg_away_goals': np.mean(away_goals),
            'simulations': num_simulations
        }
    
    def print_prediction_summary(self, prediction):
        """Print a comprehensive prediction summary"""
        print("=" * 60)
        print(f"MATCH PREDICTION: {prediction['home_team']} vs {prediction['away_team']}")
        print("=" * 60)
        
        # Expected goals
        home_exp = prediction['expected_goals']['home']
        away_exp = prediction['expected_goals']['away']
        print(f"\nExpected Goals: {home_exp:.2f} - {away_exp:.2f}")
        
        # Match result probabilities
        results = prediction['match_results']
        print(f"\nMatch Result Probabilities:")
        print(f"  {prediction['home_team']} Win: {results['home_win']:.1%}")
        print(f"  Draw:                     {results['draw']:.1%}")
        print(f"  {prediction['away_team']} Win: {results['away_win']:.1%}")
        
        # Most likely scorelines
        print(f"\nMost Likely Scorelines:")
        for i, scoreline in enumerate(prediction['scorelines'][:10]):
            print(f"  {i+1:2d}. {scoreline['scoreline']:>5s}: {scoreline['probability']:.1%}")
        
        # Monte Carlo validation
        mc = prediction['monte_carlo']
        print(f"\nMonte Carlo Validation ({mc['simulations']:,} simulations):")
        print(f"  Home Win: {mc['home_win']:.1%}")
        print(f"  Draw:     {mc['draw']:.1%}")
        print(f"  Away Win: {mc['away_win']:.1%}")
        print(f"  Avg Goals: {mc['avg_home_goals']:.2f} - {mc['avg_away_goals']:.2f}")

def main():
    """Main function for interactive match simulation with PyTorch model"""
    print("Soccer Match Simulator (PyTorch)")
    print("=" * 40)
    
    try:
        # Initialize simulator
        simulator = MatchSimulatorPyTorch()
    except:
        return
    
    # Get available teams
    teams_df = simulator.model.get_team_ratings(simulator.team_data)
    print(f"\nAvailable teams ({len(teams_df)} total):")
    print("Top 15 teams by rating:")
    for i, (_, team) in enumerate(teams_df.head(15).iterrows()):
        print(f"  {i+1:2d}. {team['team_name']}")
    
    # Interactive simulation
    while True:
        try:
            print("\n" + "=" * 60)
            home_input = input("Enter home team name (or 'quit'): ").strip()
            if home_input.lower() == 'quit':
                break
            
            away_input = input("Enter away team name: ").strip()
            
            # Simulate match
            prediction = simulator.simulate_match(home_input, away_input)
            
            # Print results
            simulator.print_prediction_summary(prediction)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()