#!/usr/bin/env python3
"""
Example Match Predictions

This script demonstrates the match simulator with pre-defined matchups
to show the output format and capabilities.
"""

from match_simulator import MatchSimulator
import pandas as pd

def run_example_predictions():
    """Run several example match predictions"""
    print("Soccer Match Prediction Examples")
    print("=" * 50)
    
    try:
        # Initialize simulator
        simulator = MatchSimulator()
    except:
        print("Could not load model. Please run team_rating_model.py first.")
        return
    
    # Get team ratings to find interesting matchups
    teams_df = simulator.model.get_team_ratings(simulator.team_data)
    
    # Define some interesting matchups
    example_matches = []
    
    if len(teams_df) >= 6:
        # Top team vs 2nd team
        top_team = teams_df.iloc[0]['team_name']
        second_team = teams_df.iloc[1]['team_name']
        example_matches.append((top_team, second_team, "Top vs 2nd"))
        
        # Top team vs mid-table team
        if len(teams_df) >= 20:
            mid_team = teams_df.iloc[len(teams_df)//2]['team_name']
            example_matches.append((top_team, mid_team, "Top vs Mid-table"))
        
        # Mid-table vs bottom team
        if len(teams_df) >= 10:
            bottom_team = teams_df.iloc[-1]['team_name']
            example_matches.append((second_team, bottom_team, "Strong vs Weak"))
    
    # If we don't have enough teams, use first available teams
    if not example_matches and len(teams_df) >= 2:
        team1 = teams_df.iloc[0]['team_name']
        team2 = teams_df.iloc[1]['team_name']
        example_matches = [(team1, team2, "Example Match")]
    
    # Run predictions for each match
    for i, (home_team, away_team, description) in enumerate(example_matches):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i+1}: {description}")
        print(f"{'='*60}")
        
        try:
            # Simulate the match
            prediction = simulator.simulate_match(home_team, away_team)
            
            # Print detailed results
            simulator.print_prediction_summary(prediction)
            
            # Show scoreline grid for first example
            if i == 0:
                print(f"\nDetailed Scoreline Probabilities (up to 4-4):")
                print("     ", end="")
                for away_goals in range(5):
                    print(f"{away_goals:>7}", end="")
                print()
                
                prob_matrix = prediction['probability_matrix']
                for home_goals in range(5):
                    print(f"{home_goals:>2}: ", end="")
                    for away_goals in range(5):
                        prob = prob_matrix[home_goals, away_goals] * 100
                        print(f"{prob:>6.1f}%", end="")
                    print()
            
        except Exception as e:
            print(f"Error predicting {home_team} vs {away_team}: {e}")
    
    # Summary of all teams
    print(f"\n{'='*60}")
    print("TEAM RATINGS SUMMARY")
    print(f"{'='*60}")
    print(f"Total teams in model: {len(teams_df)}")
    print(f"Home advantage: {simulator.model.home_advantage:.3f}")
    print(f"\nTop 10 teams:")
    print(teams_df.head(10)[['team_name', 'attack_rating', 'defense_rating', 'overall_strength']].to_string(index=False))
    
    print(f"\nBottom 5 teams:")
    print(teams_df.tail(5)[['team_name', 'attack_rating', 'defense_rating', 'overall_strength']].to_string(index=False))

def demonstrate_specific_match():
    """Demonstrate prediction for a specific match with detailed analysis"""
    print(f"\n{'='*60}")
    print("DETAILED MATCH ANALYSIS EXAMPLE")
    print(f"{'='*60}")
    
    try:
        simulator = MatchSimulator()
        teams_df = simulator.model.get_team_ratings(simulator.team_data)
        
        if len(teams_df) < 2:
            print("Not enough teams for demonstration")
            return
        
        # Use top 2 teams for demonstration
        home_team = teams_df.iloc[0]['team_name']
        away_team = teams_df.iloc[1]['team_name']
        
        print(f"Analyzing: {home_team} vs {away_team}")
        
        # Get detailed prediction
        prediction = simulator.simulate_match(home_team, away_team, max_goals=6)
        
        # Show comprehensive analysis
        print(f"\nComprehensive Match Analysis:")
        
        home_exp = prediction['expected_goals']['home']
        away_exp = prediction['expected_goals']['away']
        results = prediction['match_results']
        
        print(f"Expected Goals: {home_exp:.3f} - {away_exp:.3f}")
        print(f"Goal Difference: {home_exp - away_exp:+.3f} in favor of {home_team if home_exp > away_exp else away_team}")
        
        print(f"\nMatch Outcome Analysis:")
        if results['home_win'] > 0.5:
            print(f"  Prediction: {home_team} favored to win ({results['home_win']:.1%})")
        elif results['away_win'] > 0.5:
            print(f"  Prediction: {away_team} favored to win ({results['away_win']:.1%})")
        else:
            print(f"  Prediction: Close match, slight favor to {'home' if results['home_win'] > results['away_win'] else 'away'}")
        
        print(f"  Draw probability: {results['draw']:.1%}")
        
        # Over/Under analysis
        total_expected = home_exp + away_exp
        print(f"\nGoal Total Analysis:")
        print(f"  Expected total goals: {total_expected:.2f}")
        
        # Calculate over/under 2.5 goals
        prob_matrix = prediction['probability_matrix']
        under_25 = 0
        for h in range(len(prob_matrix)):
            for a in range(len(prob_matrix[0])):
                if h + a < 2.5:
                    under_25 += prob_matrix[h, a]
        
        over_25 = 1 - under_25
        print(f"  Over 2.5 goals: {over_25:.1%}")
        print(f"  Under 2.5 goals: {under_25:.1%}")
        
        # Most likely exact score
        scorelines = prediction['scorelines']
        print(f"\nMost likely exact scores:")
        for i in range(min(5, len(scorelines))):
            scoreline = scorelines[i]
            print(f"  {scoreline['scoreline']}: {scoreline['probability']:.1%}")
        
    except Exception as e:
        print(f"Error in detailed analysis: {e}")

if __name__ == "__main__":
    run_example_predictions()
    demonstrate_specific_match()