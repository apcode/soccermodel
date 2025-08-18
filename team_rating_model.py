#!/usr/bin/env python3
"""
Team Rating Model for Soccer Goal Prediction

This module implements a Poisson-based model where:
- Each team has an attacking rating and defensive rating
- Goal rates are determined by attacking team rating vs defending team rating
- Home advantage is incorporated as an additional parameter
- Model learns team ratings from historical match data

Model Structure:
- Home goals ~ Poisson(λ_home)
- Away goals ~ Poisson(λ_away)
- λ_home = exp(α_home + attack_home - defense_away + home_advantage)
- λ_away = exp(α_away + attack_away - defense_home)
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
import os

class SoccerRatingModel:
    """
    Soccer team rating model using Poisson regression
    """
    
    def __init__(self):
        self.team_encoder = LabelEncoder()
        self.teams = None
        self.n_teams = 0
        self.parameters = None
        self.is_fitted = False
        
    def load_and_prepare_data(self, db_path="data/database.sqlite"):
        """Load and prepare match data for modeling"""
        print("Loading and preparing data...")
        
        conn = sqlite3.connect(db_path)
        
        # Load matches with team information
        query = """
        SELECT 
            m.date,
            m.season,
            m.home_team_api_id,
            m.away_team_api_id,
            m.home_team_goal,
            m.away_team_goal,
            ht.team_long_name as home_team_name,
            at.team_long_name as away_team_name,
            l.name as league_name
        FROM Match m
        JOIN Team ht ON m.home_team_api_id = ht.team_api_id
        JOIN Team at ON m.away_team_api_id = at.team_api_id
        JOIN League l ON m.league_id = l.id
        WHERE m.home_team_goal IS NOT NULL 
        AND m.away_team_goal IS NOT NULL
        ORDER BY m.date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Loaded {len(df):,} matches")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique teams: {df['home_team_api_id'].nunique()}")
        
        # Prepare team encoding
        all_teams = pd.concat([df['home_team_api_id'], df['away_team_api_id']]).unique()
        self.team_encoder.fit(all_teams)
        self.teams = all_teams
        self.n_teams = len(all_teams)
        
        # Encode teams
        df['home_team_encoded'] = self.team_encoder.transform(df['home_team_api_id'])
        df['away_team_encoded'] = self.team_encoder.transform(df['away_team_api_id'])
        
        return df
    
    def create_features(self, df):
        """
        Create feature matrix for the model
        
        Features:
        - One-hot encoding for home team attack
        - One-hot encoding for home team defense  
        - One-hot encoding for away team attack
        - One-hot encoding for away team defense
        - Home advantage indicator
        """
        n_matches = len(df)
        
        # Create feature matrices
        # For home goals: home_attack - away_defense + home_advantage
        X_home = np.zeros((n_matches, 2 * self.n_teams + 1))
        
        # For away goals: away_attack - home_defense
        X_away = np.zeros((n_matches, 2 * self.n_teams))
        
        for i, (_, row) in enumerate(df.iterrows()):
            home_team = int(row['home_team_encoded'])
            away_team = int(row['away_team_encoded'])
            
            # Home goals features
            X_home[i, home_team] = 1  # Home attack
            X_home[i, self.n_teams + away_team] = -1  # Away defense (negative)
            X_home[i, -1] = 1  # Home advantage
            
            # Away goals features  
            X_away[i, away_team] = 1  # Away attack
            X_away[i, self.n_teams + home_team] = -1  # Home defense (negative)
        
        y_home = df['home_team_goal'].values
        y_away = df['away_team_goal'].values
        
        return X_home, X_away, y_home, y_away
    
    def poisson_log_likelihood(self, params, X_home, X_away, y_home, y_away):
        """
        Calculate negative log-likelihood for Poisson model
        
        Parameters:
        - params: [attack_ratings (n_teams), defense_ratings (n_teams), home_advantage]
        """
        # Split parameters
        attack_ratings = params[:self.n_teams]
        defense_ratings = params[self.n_teams:2*self.n_teams]
        home_advantage = params[-1]
        
        # Combine attack and defense into full parameter vector
        full_params = np.concatenate([attack_ratings, defense_ratings, [home_advantage]])
        
        # Calculate expected goals
        lambda_home = np.exp(X_home @ full_params)
        lambda_away = np.exp(X_away @ full_params[:-1])  # No home advantage for away
        
        # Calculate log-likelihood
        ll_home = np.sum(poisson.logpmf(y_home, lambda_home))
        ll_away = np.sum(poisson.logpmf(y_away, lambda_away))
        
        # Return negative log-likelihood (for minimization)
        return -(ll_home + ll_away)
    
    def fit(self, df, regularization_strength=0.01):
        """
        Fit the model to match data
        """
        print("Fitting team rating model...")
        
        # Create features
        X_home, X_away, y_home, y_away = self.create_features(df)
        
        # Initialize parameters
        # Start with small random values
        np.random.seed(42)
        initial_params = np.random.normal(0, 0.1, 2 * self.n_teams + 1)
        
        # Add L2 regularization to objective function
        def objective(params):
            nll = self.poisson_log_likelihood(params, X_home, X_away, y_home, y_away)
            # L2 regularization (excluding home advantage)
            l2_penalty = regularization_strength * np.sum(params[:-1]**2)
            return nll + l2_penalty
        
        print(f"Optimizing {len(initial_params)} parameters...")
        
        # Fit model
        result = minimize(
            objective,
            initial_params,
            method='BFGS',
            options={'maxiter': 1000, 'disp': True}
        )
        
        if result.success:
            self.parameters = result.x
            self.is_fitted = True
            print("Model fitted successfully!")
            
            # Extract parameters
            self.attack_ratings = self.parameters[:self.n_teams]
            self.defense_ratings = self.parameters[self.n_teams:2*self.n_teams]
            self.home_advantage = self.parameters[-1]
            
            print(f"Home advantage: {self.home_advantage:.3f}")
            print(f"Attack ratings range: [{self.attack_ratings.min():.3f}, {self.attack_ratings.max():.3f}]")
            print(f"Defense ratings range: [{self.defense_ratings.min():.3f}, {self.defense_ratings.max():.3f}]")
            
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    def predict_goals(self, home_team_id, away_team_id):
        """
        Predict expected goals for a match
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Encode teams
        home_encoded = self.team_encoder.transform([home_team_id])[0]
        away_encoded = self.team_encoder.transform([away_team_id])[0]
        
        # Calculate expected goals
        lambda_home = np.exp(
            self.attack_ratings[home_encoded] - 
            self.defense_ratings[away_encoded] + 
            self.home_advantage
        )
        
        lambda_away = np.exp(
            self.attack_ratings[away_encoded] - 
            self.defense_ratings[home_encoded]
        )
        
        return lambda_home, lambda_away
    
    def get_team_ratings(self, df):
        """
        Get team ratings as a DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting ratings")
        
        # Create team name mapping
        team_names = {}
        for _, row in df[['home_team_api_id', 'home_team_name']].drop_duplicates().iterrows():
            team_names[row['home_team_api_id']] = row['home_team_name']
        for _, row in df[['away_team_api_id', 'away_team_name']].drop_duplicates().iterrows():
            team_names[row['away_team_api_id']] = row['away_team_name']
        
        ratings_df = pd.DataFrame({
            'team_api_id': self.teams,
            'team_name': [team_names.get(tid, f'Team_{tid}') for tid in self.teams],
            'attack_rating': self.attack_ratings,
            'defense_rating': self.defense_ratings
        })
        
        # Calculate overall strength (attack - defense)
        ratings_df['overall_strength'] = ratings_df['attack_rating'] - ratings_df['defense_rating']
        
        return ratings_df.sort_values('overall_strength', ascending=False)
    
    def evaluate_model(self, df_test):
        """
        Evaluate model performance on test data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        print("Evaluating model performance...")
        
        predictions = []
        actuals = []
        
        for _, row in df_test.iterrows():
            lambda_home, lambda_away = self.predict_goals(
                row['home_team_api_id'], 
                row['away_team_api_id']
            )
            
            predictions.append([lambda_home, lambda_away])
            actuals.append([row['home_team_goal'], row['away_team_goal']])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        home_mae = np.mean(np.abs(predictions[:, 0] - actuals[:, 0]))
        away_mae = np.mean(np.abs(predictions[:, 1] - actuals[:, 1]))
        
        home_rmse = np.sqrt(np.mean((predictions[:, 0] - actuals[:, 0])**2))
        away_rmse = np.sqrt(np.mean((predictions[:, 1] - actuals[:, 1])**2))
        
        print(f"Home goals - MAE: {home_mae:.3f}, RMSE: {home_rmse:.3f}")
        print(f"Away goals - MAE: {away_mae:.3f}, RMSE: {away_rmse:.3f}")
        
        # Calculate log-likelihood on test set
        X_home, X_away, y_home, y_away = self.create_features(df_test)
        test_ll = -self.poisson_log_likelihood(self.parameters, X_home, X_away, y_home, y_away)
        test_ll_per_match = test_ll / len(df_test)
        
        print(f"Test log-likelihood per match: {test_ll_per_match:.3f}")
        
        return {
            'home_mae': home_mae,
            'away_mae': away_mae,
            'home_rmse': home_rmse,
            'away_rmse': away_rmse,
            'test_log_likelihood': test_ll_per_match
        }
    
    def save_model(self, filepath="models/soccer_rating_model.pkl"):
        """
        Save the trained model to disk
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'parameters': self.parameters,
            'attack_ratings': self.attack_ratings,
            'defense_ratings': self.defense_ratings,
            'home_advantage': self.home_advantage,
            'teams': self.teams,
            'n_teams': self.n_teams,
            'team_encoder': self.team_encoder,
            'is_fitted': self.is_fitted
        }
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
        
        # Also save a human-readable summary
        summary_path = filepath.replace('.pkl', '_summary.json')
        summary = {
            'n_teams': self.n_teams,
            'home_advantage': float(self.home_advantage),
            'attack_ratings_stats': {
                'min': float(self.attack_ratings.min()),
                'max': float(self.attack_ratings.max()),
                'mean': float(self.attack_ratings.mean()),
                'std': float(self.attack_ratings.std())
            },
            'defense_ratings_stats': {
                'min': float(self.defense_ratings.min()),
                'max': float(self.defense_ratings.max()),
                'mean': float(self.defense_ratings.mean()),
                'std': float(self.defense_ratings.std())
            },
            'saved_at': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Model summary saved to {summary_path}")
    
    def load_model(self, filepath="models/soccer_rating_model.pkl"):
        """
        Load a trained model from disk
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model state
        self.parameters = model_data['parameters']
        self.attack_ratings = model_data['attack_ratings']
        self.defense_ratings = model_data['defense_ratings']
        self.home_advantage = model_data['home_advantage']
        self.teams = model_data['teams']
        self.n_teams = model_data['n_teams']
        self.team_encoder = model_data['team_encoder']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
        print(f"Teams: {self.n_teams}, Home advantage: {self.home_advantage:.3f}")
    
    def export_team_ratings_csv(self, df, filepath="models/team_ratings.csv"):
        """
        Export team ratings to CSV file
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before exporting ratings")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        ratings_df = self.get_team_ratings(df)
        ratings_df.to_csv(filepath, index=False)
        
        print(f"Team ratings exported to {filepath}")

def main():
    """Main function to train and evaluate the model"""
    print("Soccer Team Rating Model")
    print("=" * 50)
    
    # Initialize model
    model = SoccerRatingModel()
    
    # Load data
    df = model.load_and_prepare_data()
    
    # Split data chronologically (use earlier seasons for training)
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = '2015-01-01'  # Use 2015+ for testing
    
    df_train = df[df['date'] < cutoff_date].copy()
    df_test = df[df['date'] >= cutoff_date].copy()
    
    print(f"\nData split:")
    print(f"Training: {len(df_train):,} matches ({df_train['date'].min()} to {df_train['date'].max()})")
    print(f"Testing:  {len(df_test):,} matches ({df_test['date'].min()} to {df_test['date'].max()})")
    
    # Fit model
    model.fit(df_train)
    
    # Save the trained model
    model.save_model()
    model.export_team_ratings_csv(df)
    
    # Get team ratings
    print("\n" + "=" * 30)
    print("TOP 10 TEAMS BY OVERALL STRENGTH")
    print("=" * 30)
    ratings = model.get_team_ratings(df)
    print(ratings.head(10)[['team_name', 'attack_rating', 'defense_rating', 'overall_strength']])
    
    print("\n" + "=" * 30)
    print("BOTTOM 10 TEAMS BY OVERALL STRENGTH") 
    print("=" * 30)
    print(ratings.tail(10)[['team_name', 'attack_rating', 'defense_rating', 'overall_strength']])
    
    # Evaluate model
    print("\n" + "=" * 30)
    print("MODEL EVALUATION")
    print("=" * 30)
    metrics = model.evaluate_model(df_test)
    
    # Example predictions
    print("\n" + "=" * 30)
    print("EXAMPLE PREDICTIONS")
    print("=" * 30)
    
    # Get some example teams
    top_teams = ratings.head(3)['team_api_id'].values
    if len(top_teams) >= 2:
        team1, team2 = top_teams[0], top_teams[1]
        lambda_home, lambda_away = model.predict_goals(team1, team2)
        
        team1_name = ratings[ratings['team_api_id'] == team1]['team_name'].iloc[0]
        team2_name = ratings[ratings['team_api_id'] == team2]['team_name'].iloc[0]
        
        print(f"{team1_name} vs {team2_name}")
        print(f"Expected goals: {lambda_home:.2f} - {lambda_away:.2f}")
    
    print("\nModel training completed!")

if __name__ == "__main__":
    main()