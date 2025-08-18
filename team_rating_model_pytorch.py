#!/usr/bin/env python3
"""
PyTorch Team Rating Model for Soccer Goal Prediction

This module implements a Poisson-based model using PyTorch for GPU acceleration:
- Each team has an attacking rating and defensive rating (learnable parameters)
- Goal rates are determined by attacking team rating vs defending team rating
- Home advantage is incorporated as an additional parameter
- Model learns team ratings from historical match data using gradient descent
- Supports GPU training for faster convergence

Model Structure:
- Home goals ~ Poisson(λ_home)
- Away goals ~ Poisson(λ_away)
- λ_home = exp(attack_home - defense_away + home_advantage)
- λ_away = exp(attack_away - defense_home)
"""

import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
import os
from tqdm import tqdm

class SoccerRatingNet(nn.Module):
    """
    PyTorch neural network for soccer team ratings
    """
    
    def __init__(self, n_teams, device='cpu'):
        super(SoccerRatingNet, self).__init__()
        
        self.n_teams = n_teams
        self.device = device
        
        # Learnable parameters
        self.attack_ratings = nn.Parameter(torch.randn(n_teams, device=device) * 0.1)
        self.defense_ratings = nn.Parameter(torch.randn(n_teams, device=device) * 0.1)
        self.home_advantage = nn.Parameter(torch.tensor(0.3, device=device))
        
        # Move to device
        self.to(device)
    
    def forward(self, home_teams, away_teams):
        """
        Forward pass to calculate expected goals
        
        Args:
            home_teams: tensor of home team indices
            away_teams: tensor of away team indices
        
        Returns:
            lambda_home, lambda_away: expected goals for home and away teams
        """
        # Get team ratings
        home_attack = self.attack_ratings[home_teams]
        home_defense = self.defense_ratings[home_teams]
        away_attack = self.attack_ratings[away_teams]
        away_defense = self.defense_ratings[away_teams]
        
        # Calculate expected goals
        # λ_home = exp(attack_home - defense_away + home_advantage)
        lambda_home = torch.exp(home_attack - away_defense + self.home_advantage)
        
        # λ_away = exp(attack_away - defense_home)
        lambda_away = torch.exp(away_attack - home_defense)
        
        return lambda_home, lambda_away

class PoissonLoss(nn.Module):
    """
    Poisson negative log-likelihood loss
    """
    
    def __init__(self):
        super(PoissonLoss, self).__init__()
    
    def forward(self, lambda_pred, targets):
        """
        Calculate Poisson negative log-likelihood
        
        Args:
            lambda_pred: predicted rates (λ)
            targets: actual goal counts
        
        Returns:
            negative log-likelihood
        """
        # Poisson log-likelihood: log(λ^k * exp(-λ) / k!)
        # = k * log(λ) - λ - log(k!)
        # We ignore log(k!) as it's constant w.r.t. parameters
        
        log_likelihood = targets * torch.log(lambda_pred + 1e-8) - lambda_pred
        
        # Return negative log-likelihood for minimization
        return -torch.mean(log_likelihood)

class SoccerRatingModelPyTorch:
    """
    PyTorch-based soccer team rating model
    """
    
    def __init__(self, device=None):
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Model components
        self.team_encoder = LabelEncoder()
        self.teams = None
        self.n_teams = 0
        self.model = None
        self.is_fitted = False
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
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
    
    def create_datasets(self, df, val_split=0.2, batch_size=512):
        """
        Create PyTorch datasets for training and validation
        """
        print(f"Creating datasets with batch size {batch_size}...")
        
        # Convert to tensors
        home_teams = torch.tensor(df['home_team_encoded'].values, dtype=torch.long, device=self.device)
        away_teams = torch.tensor(df['away_team_encoded'].values, dtype=torch.long, device=self.device)
        home_goals = torch.tensor(df['home_team_goal'].values, dtype=torch.float32, device=self.device)
        away_goals = torch.tensor(df['away_team_goal'].values, dtype=torch.float32, device=self.device)
        
        # Split data chronologically
        n_matches = len(df)
        split_idx = int(n_matches * (1 - val_split))
        
        # Training data
        train_home_teams = home_teams[:split_idx]
        train_away_teams = away_teams[:split_idx]
        train_home_goals = home_goals[:split_idx]
        train_away_goals = away_goals[:split_idx]
        
        # Validation data
        val_home_teams = home_teams[split_idx:]
        val_away_teams = away_teams[split_idx:]
        val_home_goals = home_goals[split_idx:]
        val_away_goals = away_goals[split_idx:]
        
        # Create datasets
        train_dataset = TensorDataset(train_home_teams, train_away_teams, train_home_goals, train_away_goals)
        val_dataset = TensorDataset(val_home_teams, val_away_teams, val_home_goals, val_away_goals)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        
        return train_loader, val_loader
    
    def fit(self, df, epochs=1000, lr=0.01, weight_decay=1e-4, batch_size=512, patience=50):
        """
        Fit the model using PyTorch optimization
        """
        print("Fitting PyTorch team rating model...")
        print(f"Training for up to {epochs} epochs with early stopping (patience={patience})")
        
        # Create datasets
        train_loader, val_loader = self.create_datasets(df, batch_size=batch_size)
        
        # Initialize model
        self.model = SoccerRatingNet(self.n_teams, device=self.device)
        
        # Initialize with better starting values
        self._initialize_parameters(df)
        
        # Loss function and optimizer
        criterion = PoissonLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        
        print(f"\nStarting training on {self.device}")
        print("Epoch | Train Loss | Val Loss | LR | Home Adv")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_home, batch_away, batch_home_goals, batch_away_goals in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                lambda_home, lambda_away = self.model(batch_home, batch_away)
                
                # Calculate loss for both home and away goals
                loss_home = criterion(lambda_home, batch_home_goals)
                loss_away = criterion(lambda_away, batch_away_goals)
                loss = loss_home + loss_away
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_home, batch_away, batch_home_goals, batch_away_goals in val_loader:
                    lambda_home, lambda_away = self.model(batch_home, batch_away)
                    
                    loss_home = criterion(lambda_home, batch_home_goals)
                    loss_away = criterion(lambda_away, batch_away_goals)
                    loss = loss_home + loss_away
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            # Calculate average losses
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print progress
            if epoch % 10 == 0 or epoch < 20:
                home_adv = self.model.home_advantage.item()
                print(f"{epoch:5d} | {avg_train_loss:10.4f} | {avg_val_loss:8.4f} | {current_lr:.2e} | {home_adv:6.3f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.best_state_dict = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_state_dict)
        
        self.is_fitted = True
        print("\nModel training completed!")
        
        # Extract final parameters
        with torch.no_grad():
            self.attack_ratings = self.model.attack_ratings.cpu().numpy()
            self.defense_ratings = self.model.defense_ratings.cpu().numpy()
            self.home_advantage = self.model.home_advantage.cpu().item()
        
        print(f"Final home advantage: {self.home_advantage:.3f}")
        print(f"Attack ratings range: [{self.attack_ratings.min():.3f}, {self.attack_ratings.max():.3f}]")
        print(f"Defense ratings range: [{self.defense_ratings.min():.3f}, {self.defense_ratings.max():.3f}]")
    
    def _initialize_parameters(self, df):
        """Initialize model parameters with reasonable starting values"""
        print("Initializing parameters with data-driven values...")
        
        with torch.no_grad():
            for i in range(self.n_teams):
                team_id = self.teams[i]
                
                # Calculate average goals scored and conceded
                home_goals = df[df['home_team_api_id'] == team_id]['home_team_goal']
                away_goals = df[df['away_team_api_id'] == team_id]['away_team_goal']
                home_conceded = df[df['home_team_api_id'] == team_id]['away_team_goal']
                away_conceded = df[df['away_team_api_id'] == team_id]['home_team_goal']
                
                total_matches = len(home_goals) + len(away_goals)
                
                if total_matches > 0:
                    # Attack rating based on goals scored
                    total_scored = home_goals.sum() + away_goals.sum()
                    avg_scored = total_scored / total_matches
                    self.model.attack_ratings[i] = torch.log(torch.tensor(max(avg_scored, 0.1)))
                    
                    # Defense rating based on goals conceded (negative is better)
                    total_conceded = home_conceded.sum() + away_conceded.sum()
                    avg_conceded = total_conceded / total_matches
                    self.model.defense_ratings[i] = -torch.log(torch.tensor(max(avg_conceded, 0.1)))
    
    def predict_goals(self, home_team_id, away_team_id):
        """Predict expected goals for a match"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Encode teams
        home_encoded = self.team_encoder.transform([home_team_id])[0]
        away_encoded = self.team_encoder.transform([away_team_id])[0]
        
        # Convert to tensors
        home_tensor = torch.tensor([home_encoded], dtype=torch.long, device=self.device)
        away_tensor = torch.tensor([away_encoded], dtype=torch.long, device=self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            lambda_home, lambda_away = self.model(home_tensor, away_tensor)
            
        return lambda_home.cpu().item(), lambda_away.cpu().item()
    
    def get_team_ratings(self, df):
        """Get team ratings as a DataFrame"""
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
        
        # Calculate overall strength
        ratings_df['overall_strength'] = ratings_df['attack_rating'] - ratings_df['defense_rating']
        
        return ratings_df.sort_values('overall_strength', ascending=False)
    
    def save_model(self, filepath="models/soccer_rating_model_pytorch.pkl"):
        """Save the trained PyTorch model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'attack_ratings': self.attack_ratings,
            'defense_ratings': self.defense_ratings,
            'home_advantage': self.home_advantage,
            'teams': self.teams,
            'n_teams': self.n_teams,
            'team_encoder': self.team_encoder,
            'is_fitted': self.is_fitted,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'device': str(self.device)
        }
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"PyTorch model saved to {filepath}")
        
        # Save training plots
        self.plot_training_history(filepath.replace('.pkl', '_training.png'))
    
    def load_model(self, filepath="models/soccer_rating_model_pytorch.pkl"):
        """Load a trained PyTorch model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model state
        self.teams = model_data['teams']
        self.n_teams = model_data['n_teams']
        self.team_encoder = model_data['team_encoder']
        self.attack_ratings = model_data['attack_ratings']
        self.defense_ratings = model_data['defense_ratings']
        self.home_advantage = model_data['home_advantage']
        self.is_fitted = model_data['is_fitted']
        self.train_losses = model_data.get('train_losses', [])
        self.val_losses = model_data.get('val_losses', [])
        
        # Recreate model
        self.model = SoccerRatingNet(self.n_teams, device=self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        
        print(f"PyTorch model loaded from {filepath}")
        print(f"Teams: {self.n_teams}, Home advantage: {self.home_advantage:.3f}")
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation loss curves"""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()

def main():
    """Main function to train and evaluate the PyTorch model"""
    print("PyTorch Soccer Team Rating Model")
    print("=" * 50)
    
    # Initialize model
    model = SoccerRatingModelPyTorch()
    
    # Load data
    df = model.load_and_prepare_data()
    
    # Split data chronologically
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = '2015-01-01'
    
    df_train = df[df['date'] < cutoff_date].copy()
    df_test = df[df['date'] >= cutoff_date].copy()
    
    print(f"\nData split:")
    print(f"Training: {len(df_train):,} matches ({df_train['date'].min()} to {df_train['date'].max()})")
    print(f"Testing:  {len(df_test):,} matches ({df_test['date'].min()} to {df_test['date'].max()})")
    
    # Fit model
    model.fit(df_train, epochs=500, lr=0.001, batch_size=1024)
    
    # Save model
    model.save_model()
    
    # Get team ratings
    print("\n" + "=" * 40)
    print("TOP 10 TEAMS BY OVERALL STRENGTH")
    print("=" * 40)
    ratings = model.get_team_ratings(df)
    print(ratings.head(10)[['team_name', 'attack_rating', 'defense_rating', 'overall_strength']])
    
    print("\nPyTorch model training completed!")

if __name__ == "__main__":
    main()