#!/usr/bin/env python3
"""
Betting Evaluation Script

This script evaluates betting profitability by:
1. Comparing model predictions against bookmaker odds
2. Identifying value bets where bookmaker odds exceed model probabilities
3. Calculating betting returns and profitability analysis
4. Providing comprehensive betting strategy evaluation

The script uses the same test data split as the model evaluation for fair comparison.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from team_rating_model_pytorch import SoccerRatingModelPyTorch
from match_simulator_pytorch import MatchSimulatorPyTorch
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class BettingEvaluator:
    """
    Evaluate betting strategies using model predictions vs bookmaker odds
    """

    def __init__(self, model_path="models/soccer_rating_model_pytorch.pkl"):
        """Initialize with trained model"""
        try:
            self.simulator = MatchSimulatorPyTorch(model_path)
            print("Model loaded successfully for betting evaluation")
        except:
            print("Could not load model. Please train model first.")
            raise

        self.bookmakers = {
            "B365": ["B365H", "B365D", "B365A"],  # Bet365
            "BS": ["BSH", "BSD", "BSA"],  # BlueSq
            "BW": ["BWH", "BWD", "BWA"],  # BetWin
            "GB": ["GBH", "GBD", "GBA"],  # Gamebookers
            "IW": ["IWH", "IWD", "IWA"],  # Interwetten
            "LB": ["LBH", "LBD", "LBA"],  # Ladbrokes
            "PS": ["PSH", "PSD", "PSA"],  # Pinnacle
            "SJ": ["SJH", "SJD", "SJA"],  # Stan James
            "VC": ["VCH", "VCD", "VCA"],  # VC Bet
            "WH": ["WHH", "WHD", "WHA"],  # William Hill
        }

    def load_test_data_with_odds(
        self, db_path="data/database.sqlite", cutoff_date="2015-01-01"
    ):
        """
        Load test data with bookmaker odds
        """
        print("Loading test data with bookmaker odds...")

        conn = sqlite3.connect(db_path)

        # Build query with all odds columns
        odds_columns = []
        for bookmaker, cols in self.bookmakers.items():
            odds_columns.extend(cols)

        odds_cols_str = ", ".join([f"m.{col}" for col in odds_columns])

        query = f"""
        SELECT 
            m.id as match_id,
            m.date,
            m.season,
            m.home_team_api_id,
            m.away_team_api_id,
            m.home_team_goal,
            m.away_team_goal,
            ht.team_long_name as home_team_name,
            at.team_long_name as away_team_name,
            l.name as league_name,
            c.name as country_name,
            {odds_cols_str}
        FROM Match m
        JOIN Team ht ON m.home_team_api_id = ht.team_api_id
        JOIN Team at ON m.away_team_api_id = at.team_api_id
        JOIN League l ON m.league_id = l.id
        JOIN Country c ON l.country_id = c.id
        WHERE m.home_team_goal IS NOT NULL 
        AND m.away_team_goal IS NOT NULL
        AND m.date >= ?
        ORDER BY m.date
        """

        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()

        df["date"] = pd.to_datetime(df["date"])

        print(f"Loaded {len(df):,} test matches with odds data")

        # Add actual match result
        df["actual_result"] = "D"  # Default to draw
        df.loc[df["home_team_goal"] > df["away_team_goal"], "actual_result"] = "H"
        df.loc[df["home_team_goal"] < df["away_team_goal"], "actual_result"] = "A"

        return df

    def get_model_predictions(self, df):
        """
        Get model predictions for all matches
        """
        print("Generating model predictions...")

        predictions = []

        for i, row in df.iterrows():
            if i % 100 == 0:
                print(f"  Processing match {i+1}/{len(df)}")

            try:
                prediction = self.simulator.simulate_match(
                    row["home_team_api_id"], row["away_team_api_id"]
                )

                match_results = prediction["match_results"]

                predictions.append(
                    {
                        "match_id": row["match_id"],
                        "model_prob_H": match_results["home_win"],
                        "model_prob_D": match_results["draw"],
                        "model_prob_A": match_results["away_win"],
                        "model_odds_H": (
                            1.0 / match_results["home_win"]
                            if match_results["home_win"] > 0
                            else 999
                        ),
                        "model_odds_D": (
                            1.0 / match_results["draw"]
                            if match_results["draw"] > 0
                            else 999
                        ),
                        "model_odds_A": (
                            1.0 / match_results["away_win"]
                            if match_results["away_win"] > 0
                            else 999
                        ),
                    }
                )

            except Exception as e:
                print(f"Error predicting match {row['match_id']}: {e}")
                # Add default values for failed predictions
                predictions.append(
                    {
                        "match_id": row["match_id"],
                        "model_prob_H": 0.33,
                        "model_prob_D": 0.33,
                        "model_prob_A": 0.33,
                        "model_odds_H": 3.0,
                        "model_odds_D": 3.0,
                        "model_odds_A": 3.0,
                    }
                )

        pred_df = pd.DataFrame(predictions)
        return pred_df

    def identify_value_bets(
        self,
        df,
        pred_df,
        min_edge=0.05,
        min_odds=1.1,
        max_odds=10.0,
        best_odds_only=True,
    ):
        """
        Identify value bets where bookmaker odds exceed model fair odds

        Args:
            df: Match data with bookmaker odds
            pred_df: Model predictions
            min_edge: Minimum edge required (e.g., 0.05 = 5%)
            min_odds: Minimum acceptable odds
            max_odds: Maximum acceptable odds (to avoid extreme longshots)
            best_odds_only: If True, only take the best odds across all bookmakers per match
        """
        print(f"Identifying value bets with min edge: {min_edge:.1%}")
        print(f"Test dataset: {len(df):,} matches")

        # Merge predictions with match data
        full_df = df.merge(pred_df, on="match_id", how="inner")
        print(f"Matches with predictions: {len(full_df):,}")

        value_bets = []

        for _, row in full_df.iterrows():
            match_id = row["match_id"]
            actual_result = row["actual_result"]

            # Find best odds across all bookmakers for this match
            best_odds = {"H": 0, "D": 0, "A": 0}
            best_bookmaker = {"H": None, "D": None, "A": None}

            # Check each bookmaker to find best odds
            for bookmaker, cols in self.bookmakers.items():
                bm_odds_H = row[cols[0]]  # Home
                bm_odds_D = row[cols[1]]  # Draw
                bm_odds_A = row[cols[2]]  # Away

                # Skip if odds are missing
                if pd.isna(bm_odds_H) or pd.isna(bm_odds_D) or pd.isna(bm_odds_A):
                    continue

                # Update best odds
                if bm_odds_H > best_odds["H"]:
                    best_odds["H"] = bm_odds_H
                    best_bookmaker["H"] = bookmaker
                if bm_odds_D > best_odds["D"]:
                    best_odds["D"] = bm_odds_D
                    best_bookmaker["D"] = bookmaker
                if bm_odds_A > best_odds["A"]:
                    best_odds["A"] = bm_odds_A
                    best_bookmaker["A"] = bookmaker

            if best_odds_only:
                # Only consider best odds for each outcome
                outcomes = [
                    (
                        "H",
                        best_odds["H"],
                        row["model_odds_H"],
                        row["model_prob_H"],
                        best_bookmaker["H"],
                    ),
                    (
                        "D",
                        best_odds["D"],
                        row["model_odds_D"],
                        row["model_prob_D"],
                        best_bookmaker["D"],
                    ),
                    (
                        "A",
                        best_odds["A"],
                        row["model_odds_A"],
                        row["model_prob_A"],
                        best_bookmaker["A"],
                    ),
                ]

                for outcome, bm_odds, model_odds, model_prob, bookmaker in outcomes:
                    # Skip if no bookmaker found or invalid odds
                    if bookmaker is None or bm_odds < min_odds or bm_odds > max_odds:
                        continue

                    if model_odds < min_odds or model_prob <= 0:
                        continue

                    # Calculate edge: (bookmaker_odds * model_probability) - 1
                    edge = (bm_odds * model_prob) - 1.0

                    # Value bet if edge exceeds minimum threshold
                    if edge >= min_edge:
                        bet_won = outcome == actual_result
                        profit = (bm_odds - 1.0) if bet_won else -1.0

                        value_bets.append(
                            {
                                "match_id": match_id,
                                "date": row["date"],
                                "home_team": row["home_team_name"],
                                "away_team": row["away_team_name"],
                                "league": row["league_name"],
                                "country": row["country_name"],
                                "bookmaker": bookmaker,
                                "bet_outcome": outcome,
                                "bookmaker_odds": bm_odds,
                                "model_odds": model_odds,
                                "model_prob": model_prob,
                                "edge": edge,
                                "actual_result": actual_result,
                                "bet_won": bet_won,
                                "profit": profit,
                                "stake": 1.0,  # Unit stake
                            }
                        )
            else:
                # Check all bookmakers (original behavior)
                for bookmaker, cols in self.bookmakers.items():
                    bm_odds_H = row[cols[0]]  # Home
                    bm_odds_D = row[cols[1]]  # Draw
                    bm_odds_A = row[cols[2]]  # Away

                    # Skip if odds are missing
                    if pd.isna(bm_odds_H) or pd.isna(bm_odds_D) or pd.isna(bm_odds_A):
                        continue

                    # Check each outcome for value
                    outcomes = [
                        ("H", bm_odds_H, row["model_odds_H"], row["model_prob_H"]),
                        ("D", bm_odds_D, row["model_odds_D"], row["model_prob_D"]),
                        ("A", bm_odds_A, row["model_odds_A"], row["model_prob_A"]),
                    ]

                    for outcome, bm_odds, model_odds, model_prob in outcomes:
                        # Skip invalid odds
                        if bm_odds < min_odds or bm_odds > max_odds:
                            continue

                        if model_odds < min_odds or model_prob <= 0:
                            continue

                        # Calculate edge: (bookmaker_odds * model_probability) - 1
                        edge = (bm_odds * model_prob) - 1.0

                        # Value bet if edge exceeds minimum threshold
                        if edge >= min_edge:
                            bet_won = outcome == actual_result
                            profit = (bm_odds - 1.0) if bet_won else -1.0

                            value_bets.append(
                                {
                                    "match_id": match_id,
                                    "date": row["date"],
                                    "home_team": row["home_team_name"],
                                    "away_team": row["away_team_name"],
                                    "league": row["league_name"],
                                    "country": row["country_name"],
                                    "bookmaker": bookmaker,
                                    "bet_outcome": outcome,
                                    "bookmaker_odds": bm_odds,
                                    "model_odds": model_odds,
                                    "model_prob": model_prob,
                                    "edge": edge,
                                    "actual_result": actual_result,
                                    "bet_won": bet_won,
                                    "profit": profit,
                                    "stake": 1.0,  # Unit stake
                                }
                            )

        value_bets_df = pd.DataFrame(value_bets)

        if len(value_bets_df) > 0:
            print(
                f"Found {len(value_bets_df):,} value bets from {len(full_df):,} matches"
            )
            unique_matches = value_bets_df["match_id"].nunique()
            print(f"Value bets found in {unique_matches:,} different matches")
        else:
            print("No value bets found with current criteria")

        return value_bets_df

    def analyze_betting_performance(self, value_bets_df):
        """
        Analyze betting performance and profitability
        """
        if len(value_bets_df) == 0:
            print("No bets to analyze")
            return {}

        print("\n" + "=" * 60)
        print("BETTING PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Overall statistics
        total_bets = len(value_bets_df)
        total_wins = value_bets_df["bet_won"].sum()
        win_rate = total_wins / total_bets

        total_stake = value_bets_df["stake"].sum()
        total_return = value_bets_df.apply(
            lambda row: row["stake"] * row["bookmaker_odds"] if row["bet_won"] else 0,
            axis=1,
        ).sum()
        total_profit = total_return - total_stake
        roi = (total_profit / total_stake) * 100

        print(f"Overall Performance:")
        print(f"  Total Bets: {total_bets:,}")
        print(f"  Wins: {total_wins:,} ({win_rate:.1%})")
        print(f"  Total Stake: £{total_stake:,.2f}")
        print(f"  Total Return: £{total_return:,.2f}")
        print(f"  Total Profit: £{total_profit:,.2f}")
        print(f"  ROI: {roi:+.2f}%")

        # Performance by outcome
        print(f"\nPerformance by Outcome:")
        outcome_stats = (
            value_bets_df.groupby("bet_outcome")
            .agg(
                {
                    "bet_won": ["count", "sum", "mean"],
                    "profit": "sum",
                    "edge": "mean",
                    "bookmaker_odds": "mean",
                }
            )
            .round(3)
        )

        for outcome in ["H", "D", "A"]:
            if outcome in outcome_stats.index:
                stats = outcome_stats.loc[outcome]
                bets = int(stats[("bet_won", "count")])
                wins = int(stats[("bet_won", "sum")])
                win_rate = stats[("bet_won", "mean")]
                profit = stats[("profit", "sum")]
                avg_edge = stats[("edge", "mean")]
                avg_odds = stats[("bookmaker_odds", "mean")]

                print(
                    f"  {outcome}: {bets} bets, {wins} wins ({win_rate:.1%}), "
                    f"£{profit:+.2f} profit, {avg_edge:.1%} avg edge, {avg_odds:.2f} avg odds"
                )

        # Performance by bookmaker
        print(f"\nPerformance by Bookmaker:")
        bm_stats = (
            value_bets_df.groupby("bookmaker")
            .agg({"bet_won": ["count", "sum", "mean"], "profit": "sum", "edge": "mean"})
            .round(3)
        )

        for bm in sorted(bm_stats.index):
            stats = bm_stats.loc[bm]
            bets = int(stats[("bet_won", "count")])
            wins = int(stats[("bet_won", "sum")])
            win_rate = stats[("bet_won", "mean")]
            profit = stats[("profit", "sum")]
            avg_edge = stats[("edge", "mean")]

            print(
                f"  {bm}: {bets} bets, {wins} wins ({win_rate:.1%}), "
                f"£{profit:+.2f} profit, {avg_edge:.1%} avg edge"
            )

        # Performance by country
        print(f"\nPerformance by Country:")
        country_stats = (
            value_bets_df.groupby("country")
            .agg({"bet_won": ["count", "sum", "mean"], "profit": "sum", "edge": "mean"})
            .round(3)
        )

        # Sort by profit
        country_profit = country_stats[("profit", "sum")].sort_values(ascending=False)

        for country in country_profit.index:
            stats = country_stats.loc[country]
            bets = int(stats[("bet_won", "count")])
            wins = int(stats[("bet_won", "sum")])
            win_rate = stats[("bet_won", "mean")]
            profit = stats[("profit", "sum")]
            avg_edge = stats[("edge", "mean")]

            print(
                f"  {country}: {bets} bets, {wins} wins ({win_rate:.1%}), "
                f"£{profit:+.2f} profit, {avg_edge:.1%} avg edge"
            )

        # Performance by league
        print(f"\nPerformance by League:")
        league_stats = (
            value_bets_df.groupby("league")
            .agg({"bet_won": ["count", "sum", "mean"], "profit": "sum", "edge": "mean"})
            .round(3)
        )

        # Sort by profit and show top leagues
        league_profit = league_stats[("profit", "sum")].sort_values(ascending=False)

        for league in league_profit.head(15).index:  # Show top 15 leagues
            stats = league_stats.loc[league]
            bets = int(stats[("bet_won", "count")])
            wins = int(stats[("bet_won", "sum")])
            win_rate = stats[("bet_won", "mean")]
            profit = stats[("profit", "sum")]
            avg_edge = stats[("edge", "mean")]

            print(
                f"  {league}: {bets} bets, {wins} wins ({win_rate:.1%}), "
                f"£{profit:+.2f} profit, {avg_edge:.1%} avg edge"
            )

        if len(league_profit) > 15:
            print(f"  ... and {len(league_profit) - 15} more leagues")

        # Edge analysis
        print(f"\nEdge Analysis:")
        edge_bins = pd.cut(
            value_bets_df["edge"],
            bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
            labels=["5-10%", "10-20%", "20-30%", "30-50%", "50%+"],
        )
        edge_analysis = (
            value_bets_df.groupby(edge_bins)
            .agg({"bet_won": ["count", "sum", "mean"], "profit": "sum"})
            .round(3)
        )

        for edge_range in edge_analysis.index:
            if pd.notna(edge_range):
                stats = edge_analysis.loc[edge_range]
                bets = int(stats[("bet_won", "count")])
                wins = int(stats[("bet_won", "sum")])
                win_rate = stats[("bet_won", "mean")]
                profit = stats[("profit", "sum")]

                print(
                    f"  {edge_range} edge: {bets} bets, {wins} wins ({win_rate:.1%}), £{profit:+.2f} profit"
                )

        # Monthly performance
        value_bets_df["month"] = value_bets_df["date"].dt.to_period("M")
        monthly_stats = (
            value_bets_df.groupby("month")
            .agg({"bet_won": ["count", "sum"], "profit": "sum"})
            .round(2)
        )

        print(f"\nMonthly Performance (showing first 12 months):")
        for month in monthly_stats.head(12).index:
            stats = monthly_stats.loc[month]
            bets = int(stats[("bet_won", "count")])
            wins = int(stats[("bet_won", "sum")])
            profit = stats[("profit", "sum")]
            win_rate = wins / bets if bets > 0 else 0

            print(
                f"  {month}: {bets} bets, {wins} wins ({win_rate:.1%}), £{profit:+.2f} profit"
            )

        return {
            "total_bets": total_bets,
            "total_wins": total_wins,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "roi": roi,
            "total_stake": total_stake,
            "total_return": total_return,
        }

    def create_betting_visualizations(
        self, value_bets_df, save_path_prefix="models/betting_analysis"
    ):
        """
        Create visualizations for betting analysis
        """
        if len(value_bets_df) == 0:
            print("No betting data to visualize")
            return

        print("Creating betting analysis visualizations...")

        # Figure 1: Overview charts
        plt.figure(figsize=(15, 10))

        # 1. Profit over time
        plt.subplot(2, 3, 1)
        value_bets_df_sorted = value_bets_df.sort_values("date")
        cumulative_profit = value_bets_df_sorted["profit"].cumsum()
        plt.plot(value_bets_df_sorted["date"], cumulative_profit)
        plt.title("Cumulative Profit Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Profit (£)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 2. Win rate by edge
        plt.subplot(2, 3, 2)
        edge_bins = pd.cut(value_bets_df["edge"], bins=10)
        edge_win_rates = value_bets_df.groupby(edge_bins)["bet_won"].mean()
        edge_win_rates.plot(kind="bar")
        plt.title("Win Rate by Edge")
        plt.xlabel("Edge Range")
        plt.ylabel("Win Rate")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 3. ROI by bookmaker
        plt.subplot(2, 3, 3)
        bm_profit = value_bets_df.groupby("bookmaker")["profit"].sum()
        bm_stake = value_bets_df.groupby("bookmaker")["stake"].sum()
        bm_roi = (bm_profit / bm_stake * 100).sort_values(ascending=False)
        bm_roi.plot(kind="bar")
        plt.title("ROI by Bookmaker")
        plt.xlabel("Bookmaker")
        plt.ylabel("ROI (%)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 4. Profit by country
        plt.subplot(2, 3, 4)
        country_profit = (
            value_bets_df.groupby("country")["profit"]
            .sum()
            .sort_values(ascending=False)
        )
        country_profit.plot(kind="bar")
        plt.title("Total Profit by Country")
        plt.xlabel("Country")
        plt.ylabel("Total Profit (£)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 5. ROI by country
        plt.subplot(2, 3, 5)
        country_stake = value_bets_df.groupby("country")["stake"].sum()
        country_roi = (country_profit / country_stake * 100).sort_values(
            ascending=False
        )
        country_roi.plot(kind="bar")
        plt.title("ROI by Country")
        plt.xlabel("Country")
        plt.ylabel("ROI (%)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 6. Bet volume by outcome
        plt.subplot(2, 3, 6)
        outcome_counts = value_bets_df["bet_outcome"].value_counts()
        outcome_counts.plot(kind="pie", autopct="%1.1f%%")
        plt.title("Bet Distribution by Outcome")
        plt.ylabel("")

        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_overview.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Figure 2: League analysis
        plt.figure(figsize=(15, 8))

        # Top leagues by profit
        plt.subplot(2, 2, 1)
        league_profit = (
            value_bets_df.groupby("league")["profit"].sum().sort_values(ascending=False)
        )
        league_profit.head(10).plot(kind="bar")
        plt.title("Top 10 Leagues by Total Profit")
        plt.xlabel("League")
        plt.ylabel("Total Profit (£)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Top leagues by ROI
        plt.subplot(2, 2, 2)
        league_stake = value_bets_df.groupby("league")["stake"].sum()
        league_roi = league_profit / league_stake * 100
        # Filter leagues with at least 10 bets for reliable ROI
        league_bet_counts = value_bets_df.groupby("league").size()
        league_roi_filtered = league_roi[league_bet_counts >= 10].sort_values(
            ascending=False
        )
        league_roi_filtered.head(10).plot(kind="bar")
        plt.title("Top 10 Leagues by ROI (≥10 bets)")
        plt.xlabel("League")
        plt.ylabel("ROI (%)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # League bet volume
        plt.subplot(2, 2, 3)
        league_bet_counts.sort_values(ascending=False).head(10).plot(kind="bar")
        plt.title("Top 10 Leagues by Bet Volume")
        plt.xlabel("League")
        plt.ylabel("Number of Bets")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Win rate by league (for leagues with ≥10 bets)
        plt.subplot(2, 2, 4)
        league_win_rates = value_bets_df.groupby("league")["bet_won"].mean()
        league_win_rates_filtered = league_win_rates[
            league_bet_counts >= 10
        ].sort_values(ascending=False)
        league_win_rates_filtered.head(10).plot(kind="bar")
        plt.title("Top 10 Leagues by Win Rate (≥10 bets)")
        plt.xlabel("League")
        plt.ylabel("Win Rate")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_leagues.png", dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """
    Main function to run betting evaluation
    """
    print("Soccer Betting Evaluation")
    print("=" * 50)

    # Initialize evaluator
    try:
        evaluator = BettingEvaluator()
    except:
        return

    # Load test data with odds
    df = evaluator.load_test_data_with_odds()

    if len(df) == 0:
        print("No test data found")
        return

    # Get model predictions
    pred_df = evaluator.get_model_predictions(df)

    # Test different edge thresholds
    edge_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    print("\n" + "=" * 80)
    print("TESTING DIFFERENT EDGE THRESHOLDS")
    print("=" * 80)

    best_roi = -float("inf")
    best_threshold = None
    best_results = None

    for edge in edge_thresholds:
        print(f"\n--- Testing {edge:.0%} minimum edge ---")

        # Use best odds only to avoid multiple bets per match
        value_bets = evaluator.identify_value_bets(
            df, pred_df, min_edge=edge, best_odds_only=True
        )

        if len(value_bets) > 0:
            results = evaluator.analyze_betting_performance(value_bets)

            if results["roi"] > best_roi:
                best_roi = results["roi"]
                best_threshold = edge
                best_results = (value_bets, results)
        else:
            print(f"No value bets found with {edge:.0%} edge threshold")

    # Show best strategy
    if best_results:
        print(f"\n" + "=" * 80)
        print(f"BEST STRATEGY: {best_threshold:.0%} MINIMUM EDGE")
        print("=" * 80)

        best_bets, best_stats = best_results
        print(f"ROI: {best_stats['roi']:+.2f}%")
        print(f"Total Profit: £{best_stats['total_profit']:+,.2f}")
        print(f"Win Rate: {best_stats['win_rate']:.1%}")

        # Create visualizations for best strategy
        evaluator.create_betting_visualizations(best_bets)
    else:
        print("\nNo profitable betting strategy found with current parameters")

    print("\nBetting evaluation completed!")


if __name__ == "__main__":
    main()
