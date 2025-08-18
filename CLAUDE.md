# Soccer Model Project - Claude Context

## Project Overview
This project builds a predictive model for soccer games using team attacking/defensive ratings and Poisson distributions for goal scoring.

## Current Phase: Step 1 - Datasets
- Need to find and process soccer data sources
- Kaggle has a 25,000 game Euro soccer dataset mentioned
- Focus on data that provides comprehensive game event information

## Project Structure & Development Guidelines

### Testing
- Check for existing test framework in codebase before running tests
- Use `pytest` or equivalent based on project setup

### Linting & Code Quality
- Run linting commands before finalizing changes
- Check for `requirements.txt` or `pyproject.toml` for dependencies

### Key Technical Components to Implement
1. **Data Processing Pipeline**
   - Data ingestion from Kaggle dataset
   - Data cleaning and preprocessing
   - Feature extraction for team ratings

2. **Basic Model (Step 2)**
   - Team attacking/defensive rating system
   - Poisson distribution for goal prediction
   - Home advantage offset
   - Model training and validation

3. **Future Enhancements (Step 3+)**
   - Expected points calculation
   - League table predictions
   - Time-based goal rate modeling
   - Player-level ratings

## Development Notes
- Focus on building basic infrastructure first
- Prioritize model accuracy with held-out test games
- Consider xG (expected goals) integration if consistent model available
- Use all available data features for maximum predictive power

## Getting Started

### 1. Database Setup
```bash
# Activate conda environment
conda activate soccer

# Set up database (downloads Kaggle data if needed)
python create_database.py

# Force re-download if needed
python create_database.py --force
```

### 2. Data Exploration
```bash
python explore_data.py
```

### 3. Model Training
```bash
python team_rating_model.py
```

### 4. Match Predictions
```bash
python predict_match.py
```

## File Structure
- `create_database.py` - Downloads and sets up the SQLite database
- `explore_data.py` - Explores the dataset structure and creates visualizations
- `team_rating_model.py` - Implements and trains the Poisson-based rating model
- `predict_match.py` - Loads trained model and makes predictions
- `data/database.sqlite` - Main SQLite database (not in git)
- `models/` - Saved models and team ratings (not in git)