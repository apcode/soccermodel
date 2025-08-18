# Soccer Data Sources

## Primary Data Sources

### 1. Kaggle European Soccer Database (Recommended)
**URL**: https://www.kaggle.com/datasets/hugomathien/soccer  
**Alternative**: https://www.kaggle.com/datasets/prajitdatta/ultimate-25k-matches-football-database-european

**Dataset Details**:
- **Size**: 25,000+ matches
- **Coverage**: 11 European countries, seasons 2008-2016
- **Players**: 10,000+ players with attributes from EA Sports FIFA
- **Features**:
  - Team lineups with squad formation (X, Y coordinates)
  - Betting odds from up to 10 providers
  - Detailed match events (goals, possession, corners, fouls, cards)
  - Player and team attributes with weekly updates
  - 7 seasons of comprehensive data (2009/2010 to 2015/2016)

**Download Instructions**:
1. Create Kaggle account
2. Install Kaggle API: `pip install kaggle`
3. Setup API credentials (kaggle.json)
4. Download: `kaggle datasets download -d hugomathien/soccer`

### 2. StatsBomb Open Data (High Quality)
**URL**: https://github.com/statsbomb/open-data  
**Company**: https://statsbomb.com/what-we-do/hub/free-data/

**Dataset Details**:
- **Format**: JSON files from StatsBomb Data API
- **Coverage**: Multiple competitions including:
  - French Ligue 1 (multiple seasons)
  - Major League Soccer (US)
  - Women's football matches
- **Features**:
  - Event-level data with timestamps
  - Player tracking data (StatsBomb 360)
  - Advanced analytics metrics
  - Competition and season metadata

**Download Instructions**:
1. Clone repository: `git clone https://github.com/statsbomb/open-data.git`
2. Data structure:
   - `data/competitions.json` - competitions and seasons
   - `data/matches/` - match data by competition
   - `data/events/` - detailed event data
   - `data/three-sixty/` - tracking data
3. **Attribution Required**: Must credit StatsBomb in any published work

## Additional Data Sources

### 3. API Football (Real-time Data)
**URL**: API Football service  
**Type**: Freemium API
- **Free Tier**: 30 requests/minute, 100 requests/day
- **Features**: Live scores, fixtures, standings, player statistics
- **Best For**: Real-time data and current season updates

### 4. Football.db (Open Source)
**URL**: http://openfootball.github.io/
- **Type**: Open source, public domain
- **Coverage**: Historical results, multiple leagues
- **Format**: Plain text datasets, CSV
- **Best For**: Simple match results and league tables

### 5. International Football Results
**URL**: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- **Coverage**: 40,000+ international matches (1872-2018)
- **Best For**: International competition analysis

### 6. FiveThirtyEight Soccer Data
**Type**: Publicly available datasets
- **Focus**: Statistical analysis and predictions
- **Best For**: Advanced analytics and modeling examples

## Recommended Setup for This Project

**Primary**: Start with Kaggle European Soccer Database
- Most comprehensive for team rating model development
- Includes all necessary features for Poisson goal modeling
- Well-documented and widely used in research

**Secondary**: Add StatsBomb data for validation
- Higher quality event data
- Good for testing model accuracy
- Provides modern competition data

**Future**: Integrate API Football for live predictions
- Use trained model on current season data
- Real-time prediction capabilities

## Data Processing Notes

1. **Kaggle Dataset**: SQLite database format, requires pandas/sqlite3
2. **StatsBomb**: JSON format, use their Python/R packages
3. **Consistent Schema**: Plan data normalization across sources
4. **Feature Engineering**: Focus on team strength indicators, home advantage, recent form