# Soccer Model

The goal of this project is to build a model of soccer games that is predictive of future games.

Some guidance for such models.
- Teams have parameters that help define goal rates
- Basic models use Poisson distributions for goals scored and the mean
- The mean is often defined by a teams attacking and defensive rating
- It should incorporate the difference in attack and defending strengths
- it can include a home advantage offet


Step 1: Datasets
- find sources of soccer data that will provide the most information of the events of a soccer game
  - Kaggle has a 25000 game euro soccer dataset
- write code to read in, cleanup and process this dataset

Step 2: Basic model
- assume teams have a attacking and defensive rating
- build a forward model of these ratings, their diff and home advantage, and a poisson model to predict actual goals, and learn the ratings of every team
- Test model with held out games
- use any and all data features available. If using expected goals, we will need a consistent xG model.

Step 3:
- build a process for expected points and expected tables for leagues

Step 3+: future ideas
- A better model than poisson for goals.
  - probability of goals is not uniform across the match
  - build a model of rate of goals over 90 minutes and injury time
  - incorporate this model over poisson
- break down team ratings into individual players with player ratings instead
  - player ratings can start with attack, defence ratings
  - add in other ratings that might add value
  - team combination for players
  - assess the impact of different lineups on result

And more. But let's start getting the basic infrastructure in place.
