# VAEP

## Introduction
VAEP is a Python code based on ML-KULeuven's and SALab's VAEP. The project uses Wyscout free data to develop the VAEP values for the English Premier League. To manage the process, the d6tflow library is used. 

The project uses data from the Spanish, French, and Italian leagues to train the model and generate VAEP values for the English Premier League. 

## Data Requirements
To run VAEP, you will need access to soccer match data in the Wyscout format. The project has been developed using Wyscout data from the Spanish, French, Italian, and English leagues, but it is possible to adapt the code to work with data from other leagues. All data files should be on the same directory.

The following data is required:
- Event data: [events_England.json, events_Spain.json, events_France.json, events_Italy.json].
- Players data: [players.json, minutes_played_per_game_England.json]
- Teams data: teams.json

## Files
The project contains the following files:
- `loaders.py`: This file loads the Wyscout data and converts it to the SPADL format.
- `labels.py`: This file creates the 'scores' and 'concedes' labels that are used in the VAEP model.
- `features.py`: This file gets the features for the actions to use in the VAEP model.
- `training.py`: This file trains the model using the XGBoost algorithm.
- `computeVAEP.py`: This file calculates the scores, concedes, and VAEP values for all actions.
- `wyVAEP.py`: This file joins the VAEP values to the complete actions dataframe, including player and team names.
- `wyDataLoader.py`: This file loads additional Wyscout data, such as minutes played per game per player.
- `functions.py`: This file contains additional functions to get summarized tables with rating values (VAEP by 90).
- `main.py`: This file runs the code.

## How to run
To generate a complete Pandas DataFrame with all actions and their values for the entire English Premier League 2017/2018 season, run the w3 workflow in `main.py`. 

To get summarized tables, run the other functions. 
