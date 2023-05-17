# VAEP

## Description
This is a VAEP model based on ML-KULeuven's and SALab's VAEP. This model uses Xgboost to predict the value of every SPADL action on a game, using Wyscout data to train and test. The project uses data from the German, French, and Italian leagues to train the model and generate VAEP values for the English and Spanish first division. 

## Model
### Xgboost


### Features


## Data
###Train Data
To train the model, all actions in the Italian, French and German League on the 17/18 season were used. This train data resulted in X total acions.

###Test Data
To test the model, all actions in the English and Spanish League on the 17/18 season were used. This test data resulted in X total actions.

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

## Results
### Train Data

### Test Data

## To Do
* Test features differences
* Get features values
* refactor tqdm loops
* refactor main.py
