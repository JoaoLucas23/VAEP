# VAEP

## Introduction
This is a Python code, based on ML-KULeuven's and SALab's VAEP. I used Wyscout free data to develop this project. Using Spanish league, France league and Italy league for training to get the VAEP values for the English league.
## Files
loaders.py -> load the wyscout data and convert to SPADL format
labels.py -> create the 'scores' and 'concedes' labels to use in the VAEP model
features.py -> get the features for the actions to use on the VAEP model
training.py -> train the model using the xgboost algorithm
computeVAEP.py -> calculate the scores, concedes and VAEP values for all actions
wyVAEP.py -> join the VAEP values to the complete actions dataframe including players and teams names
wyDataLoader.py -> load adittional wyscout data, such as minutes played per game per player
functions.py -> some aditional functions to get summarized tables with rating values (VAEP by 90)
main.py -< run the code
## How to run
On the main.py run the w3 Workflow to get a complete pandas DataFrame with all actions and their values for the entire English 17/18 season.
Run the other functions to get some summarized tables.
