# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:28:15 2023

@author: jllgo
"""

import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
from socceraction.data.wyscout import PublicWyscoutLoader
import socceraction.spadl as spadl
from socceraction.vaep import features as ft
import socceraction.vaep.labels as lb
import xgboost as xgb
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import socceraction.vaep.formula as vaepformula
import numpy as np
from functions import *
from wyDataLoader import wyLoadTimePlayed

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'
COMPETITION_NAME = 'English first division'

WYL = PublicWyscoutLoader(root=DATA_DIR)
competitions = WYL.competitions()
selected_competitions = competitions[competitions.competition_name==COMPETITION_NAME]
games = pd.concat([
        WYL.games(row.competition_id, row.season_id)
        for row in selected_competitions.itertuples()
])

games_verbose = list(games.itertuples())
actions = []
for game in tqdm(games_verbose, desc="Converting to SPADL ({} games)".format(len(games_verbose)), total=len(games_verbose)):
        events = WYL.events(game.game_id)
        events = events.rename(columns={'id': 'event_id', 'eventId': 'type_id', 'subEventId': 'subtype_id',
                                'teamId': 'team_id', 'playerId': 'player_id', 'matchId': 'game_id'})
        actions_game = spadl.wyscout.convert_to_actions(events, game.home_team_id)
        actions_game = spadl.play_left_to_right(actions=actions_game, home_team_id=game.home_team_id)
        actions_game = spadl.add_names(actions_game)
        actions_game['home_team_id'] = game.home_team_id
        actions.append(actions_game)

actions = pd.concat(actions).reset_index(drop=True)

def createFeatures(actions):
    actions.loc[actions.result_id.isin([2, 3]), ['result_id']] = 0
    actions.loc[actions.result_name.isin(['offside', 'owngoal']), ['result_name']] = 'fail'
    xfns = [
        ft.actiontype_onehot,
        ft.bodypart_onehot,
        ft.result_onehot,
        ft.goalscore,
        ft.startlocation,
        ft.endlocation,
        ft.movement,
        ft.space_delta,
        ft.startpolar,
        ft.endpolar,
        ft.team,
        ft.time,
        ft.time_delta
    ]
    
    features = []
    for game in tqdm(actions.game_id.unique(), desc="Creating features"):
        actions_game = actions[actions.game_id==game].reset_index(drop=True)
        match_states = ft.gamestates(actions=actions_game,nb_prev_actions=3)
        match_states = ft.play_left_to_right(match_states,actions_game.home_team_id.unique()[0])
        match_features = pd.concat([fn(match_states) for fn in xfns], axis=1)
        features.append(match_features)
    
    features = pd.concat(features).reset_index(drop=True)
    bp_other = [c for c in list(features.columns) if 'bodypart' in c and 'foot' not in c]
    non_success = [c for c in list(features.columns) if 'result' in c and 'success' not in c]
    non_action = [c for c in list(features.columns) if 'non_action' in c]
    keeper = [c for c in list(features.columns) if 'keeper' in c and 'save' not in c]
    
    features = features.drop(bp_other + non_success + non_action + keeper, axis=1)
    return features

def createLabels(actions):
    yfns = [
         lb.scores,
         lb.concedes
     ]
    
    labels = []
    for game in tqdm(actions.game_id.unique(), desc="Creating labels"):
        actions_game = actions[actions.game_id==game].reset_index(drop=True)
        labels.append(pd.concat([fn(actions=actions_game) for fn in yfns], axis=1))
    
    labels = pd.concat(labels).reset_index(drop=True)
    return labels

def evaluate(y, y_hat):
    p = sum(y) / len(y)
    base = [p] * len(y)
    brier = brier_score_loss(y, y_hat)
    print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
    ll = log_loss(y, y_hat)
    print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

def trainModel(features, labels):
    x_train = []
    y_train = []
    
    x_train.append(features)
    y_train.append(labels)
    x_train = pd.concat(x_train).reset_index(drop=True)
    y_train = pd.concat(y_train).reset_index(drop=True)
    
    models = {}
    for label in tqdm(y_train.columns):
        print("Training model for {}".format(label))
        models[label] = xgb.XGBClassifier(n_estimators=100, max_depth=9, n_jobs=-3, verbosity=1)
        models[label].fit(x_train, y_train[label])
    
    Y_hat = pd.DataFrame()
    for col in y_train.columns:
        Y_hat[col] = [p[1] for p in models[col].predict_proba(x_train)]
        print(f"### Y: {col} ###")
        evaluate(y_train[col], Y_hat[col])
        
    return models
    
def calculateVaep(actions, training_actions):
    features = createFeatures(actions=actions)
    training_labels = createLabels(actions=training_actions)
    training_features = createFeatures(actions=training_actions)
    models = trainModel(training_features, training_labels)
    
    predictions = {}
    for model in tqdm(['scores', 'concedes'], desc="Predicting scores and concedes"):
        predictions[model] = models[model].predict_proba(features)[:,1]
    
    predictions = pd.DataFrame(predictions)
    predictions = vaepformula.value(actions, predictions['scores'], predictions['concedes'])
    return predictions

n_treino = int(np.ceil(len(actions) * 0.7))
n_test = len(actions) - n_treino

training_actions = actions[:n_treino].reset_index(drop=True)
actions = actions[n_treino:].reset_index(drop=True)

preds = calculateVaep(actions=actions, training_actions=training_actions)

VAEPactions = pd.concat([actions, preds], axis=1).reset_index(drop=True)
VAEPactions = VAEPactions.sort_values(by=['game_id','period_id','time_seconds'])

pt = d6t.Workflow(wyLoadTimePlayed, params={'data_dir': DATA_DIR})
pt.run()
played_time = pt.outputLoad()
minutes_table = played_time.groupby('player_id')['minutes_played'].sum().reset_index(drop=False)
minutes_table = minutes_table.rename(columns={'sum': 'minutes_played'})

player_summ_table = oneColumnGroupedVAEP(VAEPactions,column='player_id',by90=True,minutes_table=minutes_table)
action_summ_table = oneColumnGroupedVAEP(df=VAEPactions, column='type_name',by90=False)
result_summ_table = oneColumnGroupedVAEP(df=VAEPactions, column='result_name',by90=False)

