import d6tflow as d6t
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

from features import CreateVAEPFeatures
from labels import CreateVAEPLabels

class TrainXgboostVAEPModel(d6t.tasks.TaskPickle):
    train_competitions = d6t.ListParameter()

    def requires(self):
        reqs = {'labels': {}, 'features': {}}

        for competition in self.train_competitions:
            reqs['labels'][competition] = CreateVAEPLabels(competition_name=competition)
            reqs['features'][competition] = CreateVAEPFeatures(competition_name=competition)

        return reqs
    
    def run(self):
        x_train = []
        y_train = []

        for comp in self.training_competitions:
            x_train.append(self.input()['features'][comp].load())
            y_train.append(self.input()['labels'][comp].load())
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

        self.save(models)


def evaluate(y, y_hat):
    p = sum(y) / len(y)
    base = [p] * len(y)
    brier = brier_score_loss(y, y_hat)
    print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
    ll = log_loss(y, y_hat)
    print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))