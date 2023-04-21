import d6tflow as d6t
import pandas as pd
from tqdm import tqdm
import socceraction.vaep.formula as vaepformula

from features import CreateVAEPFeatures
from training import TrainXgboostVAEPModel
from loaders import WyscoutToSPADL

class ComputeVAEP(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()
    train_competitions = d6t.ListParameter()

    persist = ['nVAEP', 'predictions']

    def requires(self):
        return {
            'features': CreateVAEPFeatures(competition_name=self.competition_name),
            'models': TrainXgboostVAEPModel(train_competitions=self.train_competitions),
            'actions': WyscoutToSPADL(competition_name=self.competition_name)
        }
    
    def run(self):
        features = self.input()['features'].load()
        models = self.input()['models'].load()
        actions = self.input()['actions'].load()

        predictions = {}
        for model in tqdm(['scores', 'concedes'], desc="Predicting scores and concedes"):
            predictions[model] = models[model].predict_proba(features)[:,1]
        predictions = pd.DataFrame(predictions)

        nVAEP = []
        for game in tqdm(actions.game_id.unique(), desc="Computing VAEP"):
            values = vaepformula.value(actions, predictions['scores'], predictions['concedes'])
            nVAEP.append(pd.concat([actions, predictions, values], axis=1))

        nVAEP = pd.concat(nVAEP).sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
        predictions['VAEP'] = predictions['scores'] - predictions['concedes']

        self.save({'predictions': predictions, 'nVAEP': nVAEP})

