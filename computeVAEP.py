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

    def requires(self):
        return CreateVAEPFeatures(competition_name=self.competition_name), TrainXgboostVAEPModel(train_competitions=self.train_competitions),WyscoutToSPADL(competition_name=self.competition_name)
    
    def run(self):
        features = self.input()[0].load()
        models = self.input()[1].load()
        actions = self.input()[2].load()
        predictions = {}
        for model in tqdm(['scores', 'concedes'], desc="Predicting scores and concedes"):
            predictions[model] = models[model].predict_proba(features)[:,1]

        predictions = pd.DataFrame(predictions)
        predictions = vaepformula.value(actions, predictions['scores'], predictions['concedes'])
        #predictions['VAEP'] = predictions['scores'] - predictions['concedes']
        self.save(predictions)
