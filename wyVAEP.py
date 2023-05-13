import d6tflow as d6t
import pandas as pd
from socceraction.data.wyscout import PublicWyscoutLoader

from loaders import WyscoutToSPADL
from computeVAEP import ComputeVAEP
from wyDataLoader import wyLoadData

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'

class wyVAEP(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()
    train_competitions = d6t.ListParameter()
    data_dir = DATA_DIR
    def requires(self):
        return ComputeVAEP(competition_name=self.competition_name, train_competitions=self.train_competitions), WyscoutToSPADL(competition_name=self.competition_name), wyLoadData(data_dir=self.data_dir)


    def run(self):
        predictions = self.input()[0].load()
        actions = self.input()[1].load()
        wyData = self.input()[2].load()

        actions = actions.merge(wyData)
        VAEPactions = pd.concat([actions, predictions], axis=1).reset_index(drop=True)
        VAEPactions = VAEPactions.sort_values(by=['game_id','period_id','time_seconds'])

        self.save(VAEPactions)