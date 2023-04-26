import d6tflow as d6t
import pandas as pd
from socceraction.data.wyscout import PublicWyscoutLoader

from loaders import WyscoutToSPADL
from computeVAEP import ComputeVAEP

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'
selected_competitions = ['Spanish first division', 'Italian first division', 'French first division']
class wyVAEP(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()
    num_prev_actions = d6t.Parameter()

    def requires(self):
        return ComputeVAEP(competition_name=self.competition_name, train_competitions=selected_competitions, num_prev_actions=self.num_prev_actions)
    #,WyscoutToSPADL(competition_name=self.competition_name)


    def run(self):
        predictions = self.input().load()
        #actions = self.input()[1].load()

        #VAEPactions = pd.concat([actions, predictions], axis=1).reset_index(drop=True)
        #VAEPactions = VAEPactions.sort_values(by=['game_id','period_id','time_seconds'])

        self.save(predictions)