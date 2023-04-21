import d6tflow as d6t
import pandas as pd
from socceraction.data.wyscout import PublicWyscoutLoader

from loaders import WyscoutToSPADL
from computeVAEP import ComputeVAEP

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'

class wyVAEP(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()

    persist = ['VAEP', 'nVAEP']

    def requires(self):
        WYL = PublicWyscoutLoader(root=DATA_DIR)
        competitions = WYL.competitions()
        selected_competitions = competitions[competitions.competition_name!=self.competition_name]
        selected_competitions = selected_competitions.competition_name.unique().tolist()
        return {
            'VAEPs': ComputeVAEP(competition_name=self.competition_name, train_competitions=selected_competitions),
            'actions': WyscoutToSPADL(competition_name=self.competition_name)
        }

    def run(self):
        predictions = self.input()['VAEPs']['predictions'].load()
        nVAEP = self.input()['VAEPs']['nVAEP'].load()
        actions = self.input()['actions'].load()

        VAEPactions = pd.concat([actions, predictions], axis=1).reset_index(drop=True)

        self.save({'VAEP': VAEPactions, 'nVAEP': nVAEP})
