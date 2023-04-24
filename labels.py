import d6tflow as d6t
import pandas as pd
from tqdm import tqdm
import socceraction.vaep.labels as lb

from loaders import WyscoutToSPADL

class CreateVAEPLabels(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()

    def requires(self):
        return WyscoutToSPADL(competition_name=self.competition_name)

    def run(self):
        actions = self.inputLoad()
        
        yfns = [
            lb.scores,
            lb.concedes
        ]

        labels = []
        for game in tqdm(actions.game_id.unique(), desc="Creating labels"):
            actions_game = actions[actions.game_id==game].reset_index(drop=True)
            labels.append(pd.concat([fn(actions=actions_game) for fn in yfns], axis=1))

        labels = pd.concat(labels).reset_index(drop=True)
        self.save(labels)