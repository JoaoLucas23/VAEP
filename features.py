import d6tflow as d6t
import pandas as pd
from tqdm import tqdm
import socceraction.vaep.features as ft

from loaders import WyscoutToSPADL

class CreateVAEPFeatures(d6t.tasks.TaskCSVPandas):
    competition_name = d6t.Parameter()

    def requires(self):
        return WyscoutToSPADL(competition_name=self.competition_name)

    def run(self):
        actions = self.input().load()
        
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
        self.save(features)