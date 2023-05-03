import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
from socceraction.data.wyscout import PublicWyscoutLoader
import socceraction.spadl as spadl

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'

class WyscoutToSPADL(d6t.tasks.TaskCSVPandas):
        competition_name = d6t.Parameter()

        def run(self):
                WYL = PublicWyscoutLoader(root=DATA_DIR)
                competitions = WYL.competitions()
                selected_competitions = competitions[competitions.competition_name==self.competition_name]
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
                self.save(actions)
