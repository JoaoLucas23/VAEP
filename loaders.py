import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
from socceraction.data.wyscout import PublicWyscoutLoader
import socceraction.spadl as spadl

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'

class WyscoutToSPADL(d6t.tasks.TaskCSVPandas):
        competition_name = d6t.params.Parameter()
        WYL = PublicWyscoutLoader(root=DATA_DIR)

        def requires(self):
                return []

        def run(self):
                competitions = self.WYL.competitions()
                selected_competitions = competitions[competitions.competition_name==self.competition_name]
                games = pd.concat([
                        self.WYL.games(row.competition_id, row.season_id)
                        for row in selected_competitions.itertuples()
                ])

                games_verbose = tqdm(list(games.itertuples()), desc="Loading games")
                teams, players = [], []
                actions = {}
                for game in tqdm(games_verbose, desc="Converting to SPADL ({} games)".format(len(games_verbose)), total=len(games_verbose)):
                # load data
                        teams.append(self.WYL.teams(game.game_id))
                        players.append(self.WYL.players(game.game_id))
                        events = self.WYL.events(game.game_id)
                        actions[game.game_id] = spadl.wyscout.convert_to_actions(events, game.home_team_id)

                actions = pd.DataFrame(data=actions['Value'])

                teams = pd.concat(teams).drop_duplicates(subset="team_id")
                players = pd.concat(players)

                self.save(competitions)
                self.save(teams)
                self.save(players)
                self.save(actions)