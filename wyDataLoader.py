import pandas as pd
import d6tflow as d6t

class wyLoadData(d6t.tasks.TaskCSVPandas):
    data_dir = d6t.Parameter()

    persist = ['players', 'teams', 'minutes_played']
    
    def run(self):
        players = pd.read_json(path_or_buf=self.data_dir+'\players.json')
        teams = pd.read_json(path_or_buf=self.data_dir+'\\teams.json')
        minutes_played_england = pd.read_json(path_or_buf=self.data_dir+'\minutes_played_per_game_England.json')

        players = players.rename(columns={'wyId': 'player_id', 'shortName': 'player_name'})
        teams = teams.rename(columns={'wyId': 'team_id', 'name': 'team_name'})
        minutes_played_england = minutes_played_england.rename(columns={'playerId': 'player_id','minutesPlayed':'minutes_played'})

        players = players[['player_id','player_name']]
        teams = teams[['team_id','team_name']]
        minutes_played_england = minutes_played_england[['player_id', 'minutes_played']]

        self.save({
            'players': players,
            'teams': teams,
            'minutes_played': minutes_played_england
        })