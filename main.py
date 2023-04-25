import d6tflow as d6t
import pandas as pd
from features import CreateVAEPFeatures
from wyVAEP import wyVAEP
from computeVAEP import ComputeVAEP

COMPETITION_NAME = 'English first division'
TRAIN_COMPETITIONS = ['Spanish first division', 'Italian first division', 'French first division']

wy = d6t.Workflow(wyVAEP, params={'competition_name': COMPETITION_NAME})
wy.run()
VAEPactions = wy.outputLoad()

minutes = pd.read_json("H:\Documentos\SaLab\Soccermatics\Wyscout Data\minutes_played_per_game_England.json",encoding='raw_unicode_escape')
playing_time = minutes.rename(columns={'playerId': 'player_id'})
playing_time = playing_time.groupby('player_id')['minutesPlayed'].sum().reset_index()
vaep_by_player = VAEPactions.groupby('player_id')['VAEP'].agg(['count','mean','sum','max','min']).sort_values(by='mean', ascending=False).reset_index(drop=False)
vaep_by_player = vaep_by_player.merge(playing_time, on='player_id')
vaep_by_player['rating'] = (vaep_by_player['sum']*90) / vaep_by_player['minutesPlayed']
vaep_by_player = vaep_by_player.loc[vaep_by_player.minutesPlayed>450].sort_values(by='rating', ascending=False)
