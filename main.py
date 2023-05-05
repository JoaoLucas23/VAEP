import d6tflow as d6t
import pandas as pd

from wyVAEP import wyVAEP
from computeVAEP import ComputeVAEP
from wyDataLoader import wyLoadTimePlayed

from functions import *

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'
ENGLAND = 'English first division'
SPAIN = 'Spanish first division'
TRAIN_COMPETITIONS = ['German first division', 'Italian first division', 'French first division']

pt = d6t.Workflow(wyLoadTimePlayed, params={'data_dir': DATA_DIR})
pt.run()
played_time = pt.outputLoad()
minutes_table = played_time.groupby('player_name')['minutes_played'].sum().reset_index(drop=False)
minutes_table = minutes_table.rename(columns={'sum': 'minutes_played'})

en = d6t.Workflow(wyVAEP, params={'competition_name': ENGLAND, 'num_prev_actions': 3, 'train_competitions': TRAIN_COMPETITIONS})
en.run()
englandVAEP = en.outputLoad()

sp = d6t.Workflow(wyVAEP, params={'competition_name': SPAIN, 'num_prev_actions': 3, 'train_competitions': TRAIN_COMPETITIONS})
sp.run()
spainVAEP = sp.outputLoad()

player_summ_table = oneColumnGroupedVAEP(englandVAEP,column='player_name',by90=True,minutes_table=minutes_table)
action_summ_table = oneColumnGroupedVAEP(df=englandVAEP, column='type_name',by90=False)
result_summ_table = oneColumnGroupedVAEP(df=englandVAEP, column='result_name',by90=False)

player_game_table = multipleColumnsGroupedVAEP(df=englandVAEP,columns=['player_name','game_id'])
player_action_table = multipleColumnsGroupedVAEP(df=englandVAEP,columns=['player_name','type_name'])
