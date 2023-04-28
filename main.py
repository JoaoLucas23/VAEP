import d6tflow as d6t
import pandas as pd
from features import CreateVAEPFeatures
from wyVAEP import wyVAEP

from computeVAEP import ComputeVAEP
from wyDataLoader import wyLoadTimePlayed

from functions import *

DATA_DIR = 'H:\Documentos\SaLab\Soccermatics\Wyscout Data'
COMPETITION_NAME = 'English first division'
TRAIN_COMPETITIONS = ['Spanish first division', 'Italian first division', 'French first division']

pt = d6t.Workflow(wyLoadTimePlayed, params={'data_dir': DATA_DIR})
pt.run()
played_time = pt.outputLoad()
minutes_table = played_time.groupby('player_name')['minutes_played'].sum().reset_index(drop=False)
minutes_table = minutes_table.rename(columns={'sum': 'minutes_played'})

wy3 = d6t.Workflow(wyVAEP, params={'competition_name': COMPETITION_NAME, 'num_prev_actions': 3})
wy3.run()
VAEP3actions = wy3.outputLoad()

player_summ_table = oneColumnGroupedVAEP(VAEP3actions,column='player_name',by90=True,minutes_table=minutes_table)
action_summ_table = oneColumnGroupedVAEP(df=VAEP3actions, column='type_name',by90=False)
result_summ_table = oneColumnGroupedVAEP(df=VAEP3actions, column='result_name',by90=False)

player_game_table = multipleColumnsGroupedVAEP(df=VAEP3actions,columns=['player_name','game_id'])
player_action_table = multipleColumnsGroupedVAEP(df=VAEP3actions,columns=['player_name','type_name'])
