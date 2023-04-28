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

player_summ_table = groupedVAEP(VAEP3actions, minutes_table, column='player_name')
getVAEPByPlayer(player_summ_table, 'M. Salah')

action_summ_table = groupedVAEP(VAEP3actions, minutes_table, column='action_name')
