import d6tflow as d6t
import pandas as pd
from features import CreateVAEPFeatures
from wyVAEP import wyVAEP
from computeVAEP import ComputeVAEP
from wyDataLoader import wyLoadTimePlayed

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

vaep_table = VAEP3actions.groupby('player_name')['VAEP'].agg(['count','sum']).reset_index(drop=False)
vaep_table = vaep_table.rename(columns={'sum': 'VAEP'})
scores_table = VAEP3actions.groupby('player_name')['scores'].sum().reset_index(drop=False)
scores_table = scores_table.rename(columns={'sum': 'scores'})
concedes_table = VAEP3actions.groupby('player_name')['concedes'].sum().reset_index(drop=False)
concedes_table = concedes_table.rename(columns={'sum': 'concedes'})

summ_table = vaep_table.merge(scores_table).merge(concedes_table).merge(minutes_table).reset_index(drop=True)

summ_table['rating'] = (summ_table['VAEP']*90) / summ_table['minutes_played']
summ_table['offensive'] = (summ_table['scores']*90) / summ_table['minutes_played']
summ_table['defensive'] = (summ_table['concedes']*90) / summ_table['minutes_played']
summ_table = summ_table.sort_values(by=['rating','offensive','defensive'], ascending=[False,False,True])
summ_table = summ_table.loc[summ_table.minutes_played>450]
