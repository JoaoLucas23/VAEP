import pandas as pd
import numpy as np

def getVAEPByPlayer(summ_table, player):
    summ_table = summ_table.loc[summ_table.player_name==player]
    return summ_table

def oneColumnGroupedVAEP(df=pd.DataFrame(), column='player_name', by90=True, minutes_table=pd.DataFrame()):
    vaep_table = df.groupby(column)['VAEP'].agg(['count','sum']).reset_index(drop=False)
    vaep_table = vaep_table.rename(columns={'sum': 'VAEP'})
    scores_table = df.groupby(column)['scores'].sum().reset_index(drop=False)
    scores_table = scores_table.rename(columns={'sum': 'scores'})
    concedes_table = df.groupby(column)['concedes'].sum().reset_index(drop=False)
    concedes_table = concedes_table.rename(columns={'sum': 'concedes'})

    if by90==True:
        summ_table = vaep_table.merge(scores_table).merge(concedes_table).merge(minutes_table).reset_index(drop=True)
        summ_table['rating'] = (summ_table['VAEP']*90) / summ_table['minutes_played']
        summ_table['offensive'] = (summ_table['scores']*90) / summ_table['minutes_played']
        summ_table['defensive'] = (summ_table['concedes']*90) / summ_table['minutes_played']
        summ_table = summ_table.sort_values(by=['rating','offensive','defensive'], ascending=[False,False,True])
        summ_table = summ_table.loc[summ_table.minutes_played>450]
    else:
        summ_table = vaep_table.merge(scores_table).merge(concedes_table).reset_index(drop=True)
        summ_table['rating'] = summ_table['VAEP'] / summ_table['count']
        summ_table['offensive'] = summ_table['scores'] / summ_table['count']
        summ_table['defensive'] = summ_table['concedes'] / summ_table['count']
        summ_table = summ_table.sort_values(by=['rating','offensive','defensive'], ascending=[False,False,True])

    return summ_table

def multipleColumnsGroupedVAEP(df=pd.DataFrame(), columns=['player_name']):
    vaep_table = df.groupby(columns)['VAEP'].agg(['count','sum']).reset_index(drop=False)
    vaep_table = vaep_table.rename(columns={'sum': 'VAEP'})
    scores_table = df.groupby(columns)['scores'].sum().reset_index(drop=False)
    scores_table = scores_table.rename(columns={'sum': 'scores'})
    concedes_table = df.groupby(columns)['concedes'].sum().reset_index(drop=False)
    concedes_table = concedes_table.rename(columns={'sum': 'concedes'})
    summ_table = vaep_table.merge(scores_table).merge(concedes_table).reset_index(drop=True)
    summ_table['rating'] = summ_table['VAEP'] / summ_table['count']
    summ_table['offensive'] = summ_table['scores'] / summ_table['count']
    summ_table['defensive'] = summ_table['concedes'] / summ_table['count']
    summ_table = summ_table.sort_values(by=['rating','offensive','defensive'], ascending=[False,False,True])
    summ_table = summ_table.loc[summ_table['count']>=10]
    
    return summ_table