import pandas as pd
import numpy as np

def getVAEPByPlayer(summ_table, player):
    summ_table = summ_table.loc[summ_table.player_name==player]
    return summ_table

def oneColumnGroupedVAEP(df=pd.DataFrame(), column='player_name', by90=True, minutes_table=pd.DataFrame()):
    vaep_table = df.groupby(column)['vaep_value'].agg(['count','sum']).reset_index(drop=False)
    vaep_table = vaep_table.rename(columns={'sum': 'vaep_value'})
    scores_table = df.groupby(column)['offensive_value'].sum().reset_index(drop=False)
    concedes_table = df.groupby(column)['defensive_value'].sum().reset_index(drop=False)

    if by90==True:
        summ_table = vaep_table.merge(scores_table).merge(concedes_table).merge(minutes_table).reset_index(drop=True)
        summ_table['rating'] = (summ_table['vaep_value']*90) / summ_table['minutes_played']
        summ_table['offensive'] = (summ_table['offensive_value']*90) / summ_table['minutes_played']
        summ_table['defensive'] = (summ_table['defensive_value']*90) / summ_table['minutes_played']
        summ_table = summ_table.sort_values(by=['rating','offensive','defensive'], ascending=[False,False,True])
        summ_table = summ_table.loc[summ_table.minutes_played>450]
    else:
        summ_table = vaep_table.merge(scores_table).merge(concedes_table).reset_index(drop=True)
        summ_table['rating'] = summ_table['vaep_value'] / summ_table['count']
        summ_table['offensive'] = summ_table['offensive_value'] / summ_table['count']
        summ_table['defensive'] = summ_table['defensive_value'] / summ_table['count']
        summ_table = summ_table.sort_values(by=['rating','offensive','defensive'], ascending=[False,False,True])

    return summ_table

def multipleColumnsGroupedVAEP(df=pd.DataFrame(), columns=['player_name']):
    vaep_table = df.groupby(columns)['vaep_value'].agg(['count','sum']).reset_index(drop=False)
    vaep_table = vaep_table.rename(columns={'sum': 'vaep_value'})
    scores_table = df.groupby(columns)['offensive_value'].sum().reset_index(drop=False)
    concedes_table = df.groupby(columns)['defensive_value'].sum().reset_index(drop=False)
    summ_table = vaep_table.merge(scores_table).merge(concedes_table).reset_index(drop=True)
    summ_table['rating'] = summ_table['vaep_value'] / summ_table['count']
    summ_table['offensive'] = summ_table['offensive_value'] / summ_table['count']
    summ_table['defensive'] = summ_table['defensive_value'] / summ_table['count']
    summ_table = summ_table.sort_values(by=['rating','offensive','defensive'], ascending=[False,False,True])
    summ_table = summ_table.loc[summ_table['count']>=10]
    
    return summ_table