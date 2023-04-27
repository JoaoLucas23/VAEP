import pandas as pd
import numpy as np

def getVAEPByPlayer(df, player):
    return 0

def groupedVAEP(df, column='player_id'):
    summ_table = df.groupby(column).agg({'VAEP':['count','sum'],'scores':['count','sum'],'concedes':['count','sum'],'minutes_played':'sum'})

    return summ_table