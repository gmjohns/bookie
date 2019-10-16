import numpy as np
import pandas as pd

def preprocess():
    df = pd.read_csv('2017-regularLabeledRaw.csv')
    df = df.iloc[200:]
    df.loc[df['home_pitcher_prev_ip'] < 0, 'home_pitcher_prev_ip'] = 0
    df.loc[df['home_pitcher_prev_ip'] < 20, 'home_pitcher_prev_era'] = 6.00
    df.loc[df['home_pitcher_curr_ip'] < 20, 'home_pitcher_curr_era'] = df['home_pitcher_prev_era']
    df.loc[df['away_pitcher_prev_ip'] < 0, 'away_pitcher_prev_ip'] = 0
    df.loc[df['away_pitcher_prev_ip'] < 20, 'away_pitcher_prev_era'] = 6.00
    df.loc[df['away_pitcher_curr_ip'] < 20, 'away_pitcher_curr_era'] = df['away_pitcher_prev_era']
    df.to_csv('2017-regularPP.csv')


if __name__ == '__main__':
    preprocess()