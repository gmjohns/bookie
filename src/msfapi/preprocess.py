import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def preprocess():
    df = pd.read_csv('2017-regularLabeledRaw.csv')
    df = df.iloc[200:]
    df.loc[df['home_pitcher_prev_ip'] < 0, 'home_pitcher_prev_ip'] = 0
    df.loc[df['home_pitcher_prev_ip'] < 20, 'home_pitcher_prev_era'] = 6.00
    df.loc[df['home_pitcher_curr_ip'] < 20, 'home_pitcher_curr_era'] = df['home_pitcher_prev_era']
    df.loc[df['away_pitcher_prev_ip'] < 0, 'away_pitcher_prev_ip'] = 0
    df.loc[df['away_pitcher_prev_ip'] < 20, 'away_pitcher_prev_era'] = 6.00
    df.loc[df['away_pitcher_curr_ip'] < 20, 'away_pitcher_curr_era'] = df['away_pitcher_prev_era']
    df.drop(columns=['home_pitcher_curr_ip', 'away_pitcher_curr_ip'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

 
def normalize(df, columns):
    x = df.values
    x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=columns)
    return x_scaled, df


def standardize(df, columns):
    x = df.values
    x_scaled = preprocessing.StandardScaler().fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=columns)
    return x_scaled, df


def principal_component_analysis(x, n=None):
    if n == None:
        n = min(x.shape[0], x.shape[1])
    pca = PCA(n_components=n)
    p_comp = pca.fit_transform(x)
    columns = []
    nums = []
    for i in range(n):
        nums.append(i)
        columns.append('p_c_'+str(i+1))
    df = pd.DataFrame(data=p_comp, columns=columns)
    ev = pca.explained_variance_ratio_
    plt.plot(ev, marker='o')
    plt.xticks(nums, labels=list(np.array(nums)+1))
    plt.show()
    return df


if __name__ == '__main__':
    df = preprocess()
    df.to_csv('2017-regularPP.csv')
    to_process = df.drop(columns = ['date', 'home_team', 'away_team', 'fir_result'])
    x_norm, df_norm = normalize(to_process, to_process.columns.values)
    df_norm_cent = df_norm - df_norm.mean()
    df1 = df.copy()
    df1.update(df_norm_cent)
    df1.to_csv('2017-norm-centered.csv')
    x_std, df_std = standardize(to_process, to_process.columns.values)
    df2 = df.copy()
    df2.update(df_std)
    df2.to_csv('2017-standardized.csv')
    df_pc = principal_component_analysis(x_std)
    df_pc['label'] = df['fir_result']
    df_pc.to_csv('2017-pca.csv')