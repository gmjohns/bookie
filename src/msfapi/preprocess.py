import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def preprocess(raw_file):
    df = pd.read_csv(raw_file)
    df = df.iloc[200:]
    df.loc[df['home_pitcher_prev_ip'] < 0, 'home_pitcher_prev_ip'] = 0
    df.loc[df['home_pitcher_prev_ip'] < 20, 'home_pitcher_prev_era'] = 6.00
    df.loc[df['home_pitcher_curr_ip'] < 20, 'home_pitcher_curr_era'] = df['home_pitcher_prev_era']
    df.loc[df['away_pitcher_prev_ip'] < 0, 'away_pitcher_prev_ip'] = 0
    df.loc[df['away_pitcher_prev_ip'] < 20, 'away_pitcher_prev_era'] = 6.00
    df.loc[df['away_pitcher_curr_ip'] < 20, 'away_pitcher_curr_era'] = df['away_pitcher_prev_era']
    # df.insert(loc=11, column='pitcher_adv', value= df['home_pitcher_adv'] + df['away_pitcher_adv'])
    df.drop(columns=['home_pitcher_curr_ip', 'away_pitcher_curr_ip'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

 
def normalize(df, df_test, columns):
    x = df.values
    x_test = df_test.values
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled_test = scaler.transform(x_test)
    df = pd.DataFrame(x_scaled, columns=columns)
    df_test = pd.DataFrame(x_scaled_test, columns=columns)
    return x_scaled, df, x_scaled_test, df_test


def standardize(df, df_test, columns):
    x = df.values
    x_test = df_test.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled_test = scaler.transform(x_test)
    df = pd.DataFrame(x_scaled, columns=columns)
    df_test = pd.DataFrame(x_scaled_test, columns=columns)
    return x_scaled, df, x_scaled_test, df_test


def principal_component_analysis(x, x_test, column_names, n=None):
    if n == None:
        n = min(x.shape[0], x.shape[1])
    pca = PCA(n_components=n)
    p_comp = pca.fit_transform(x)
    p_comp_test = pca.transform(x_test)
    plt.matshow(pca.components_,cmap='viridis')
    plt.yticks([0,1,2,3,4],['1st Comp','2nd Comp','3rd Comp', '4th Comp', '5th Comp'],fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(column_names)),column_names,rotation=65,ha='left')
    plt.tight_layout()
    plt.show()
    plt.clf()
    columns = []
    nums = []
    for i in range(n):
        nums.append(i)
        columns.append('p_c_'+str(i+1))
    df = pd.DataFrame(data=p_comp, columns=columns)
    df_test = pd.DataFrame(data=p_comp_test, columns=columns)
    ev = pca.explained_variance_ratio_
    plt.plot(ev, marker='o')
    plt.xticks(nums, labels=list(np.array(nums)+1))
    plt.show()
    return df, df_test


if __name__ == '__main__':
    df = preprocess('2017-regularLabeledRaw.csv')
    df.to_csv('2017-regularPP.csv')
    df_test = preprocess('2018-regularLabeledRaw.csv')
    df_test.to_csv('2018-regularPP.csv')
    to_process = df.drop(columns = ['date', 'home_team', 'away_team', 'fir_result'])
    to_process_test = df_test.drop(columns = ['date', 'home_team', 'away_team', 'fir_result'])
    x_norm, df_norm, x_norm_test, df_norm_test = normalize(to_process, to_process_test, to_process.columns.values)
    df_norm_cent = df_norm - df_norm.mean()
    df_norm_cent_test = df_norm_test - df_norm.mean()
    df_norm_cent['label'] = df['fir_result']
    df_norm_cent_test['label'] = df_test['fir_result']
    df_norm_cent.to_csv('2017-norm-centered.csv')
    df_norm_cent_test.to_csv('2018-norm-centered.csv')
    x_std, df_std, x_std_test, df_std_test = standardize(to_process, to_process_test, to_process.columns.values)
    df2 = df.copy()
    df_std['label'] = df['fir_result']
    df_std_test['label'] = df_test['fir_result']
    df_std.to_csv('2017-standardized.csv')
    df_std_test.to_csv('2018-standardized.csv')
    df_pc, df_pc_test = principal_component_analysis(x=x_norm, x_test=x_norm_test, column_names=to_process.columns)
    df_pc['label'] = df['fir_result']
    df_pc_test['label'] = df_test['fir_result']
    df_pc.to_csv('2017-pca-norm.csv')
    df_pc_test.to_csv('2018-pca-norm.csv')
