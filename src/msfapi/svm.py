#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA


# In[ ]:


#Path
loc = 'C:\\Users\\Will\\Documents\\GradSchool\\CSC522\\Project\\svm\\'

# New data
df_2017 = pd.read_csv(loc + '2017-regularPP.csv')
df_2018 = pd.read_csv(loc + '2018-regularPP.csv')
df_2019 = pd.read_csv(loc + '2019-regularPP.csv')

df_2017_2018 = pd.concat([df_2017, df_2018])

X_df = df_2017_2018.iloc[:,4:27]
y_df = df_2017_2018.iloc[:,28]

test_x = df_2019.iloc[:,4:27]
test_y = df_2019.iloc[:,28]




# In[33]:


def cv_svm_pca (kernel, random_state, n_splits, n_pca_components, X_df, y_df, c, gamma='auto', coef=0.0):

    # Create SVM model
    model = SVC(kernel=kernel, C=c, gamma=gamma, coef0=coef)

    # Create classifier pipelines
    pipeline_pca_standardize = make_pipeline(preprocessing.StandardScaler(),
                                         PCA(n_components=n_pca_components), model)


    # Get CV scores
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    scores_pca_standardized = cross_val_score(pipeline_pca_standardize, X_df, y_df,
                                          cv=cv, scoring='accuracy')


    # Get mean of scores
    scores_pca_standardized_mean = scores_pca_standardized.mean()

    
    return scores_pca_standardized_mean



random_state = 30  # Random number seed
n_splits = 10   # Number of CV splits
n_pca_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]   # PCA components
c_vals = [0.1, 0.2, 0.3, 1., 5., 10., 20., 100., 200., 1000.]
coef0 = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1., 2., 5., 10.]
gamma = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1., 2., 3.]
   

svm_accuracy_data = pd.DataFrame(columns=['Kernel', 'Standardized PCA Accuracy', 'PCA Components', 'C Value', 'Gamma Value', 'Coef0 Value'])


# Loop for creating plot data
for n in n_pca_components:

    for c in c_vals:
    
        scores_pca_standardized_mean = cv_svm_pca('sigmoid', random_state, n_splits, n, X_df, y_df, c=c, gamma=.1, coef=.2)
        svm_accuracy_data = svm_accuracy_data.append({'Kernel': 'sigmoid', 'Standardized PCA Accuracy': scores_pca_standardized_mean,
        'PCA Components' : n, 'C Value' : c, 'Gamma Value': .1, 'Coef0 Value': .2}, ignore_index=True)

print(svm_accuracy_data)




# In[ ]:


# To csv
svm_stan_pca_tuning = svm_accuracy_data.sort_values(by=['Standardized PCA Accuracy'], ascending=False)
svm_stan_pca_tuning.to_csv(loc + 'results_stan_pca_tuning.csv')





