import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_training_validation_data  import Randomized_Data as RD
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
#from sklearn import cross_validation

#import preprocessing
#import processing

##############################################################################
#
#  This program performs KNN ...
#
##############################################################################


# Read in data
# Test files
#data_2017_df = pd.read_csv("2017_small_test_file.csv")
#data_2018_df = pd.read_csv("2018_small_test_file.csv") 
#data_2019_df = pd.read_csv("2019_small_test_file.csv")
# Real data
data_2017_df = pd.read_csv("2017-regularPP.csv")
data_2018_df = pd.read_csv("2018-regularPP.csv")

# test code
print("2017 data")
print(data_2017_df)
print("2018 data")
print(data_2018_df)
print("2019 data")
#print(data_2019_df)


# Combine 2017 and 2018 data
data_2017_2018_df = pd.concat([data_2017_df, data_2018_df])

# test code
print("Combined 2017/2018 data")
print(data_2017_2018_df)

# Separate X and Y
# Test files
#X_df = data_2017_2018_df[['era', 'batt_avg', 'slug_avg']]
#y_df = data_2017_2018_df[['Class']]
# Real data
X_df = data_2017_2018_df.iloc[:,4:21]
X_df = X_df.drop(columns=['pitcher_adv'])
y_df = data_2017_2018_df.iloc[:,22]

# Test code 
print("X_df")
print(X_df)
print("y_df")
print(y_df)

n_neighbors=10
random_state=30
n_splits=3
n_pca_components=5


def cv_knn (n_neighbors, random_state, n_splits, n_pca_components, X_df, y_df):
    """   """

    # Create KNN model
    KNN = knn(n_neighbors=n_neighbors)

    # Create classifier pipelines
    pipeline_normalize = make_pipeline(preprocessing.MinMaxScaler(), KNN)
    #pipelint_normalize_shift = make_pipeline(preprocessing)
    pipeline_standardize = make_pipeline(preprocessing.StandardScaler(), KNN)
    pipeline_pca_standardize = make_pipeline(preprocessing.StandardScaler(),
                                         PCA(n_components=n_pca_components), KNN)

    #princ_comps_results = preprocessing.principal_component_analysis(
    #        standardized_data[1], standardized_data[3],4)

    # Get CV scores
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    scores_normalized = cross_val_score(pipeline_normalize, X_df, y_df, cv=cv, 
                         scoring='accuracy')
    scores_standardized = cross_val_score(pipeline_standardize, X_df, y_df, cv=cv, 
                         scoring='accuracy')
    scores_pca_standardized = cross_val_score(pipeline_pca_standardize, X_df, y_df,
                                          cv=cv, scoring='accuracy')

    # Test code
    print()
    print("scores normalized")
    print(scores_normalized)
    print("scores standardized")
    print(scores_standardized)
    print("PCA scores standardized")
    print(scores_pca_standardized)

    # Get mean of scores
    scores_normalized_mean = scores_normalized.mean()
    scores_standardized_mean = scores_standardized.mean()
    scores_pca_standardized_mean = scores_pca_standardized.mean()

    # Test code
    print()
    print("scores_normalized_mean")
    print(scores_normalized_mean)
    print("scores_standardized mean")
    print(scores_standardized_mean)
    print("scores_pca_standardized_mean")
    print(scores_pca_standardized_mean)
    
    return scores_normalized_mean, scores_standardized_mean, scores_pca_standardized_mean


if __name__ == '__main__':
    
    # Parameters
    n_neighbors=50 # Max number of KNN neighbors to evaluate
    random_state=30  # Random number seed
    n_splits=3   # Number of CV splits
    n_pca_components=5   # PCA components

   
    
    # Read in data
    # Test files
    #data_2017_df = pd.read_csv("2017_small_test_file.csv")
    #data_2018_df = pd.read_csv("2018_small_test_file.csv") 
    #data_2019_df = pd.read_csv("2019_small_test_file.csv")
    # Real data
    data_2017_df = pd.read_csv("2017-regularPP.csv")
    data_2018_df = pd.read_csv("2018-regularPP.csv")

    # test code
    print("2017 data")
    print(data_2017_df)
    print("2018 data")
    print(data_2018_df)
    print("2019 data")
    #print(data_2019_df)


    # Combine 2017 and 2018 data
    data_2017_2018_df = pd.concat([data_2017_df, data_2018_df])
    
    # test code
    print("Combined 2017/2018 data")
    print(data_2017_2018_df)
    
    # Separate X and Y
    # Test files
    #X_df = data_2017_2018_df[['era', 'batt_avg', 'slug_avg']]
    #y_df = data_2017_2018_df[['Class']]
    # Real data
    X_df = data_2017_2018_df.iloc[:,4:22]
    y_df = data_2017_2018_df.iloc[:,23]

    # Test code 
    print("X_df")
    print(X_df)
    print("y_df")
    print(y_df)
    
    
    
    
    knn_accuracy_data = pd.DataFrame(columns=['k','normalized_accuracy', 'standardized_accuracy', 'standardized_pca_accuracy'])
    
    # Loop for creating plot data
    for k in range(1, n_neighbors + 1):
        scores_normalized_mean, scores_standardized_mean, scores_pca_standardized_mean = cv_knn(n_neighbors=k, 
                                    random_state=random_state, n_splits=n_splits, n_pca_components=n_pca_components, X_df=X_df, y_df=y_df)
        knn_accuracy_data = knn_accuracy_data.append({'k': k, 'normalized_accuracy' : scores_normalized_mean, 
                                                      'standardized_accuracy': scores_standardized_mean,
                                                      'standardized_pca_accuracy': scores_pca_standardized_mean}, ignore_index=True)

#    print("cv_knn_scores_normalized_mean")
#    print(cv_knn(n_neighbors=10, random_state=30, n_splits=3, n_pca_components=5, X_df=X_df, y_df=y_df)[0])
    
    print("accuracy data:")
    print(knn_accuracy_data)





# Split data into test and validation datasets using seed = 30
#X_train, X_val, Y_train, Y_val = train_test_split(X_df, Y_df, random_state=30)

# Test code
#print("X_train")
#print(X_train)
#print("X_val")
#print(X_val)
#print("Y_train")
#print(Y_train)
#print("Y_val")
#print(Y_val)
