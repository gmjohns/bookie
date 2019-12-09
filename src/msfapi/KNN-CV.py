
"""
This program performs KNN cross validation evaluation for the datasets
identified in the program.  The following global parameters are provided
by the user in the code:
    - Max number of nearest neighbors to evaluate
    - Max number of pca components to evaluate
    - Random seed
    - X data set as a csv file
    - y data set as a csv file
    - Number of splits for cross validation
The following output is produced:
    - Graph of accuracy vs. K for normalized data
    - Graph of accuracy vs. K for standardized data
    - Graph of accuracy vs. K for normalized-mean-shifted-pca data
    - Graph of accuracy vs. K for standardized-pca data

@author: mhhammer
"""
# import needed libraries
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from generate_training_validation_data  import Randomized_Data as RD
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def cv_knn_no_pca (n_neighbors, random_state, n_splits, X_df, y_df):
    """
    Function performs cross-validation KNN (no pca) using normalized and
    standardized data sets.
    Inputs:
        -  n_neighbors - number of KNN nearest neighbors
        -  random_state - seed for random number generator
        -  n_splits - number of CV folds
        -  X_df - dataframe containing X data
        -  y_df - dataframe containing y data
    Returns:
        - scores_normalized_mean - mean accuracy score of individual
          normalized accuracy scores from each CV fold
        - scores_standardized_mean - mean accuracy score of individual
          standardizded accuracy scores from each CV fold
    """
    # Create KNN model
    KNN = knn(n_neighbors=n_neighbors)

    # Set cv parameters
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state,
                         shuffle=True)

    # Cross validate - normalized
    pipeline_normalize = make_pipeline(preprocessing.MinMaxScaler(), KNN)
    scores_normalized = cross_val_score(pipeline_normalize, X_df, y_df, cv=cv,
                                        scoring='accuracy')
    scores_normalized_mean = scores_normalized.mean()

    # Cross validate - standardized
    pipeline_standardize = make_pipeline(preprocessing.StandardScaler(), KNN)
    scores_standardized = cross_val_score(pipeline_standardize, X_df, y_df,
                                          cv=cv, scoring='accuracy')
    scores_standardized_mean = scores_standardized.mean()

    return scores_normalized_mean, scores_standardized_mean


def cv_knn_pca_std(n_neighbors, random_state, n_splits, n_pca_components,
                   X_df, y_df):
    """
    Function performs cross-validation KNN with PCA using standardized data
    set.
    Inputs:
        -  n_neighbors - number of KNN nearest neighbors
        -  random_state - seed for random number generator
        -  n_splits - number of CV folds
        -  n_pca_components - number of PCA components to use
        -  X_df - dataframe containing X data
        -  y_df - dataframe containing y data
    Returns:
        - scores_pca_standardized_mean - mean accuracy score of individual
          standardizded accuracy scores from each CV pca fold"""

    # Create KNN model
    KNN = knn(n_neighbors=n_neighbors)

    # Set cv parameters
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state,
                         shuffle=True)

    # Cross validate - pca_standardized
    pipeline_pca_standardize = make_pipeline(preprocessing.StandardScaler(),
                                             PCA(n_components=n_pca_components), KNN)
    scores_pca_standardized = cross_val_score(pipeline_pca_standardize,
                                              X_df, y_df, cv=cv, scoring='accuracy')
    scores_pca_standardized_mean = scores_pca_standardized.mean()

    return scores_pca_standardized_mean


def MinMaxMeanShiftStratifiedKFoldKNNpca(X_df, y_df, n_splits, n_neighbors,
                                         n_pca_comps, random_state):
    """
    Function that takes as inputs X and y data sets and returns the mean
    KNN accuracy using cross validation and pca with normalized, mean-shifted
    data
    Inputs:
        - X_df - dataset containing X values
        - y_df - dataset containing y values
        - n_splits - number of CV folds
        - n_neighbors - number of KNN neighbors
        - n_pca_comps - number of pca components to consider
        - random_state - seed for random number generation
    Outputs:
        - mean accuracy of KNN model for number of n_neighbors and n_pca_comps
          specified as parameters
    """

    # Convert dataframe input to numpy for split determination
    x = X_df.to_numpy()
    y = y_df.to_numpy()
    # Get splits
    skf = StratifiedKFold(n_splits=n_splits, random_state = random_state,
                          shuffle = False)
    skf.get_n_splits(x, y)

    # Create dataframe for capturing accuracies for each fold
    cv_accuracy_scores = pd.DataFrame(columns = ['accuracy'])

    # Loop for conducting CV for specified n_neighbors and n_pca_comps
    for train_index, test_index in skf.split(x, y):

        # Use indexes of splits to set X and y training and test data sets
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert to data frames
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_test)

        # Normalize using X_train mean and min
        X_train_norm = preprocessing.minmax_scale(X_train_df)
        X_test_norm = (X_test_df - X_train_df.min()) / (X_train_df.max() -
                                                        X_train_df.min())

        # Shift data by mean of X_train_norm data
        X_train_norm_shift = X_train_norm - X_train_norm.mean(axis=0)
        X_test_norm_shift = X_test_norm - X_train_norm.mean(axis=0)

        # Conduct PCA
        pca = PCA(n_components=n_pca_comps)
        pca.fit(X_train_norm_shift)
        X_train_pca = pca.transform(X_train_norm_shift)
        X_test_pca = pca.transform(X_test_norm_shift)

        # Conduct KNN
        KNN = knn(n_neighbors=n_neighbors)
        KNN.fit(X_train_pca, y_train)
        y_pred = KNN.predict(X_test_pca)

        # Get accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Add fold accuracy data to cv_accuracy_scores dataframe
        cv_accuracy_scores = cv_accuracy_scores.append({'accuracy': accuracy},
                                                       ignore_index=True)

    # Get mean of fold accuracy scores
    mean_accuracy = cv_accuracy_scores.mean(axis=0)

    # Return mean_accuracy
    return mean_accuracy.accuracy


def plot_KNN(x_data_df, y_data_df, k_highlight, max_accuracy, title, filename):
    """
    Function creates a bar chart of accuracy vs. K for various values of K
    for the no-pca case.
    Inputs:
        - x_data_df - dataframe containing values of K
        - y_data_df - dataframe containing accuracy values
        - k-highlight - value of K to highlight (largest accuracy)
        - max-accuracy - value of maximum accuracy for dataset
        - title - text for title of graph
        - filename - name of file to store ooutput graph
    Output:
        - Bar graph of accuracy vs. K with bar having largest accuracy colored
          differently from others and with levl of accuracy printed on chart
    """

    # Set defalut parameters
    plt.rcParams.update(plt.rcParamsDefault)
    # Set style
    plt.style.use('bmh')
    # Create figure and one subplot
    fig, ax = plt.subplots()

    # Define x and y data
    x = x_data_df
    y = y_data_df

    # Make plot
    plt.bar(x, y, color='blue')
    plt.bar(k_highlight, y, color= 'red')
    plt.xlabel("k")  # x axis label
    plt.ylabel("Accuracy")  # y axis label

    # Text for box identifying highest accuracy
    box_text = f'Highest accuracy = {"%.4f" % round(max_accuracy, 4)}\nK = {k_highlight}'
    plt.text(0.5, 0.7, box_text, bbox=dict(facecolor='white', alpha=0.5), transform=fig.transFigure)

    plt.title(title)  #  Chart title
    plt.ylim(0,1)  # y axis range

    # Set ticks for axes
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    # Save fig
    fig1 = plt.gcf()

    # Show plot
    plt.show()

    # Save plot
    fig1.savefig(filename, bbox_inches='tight')

def plot_KNN_PCA(x_data_df, y_data_df, max_pca_standardized_mean_accuracy,
                 k_for_max_mean, n_pca_for_max_mean, title, filename):
    """
    Function creates a bar chart of accuracy vs. K for various values of K
    for the pca case.  Individual lines are provided for each number of pca
    components.
    Inputs:
        - x_data_df - dataframe containing values of K
        - y_data_df - dataframe containing accuracy values
        - max_pca_standardized_mean_accuracy - max accuracy to display
        - k_for_max_mean - k value for max accuracy
        - n_pca_for_max_mean - number of pca components to achieve max
          accuracy
        - title - text for title of graph
        - filename - name of file to store output graph
    Output:
        - Bar graph of accuracy vs. K with bar having largest accuracy colored
          differently from others and with levl of accuracy printed on chart
    """
    # Set defalut parameters
    plt.rcParams.update(plt.rcParamsDefault)
    # Set style
    plt.style.use('bmh')
    # Create figure and one subplot
    fig, ax = plt.subplots()

    # Define x data
    x = x_data_df

    # Make plot
    plt.title(title)  # Chart title
    plt.xlabel('k')   # x axis label
    plt.ylabel('Accuracy')  # y axis label
    plt.ylim(0.45,0.60)  # y axis range

    # Text for box identifying highest accuracy
    box_text = f'Highest accuracy = {"%.4f" % round(max_pca_standardized_mean_accuracy, 4)}\nK = {k_for_max_mean}\nNum PCA components = {n_pca_for_max_mean}'
    plt.text(0.5, 0.7, box_text, bbox=dict(facecolor='white', alpha=0.5), transform=fig.transFigure)

    # Set ticks for axes
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))


    # Loop for plotting PCA lines
    for i in range(0, len(y_data_df.columns)):
        # Define y data
        y = y_data_df[y_data_df.columns[i]]

        # plot line
        plt.plot(x, y, color = 'blue')

    # Get and plot data for PCA line with maximum mean accuracy
    a = y_data_df[y_data_df.columns[n_pca_for_max_mean - 1]]
    plt.plot(x, a, color = 'red')

    # Make "fig2" the current figure
    fig2 = plt.gcf()

    # Display plot
    plt.show()

    # Save plot
    fig2.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    """ 
    Main program.
    
    Provides place for user to set global variables.  Provides place for user
    to specify data files to be used in analysis.  Calls functions to perform
    analyses under specified conditions.
    
    """
    #  Global parameters
    n_neighbors=300 # Max number of KNN neighbors to evaluate
    random_state=30  # Random number seed
    n_splits=10   # Number of CV splits
    max_num_pca_comps=10  # maximum number of pca components to evaluate

    # Read in data
    data_2017_df = pd.read_csv("2017-regularPP.csv")
    data_2018_df = pd.read_csv("2018-regularPP.csv")

    # Combine 2017 and 2018 data
    data_2017_2018_df = pd.concat([data_2017_df, data_2018_df])

    # Separate X and Y
    X_df = data_2017_2018_df.iloc[:, 4:27]
    y_df = data_2017_2018_df.iloc[:, 28]

    # Data frame for holding normalized and standardized accuracy data (no pca)
    knn_accuracy_data = pd.DataFrame(columns=['k','normalized_accuracy',
                                              'standardized_accuracy'])

    # Data frame for holding PCA with normalized-mean-shifted data
    knn_pca_norm_shift_accuracy_data_df = pd.DataFrame(columns=np.arange(
        max_num_pca_comps + 1), index=np.arange(n_neighbors))

    # Code to allow labeling of data frame columns to be 1PC, 2PC, etc.
    current_name = 0
    knn_pca_norm_shift_accuracy_data_df.rename(columns={current_name : 'k'},
                                               inplace=True)

    # Loop to change columns names
    for i in range(1, max_num_pca_comps + 1):
        current_name = i

        new_name = f'{i}PC'

        knn_pca_norm_shift_accuracy_data_df.rename(columns=
                                                   {current_name : new_name}, inplace=True)


    # Data frame for holding PCA with standardized data
    knn_pca_std_accuracy_data_df = pd.DataFrame(columns=np.arange(
        max_num_pca_comps + 1), index=np.arange(n_neighbors))


    # Code to allow labeling of data frame columns to be 1PC, 2PC, etc.
    current_name = 0
    knn_pca_std_accuracy_data_df.rename(columns={current_name : 'k'},
                                        inplace=True)

    # Loop to change columns names
    for i in range(1, max_num_pca_comps + 1):
        current_name = i

        new_name = f'{i}PC'

        knn_pca_std_accuracy_data_df.rename(columns={current_name : new_name},
                                            inplace=True)


    # Initialize counters
    max_normalized_mean_accuracy = 0
    k_value_for_max_normalized = 0

    max_standardized_mean_accuracy = 0
    k_value_for_max_standardized = 0

    max_norm_shift_pca_mean_accuracy = 0
    k_value_for_max_pca_norm_shift = 0
    num_pca_comps_for_max_mean_norm_shift_accuracy = 0

    max_standardized_pca_mean_accuracy = 0
    k_value_for_max_pca_standardized = 0
    num_pca_comps_for_max_mean_std_accuracy = 0

    # Loop for creating plot data for KNN (normalized and standardized)
    # without pca accuracy data
    for k in range(1, n_neighbors + 1):
        scores_normalized_mean, scores_standardized_mean = cv_knn_no_pca(
            n_neighbors=k,
            random_state=random_state,
            n_splits=n_splits,
            X_df=X_df,
            y_df=y_df)
        knn_accuracy_data = knn_accuracy_data.append({'k': k,
                                                      'normalized_accuracy' : scores_normalized_mean,
                                                      'standardized_accuracy': scores_standardized_mean},
                                                     ignore_index=True)

        # Keep track of max mean and associated k value for non-pca case
        if scores_normalized_mean > max_normalized_mean_accuracy:
            max_normalized_mean_accuracy = scores_normalized_mean
            k_value_for_max_normalized = k
        if scores_standardized_mean > max_standardized_mean_accuracy:
            max_standardized_mean_accuracy = scores_standardized_mean
            k_value_for_max_standardized = k


    # Loop for creating pca-normalized-mean-shifted accuracy
    for n_pca in range(1, max_num_pca_comps + 1):
        for k in range(1, n_neighbors + 1):
            # Set first column to correct k value
            knn_pca_norm_shift_accuracy_data_df.iat[k-1, 0] = k
            # Get mean accuracy for n_pca and k values
            scores_pca_normalized_mean = MinMaxMeanShiftStratifiedKFoldKNNpca(
                X_df=X_df, y_df=y_df, n_splits=n_splits, n_neighbors=k,
                n_pca_comps=n_pca, random_state=random_state)
            # Update dataframe with max accuracy value for k and n_pca values
            knn_pca_norm_shift_accuracy_data_df.iat[k-1, n_pca] = scores_pca_normalized_mean

            #  Keep track of max mean and corresponding k and n_pca values
            if scores_pca_normalized_mean > max_norm_shift_pca_mean_accuracy:
                max_norm_shift_pca_mean_accuracy = scores_pca_normalized_mean
                k_value_for_max_pca_norm_shift = k
                num_pca_comps_for_max_mean_norm_shift_accuracy = n_pca

    # Loop for creating pca-standardized accuracy
    for n_pca in range(1, max_num_pca_comps + 1):
        for k in range(1, n_neighbors + 1):
            # Set first column to correct k value
            knn_pca_std_accuracy_data_df.iat[k-1, 0] = k
            # Get mean accuracy for n_pca and k values
            scores_pca_standardized_mean = cv_knn_pca_std(n_neighbors=k,
                                                          random_state=random_state, n_splits=n_splits,
                                                          n_pca_components=n_pca, X_df=X_df, y_df=y_df)
            # Update dataframe with max accuracy value for k and n_pca values
            knn_pca_std_accuracy_data_df.iat[k-1, n_pca] = scores_pca_standardized_mean

            #  Keep track of max mean and corresponding k and n_pca values
            if scores_pca_standardized_mean > max_standardized_pca_mean_accuracy:
                max_standardized_pca_mean_accuracy = scores_pca_standardized_mean
                k_value_for_max_pca_standardized = k
                num_pca_comps_for_max_mean_std_accuracy = n_pca


    # Plot normalized results (accuracy vs. K)
    title = 'KNN Accuracy\nNormalized Data, No PCA'
    filename = 'KNN_accuracy_normalized_noPCA.pdf'
    plot_KNN(knn_accuracy_data.k, knn_accuracy_data.normalized_accuracy,
             k_value_for_max_normalized, max_normalized_mean_accuracy,
             title, filename)



    # Plot standardized results (accuracy vs. K)
    title = 'KNN Accuracy\nStandardized Data, No PCA'
    filename = 'KNN_accuracy_standardized_noPCA.pdf'
    plot_KNN(knn_accuracy_data.k, knn_accuracy_data.standardized_accuracy,
             k_value_for_max_standardized, max_standardized_mean_accuracy,
             title, filename)


    # Plot normalized-mean-shift-PCA results (accuracy vs. K) -
    # lines for # of PCA components
    title = 'KNN Accuracy\nNormalized (w/mean shift) Data with PCA'
    filename = 'KNN-PCA_norm_accuracy_data_line_graph.pdf'
    plot_KNN_PCA(knn_pca_norm_shift_accuracy_data_df.k,
                 knn_pca_norm_shift_accuracy_data_df.drop(columns=['k']),
                 max_norm_shift_pca_mean_accuracy,
                 k_value_for_max_pca_norm_shift,
                 num_pca_comps_for_max_mean_norm_shift_accuracy,
                 title, filename)


    # Plot standardized-PCA results (accuracy vs. K) - lines for # of PCA components
    title = 'KNN Accuracy\nStandardized Data with PCA'
    filename = 'KNN-PCA_std_accuracy_data_line_graph.pdf'
    plot_KNN_PCA(knn_pca_std_accuracy_data_df.k,
                 knn_pca_std_accuracy_data_df.drop(columns=['k']),
                 max_standardized_pca_mean_accuracy,
                 k_value_for_max_pca_standardized,
                 num_pca_comps_for_max_mean_std_accuracy,
                 title, filename)


