"""
This program performs KNN training and test (no CV) for the datasets
identified in the program using pca-standardized data.
The following global parameters are provided
by the user in the code:
    - Max number of nearest neighbors to evaluate
    - Max number of pca components to evaluate
    - Random seed
    - Data sets as csv files (program is set up to use
      "2017-regularPP.csv" and "2018-regularPP.csv" as sources for
      training data and the "2019-regularPP.csv" as the source for
      testing data
The following output is produced:
    - Graph of accuracy vs. K for standardized data
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
import random


def knn_no_pca (n_neighbors, X_train_df, X_test_df, y_train_df, y_test_df):
    """
    Function performs KNN (no pca) using normalized and
    standardized data sets.
    Inputs:
        -  n_neighbors - number of KNN nearest neighbors
        -  n_splits - number of CV folds
        -  X_train_df - dataframe containing X data for training
        -  X_test_df - dataframe containing X data for testing
        -  y_train_df - dataframe containing y data for training
        -  y_test_df - dataframe containing y data for testing
    Returns:
        - knn accuracy
    """

    # Standardize data
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_df)
    X_train_std_df = scaler.transform(X_train_df)
    X_test_std_df = scaler.transform(X_test_df)

    # Perform KNN
    KNN = knn(n_neighbors=n_neighbors)
    KNN.fit(X_train_std_df,y_train_df)
    y_pred = KNN.predict(X_test_std_df)
    accuracy = accuracy_score(y_test_df, y_pred)

    return accuracy


def knn_pca_std(n_neighbors, n_pca_components, X_train_df, X_test_df,
                y_train_df, y_test_df):
    """
    Function performs  KNN with PCA using standardized data set.
    Inputs:
        -  n_neighbors - number of KNN nearest neighbors
        -  n_pca_components - number of PCA components to use
        -  X_train_df - dataframe containing X data for training
        -  X_test_df - dataframe containing X data for testing
        -  y_train_df - dataframe containing y data for training
        -  y_test_df - dataframe containing y data for testing
    Returns:
        - KNN accuracy for specified K and number of pca components
          values
    """

    # Standardize data
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_df)
    X_train_std_df = scaler.transform(X_train_df)
    X_test_std_df = scaler.transform(X_test_df)


    # Conduct KNN - pca_standardized
    KNN = knn(n_neighbors=n_neighbors)
    pca = PCA(n_components=n_pca_components)
    pca.fit(X_train_std_df)
    X_train_std_pca_df = pca.transform(X_train_std_df)
    X_test_std_pca_df = pca.transform(X_test_std_df)
    KNN.fit(X_train_std_pca_df, y_train_df)
    y_pred = KNN.predict(X_test_std_pca_df)
    accuracy = accuracy_score(y_test_df, y_pred)

    return accuracy


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
    max_num_pca_comps=10  # maximum number of pca components to evaluate

    # Read in data
    data_2017_df = pd.read_csv("2017-regularPP.csv")
    data_2018_df = pd.read_csv("2018-regularPP.csv")
    data_2019_df = pd.read_csv("2019-regularPP.csv")

    # Combine 2017 and 2018 data
    data_2017_2018_df = pd.concat([data_2017_df, data_2018_df])

    # Randomize the training data
    random.seed(random_state)  # Set random seed
    data_2017_2018_randomized_df = data_2017_2018_df.sample(frac=1).reset_index(drop=True)

    # Separate into X_train, X_test, y_train and y_test
    X_train_df = data_2017_2018_randomized_df.iloc[:, 4:27]
    X_test_df = data_2019_df.iloc[:, 4:27]
    y_train_df = data_2017_2018_randomized_df.iloc[:, 28]
    y_test_df = data_2019_df.iloc[:, 28]

    # Data frame for holding standardized accuracy data (no pca)
    knn_accuracy_data = pd.DataFrame(columns=['k', 'standardized_accuracy'])


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
    max_standardized_accuracy = 0
    k_value_for_max_standardized = 0

    max_standardized_pca_accuracy = 0
    k_value_for_max_pca_standardized = 0
    num_pca_comps_for_max_std_accuracy = 0

    # Loop for creating plot data for KNN (standardized)
    # without pca accuracy data
    for k in range(1, n_neighbors + 1):
        accuracy = knn_no_pca (n_neighbors=k, X_train_df=X_train_df,
                               X_test_df=X_test_df, y_train_df=y_train_df,
                               y_test_df=y_test_df)

        knn_accuracy_data = knn_accuracy_data.append({'k': k,
                                            'standardized_accuracy': accuracy},
                                            ignore_index=True)

        if accuracy > max_standardized_accuracy:
            max_standardized_accuracy = accuracy
            k_value_for_max_standardized = k


    # Loop for creating pca-standardized accuracy
    for n_pca in range(1, max_num_pca_comps + 1):
        for k in range(1, n_neighbors + 1):
            # Set first column to correct k value
            knn_pca_std_accuracy_data_df.iat[k-1, 0] = k
            # Get accuracy for n_pca and k values
            accuracy = knn_pca_std(n_neighbors=k, n_pca_components=n_pca,
                                   X_train_df=X_train_df, X_test_df=X_test_df,
                                   y_train_df=y_train_df, y_test_df=y_test_df)

            # Update dataframe with max accuracy value for k and n_pca values
            knn_pca_std_accuracy_data_df.iat[k-1, n_pca] = accuracy

            #  Keep track of max mean and corresponding k and n_pca values
            if accuracy > max_standardized_pca_accuracy:
                max_standardized_pca_accuracy = accuracy
                k_value_for_max_pca_standardized = k
                num_pca_comps_for_max_std_accuracy = n_pca


    # Plot standardized results (accuracy vs. K)
    title = 'KNN Accuracy\nStandardized Data, No PCA\n(2017+2018 Training Data, 2019 Test Data)'
    filename = 'KNN_accuracy_standardized_noPCA_test.pdf'
    plot_KNN(knn_accuracy_data.k, knn_accuracy_data.standardized_accuracy,
             k_value_for_max_standardized, max_standardized_accuracy,
             title, filename)


    # Plot standardized-PCA results (accuracy vs. K) - lines for # of PCA components
    title = 'KNN Accuracy\nStandardized Data with PCA\n(2017+2018 Training Data, 2019 Test Data)'
    filename = 'KNN-PCA_std_accuracy_data_line_graph_test.pdf'
    plot_KNN_PCA(knn_pca_std_accuracy_data_df.k,
                 knn_pca_std_accuracy_data_df.drop(columns=['k']),
                 max_standardized_pca_accuracy,
                 k_value_for_max_pca_standardized,
                 num_pca_comps_for_max_std_accuracy,
                 title, filename)

