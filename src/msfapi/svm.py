from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Path
loc = 'C:\\Users\\Will\\Documents\\GradSchool\\CSC522\\Project\\data\\'


#Reusable code for testing different data sets against model

def runModel(trainFile, validateFile, pca):
    train_X = pd.read_csv(loc + trainFile)
    val_X = pd.read_csv(loc + validateFile)

    if pca == False:
        train_y = train_X['fir_result'].tolist()
        val_y = val_X['fir_result'].tolist()
    else: 
        train_y = train_X['label'].tolist()
        val_y = val_X['label'].tolist()
    
    #Drop unnecessary columns for model fitting
    if pca == False:
        train_X = train_X.drop(columns=['date', 'home_team', 'away_team', 'fir_result'])
        val_X = val_X.drop(columns=['date', 'home_team', 'away_team', 'fir_result'])
    else:
        train_X = train_X.drop(columns=['label'])
        val_X = val_X.drop(columns=['label'])

    train_X = train_X.drop(train_X.columns[0], axis=1)
    val_X = val_X.drop(val_X.columns[0], axis=1)

    #Tune parameters for SVC + plot
    acc = np.array([])
    iterations = np.array([100, 1000, 10000, 100000])
    for i in iterations:
        model = LinearSVC(max_iter=i)
        model.fit(train_X, train_y)
        #predictions = model.predict(val_X)
        accuracy = model.score(val_X, val_y)
        acc = np.append(acc, accuracy)
        print(accuracy)


    plt.title('Max Iterations & Accuracy')
    plt.rc('font', family='serif', size=13)
    plt.plot(iterations, acc)
    axes = plt.gca()
    axes.set_xlim([100, 100000])
    axes.set_ylim([0, 1])
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Iterations")
    plt.show()

    acc = np.array([])
    c_vals = np.array([.01, .1, 1.0, 10.0, 100.0, 1000.0])
    for c in c_vals:
        model = LinearSVC(C=c)
        model.fit(train_X, train_y)
        #predictions = model.predict(val_X)
        accuracy = model.score(val_X, val_y)
        acc = np.append(acc, accuracy)
        print(accuracy)


    plt.title('Penalty Parameter & Accuracy')
    plt.rc('font', family='serif', size=13)
    plt.plot(c_vals, acc)
    axes = plt.gca()
    axes.set_xlim([.01, 1000.0])
    axes.set_ylim([0, 1])
    plt.ylabel("Accuracy")
    plt.xlabel("Penalty Parameter Value")
    plt.show()



if __name__ == "__main__":
    #Regular
    runModel('2017-regularPP.csv', '2018-regularPP.csv', pca=False)
    #Normalized
    runModel('2017-norm-centered.csv', '2018-norm-centered.csv', pca=False)
    #PCA + Normalized
    runModel('2017-pca-norm.csv', '2018-pca-norm.csv', pca=True)
    #PCA + Standardized
    runModel('2017-pca-std.csv', '2018-pca-std.csv', pca=True)
    #Standardized
    runModel('2017-standardized.csv', '2018-standardized.csv', pca=False)