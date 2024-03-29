# bookie
To run KNN-CV.py:
1)  Ensure the libraries identified in the import statements are available to 
    the program.
2)  Select KNN-CV.py from the bookie/src/mfsapi directory and run the program.  
    The KNN-CV.py file depends on two data files located in the 
    bookie/src/mfsapi/data directory ("2017-regularPP.csv" and 
    2018-regularPP.csv") and is preconfigured to go there for the data.  
    That is all that is required to run the code.
Notes:
  a) This program performs KNN for 4 scenarios using cross-validation of 2017 
     and 2018 season data.  The 4 scenarios are: 1) Normalized data, 
     2) Standardized data, 3) Normalized with mean-shifted and PCA data and
     4) Standardized with PCA data.
  b) The code is set up to run with the following global paramaters:  max 
     number of K-nearest neighbors = 300, random state = 30, 
     number of CV splits = 10, max number of PCA components to consider = 10.
  c) With these parameters, it takes 3-4 hours to run and the output is 
     4 graphs, so you can start it and go out for a nice dinner.
  d) If you want to just verify that the code works without verifying the 
     results compared to the report, you can change the global variables 
     as long as you don't make selections that cause errors (i.e., inconsistent
     parameters with the code).
     
     
To run KNN_Train_Test.py:
1)  Ensure the libraries identified in the import statements are available to 
    the program.
2)  Select KNN_Train_Test.py from the bookie/src/mfsapi directory and run 
    the program.  The KNN_Train_Test.py file depends on three data files 
    located in the bookie/src/mfsapi/data directory ("2017-regularPP.csv", 
    2018-regularPP.csv", and "2019-regularPP.csv") and is preconfigured to go 
    there for the data.  That is all that is required to run the code.
Notes:
  a) This program performs KNN for 2 scenarios using 2017 and 2018 season data 
     for training and 2019 season data for testing.  The 2 scenarios are:  
     1) Standardized data, and 2) Standardized with PCA data.
  b) The code is set up to run with the following global paramaters:  
     max number of K-nearest neighbors = 300, random state = 30, max number 
     of PCA components to consider = 10.
  c) KNN_Train_Test.py takes much less time that the KNN-CV.py file 
     to run, but it still takes over an hour. So, you don't have time
     for a nice dinner but can go out for a quick lunch.
  d) If you want to just verify that the code works without verifying the 
     results compared to the report, you can change the global variables 
     as long as you don't make selections that cause errors (i.e., inconsistent
     parameters with the code).

To run decisionTree.R:
1)  Ensure the packages identified in the install statements are available to 
    the program.
2)  Select decisionTree.R from the bookie/src/mfsapi directory and run 
    the program.  The decisionTree.R file depends on three data files 
    located in the bookie/src/mfsapi/data directory ("2017-regularPP.csv", 
    2018-regularPP.csv", and "2019-regularPP.csv") and is preconfigured to go 
    there for the data.  That is all that is required to run the code.
Notes:
  a) This program performs Decision Tree for 2 scenarios using 2017 data for training and 2018 data for testing (Midway 
     Results) 2017 and 2018 season data for training and 2019 season data for testing (Final Results).  The 2 scenarios are:        
     1) Cross-Validation, and 2) Without Cross-Validaion.
  b) This code produces seperate decision tree outputs for 2 approaches. The 2 approaches are: 1) GINI Index and 
     2) Information Gain
 
 
To run svm.py:
1)  Ensure the packages identified in the import statements are available to 
    the program.
2)  To run: python svm.py
Notes:
   - This code performs tuning of pca components on 2017/2018 data with cross-validation using optimal 
   hyperparameters for SVM with sigmoid kernel on standardized data set and outputs to csv file sorted by
   descending accuracy.


To run svm_final.py:
1)  Ensure the packages identified in the import statements are available to 
    the program.
2)  To run: python svm_final.py
Notes:
   - This code fits the optimal SVM model with 2017/2018 training data and tests on unseen 2019 test set. 
   The code outputs accuracy on the test set as well as several plots to visualize results.
   
To run nn_keras.py
1) Ensure all necessary packages are installed.
2) To run: python nn_keras
Notes:
    - hyperparameter tuning has been commented out on the current version. Currently set to trainon 2017/2018 data and test on 2019 with optimal hyperparamters chosen.
    
