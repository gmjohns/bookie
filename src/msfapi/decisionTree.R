##Project pre-processing

#Clearing the global variables
rm(list = ls())
#Installing packages
install.packages(c("rlang","caret","tidyverse","RGtk2","rattle", "esquisse", "dplyr"))
install.packages("FSelectorRcpp")

#Setting the directory
wd <- getwd()
setwd(wd) # Set the directory the same as source script
setwd("data") # name of the folder that contains the data
getwd()

# Previous Training and Test set
trainingSet_2017 <- read.csv("2017-regularPP.csv")
testSet_2018 <- read.csv("2018-regularPP.csv")

#Restructure Training and Test set to remove non-numerical columns
restructTrainingSet_2017 <- trainingSet_2017[6:length(trainingSet_2017)-1]
restructTestSet_2018 <- testSet_2018[6:length(testSet_2018)-1]

# Cross validation training set (2017 -2018 data)
combTrainingSet_2017_18 <- rbind(trainingSet_2017, testSet_2018)
finTestSet_2019 <- read.csv("2019-regularPP.csv")

#Restructure Training and Test set to remove non-numerical columns for Final Results
restructTrainingSet_2017_18 <- combTrainingSet_2017_18[6:length(combTrainingSet_2017_18)-1]
restructTestSet_2019 <- finTestSet_2019[6:length(finTestSet_2019)-1]


#Normalize function
norm <- function(x) # x is a column or vector in a dataset
{
  colMin <- (lapply(x, min)) #Min of column
  colMax <- (lapply(x, max)) #Max of column
  colMin <- colMin[[1]]
  colMax <- colMax[[1]]
  return((x- colMin) /(colMax-colMin))
}

#Centralize funtion (Subtracts the column mean from the value)
centralized <- function(x)
{
  colMean <- (lapply(x, mean))
  colMean <- colMean[[1]]
  return(x- colMean)
}

#Normalize Training Set
normalizeDataSet <- function(dataSet) {
  normTrainingSet <- data.frame()
  for (column in 1:ncol(dataSet)){
    colNorm <- as.data.frame(norm(dataSet[column]))
    normTrainingSet <- as.data.frame(append(normTrainingSet, colNorm))
  }
  return(normTrainingSet)
}

# Normalize Training Data Set for Midterm Results
norm_2017 <- normalizeDataSet(restructTrainingSet_2017)

# Normalize Training Data Set for Final Results
norm_2017_18 <- normalizeDataSet(restructTrainingSet_2017_18)

centralizeDataSet <- function(dataSet) {
  centTrainingSet <- data.frame()
  for (column in 1:ncol(dataSet)){
    colCent <- as.data.frame(centralized(dataSet[column]))
    centTrainingSet <- as.data.frame(append(centTrainingSet, colCent))
  }
  return(centTrainingSet)
}

# Centralize Training Data Set for Midterm Results
cent_2017 <- centralizeDataSet(norm_2017)

# Centralize Training Data Set for Final Results
cent_2017_18 <- centralizeDataSet(norm_2017_18)

#Normalize function for Test set
normBased <- function(y, x) # y is the test set and x is a column or vector in a dataset
{
  colMin <- (lapply(x, min)) #Min of column
  colMax <- (lapply(x, max)) #Max of column
  colMin <- colMin[[1]]
  colMax <- colMax[[1]]
  return((y- colMin) /(colMax-colMin))
}

#Centralize funtion (Subtracts the column mean from the value)
centBased <- function(y, x) # y is the test set and x is a column or vector in a dataset
{
  colMean <- (lapply(x, mean))
  colMean <- colMean[[1]]
  return(y- colMean)
}

#Normalize Test Set based on Old Min and Max
normalizeTestSet <- function(dataTestSet, dataTrainSet) {
  normTestSet <- data.frame()
  for (column in 1:ncol(dataTestSet)){
    colNormTest <- as.data.frame(normBased(dataTestSet[column], dataTrainSet[column]))
    normTestSet <- as.data.frame(append(normTestSet, colNormTest))
  }
  return(normTestSet)
}

#Normalize function for Test Set-Midterm
norm_2018 <- normalizeTestSet(restructTestSet_2018, restructTrainingSet_2017)

#Normalize function for Test Set-Final
norm_2019 <- normalizeTestSet(restructTestSet_2019, restructTrainingSet_2017_18)

#Centralize Test Set based on Old Mean
centralizedTestSet <- function(dataTestSet, dataTrainSet) {
  centTestSet <- data.frame()
  for (column in 1:ncol(dataTestSet)){
    colCentTest <- as.data.frame(centBased(dataTestSet[column], dataTrainSet[column]))
    # print(colNorm)
    centTestSet <- as.data.frame(append(centTestSet, colCentTest))
  }
  return(centTestSet)
}

#Centralize function for Test Set-Midterm
cent_2018 <- centralizedTestSet(norm_2018,norm_2017)

#Centralize function for Test Set-final
cent_2019 <- centralizedTestSet(norm_2019, norm_2017_18)


#Libraries for Decision Tree
library(rattle)
library(rpart)
library(rpart.plot)
library(RColorBrewer)

createDataTrain <- function(dataSet, rawDataSet) {
  data_train <- as.data.frame(append(dataSet, data.frame(rawDataSet[,ncol(rawDataSet)])))
  names(data_train)[ncol(data_train)] <- "fir_result"
  return(data_train)
}

# Midterm Training data for Decison Tree
data_train <- createDataTrain(cent_2017, trainingSet_2017)

# Final Training data for Decison Tree
fin_data_train <- createDataTrain(cent_2017_18, combTrainingSet_2017_18)


decisionTree <- function(dataTrainSet, dataTestSet, rawDataTestSet) {

  tree_dict <- data.frame("GINI" = 1, "InformationGain" = 2)
  sel_tree <- seq(1, 2)
  
  minD <- length(dataTestSet)
  minB <- ceiling(nrow(dataTrainSet)/minD)
  minS <- minB * 3
  
  for (value in sel_tree) {
    if (names(tree_dict[value]) == "GINI"){
        fit_Gini <- rpart(fir_result~., 
                          data = dataTrainSet, 
                          method = 'class', 
                          minsplit = minS, 
                          minbucket = minB,
                          maxdepth = minD)
        print(names(tree_dict[value])) #States that this is GINI
        fancyRpartPlot(fit_Gini, caption = "GINI")
        #summary(fit_Gini)
        GINI_unseen <- dataTestSet #Define unseen data to predict
        #GINI Prediction Model Dataframe
        prediction_df <- as.data.frame(predict(fit_Gini, GINI_unseen, type = "class"))
        #Create Error Rate Table
        GINI_compare_pred_to_test <- as.data.frame(append(prediction_df, data.frame(rawDataTestSet[,ncol(rawDataTestSet)])))
        colnames(GINI_compare_pred_to_test) <- c("predict","actual")
        GINI_error_rate <-as.data.frame(ifelse(GINI_compare_pred_to_test$predict==GINI_compare_pred_to_test$actual,1,0))
        colnames(GINI_error_rate) <- c("Error")
        #Prediction Error using normalize data
        GINI_Error <- prop.table(table(GINI_error_rate$Error))
        print(GINI_Error)
      
    } else if (names(tree_dict[value]) == "InformationGain") {
      
        fit_InfoGain <- rpart(fir_result~., 
                              data = dataTrainSet, 
                              method = 'class', 
                              parms = list(split = 'information'), 
                              minsplit = minS, 
                              minbucket = minB,
                              maxdepth = minD)
        print(names(tree_dict[value])) #States that this is InfoGain
        fancyRpartPlot(fit_InfoGain, caption = "Information Gain")
        #summary(fit_InfoGain)
        unseen <- dataTestSet #Define unseen data to predict
        #Information Gain Prediction Model
        prediction_IG_df <- as.data.frame(predict(fit_InfoGain, unseen, type = "class"))
        #Create Error Rate Table
        compare_pred_to_test <- as.data.frame(append(prediction_IG_df, data.frame(rawDataTestSet[,ncol(rawDataTestSet)])))
        colnames(compare_pred_to_test) <- c("predict","actual")
        error_rate <-as.data.frame(ifelse(compare_pred_to_test$predict==compare_pred_to_test$actual,1,0))
        colnames(error_rate) <- c("Error")
        #Prediction Error using normalize data
        Error <- prop.table(table(error_rate$Error))
        print(Error)
    }
  }
}


#Adding library to do cross validation
library(tidyverse)
library(caret)

# Midterm Descision Tree and Error/Accuracy
decisionTree(data_train, cent_2018, testSet_2018)
# Final Descision Tree and Error/Accuracy
decisionTree(fin_data_train, cent_2019, finTestSet_2019)


# Function for Decision Tree using Cross-Validation for each Fold
decisionTree_cv <- function(dataTrainSet, rawDataTrainSet, editedTrainSet, editedTestSet, rawTestSet, k=10) {
  
  n <- nrow(dataTrainSet)
  K <- k
  remain <- n%/%K
  set.seed(30) 
  randomN <- runif(n)
  rang <- rank(randomN)
  fold <- (rang-1)%/%remain + 1
  fold <- as.factor(fold)
  
  tree_dict <- data.frame("GINI" = 1, "InformationGain" = 2)
  sel_tree <- seq(1, 2)
  
  minD <- length(editedTestSet)
  minB <- ceiling((length(data_train[fold == 1, ncol(data_train)]))/minD)
  minS <- minB * 3
 
  for (value in sel_tree) {
    if (names(tree_dict[value]) == "GINI"){
      
      all.acc_G <- numeric(0)
      
      for (k in 1:K){
        
        fold_Output <- as.data.frame(rawDataTrainSet[fold == k, ncol(rawDataTrainSet)])
        colnames(fold_Output) <- c("fir_result")
        #norm_fold_Data <- normalizeDataSet(restructTrainingSet_2017_18[fold == k,])
        norm_fold_Data <- normalizeDataSet(editedTrainSet[fold == k,])
        cent_fold_Data <- centralizeDataSet(norm_fold_Data)
        
        #Normalize by Fold
        norm_Test_Data <- normalizeTestSet(editedTestSet, norm_fold_Data)
        #Centralize by Fold
        cent_Test_Data <- centralizedTestSet(norm_Test_Data, cent_fold_Data)
        
        fold_Data_Train <-as.data.frame(append(cent_fold_Data, fold_Output))
        
        fit_Gini<- rpart(fir_result~., 
                              data = fold_Data_Train[fold!=k,], 
                              method = 'class', 
                              minsplit = minS, 
                              minbucket = minB,
                              maxdepth = minD)
        
        #print(fold_Data_Train)
        unseen <- cent_Test_Data #Define unseen data to predict
        #Information Gain Prediction Model
        prediction <- predict(fit_Gini, unseen, type = "class")
        
        #Error Rate per Fold
        mc <- table(rawTestSet[,ncol(rawTestSet)], prediction)
        acc_G <- (mc[1,1] + mc[2,2])/sum(mc) #Accurracy per Fold
        all.acc_G <- rbind(all.acc_G, acc_G)
        colnames(all.acc_G) <- c("Fold Accuracy")
      }
      print(paste0("Each Fold Cross Validation Accuracy Result for ", names(tree_dict[value]), " Listed Below:"))
      row.names(all.acc_G) <- c(1:k)
      print(all.acc_G) 
      acc.cv_G <- mean(all.acc_G)
      print(paste0("The average Accuracy using CV for ", names(tree_dict[value]), " is ", acc.cv_G))
    } else if (names(tree_dict[value]) == "InformationGain") {
      
      all.acc <- numeric(0)
      
      for (k in 1:K){
        fold_Output <- as.data.frame(rawDataTrainSet[fold == k, ncol(rawDataTrainSet)])
        colnames(fold_Output) <- c("fir_result")
        #norm_fold_Data <- normalizeDataSet(restructTrainingSet_2017_18[fold == k,])
        norm_fold_Data <- normalizeDataSet(editedTrainSet[fold == k,])
        cent_fold_Data <- centralizeDataSet(norm_fold_Data)
        
        #Normalize by Fold
        norm_Test_Data <- normalizeTestSet(editedTestSet, norm_fold_Data)
        #Centralize by Fold
        cent_Test_Data <- centralizedTestSet(norm_Test_Data, cent_fold_Data)
        
        fold_Data_Train <-as.data.frame(append(cent_fold_Data, fold_Output))
        
        fit_InfoGain <- rpart(fir_result~., 
                              data = fold_Data_Train[fold!=k,], 
                              method = 'class', 
                              parms = list(split = 'information'), 
                              minsplit = minS, 
                              minbucket = minB,
                              maxdepth = minD)
        
        #print(fold_Data_Train)
        unseen <- cent_Test_Data #Define unseen data to predict
        #Information Gain Prediction Model
        prediction <- predict(fit_InfoGain, unseen, type = "class")
        
        #Error Rate per Fold
        mc <- table(rawTestSet[,ncol(rawTestSet)], prediction)
        acc <- (mc[1,1] + mc[2,2])/sum(mc) 
        all.acc <- rbind(all.acc, acc)
        colnames(all.acc) <- c("Fold Accuracy")
      }
      print(paste0("Each Fold Cross Validation Accuracy Result for ", names(tree_dict[value]), " Listed Below:"))
      row.names(all.acc) <- c(1:k)
      print(all.acc) 
      acc.cv <- mean(all.acc)
      print(paste0("The average Accuracy using CV for ", names(tree_dict[value]), " is ", acc.cv))
    }
  }
}


# Midterm Descision Tree and Error/Accuracy with CV
decisionTree_cv(data_train, trainingSet_2017, restructTrainingSet_2017, restructTestSet_2018, testSet_2018)
# Final Descision Tree and Error/Accuracy with CV
decisionTree_cv(fin_data_train, combTrainingSet_2017_18, restructTrainingSet_2017_18, restructTestSet_2019, finTestSet_2019)

