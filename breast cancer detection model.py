# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:26:25 2019

@author: Debasish.Dash
"""


#Breast Cancer Detection

#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#Get the data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#See the keys in the dataset
cancer.keys()

#create a dataframe having all columnns with column names along with target column
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


#Find out most correlated datapoints by a heatmap. 0.99 means most related
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True)

# Found Strong correlation between the mean radius and mean perimeter, mean area and mean primeter

#Assign X and Y
#X has every parameter except the last column
#y has the last(target) column
X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']

#Split data into train / test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=None)

#Start Training the model - First approach
from sklearn.svm import SVC 
model = SVC()  #Call the model
model.fit(X_train,y_train)  #Fit the model on train
y_predict = model.predict(X_test)   #Predict the X test
#confusion matrix and precision report
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test,y_predict)  #create cm matrix
sns.heatmap(cm,annot=True)  #plot heatmap
print(classification_report(y_test, y_predict))  #print the precision report

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)
#Above we see that the scaling is bad

#Start Training the model - Second approach
#First scale the data
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train

#check if the scaling is rectified
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

#if yes, Repeat this scaling on test data
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test-min_test)/range_test

#rebuild the model with scaled train and test data
model = SVC()  #Call the model
model.fit(X_train_scaled,y_train)  #Fit the model on train
y_predict = model.predict(X_test_scaled)   #Predict the X test
cm = confusion_matrix(y_test,y_predict)  #create cm matrix
sns.heatmap(cm,annot=True)  #plot heatmap
print(classification_report(y_test, y_predict))  #print the precision report


#further improving the model, find best C and gamma for SVC() model by gridsearch
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid, refit=True, verbose=10, n_jobs = -1,scoring='accuracy')
grid.fit(X_train_scaled,y_train)

#see the best parameter and best estimator
grid.best_params_
grid.best_estimator_

#reapply the new C and gamma values
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,grid_predictions))
