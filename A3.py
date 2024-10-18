# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:04:24 2024

@author: Admin
"""


#=============================A3

#=============California Housing


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


#============================================================================


'Model1==============================Linear Regression'


#=============================Step1: Data -----> X, Y


from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

x = data.data
y = data.target

data.feature_names
data.target_names


#=============================Step2: KFold


from sklearn.model_selection import KFold
kf = KFold(n_splits = 20, shuffle = True, random_state = 42)


#=============================Step3: Model Selection with Hyperparemetr


from sklearn.linear_model import LinearRegression
model = LinearRegression()

my_params = {'copy_X': [True,False], 
             'fit_intercept': [True,False], 
             'positive': [True,False]}


#=============================Step3: Fit


from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)


#=============================Step4: Validation


gs.best_score_   #-0.3174655973647021
gs.best_params_  #{'copy_X': True, 'fit_intercept': True, 'positive': False}


#============================================================================


'Model2==============================KNN'


#=============================Step1: Data -----> X, Y


from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

x = data.data
y = data.target

data.feature_names
data.target_names


#=============================Step2: KFold


from sklearn.model_selection import KFold
kf = KFold(n_splits = 20, shuffle = True, random_state = 42)


#=============================Step3: Model Selection with Hyperparemetr


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()

my_params = {'n_neighbors':[1,2,3,4,5,6,10],
            'metric':['minkowski'  , 'euclidean' , 'manhattan']}


#=============================Step3: Fit


from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)


#=============================Step4: Validation


gs.best_score_   #-0.47860690591324745
gs.best_params_  #{'metric': 'manhattan', 'n_neighbors': 5}


#============================================================================


'Model3=======================Decision Tree Regressor'


#=============================Step1: Data -----> X, Y


from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

x = data.data
y = data.target

data.feature_names
data.target_names


#=============================Step2: KFold


from sklearn.model_selection import KFold
kf = KFold(n_splits = 20, shuffle = True, random_state = 42)


#=============================Step3: Model Selection with Hyperparemetr


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 42)

my_params = {'max_depth':[1,2,3,4,5,6,7,10,20,50]}


#=============================Step3: Fit


from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)


#=============================Step4: Validation


gs.best_score_   #-0.2401089617223759
gs.best_params_  #{'max_depth': 10}


#============================================================================


'Model4=======================Random Forest Regressor'


#=============================Step1: Data -----> X, Y


from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

x = data.data
y = data.target

data.feature_names
data.target_names


#=============================Step2: KFold


from sklearn.model_selection import KFold
kf = KFold(n_splits = 20, shuffle = True, random_state = 42)


#=============================Step3: Model Selection with Hyperparemetr


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

my_params = {'n_estimators': [50, 100, 200],
             'max_depth': [None, 10, 20, 30],
             'min_samples_split': [2, 5, 10]}


#=============================Step3: Fit


from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)


#=============================Step4: Validation


gs.best_score_   #-0.17936497031056114
gs.best_params_  #{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}


#============================================================================


'Model5===========================================SVR'


#=============================Step1: Data -----> X, Y


from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

x = data.data
y = data.target

data.feature_names
data.target_names


#=============================Step2: KFold


from sklearn.model_selection import KFold
kf = KFold(n_splits = 20, shuffle = True, random_state = 42)


#=============================Step3: Model Selection with Hyperparemetr


from sklearn.svm import SVR
model = SVR()

my_params = {'kernel':['poly','rbf','linear'],
             'C':[0.001,0.01,1]}


#=============================Step3: Fit


from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')

gs.fit(x,y)


#=============================Step4: Validation


gs.best_score_   #
gs.best_params_  #




''' Results

======LR: 68.3%

gs.best_score_   #-0.3174655973647021
gs.best_params_  #{'copy_X': True, 'fit_intercept': True, 'positive': False}

=====KNN: 52.2%

gs.best_score_   #-0.47860690591324745
gs.best_params_  #{'metric': 'manhattan', 'n_neighbors': 5}

=====DTR: 66%

gs.best_score_   #-0.2401089617223759
gs.best_params_  #{'max_depth': 10}

=====RFR: 72.1%

gs.best_score_   #-0.17936497031056114
gs.best_params_  #{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200} 
    
=====SVR: 



'''




