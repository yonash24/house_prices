import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import logging
from typing import Tuple
from sklearn.base import BaseEstimator

"""
craete class that build train and evaluate the regression models 
for the quest 
"""

class Models:


    
    #create constructor to the class
    def __init__(self, x_train_df:pd.DataFrame, test_df:pd.DataFrame, y_train:pd.Series):
        self.x_train_df = x_train_df
        self.y_train_df = y_train
        self.test_df = test_df
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.models = {
            "Ridge" : Ridge(alpha=10),
            "Lasso" : Lasso(alpha=0.001, max_iter=5000),
            "RandomForest" : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGB" : XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='rmse'),
            "LightGBM" : LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        }

    #create and train the models make a predictions and evaluate them
    def choose_model(self):
        
        cv_result = {}
        best_model = float('inf') 

        for model_name, model in self.models.items():
            scores = cross_val_score(
                model, 
                self.x_train_df,
                self.y_train_df,
                cv= self.kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            
            rmse_score = -scores
            mean_rmse_score = rmse_score.mean()

            if mean_rmse_score < best_model:
                best_model = mean_rmse_score
                print(f"the best model is: {model_name} with mrmse of {best_model}")
                self.best_model_score = mean_rmse_score
                self.best_model_name = model_name

    #optimize ridge models
    def ridge_optimaizer(self):
        param_grid = {"alpha" : [0.0001, 0.001, 0.01, 0.1, 1, 10]}

        ridge_model = Ridge()

        grid_search = GridSearchCV(
            estimator=ridge_model,
            param_grid = param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        grid_search.fit(self.x_train_df, self.y_train_df)
        self.ridge_model = grid_search.best_estimator_
        self.best_model_score = best_score = grid_search.best_score_

        logging.info(f"lasso best evaluation is: {self.best_model_score}")

    #optimaize xgb model
    def xgb_optimaizer(self):
        param_grid = {
            'n_estimators': [300, 500, 700],   
            'learning_rate': [0.01, 0.05, 0.1],   
            'max_depth': [3, 5, 7]                
        }

        model = XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        grid_search.fit(self.x_train_df, self.y_train_df)
        self.XGB_model = grid_search.best_estimator_
        self.second_best_score = grid_search.best_score_

        logging.info(f"the best xgb model rmse is: {-self.second_best_score}")

        