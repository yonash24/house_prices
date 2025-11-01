"""
creating house prediction programm to the kaggle house pricing competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
import pathlib
import logging
from typing import Dict, Tuple
import seaborn as sns

"""
create class to import the required data from kaggle into a directory
transfer it into a dictionary of data frames
"""

class ImportData():

    #import the data from kaggle
    @staticmethod
    def import_data(path = "housing_data"):
        data_path = pathlib.Path(path)
        if data_path.exists():
            logging.info(f"the dictionary {data_path} is already exist")
            return
        else:
            try:
                data_path.mkdir()
                api = KaggleApi()
                api.authenticate()
                api.competition_download_files(
                    'house-prices-advanced-regression-techniques',
                    path=data_path,
                    unzip=True
                )
                logging.info("successfully imported the data")
            except Exception as e:
                logging.error(f"exception {e} has occured")
                

    #tranform the csv files into data frame and store them in a dictionary
    @staticmethod
    def to_df(path = "housing_data"):
        data_dict = {}
        data_path = pathlib.Path(path)
        if not data_path.exists():
            logging.error(f"Data directory '{data_path}' not found. Please run import_data() first.")
            return {}
        dir_path = data_path
        for file in dir_path.glob("*.csv"):
            try:
                key = file.stem
                df = pd.read_csv(file)
                data_dict[key] = df
            except Exception as e:
                logging.error(f"exception {e} has occured in file: {file}")
        
        logging.info("successfully transfered the data into data frames")
        return data_dict

    
"""
create a class that give us information about the data on our data set
"""

class DataInfo:

    #create constructor for the class    
    def __init__(self,data_dict:Dict[str, pd.DataFrame]):
        self.data_dict = data_dict
        self.train_df = data_dict["train"]
        self.test = data_dict["test"]
        self.sample_sub_mission = data_dict["sample_submission"]
    
    #create a histogram of salePrise
    def salePrise_info(self):
        plt.hist(self.train_df["SalePrice"], bins=30, edgecolor="black", alpha=0.7)
        plt.title("SalePrice distribution")
        plt.xlabel("SalePrice")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle = '--', alpha=0.7)
        plt.show()

    #log the target SalePrice to see his normal distribution
    def log_distributoin(self):
        log_sale_price = np.log1p(self.train_df["SalePrice"])
        plt.hist(log_sale_price, bins=30, edgecolor="black", alpha=0.7)
        plt.title("SalePrice distribution")
        plt.xlabel("SalePrice")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle = '--', alpha=0.7)
        plt.show()


    #create table of values represent how many missing values there is in any column
    def get_missing_vals(self):
        for key, df in self.data_dict.items():
            all_missing_vals = df.isnull().sum()
            missing_cols = all_missing_vals[all_missing_vals > 0]
            if missing_cols.empty:
                logging.info("theres no missing values")
            else:
                missing_df = pd.DataFrame({
                    "Missing Count" : missing_cols,
                    "Percentage %": (missing_cols / len(df) * 100).round(2) 
                })
                sorted_df = missing_df.sort_values(by = "Missing Count", ascending=False)
                logging.info("\n" + sorted_df.to_string())

    # create heat map to recognaize corelation between features
    def features_coreleation(self):
        corr_matrix = self.train_df.corr(numeric_only=True)
        sns.heatmap(corr_matrix, 
                    annot=True, cmap="coolwarm", 
                    fmt='.2f', 
                    linewidths=0.5)
        plt.title("corelation heat map")
        plt.show()

    
        # create scatter plots between tow features

    # create scatter plots between tow features
    def features_scatter(self, feature1, feature2):
        plot_df = self.train_df[[feature1, feature2]].copy()
        sns.scatterplot(
            x=feature1, 
            y=feature2, # Use the transformed y-column if log_y is True
            data=plot_df,
            alpha=0.6 # Make points slightly transparent if there's a lot of overlap
        )
        
        plt.title(f'Relationship between {feature1} and {feature2} in train data frame')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


    #create a box plot to our data
    def categorical_features_analysis(self, categorical_feature, numerical_feature):

        if categorical_feature not in self.train_df.columns or numerical_feature not in self.train_df.columns:
            logging.error("the features are not in the data frame")
            return
        if not pd.api.types.is_numeric_dtype(self.train_df[numerical_feature]):
            logging.error("the numerical feature you entered is not numerical")
            return
        try:
            sns.boxplot(x=categorical_feature, y=numerical_feature, data=self.train_df)
            plt.title(f"{categorical_feature} and {numerical_feature} box plot")
            plt.show()
        except Exception as e:
            logging.error(f"exception has occured {e}")

"""
create a class for cleaning the data and make high quality abd efficient for the models
"""

class DataCleaning:
    
    #create constructor for the class    
    def __init__(self,data_dict:Dict[str, pd.DataFrame]):
        self.data_dict = data_dict

    # fill missing values
    def fill_miss_vals(self):
        for key, df in self.data_dict.items():
            cols_fill_none = [
                'Alley', 'Fence', 'PoolQC', 'MiscFeature', 'FireplaceQu',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType'
            ]
            df[cols_fill_none] = df[cols_fill_none].fillna('None')

            cols_fill_zero = [
                'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageYrBlt', 
                'GarageCars', 'GarageArea'
            ]
            df[cols_fill_zero] = df[cols_fill_zero].fillna(0)

            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
            cols_fill_mode = [
                'Electrical', 'Utilities', 'Exterior1st', 
                'Exterior2nd', 'KitchenQual', 'SaleType', 'MSZoning'
            ]
            
            for col in cols_fill_mode:
                if col in df.columns:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
            self.data_dict[key] = df
            
        logging.info("Successfully filled missing values based on domain logic.")

    #handle outliers remove extrem values
    def outliers_handler(self):
        self.data_dict["train"] = self.data_dict["train"].drop(self.data_dict["train"][(self.data_dict["train"]["GrLivArea"] > 4000) & (self.data_dict["train"]["SalePrice"] < 300000)].index)
        
    #remove irrelevant columns
    def remove_nois(self):
        cols_to_remove = ["Id", "Utilities", "Street"]
        self.data_dict["train"] = self.data_dict["train"].drop(columns=cols_to_remove)
        self.data_dict["test"] = self.data_dict["test"].drop(columns=cols_to_remove)
        logging.info(f"Removed irrelevant columns: {cols_to_remove}")

"""
create class that preprocess the data 
and make 
"""