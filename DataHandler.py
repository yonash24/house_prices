"""
creating house prediction programm to the kaggle house pricing competition
"""

import pandas as pd
import numpy as np
import matplotlib as plt
from kaggle.api.kaggle_api_extended import KaggleApi
import pathlib
import logging
from typing import Dict, Tuple

"""
create class to import the required data from kaggle into a directory
transfer it into a dictionary of data frames
"""

class ImportData():

    #import the data from kaggle
    def import_data():
        path = "housing_data"
        if pathlib.Path(path).exists():
            logging.warning(f"the dictionary {path} is already exist")
            return
        else:
            pathlib.Path(path).mkdir()
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files("house-prices-advanced-regression-techniques/data",path=path,unzip=True)
            logging.info("successfully imported the data")

    #tranform the csv files into data frame and store them in a dictionary
    def to_df(path = "housing_data"):
        data_dict = {}
        dir_path = pathlib.Path(path)
        for file in dir_path.glob(".csv"):
            key = pathlib.Path(file).stem
            df = pd.read_csv(file)
            data_dict[key] = df
        logging.info("successfully transfered the data into data frames")
        return data_dict