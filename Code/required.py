# Import Required libraries

import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
import configparser
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
from datetime import datetime
from termcolor import colored
from pickle import dump
import pickle
from sklearn.preprocessing import Binarizer, OneHotEncoder, PowerTransformer, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBRegressor

from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error

set_config(display="diagram")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

warnings.filterwarnings("ignore")


# ===========================================================================================
def config_fetch(config_element = None):
    """
    to get config elements

    Parameters
    ----------
    config_element : config element needed. default = None

    Return
    ------
    a list if no config_element given or dictionary with config_elements
    """

    # Initialize configparser
    config = configparser.ConfigParser()
    # read config file
    config.read("config.ini")

    # input location for reading raw files
    input_dest = config.get("Input_Path", "input_dest")

    # input requirements file location
    input_req = config.get("Output_Path", "output_dest")

    # destination location for output files
    output_dest = config.get("Output_Path", "output_dest")

    # dictinary to store other section results
    if config_element:
        dict_out = {}
        if config.has_section(config_element):
            dict_out = {key:value for (key, value) in config.items(config_element)}
        else:
            print(f'Config file does not have "{config_element}", please select from available elements: {str(config.sections()[2:])}')
            return dict_out
    else:
        return input_dest, input_req, output_dest