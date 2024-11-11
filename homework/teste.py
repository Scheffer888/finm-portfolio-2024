# Import Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import datetime
from typing import Union, List, Callable

import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = "{:,.4f}".format
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

import statsmodels.api as sm
from scipy.stats import t

import os
import sys

#parent_path = os.path.dirname(os.getcwd()) # Get parent path (if using .ipynb file)
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get parent path (if using .py file)
os.chdir(parent_path) # Set parent path as working directory (for reading and writing files)
sys.path.insert(0, parent_path) # Add parent path to system path (for importing modules)

import utils.portfolio_management_functions as pm

# Check data in the file (sheets, columns, data):
INFILE = "data/momentum_data.xlsx"
try:
    pm.read_excel_default(INFILE, print_sheets = True)
except FileNotFoundError as e:
    print(f'{e}.\nCheck file in {parent_path}')

# Import data from the file:
description = pd.read_excel(INFILE, sheet_name='descriptions',index_col=0)
factors_returns = pd.read_excel(INFILE, sheet_name='factors (excess returns)',index_col=0)
momentum_returns = pd.read_excel(INFILE, sheet_name='momentum (excess returns)',index_col=0)

  # Calculate the factor returns statistics:
factors_returns_all = pd.concat([factors_returns, momentum_returns], axis=1)
momentum_performance = pm.calc_returns_statistics(factors_returns_all,
                                    provided_excess_returns=True,
                                    annual_factor=12,
                                    var_quantile = 0.05,
                                    correlations=['HML', 'MKT'],
                                    return_tangency_weights=True,
                                    timeframes={
                                        "1927-2024": ["1927", "2024"],
                                        "1927-1992": ["1927", "1992"],
                                        "1993-2008": ["1993", "2008"],
                                        "2009-2024": ["2009", "2024"]

                                    },
                                    keep_columns=['Annualized Mean', 'Annualized Vol', 'Annualized Sharpe', 'Skewness'],
                                    keep_indexes=['UMD'])

momentum_performance.index = [idx.replace('UMD (', '') for idx in momentum_performance.index]
momentum_performance.index = [idx.replace(')', '') for idx in momentum_performance.index]
print(momentum_performance)