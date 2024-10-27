# Import Libraries:
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_path) # Set parent path as working directory (for reading and writing files)
sys.path.insert(0, parent_path) # Add parent path to system path (for importing modules)

import utils.portfolio_management_functions as pm

# Check data in the file:

    
# Import data from the file:
prices = pm.read_excel_default("data/multi_asset_etf_data.xlsx", sheet_name="prices", index_col=0)
excess_returns = pm.read_excel_default("data/multi_asset_etf_data.xlsx", sheet_name="excess returns", index_col=0)
total_returns = pm.read_excel_default("data/multi_asset_etf_data.xlsx", sheet_name="total returns", index_col=0)

risk_free_rate = total_returns.loc[:,['SHV']]
total_returns = total_returns.drop('SHV', axis = 1)

pm.calc_returns_statistics(
    returns = excess_returns,
    annual_factor=12,
    rf = risk_free_rate,
    provided_excess_returns=False,
    keep_columns=['Annualized Vol', 'Annualized Mean', 'Annualized Sharpe'],
    var_quantile=0.05
)

