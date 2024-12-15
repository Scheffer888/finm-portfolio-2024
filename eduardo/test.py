# Import Libraries:
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import re
#import datetime
#from typing import Union, List, Callable

#import warnings
#warnings.filterwarnings("ignore")

#pd.options.display.float_format = "{:,.4f}".format
#pd.set_option('display.width', 200)
#pd.set_option('display.max_columns', 30)

#import statsmodels.api as sm
#from scipy.stats import t

import os
import sys

#parent_path = os.path.dirname(os.getcwd()) # Get parent path (if using .ipynb file)
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get parent path (if using .py file)
os.chdir(parent_path) # Set parent path as working directory (for reading and writing files)
sys.path.insert(0, parent_path) # Add parent path to system path (for importing modules)
'''
import utils.portfolio_management_functions as pm

# Check data in the file (sheets, columns, data):
INFILE = "data/gmo_data.xlsx"
try:
    pm.read_excel_default(INFILE, print_sheets = True)
except FileNotFoundError as e:
    print(f'{e}.\nCheck file in {parent_path}')

# Import data from the file:
description = pd.read_excel(INFILE, sheet_name='info',index_col=0)
signals_returns = pd.read_excel(INFILE, sheet_name='signals',index_col=0)
risk_free_returns = pd.read_excel(INFILE, sheet_name='risk-free rate',index_col=0)
total_returns = pd.read_excel(INFILE, sheet_name='total returns',index_col=0)

excess_returns = pd.merge(total_returns, risk_free_returns, left_index=True, right_index=True)
excess_returns = excess_returns.sub(excess_returns['TBill 3M'], axis=0).drop(columns='TBill 3M')



summary_stats = pm.calc_returns_statistics(returns=total_returns,
                                           annual_factor=12,
                                           provided_excess_returns=False,
                                           rf_returns=risk_free_returns,
                                           timeframes={'1996-2011': ['1996', '2011'],
                                                       '2012-2024': ['2012', '2024'],
                                                       '1996-2024': ['1996', '2024']},
                                           keep_columns=['Annualized Mean', 'Annualized Vol', 'Annualized Sharpe']
                                           )
print(summary_stats)
'''

file_path = 'homework/Homework 7.pdf'
from docling.document_converter import DocumentConverter

source = file_path  # PDF path or URL
converter = DocumentConverter()
result = converter.convert(source)
with open("output.md", "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())
