import pandas as pd
import numpy as np
from arch import arch_model
import math
import datetime
pd.options.display.float_format = "{:,.4f}".format
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
from typing import Union, List
from pandas import Timestamp

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

from scipy.stats import norm

import re

def read_excel_default(excel_name: str, sheet_name: str = None, index_col : int = 0, parse_dates: bool =True, print_sheets: bool = False, **kwargs):
    """
    Reads an Excel file and returns a DataFrame with specified options.

    Parameters:
    excel_name (str): The path to the Excel file.
    index_col (int, default=0): Column to use as the row index labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_sheets (bool, default=False): If True, prints the names and first few rows of all sheets.
    sheet_name (str or int, default=None): Name or index of the sheet to read. If None, reads the first sheet.
    **kwargs: Additional arguments passed to `pd.read_excel`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the specified Excel sheet.

    Notes:
    - If `print_sheets` is True, the function will print the names and first few rows of all sheets and return None.
    - The function ensures that the index name is set to 'date' if the index column name is 'date' or 'dates', or if the index contains date-like values.
    """
    if print_sheets:
        excel_file = pd.ExcelFile(excel_name)  # Load the Excel file to get sheet names
        sheet_names = excel_file.sheet_names
        n = 0
        while True:
            try:
                sheet = pd.read_excel(excel_name, sheet_name=n)
                print(f'Sheet name: {sheet_names[n]}')
                print("Columns: " + ", ".join(list(sheet.columns)))
                print(sheet.head(3))
                n += 1
                print('-' * 70)
                print('\n')
            except:
                return
    sheet_name = 0 if sheet_name is None else sheet_name
    data = pd.read_excel(excel_name, index_col=index_col, parse_dates=parse_dates,  sheet_name=sheet_name, **kwargs)
    if data.index.name is not None:
        if data.index.name.lower() in ['date', 'dates']:
            data.index.name = 'date'
    elif isinstance(data.index[0], (datetime.date, datetime.datetime)):
        data.index.name = 'date'
    return data

def return_to_df(returns: Union[pd.DataFrame, pd.Series, List[pd.Series]]):
    """
    Converts returns to a DataFrame if it is a Series or a list of Series.

    Parameters:
    returns (pd.DataFrame, pd.Series, or list): Time series of returns.

    Returns:
    pd.DataFrame: DataFrame of returns.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    elif isinstance(returns, list):
        returns_list = returns.copy()
        returns = pd.DataFrame({})

        for series in returns_list:
            if isinstance(series, pd.Series):
                returns = returns.merge(series, right_index=True, left_index=True, how='outer')
            else:
                raise TypeError('Returns must be either a pd.DataFrame or a list of pd.Series')
            
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    return returns


def fix_dates_index(returns: pd.DataFrame):
    """
    Fixes the date index of a DataFrame if it is not in datetime format and convert returns to float.

    Parameters:
    returns (pd.DataFrame): DataFrame of returns.

    Returns:
    pd.DataFrame: DataFrame with datetime index.
    """
    # Check if 'date' is in the columns and set it as the index

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    # Convert dates to datetime
    try:
        returns.index = pd.to_datetime(returns.index.map(lambda x: x.date()))
    except ValueError:
        print('Could not convert the index to datetime. Check the index format for invalid dates.')
    
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    return returns

def calc_cummulative_returns(
    returns: Union[pd.DataFrame, pd.Series],
    return_plot: bool = True,
    fig_size: Union[int, float] = 7,
    return_series: bool = False,
    name: str = None,
    timeframes: Union[None, dict] = None,
):
    """
    Calculates cumulative returns from a time series of returns.

    Parameters:
    returns (pd.DataFrame or pd.Series): Time series of returns.
    return_plot (bool, default=True): If True, plots the cumulative returns.
    fig_size (int or float, default = 7): Size of the plot for cumulative returns. Scale: 1.5
    return_series (bool, default=False): If True, returns the cumulative returns as a DataFrame.
    name (str, default=None): Name for the title of the plot or the cumulative return series.
    timeframes (dict or None, default=None): Dictionary of timeframes to calculate cumulative returns for each period.

    Returns:
    pd.DataFrame or None: Returns cumulative returns DataFrame if `return_series` is True.
    """
    if timeframes is not None:
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_returns = returns.loc[:timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f'No returns data for {name} timeframe')
            calc_cummulative_returns(
                timeframe_returns,
                return_plot=True,
                fig_size=fig_size,
                return_series=False,
                name=name,
                timeframes=None
            )
        return
    returns = returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    returns = (1 + returns).cumprod()
    returns = returns - 1
    title = f'Cummulative Returns {name}' if name else 'Cummulative Returns'
    if return_plot:
        returns.plot(
            title=title,
            figsize=(fig_size*1.5, fig_size),
            grid=True,
            xlabel='Date',
            ylabel='Cummulative Returns'
        )
    if return_series == True or return_plot == False:
        return returns

def calc_returns_statistics(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    annual_factor: int = None,
    provided_excess_returns: bool = None,
    rf: Union[pd.Series, pd.DataFrame] = None,
    var_quantile: Union[float , List] = .05,
    timeframes: Union[None, dict] = None,
    return_tangency_weights: bool = True,
    correlations: Union[bool, List] = True,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    _timeframe_name: str = None,
):
    """
    Calculates summary statistics for a time series of returns.   

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    annual_factor (int, default=None): Factor for annualizing returns.
    provided_excess_returns (bool, default=None): Whether excess returns are already provided.
    rf (pd.Series or pd.DataFrame, default=None): Risk-free rate data.
    var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
    timeframes (dict or None, default=None): Dictionary of timeframes [start, finish] to calculate statistics for each period.
    return_tangency_weights (bool, default=True): If True, returns tangency portfolio weights.
    correlations (bool or list, default=True): If True, returns correlations, or specify columns for correlations.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary statistics of the returns.
    """

    # Check if returns is a DataFrame, Series or a list of Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.copy()
    elif(isinstance(returns, pd.Series)):
        returns = returns.to_frame()
    elif isinstance(returns, list):
        returns_list = returns.copy()
        returns = pd.DataFrame({})
        for series in returns_list:
            if isinstance(series, pd.Series):
                returns = returns.merge(series, right_index=True, left_index=True, how='outer')
            else:
                raise TypeError('Returns must be either a pd.DataFrame or a list of pd.Series')
    else:
        raise TypeError('Returns must be either a pd.DataFrame or a list of pd.Series')

    
    # Check if 'date' is in the columns and set it as the index
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    # Convert dates to datetime
    try:
        returns.index = pd.to_datetime(returns.index.map(lambda x: x.date()))
    except ValueError:
        print('Could not convert the index to datetime. Check the index format for invalid dates.')

    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    # Assume annualization factor of 12 for monthly returns if None and notify user
    if annual_factor is None:
        print('Assuming monthly returns with annualization term of 12')
        annual_factor = 12

    if keep_columns is None:
        keep_columns = ['Accumulated Return', 'Annualized Mean', 'Annualized Vol', 'Annualized Sharpe', 'Min', 'Mean', 'Max',
                        'Skewness', 'Excess Kurtosis', 'Historical VaR (5.0%)', 'Historical CVaR (5.0%)', 'Max Drawdown', 
                        'Peak Date', 'Bottom Date', 'Recovery', 'Duration (days)']

    # Iterate to calculate statistics for multiple timeframes
    if isinstance(timeframes, dict):
        all_timeframes_summary_statistics = pd.DataFrame({})
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_returns = returns.loc[:timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f'No returns data for {name} timeframe')
            
            timeframe_returns = timeframe_returns.rename(columns=lambda col: col + f' ({name})')
            timeframe_summary_statistics = calc_returns_statistics(
                returns=timeframe_returns,
                annual_factor=annual_factor,
                provided_excess_returns=provided_excess_returns,
                rf=rf,
                var_quantile=var_quantile,
                timeframes=None,
                correlations=correlations,
                _timeframe_name=name,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes
            )
            all_timeframes_summary_statistics = pd.concat(
                [all_timeframes_summary_statistics, timeframe_summary_statistics],
                axis=0
            )
        return all_timeframes_summary_statistics

    # Calculate summary statistics for a single timeframe
    summary_statistics = pd.DataFrame(index=returns.columns)
    summary_statistics['Mean'] = returns.mean()
    summary_statistics['Annualized Mean'] = returns.mean() * annual_factor
    summary_statistics['Vol'] = returns.std()
    summary_statistics['Annualized Vol'] = returns.std() * np.sqrt(annual_factor)

    if provided_excess_returns is True:
        if rf is not None:
            print('Excess returns and risk-free were both provided.'
                ' Excess returns will be consider as is, and risk-free rate given will be ignored.\n'
            )
        summary_statistics['Sharpe'] = returns.mean() / returns.std()
    else:
        try:
            if rf is None:
                print('No risk-free rate provided. Interpret "Sharpe" as "Mean/Volatility".\n')
                summary_statistics['Sharpe'] = returns.mean() / returns.std()
            elif isinstance(rf, (pd.Series, pd.DataFrame)):
                rf = rf.copy()
                if len(rf.index) != len(returns.index):
                    raise Exception('"rf" index must be the same lenght as "returns"')
                if type(rf) == pd.DataFrame:
                    rf = rf.iloc[:, 0].to_list()
                elif type(rf) == pd.Series:
                    rf = rf.to_list()
                
                excess_returns = returns.apply(lambda x: x - rf)
                summary_statistics['Sharpe'] = excess_returns.mean() / returns.std()
            else:
                raise TypeError('Returns must be either a pd.DataFrame or a list of pd.Series')
        except Exception as e:
            print(f'Could not calculate Sharpe: {e}')

    summary_statistics['Annualized Sharpe'] = summary_statistics['Sharpe'] * np.sqrt(annual_factor)
    summary_statistics['Min'] = returns.min()
    summary_statistics['Max'] = returns.max()
    
    tail_risk_stats = stats_tail_risk(returns,
                                      annual_factor=annual_factor,
                                      var_quantile=var_quantile,
                                      keep_indexes=keep_indexes,
                                      drop_indexes=drop_indexes)

    summary_statistics = summary_statistics.join(tail_risk_stats)
    
    print('Summary Statistics:')
    print(summary_statistics)

    if return_tangency_weights is True:
        tangency_weights = calc_tangency_port(returns)
        summary_statistics = summary_statistics.join(tangency_weights)

    if correlations is True or isinstance(correlations, list):
        returns_corr = returns.corr()
        if _timeframe_name:
            returns_corr = returns_corr.rename(columns=lambda col: col.replace(f' {_timeframe_name}', ''))
        returns_corr = returns_corr.rename(columns=lambda col: col + ' Correlation')
        if isinstance(correlations, list):
            correlation_names = [col + ' Correlation' for col  in correlations]
            # Check if all selected columns exist in returns_corr
            not_in_returns_corr = [col for col in correlation_names if col not in returns_corr.columns]
            if len(not_in_returns_corr) > 0:
                not_in_returns_corr = ", ".join([c.replace(' Correlation', '') for c in not_in_returns_corr])
                raise Exception(f'{not_in_returns_corr} not in returns columns')
            returns_corr = returns_corr[[col + ' Correlation' for col  in correlations]]
        summary_statistics = summary_statistics.join(returns_corr)
    
    return filter_columns_and_indexes(
        summary_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def stats_tail_risk(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    annual_factor: int = None,
    var_quantile: Union[float , List] = .05,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
):
    """
    Calculates tail risk summary statistics for a time series of returns.   

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    annual_factor (int, default=None): Factor for annualizing returns.
    var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: tail risk summary statistics of the returns.
    """

    # Check if returns is a DataFrame, Series or a list of Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.copy()
    elif(isinstance(returns, pd.Series)):
        returns = returns.to_frame()
    elif isinstance(returns, list):
        returns_list = returns.copy()
        returns = pd.DataFrame({})
        for series in returns_list:
            if isinstance(series, pd.Series):
                returns = returns.merge(series, right_index=True, left_index=True, how='outer')
            else:
                raise TypeError('Returns must be either a pd.DataFrame or a list of pd.Series')
    else:
        raise TypeError('Returns must be either a pd.DataFrame or a list of pd.Series')

    
    # Check if 'date' is in the columns and set it as the index
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    # Convert dates to datetime
    try:
        returns.index = pd.to_datetime(returns.index.map(lambda x: x.date()))
    except ValueError:
        print('Could not convert the index to datetime. Check the index format for invalid dates.')

    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    tail_risk_stats = pd.DataFrame(index=returns.columns)

    tail_risk_stats['Skewness'] = returns.skew()
    tail_risk_stats['Excess Kurtosis'] = returns.kurtosis()
    var_quantile = [var_quantile] if isinstance(var_quantile, (float, int)) else var_quantile
    for var_q in var_quantile:
        tail_risk_stats[f'Historical VaR ({var_q:.1%})'] = returns.quantile(var_q, axis = 0)
        tail_risk_stats[f'Historical CVaR ({var_q:.1%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean()
        if annual_factor:
            tail_risk_stats[f'Annualized Historical VaR ({var_q:.1%})'] = returns.quantile(var_q, axis = 0) * np.sqrt(annual_factor)
            tail_risk_stats[f'Annualized Historical CVaR ({var_q:.1%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean() * np.sqrt(annual_factor)
    
    cum_returns = (1 + returns).cumprod()
    maximum = cum_returns.cummax()
    drawdown = cum_returns / maximum - 1

    tail_risk_stats['Accumulated Return'] = cum_returns.iloc[-1] - 1
    tail_risk_stats['Max Drawdown'] = drawdown.min()
    tail_risk_stats['Peak Date'] = [maximum[col][:drawdown[col].idxmin()].idxmax() for col in maximum.columns]
    tail_risk_stats['Bottom Date'] = drawdown.idxmin()
    

    recovery_date = []
    for col in cum_returns.columns:
        prev_max = maximum[col][:drawdown[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([cum_returns[col][drawdown[col].idxmin():]]).T
        recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
    tail_risk_stats['Recovery'] = recovery_date

    tail_risk_stats["Duration (days)"] = [
        (i - j).days if i != pd.NaT else "-" for i, j in
        zip(tail_risk_stats["Recovery"], tail_risk_stats["Bottom Date"])
    ]

    return filter_columns_and_indexes(
        tail_risk_stats,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def calc_neg_pos_pct(
    returns: Union[pd.DataFrame, pd.Series, list],
    calc_positive: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Calculates the percentage of negative and positive returns in the provided data.

    Parameters:
    returns (pd.DataFrame, pd.Series, or list): Time series of returns.
    calc_positive (bool, default=False): If True, calculates the percentage of positive returns.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with the percentage of negative or positive returns, number of returns, and the count of negative/positive returns.
    """
    returns = returns.copy()
    if isinstance(returns, list):
        returns_list = returns[:]
        returns = pd.DataFrame({})
        for series in returns_list:
            returns = returns.merge(series, right_index=True, left_index=True, how='outer')

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
        
    returns.index.name = 'date'

    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    returns = returns.apply(lambda x: x.astype(float))
    prev_len_index = returns.apply(lambda x: len(x))
    returns  =returns.dropna(axis=0)
    new_len_index = returns.apply(lambda x: len(x))
    if not (prev_len_index == new_len_index).all():
        print('Some columns had NaN values and were dropped')
    if calc_positive:
        returns = returns.applymap(lambda x: 1 if x > 0 else 0)
    else:
        returns = returns.applymap(lambda x: 1 if x < 0 else 0)

    negative_statistics = (
        returns
        .agg(['mean', 'count', 'sum'])
        .set_axis(['% Negative Returns', 'Nº Returns', 'Nº Negative Returns'], axis=0)
    )

    if calc_positive:
        negative_statistics = negative_statistics.rename(lambda i: i.replace('Negative', 'Positive'), axis=0)

    return filter_columns_and_indexes(
        negative_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def filter_columns_and_indexes(
    df: pd.DataFrame,
    keep_columns: Union[list, str],
    drop_columns: Union[list, str],
    keep_indexes: Union[list, str],
    drop_indexes: Union[list, str]
):
    """
    Filters a DataFrame based on specified columns and indexes.

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    keep_columns (list or str): Columns to keep in the DataFrame.
    drop_columns (list or str): Columns to drop from the DataFrame.
    keep_indexes (list or str): Indexes to keep in the DataFrame.
    drop_indexes (list or str): Indexes to drop from the DataFrame.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        return df
    
    df = df.copy()

    # Columns
    if keep_columns is not None:
        keep_columns = "(?i)" + "|".join(keep_columns) if isinstance(keep_columns, list) else "(?i)" + keep_columns
        df = df.filter(regex=keep_columns)
        if drop_columns is not None:
            print('Both "keep_columns" and "drop_columns" were specified. "drop_columns" will be ignored.')

    elif drop_columns is not None:
        drop_columns = "(?i)" + "|".join(drop_columns) if isinstance(drop_columns, list) else "(?i)" + drop_columns
        df = df.drop(columns=df.filter(regex=drop_columns).columns)

    # Indexes
    if keep_indexes is not None:
        keep_indexes = "(?i)" + "|".join(keep_indexes) if isinstance(keep_indexes, list) else "(?i)" + keep_indexes
        df = df.filter(regex=keep_indexes, axis=0)
        if drop_indexes is not None:
            print('Both "keep_indexes" and "drop_indexes" were specified. "drop_indexes" will be ignored.')

    elif drop_indexes is not None:
        drop_indexes = "(?i)" + "|".join(drop_indexes) if isinstance(drop_indexes, list) else "(?i)" + drop_indexes
        df = df.filter(regex=keep_indexes, axis=0)
    
    return df


def calc_cross_section_regression(
    returns: Union[pd.DataFrame, List],
    factors: Union[pd.DataFrame, List],
    annual_factor: int = None,
    provided_excess_returns: bool = None,
    rf: pd.Series = None,
    return_model: bool = False,
    name: str = None,
    return_mae: bool = True,
    intercept_cross_section: bool = True,
    return_historical_premium: bool = True,
    return_annualized_premium: bool = True,
    compare_premiums: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Performs a cross-sectional regression on the provided returns and factors.

    Parameters:
    returns (pd.DataFrame or list): Time series of returns.
    factors (pd.DataFrame or list): Time series of factor data.
    annual_factor (int, default=None): Factor for annualizing returns.
    provided_excess_returns (bool, default=None): Whether excess returns are already provided.
    rf (pd.Series, default=None): Risk-free rate data for subtracting from returns.
    return_model (bool, default=False): If True, returns the regression model.
    name (str, default=None): Name for labeling the regression.
    return_mae (bool, default=True): If True, returns the mean absolute error of the regression.
    intercept_cross_section (bool, default=True): If True, includes an intercept in the cross-sectional regression.
    return_historical_premium (bool, default=True): If True, returns the historical premium of factors.
    return_annualized_premium (bool, default=True): If True, returns the annualized premium of factors.
    compare_premiums (bool, default=False): If True, compares the historical and estimated premiums.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame or model: Cross-sectional regression output or the model if `return_model` is True.
    """
    returns = returns.copy()
    factors = factors.copy()
    if isinstance(rf, (pd.Series, pd.DataFrame)):
        rf = rf.copy()

    if compare_premiums:
        return_historical_premium = True
        return_annualized_premium = True

    if isinstance(returns, list):
        returns_list = returns[:]
        returns = pd.DataFrame({})
        for series in returns_list:
            returns = returns.merge(series, right_index=True, left_index=True, how='outer')

    if annual_factor is None:
        print('Assuming monthly returns with annualization term of 12')
        annual_factor = 12

    if isinstance(factors, list):
        factors_list = returns[:]
        factors = pd.DataFrame({})
        for series in factors_list:
            factors = factors.merge(series, right_index=True, left_index=True, how='outer')

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    if provided_excess_returns is None:
        print('Assuming excess returns were provided')
        provided_excess_returns = True
    elif provided_excess_returns is False:
        if rf is not None:
            if len(rf.index) != len(returns.index):
                raise Exception('"rf" index must be the same lenght as "returns"')
            print('"rf" is used to subtract returns')
            returns = returns.sub(rf, axis=0)
    
    time_series_regressions = calc_iterative_regression(returns, factors, annual_factor=annual_factor, warnings=False)
    time_series_betas = time_series_regressions.filter(regex='Beta$', axis=1)
    time_series_historical_returns = time_series_regressions[['Fitted Mean']]
    cross_section_regression = calc_regression(
        time_series_historical_returns, time_series_betas,
        annual_factor=annual_factor, intercept=intercept_cross_section,
        return_model=return_model, warnings=False
    )

    if return_model:
        return cross_section_regression
    cross_section_regression = cross_section_regression.rename(columns=lambda c: c.replace(' Beta Beta', ' Lambda').replace('Alpha', 'Eta'))
    if name is None:
        name = " + ".join([c.replace(' Lambda', '') for c in cross_section_regression.filter(regex=' Lambda$', axis=1).columns])
    cross_section_regression.index = [f'{name} Cross-Section Regression']
    cross_section_regression.drop([
        'Information Ratio', 'Annualized Information Ratio', 'Tracking Error', 'Annualized Tracking Error', 'Fitted Mean', 'Annualized Fitted Mean'
    ], axis=1, inplace=True)
    if return_annualized_premium:
        factors_annualized_premium = (
            cross_section_regression
            .filter(regex=' Lambda$', axis=1)
            .apply(lambda x: x * annual_factor)
            .rename(columns=lambda c: c.replace(' Lambda', ' Annualized Lambda'))
        )
        cross_section_regression = cross_section_regression.join(factors_annualized_premium)

    if return_historical_premium:
        print('Lambda represents the premium calculated by the cross-section regression and the historical premium is the average of the factor excess returns')
        factors_historical_premium = factors.mean().to_frame(f'{name} Cross-Section Regression').transpose().rename(columns=lambda c: c + ' Historical Premium')
        cross_section_regression = cross_section_regression.join(factors_historical_premium)
        if return_annualized_premium:
            factors_annualized_historical_premium = (
                factors_historical_premium
                .apply(lambda x: x * annual_factor)
                .rename(columns=lambda c: c.replace(' Historical Premium', ' Annualized Historical Premium'))
            )
            cross_section_regression = cross_section_regression.join(factors_annualized_historical_premium)

    if compare_premiums:
        cross_section_regression = cross_section_regression.filter(regex='Lambda$|Historical Premium$', axis=1)
        cross_section_regression = cross_section_regression.transpose()
        cross_section_regression['Factor'] = cross_section_regression.index.str.extract(f'({"|".join(list(factors.columns))})').values
        cross_section_regression['Premium Type'] = cross_section_regression.index.str.replace(f'({"|".join(list(factors.columns))})', '')
        premiums_comparison = cross_section_regression.pivot(index='Factor', columns='Premium Type', values=f'{name} Cross-Section Regression')
        premiums_comparison.columns.name = None
        premiums_comparison.index.name = None
        premiums_comparison.join(calc_tangency_port(factors))
        premiums_comparison = premiums_comparison.join(factors.corr().rename(columns=lambda c: c + ' Correlation'))
        return filter_columns_and_indexes(
            premiums_comparison,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes
        )
    
    if return_mae:
        cross_section_regression['TS MAE'] = time_series_regressions['Alpha'].abs().mean()
        cross_section_regression['TS Annualized MAE'] = time_series_regressions['Annualized Alpha'].abs().mean()
        cross_section_regression_model = calc_regression(
            time_series_historical_returns, time_series_betas,
            annual_factor=annual_factor, intercept=intercept_cross_section,
            return_model=True, warnings=False
        )
        cross_section_regression['CS MAE'] = cross_section_regression_model.resid.abs().mean()
        cross_section_regression['CS Annualized MAE'] = cross_section_regression['CS MAE'] * annual_factor

    return filter_columns_and_indexes(
        cross_section_regression,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )

def get_best_worse_sharpe(
        summary_statistics: pd.DataFrame,
        stat: str = 'Annualized Sharpe'):
    """
    Get the best and worst Sharpe ratio from a DataFrame of Sharpe ratios.

    Parameters:
    summary_statistics (pd.DataFrame): DataFrame containing summary statistics.
    stat (str, default='Annualized Sharpe'): The statistic to compare assets by.

    Returns:
    pd.DtaFrame: Best and worst Sharpe ratios.
    """

    best_worse_sharpe = summary_statistics.copy().sort_values("Annualized Sharpe", ascending=False)
    best_worse_sharpe = best_worse_sharpe.iloc[[0, -1]]
    best_worse_sharpe = best_worse_sharpe.assign(Label=["Best Sharpe", "Worst Sharpe"])
    best_worse_sharpe = best_worse_sharpe[['Annualized Sharpe', 'Label']]
    best_worse_sharpe
    return best_worse_sharpe

# CHECK THIS FUNCTION
def get_best_and_worst_old(
    summary_statistics: pd.DataFrame,
    stat: str = 'Annualized Sharpe',
    return_df: bool = True
):
    """
    Identifies the best and worst assets based on a specified statistic.

    Parameters:
    summary_statistics (pd.DataFrame): DataFrame containing summary statistics.
    stat (str, default='Annualized Sharpe'): The statistic to compare assets by.
    return_df (bool, default=True): If True, returns a DataFrame with the best and worst assets.

    Returns:
    pd.DataFrame or None: DataFrame with the best and worst assets if `return_df` is True.
    """
    summary_statistics = summary_statistics.copy()

    if len(summary_statistics.index) < 2:
        raise Exception('"summary_statistics" must have at least two lines in order to do comparison')

    if stat not in summary_statistics.columns:
        raise Exception(f'{stat} not in "summary_statistics"')
    summary_statistics.rename(columns=lambda c: c.replace(' ', '').lower())
    best_stat = summary_statistics[stat].max()
    worst_stat = summary_statistics[stat].min()
    asset_best_stat = summary_statistics.loc[lambda df: df[stat] == df[stat].max()].index[0]
    asset_worst_stat = summary_statistics.loc[lambda df: df[stat] == df[stat].min()].index[0]
    print(f'The asset with the highest {stat} is {asset_best_stat}: {best_stat:.5f}')
    print(f'The asset with the lowest {stat} is {asset_worst_stat}: {worst_stat:.5f}')
    if return_df:
        return pd.concat([
            summary_statistics.loc[lambda df: df.index == asset_best_stat],
            summary_statistics.loc[lambda df: df.index == asset_worst_stat]
        ])
    

def calc_correlations(
    returns: pd.DataFrame,
    print_highest_lowest: bool = True,
    matrix_size: Union[int, float] = 7,
    return_heatmap: bool = True,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Calculates the correlation matrix of the provided returns and optionally prints or visualizes it.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    print_highest_lowest (bool, default=True): If True, prints the highest and lowest correlations.
    matrix_size (int or float, default=7): Size of the heatmap for correlation matrix visualization.
    return_heatmap (bool, default=True): If True, returns a heatmap of the correlation matrix.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    sns.heatmap or pd.DataFrame: Heatmap of the correlation matrix or the correlation matrix itself.
    """

    returns = returns.copy()

    returns = filter_columns_and_indexes(
        returns,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    correlation_matrix = returns.corr()

    if print_highest_lowest:
        highest_lowest_corr = (
            correlation_matrix
            .unstack()
            .sort_values()
            .reset_index()
            .set_axis(['asset_1', 'asset_2', 'corr'], axis=1)
            .loc[lambda df: df.asset_1 != df.asset_2]
        )
        highest_corr = highest_lowest_corr.iloc[lambda df: len(df)-1, :]
        lowest_corr = highest_lowest_corr.iloc[0, :]
        print(f'The highest correlation ({highest_corr["corr"]:.4f}) is between {highest_corr.asset_1} and {highest_corr.asset_2}')
        print(f'The lowest correlation ({lowest_corr["corr"]:.4f}) is between {lowest_corr.asset_1} and {lowest_corr.asset_2}')
    

    if return_heatmap:
        fig, ax = plt.subplots(figsize=(matrix_size * 1.5, matrix_size))
        heatmap = sns.heatmap(
            correlation_matrix, 
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,
            annot=True,
        )
        return heatmap

    
    return correlation_matrix

    

def calc_tangency_port(
    returns: pd.DataFrame,
    cov_matrix_factor: str = 1,
    return_graphic: bool = False,
    return_port_returns: bool = False,
    target_return: Union[None, float] = None,
    annual_factor: int = 12,
    name: str = 'Tangency'
):
    """
    Calculates tangency portfolio weights based on the covariance matrix of returns.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    cov_matrix_factor (str, default=1): Weight for the covariance matrix. If 1, uses the sample covariance matrix, otherwise uses a shrinkage estimator.
    return_graphic (bool, default=False): If True, plots the tangency weights.
    return_port_returns (bool, default=False): If True, returns the portfolio returns. Otherwise, returns portfolio weights.
    target_return (float or None, default=None): Target return for rescaling weights (annualized).
    annual_factor (int, default=12): Factor for annualizing returns.
    name (str, default='Tangency'): Name for labeling the weights and portfolio.

    Returns:
    pd.DataFrame or pd.Series: Tangency portfolio weights or portfolio returns if `return_port_ret` is True.
    """
    returns = returns.copy()
    
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    # Calculate the covariance matrix
    if cov_matrix_factor == 1:
        cov_matrix = returns.cov()
    else:
        cov_matrix = returns.cov()
        cov_matrix_diag = np.diag(np.diag(cov_matrix))
        cov_matrix = cov_matrix_factor * cov_matrix + (1-cov_matrix_factor) * cov_matrix_diag
    
    cov_matrix_inv = np.linalg.pinv(cov_matrix)
    ones = np.ones(len(returns.columns))
    mu = returns.mean() # Use mean monthly excess returns as a proxy for expected excess returns: (mu)

    # Calculate the tangency portfolio weights
    scaling = 1 / (ones.T @ cov_matrix_inv @ mu)
    tangency_wts = scaling * (cov_matrix_inv @ mu)
    tangency_wts = pd.DataFrame(index=returns.columns, data=tangency_wts, columns=[f'{name} Weights'])
    
    # Calculate the portfolio returns
    port_returns = returns @ tangency_wts.rename({f'{name} Weights': f'{name} Portfolio'}, axis=1)

    # Rescale weights to target return
    if isinstance(target_return, (float, int)):
        scaler = target_return / (port_returns[f'{name} Portfolio'].mean() * annual_factor)
        tangency_wts[[f'{name} Weights']] *= scaler
        port_returns *= scaler
        '''
        tangency_wts = tangency_wts.rename(
            {f'{name} Weights': f'{name} Weights Rescaled Target {target_return:.2%}'}, axis=1)
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_return:.2%}'}, axis=1)
        '''
    
    # Plot the tangency weights
    if return_graphic:
        ax = tangency_wts.plot(kind='bar', title=f'{name} Weights')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    if cov_matrix_factor != 1:
        port_returns = port_returns.rename(columns=lambda c: c.replace('Tangency', f'Tangency Regularized ({cov_matrix_factor:.2f})'))
        tangency_wts = tangency_wts.rename(columns=lambda c: c.replace('Tangency', f'Tangency Regularized ({cov_matrix_factor:.2f})'))
        
    if return_port_returns:
        return port_returns
    return tangency_wts


def calc_equal_weights_port(
    returns: pd.DataFrame,
    return_graphic: bool = False,
    return_port_returns: bool = False,
    target_return: Union[float, None] = None,
    annual_factor: int = 12,
    name: str = 'Equal Weights'
):
    """
    Calculates equal weights for the portfolio based on the provided returns.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    return_graphic (bool, default=False): If True, plots the equal weights.
    return_port_returns (bool, default=False): If True, returns the portfolio returns. Otherwise, returns portfolio weights.
    target_return (float or None, default=None): Target return for rescaling weights (annualized).
    name (str, default='Equal Weights'): Name for labeling the portfolio.

    Returns:
    pd.DataFrame or pd.Series: Equal portfolio weights or portfolio returns if `return_port_returns` is True.
    """
    returns = returns.copy()

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'


    equal_wts = pd.DataFrame(
        index=returns.columns,
        data=[1 / len(returns.columns)] * len(returns.columns),
        columns=[f'{name}']
    )
    port_returns = returns @ equal_wts.rename({f'{name}': f'{name} Portfolio'}, axis=1)

    if isinstance(target_return, (float, int)):
        scaler = target_return / (port_returns[f'{name} Portfolio'].mean() * annual_factor)
        equal_wts[[f'{name}']] *= scaler
        port_returns *= scaler
        '''
        equal_wts = equal_wts.rename(
            {f'{name}': f'{name} Rescaled Target {target_return:.2%}'},axis=1)
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_return:.2%}'},axis=1)
        '''

    # Plot the equal weights
    if return_graphic:
        ax = equal_wts.plot(kind='bar', title=f'{name}')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        
    if return_port_returns:
        return port_returns
    return equal_wts


def calc_risk_parity_port(
    returns: pd.DataFrame,
    optimized: bool = False,
    return_graphic: bool = False,
    return_port_returns: bool = False,
    target_return: Union[None, float] = None,
    annual_factor: int = 12,
    name: str = 'Risk Parity'
):
    """
    Calculates risk parity portfolio weights based on the variance of each asset.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    optimized (bool, default=False): If True, uses an optimization algorithm to calculate the risk parity weights.
    return_graphic (bool, default=False): If True, plots the risk parity weights.
    return_port_returns (bool, default=False): If True, returns the portfolio returns. Otherwise, returns portfolio weights.
    target_return (float or None, default=None): Target return for rescaling weights (annualized).
    name (str, default='Risk Parity'): Name for labeling the portfolio.

    Returns:
    pd.DataFrame or pd.Series: Risk parity portfolio weights or portfolio returns if `return_port_ret` is True.
    """

    # Objective function for risk parity optimization
    #  - Calculate individual asset risk contributions
    #  - The objective is to minimize the squared differences in risk contributions
    def objective_function_RP(weights, cov_matrix):    
        marginal_contributions = cov_matrix @ weights
        risk_contributions = weights * marginal_contributions
        target_risk = np.mean(risk_contributions)
        return np.sum((risk_contributions - target_risk) ** 2)


    returns = returns.copy()

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    # Calculaye weights for risk parity
    weights = [1 / returns[asset].var() for asset in returns.columns] # Inverse of the variance (simple approach)
    if optimized: # Optimized approach
        cov_matrix = returns.cov()
        weights = minimize(objective_function_RP,
                        x0=weights,  # Initial guess (equal weights)
                        args=(cov_matrix,),  # Additional arguments passed to the objective function
                        bounds=None,  # No bounds, allowing for leverage
                        constraints=None,  # No constraints, allowing for leverage
                        tol=1e-13  # Precision tolerance
                        ).x
        
    risk_parity_wts = pd.DataFrame(
        index=returns.columns,
        data=weights,
        columns=[f'{name} Weights']
    )

    port_returns = returns @ risk_parity_wts.rename({f'{name} Weights': f'{name} Portfolio'}, axis=1)

    if isinstance(target_return, (float, int)):
        scaler = target_return / (port_returns[f'{name} Portfolio'].mean() * annual_factor)
        risk_parity_wts[[f'{name} Weights']] *= scaler
        port_returns *= scaler
        '''
        risk_parity_wts = risk_parity_wts.rename(
            {f'{name} Weights': f'{name} Weights Rescaled Target {target_return:.2%}'},
            axis=1
        )
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_return:.2%}'},
            axis=1
        )
        '''
    
    if optimized:
        port_returns = port_returns.rename({f'{name} Portfolio': f'{name} Portfolio (Optimized)'}, axis = 1)
        risk_parity_wts = risk_parity_wts.rename({f'{name} Weights': f'{name} Weights (Optimized)'}, axis = 1)

    # Plot the risk parity weights
    if return_graphic:
        ax = risk_parity_wts.plot(kind='bar', title=f'{name} Weights')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        
    if return_port_returns:
        return port_returns
    return risk_parity_wts


def calc_gmv_port(
    returns: pd.DataFrame,
    cov_matrix_factor: str = 1,
    return_graphic: bool = False,
    return_port_returns: bool = False,
    target_return: Union[float, None] = None,
    annual_factor: int = 12,
    name: str = 'GMV'
):
    """
    Calculates Global Minimum Variance (GMV) portfolio weights.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    cov_matrix_factor (str, default=1): Weight for the covariance matrix. If 1, uses the sample covariance matrix, otherwise uses a shrinkage estimator.
    return_graphic (bool, default=False): If True, plots the GMV weights.
    return_port_returns (bool, default=False): If True, returns the portfolio returns. Otherwise, returns portfolio weights.
    target_return (float or None, default=None): Target return for rescaling weights (annualized).
    name (str, default='GMV'): Name for labeling the portfolio.

    Returns:
    pd.DataFrame or pd.Series: GMV portfolio weights or portfolio returns if `return_port_ret` is True.
    """
    returns = returns.copy()

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    # Calculate the covariance matrix
    if cov_matrix_factor == 1:
        cov_matrix = returns.cov()
    else:
        cov_matrix = returns.cov()
        cov_matrix_diag = np.diag(np.diag(cov_matrix))
        cov_matrix = cov_matrix_factor * cov_matrix + (1-cov_matrix_factor) * cov_matrix_diag
    
    cov_matrix_inv = np.linalg.pinv(cov_matrix)
    ones = np.ones(len(returns.columns))

    # Calculate the GMV portfolio weights
    scaling = 1 / (ones.T @ cov_matrix_inv @ ones)
    gmv_wts = scaling * cov_matrix_inv @ ones
    gmv_wts = pd.DataFrame(index=returns.columns, data=gmv_wts, columns=[f'{name} Weights'])
    
    # Calculate the portfolio returns
    port_returns = returns @ gmv_wts.rename({f'{name} Weights': f'{name} Portfolio'}, axis=1)

    # Rescale weights to target return
    if isinstance(target_return, (float, int)):
        scaler = target_return / (port_returns[f'{name} Portfolio'].mean() * annual_factor)
        gmv_wts[[f'{name} Weights']] *= scaler
        port_returns *= scaler
        '''
        gmv_wts = gmv_wts.rename(
            {f'{name} Weights': f'{name} Weights Rescaled Target {target_return:.2%}'}, axis=1)
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_return:.2%}'}, axis=1)
        '''

    # Plot the Global Minimum Variance weights
    if return_graphic:
        ax = gmv_wts.plot(kind='bar', title=f'{name} Weights')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    if cov_matrix_factor != 1:
        port_returns = port_returns.rename(columns=lambda c: c.replace('Tangency', f'Tangency Regularized ({cov_matrix_factor:.2f})'))
        tangency_wts = tangency_wts.rename(columns=lambda c: c.replace('Tangency', f'Tangency Regularized ({cov_matrix_factor:.2f})'))
     
    if return_port_returns:
        return port_returns

    return gmv_wts


def calc_target_ret_weights(
    target_ret: float,
    returns: pd.DataFrame,
    return_graphic: bool = False,
    return_port_ret: bool = False
):
    # Have not checked this function yet

    return
    """
    Calculates the portfolio weights to achieve a target return by combining Tangency and GMV portfolios.

    Parameters:
    target_ret (float): Target return for the portfolio.
    returns (pd.DataFrame): Time series of asset returns.
    return_graphic (bool, default=False): If True, plots the portfolio weights.
    return_port_returns (bool, default=False): If True, returns the portfolio returns. Otherwise, returns portfolio weights.

    Returns:
    pd.DataFrame: Weights of the Tangency and GMV portfolios, along with the combined target return portfolio.
    """
    returns = returns.copy()
    
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'
    
    mu_tan = returns.mean() @ calc_tangency_port(returns, cov_mat = 1)
    mu_gmv = returns.mean() @ calc_gmv_weights(returns)
    
    delta = (target_ret - mu_gmv[0]) / (mu_tan[0] - mu_gmv[0])
    mv_weights = (delta * calc_tangency_port(returns, cov_mat=1)).values + ((1 - delta) * calc_gmv_weights(returns)).values
    
    mv_weights = pd.DataFrame(
        index=returns.columns,
        data=mv_weights,
        columns=[f'Target {target_ret:.2%} Weights']
    )
    port_returns = returns @ mv_weights.rename({f'Target {target_ret:.2%} Weights': f'Target {target_ret:.2%} Portfolio'}, axis=1)

    if return_graphic:
        mv_weights.plot(kind='bar', title=f'Target Return of {target_ret:.2%} Weights')

    if return_port_ret:
        return port_returns

    mv_weights['Tangency Weights'] = calc_tangency_port(returns, cov_mat=1).values
    mv_weights['GMV Weights'] = calc_gmv_weights(returns).values

    return mv_weights


def get_regression_statistics(
    y: Union[pd.DataFrame, pd.Series],
    x: Union[pd.DataFrame, pd.Series],
    intercept: bool = True,
    annual_factor: Union[None, int] = None,
    return_model: bool = False,
    return_fitted_values: bool = False,
    p_values: bool = True,
    tracking_error: bool = True,
    treynor_info_ratio: bool = True,
    sortino_ratio: bool = False,
    timeframes: Union[None, dict] = None,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
    ):
    
    """
    Performs an OLS regression on the provided return data with optional intercept, timeframes, statistical ratios, and performance ratios.

    Parameters:
    y (pd.DataFrame or pd.Series): Dependent variable(s) for the regression.
    X (pd.DataFrame or pd.Series): Independent variable(s) for the regression.
    intercept (bool, default=True): If True, includes an intercept in the regression.
    annual_factor (int or None, default=None): Factor for annualizing regression statistics.
    return_model (bool, default=False): If True, returns the regression model object.
    return_fitted_values (bool, default=False): If True, returns the fitted values of the regression.
    p_values (bool, default=True): If True, displays p-values for the regression coefficients.
    tracking_error (bool, default=True): If True, calculates the tracking error of the regression.
    treynor_info_ratios (bool, default=True): If True, calculates Treynor and Information ratios.
    sortino_ratio (bool, default=False): If True, calculates the Sortino ratio.
    timeframes (dict or None, default=None): Dictionary of timeframes to run separate regressions for each period.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    calc_sortino_ratio (bool, default=False): If True, calculates the Sortino ratio.

    Returns:
    pd.DataFrame or model: Regression summary statistics or the model if `return_model` is True.
    """

    x = x.copy()
    y = y.copy()

    if isinstance(x, pd.Series):
        x = x.to_frame()
    elif not isinstance(x, pd.DataFrame):
        raise TypeError('x must be either a pd.DataFrame or a pd.Series')
    
    
    if isinstance(y, pd.Series):
        y = y.to_frame()
    elif not isinstance(y, pd.DataFrame):
        raise TypeError('y must be either a pd.DataFrame or a pd.Series')

    if 'date' in x.columns.str.lower():
        x = x.rename({'Date': 'date'}, axis=1)
        x = x.set_index('date')
    x.index.name = 'date'

    if 'date' in y.columns.str.lower():
        y = y.rename({'Date': 'date'}, axis=1)
        y = y.set_index('date')
    y.index.name = 'date'


    if annual_factor is None:
        print("Regression assumes 'annual_factor' equals to 12 since it was not provided")
        annual_factor = 12

    # Add the intercept
    if intercept:
        X = sm.add_constant(x)
    else:
        X = x
    
    y_names = list(y.columns) if isinstance(y, pd.DataFrame) else [y.name]
    X_names = " + ".join(list(x.columns))
    X_names = "Intercept + " + X_names if intercept else X_names


    # Check if y and X have the same length
    if len(X.index) != len(y.index):
        print(f'y has lenght {len(y.index)} and X has lenght {len(X.index)}. Joining y and X by y.index...')
        df = y.join(X, how='left')
        df = df.dropna()
        y = df[y_names]
        X = df.drop(columns=y_names)
        if len(X.index) < len(X.columns) + 1:
            raise Exception('Indexes of y and X do not match and there are less observations than degrees of freedom. Cannot calculate regression')

    if isinstance(timeframes, dict):
        all_timeframes_regressions = pd.DataFrame()
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_y = y.loc[timeframe[0]:timeframe[1]]
                timeframe_X = X.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_y = y.loc[timeframe[0]:]
                timeframe_X = X.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_y = y.loc[:timeframe[1]]
                timeframe_X = X.loc[:timeframe[1]]
            else:
                timeframe_y = y.copy()
                timeframe_X = X.copy()
            if len(timeframe_y.index) == 0 or len(timeframe_X.index) == 0:
                raise Exception(f'No returns data for {name} timeframe')
            
            timeframe_y = timeframe_y.rename(columns=lambda col: col + f' ({name})')
            timeframe_regression = calc_regression(
                y=timeframe_y,
                X=timeframe_X,
                intercept=intercept,
                annual_factor=annual_factor,
                warnings=False,
                return_model=False,
                return_fitted_values=False,
                p_values=p_values,
                tracking_error=tracking_error,
                treynor_info_ratio=treynor_info_ratio,
                timeframes=None,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes
            )
            timeframe_regression.index = [f"{timeframe_regression.index} ({name})"]
            all_timeframes_regressions = pd.concat(
                [all_timeframes_regressions, timeframe_regression],
                axis=0
            )
        return all_timeframes_regressions
    
    regression_statistics = pd.DataFrame(index=y_names, columns=[])	
    fitted_values_all = pd.DataFrame(index=y.index, columns=y_names)
    for y_asset in y_names:
        # Fit the regression model: 
        Y = y[y_asset] if isinstance(y, pd.DataFrame) else y
        try:
            ols_model = sm.OLS(Y, X, missing="drop")
        except ValueError:
            Y = Y.reset_index(drop=True)
            X = X.reset_index(drop=True)
            ols_model = sm.OLS(Y, X, missing="drop", hasconst=intercept)
            print(f'"{y_asset}" Required to reset indexes to make regression work. Try passing "y" and "X" as pd.DataFrame')
        
        ols_results = ols_model.fit()

        if return_model:
            print(ols_results.summary())

        elif return_fitted_values:
            fitted_values = ols_results.fittedvalues
            fitted_values = fitted_values.rename(f'{y_asset}^')
            fitted_values_all[y_asset] = fitted_values

        else:
            # Calculate/get statistics:
            if intercept == True:
                regression_statistics.loc[y_asset, 'Alpha'] = ols_results.params.iloc[0]
                regression_statistics.loc[y_asset, 'Annualized Alpha'] = ols_results.params.iloc[0] * annual_factor # Annualized Alpha

                if p_values == True: 
                    regression_statistics.loc[y_asset, 'P-value (Alpha)'] = ols_results.pvalues.iloc[0] # Alpha p-value

            regression_statistics.loc[y_asset, 'R-squared'] = ols_results.rsquared # R-squared

            if isinstance(X, pd.Series):
                X = pd.DataFrame(X)
            
            X_names = list(X.columns[1:]) if intercept else list(X.columns)
            betas = ols_results.params[1:] if intercept else ols_results.params
            betas_p_values = ols_results.pvalues[1:] if intercept else ols_results.pvalues
            
            for i in range(len(X_names)):
                regression_statistics.loc[y_asset, f"Beta ({X_names[i]})"] = betas[i] # Betas
                if p_values == True: 
                    regression_statistics.loc[y_asset, f"P-Value ({X_names[i]})"] = betas_p_values[i] # Beta p-values
        
            if tracking_error == True:
                regression_statistics.loc[y_asset, 'Tracking Error'] = ols_results.resid.std() * (annual_factor ** 0.5) # Annualized Residuals Volatility
                regression_statistics.loc[y_asset, 'Annualized Tracking Error'] = regression_statistics.loc[y_asset, 'Tracking Error'] * (annual_factor ** 0.5) # Annualized Residuals Volatility

            if treynor_info_ratio == True:
                try:
                    #Y_projected = ols_model.predict(X)
                    regression_statistics[y_asset, 'Treynor Ratio'] = Y.mean() / regression_statistics.loc[y_asset, 'Beta (SPY US Equity)'] # Treynor Ratio
                    regression_statistics[y_asset, 'Annualized Treynor Ratio'] = regression_statistics[y_asset, 'Treynor Ratio'] * annual_factor # Annualized Treynor Ratio
                except:
                    print('SPY is not a factor in the regression. Treynor Ratio cannot be calculated.')

                regression_statistics.loc[y_asset, 'Information Ratio'] = regression_statistics.loc[y_asset, 'Alpha'] / ols_results.resid.std() # Information Ratio
                regression_statistics.loc[y_asset, 'Annualized Information Ratio'] = regression_statistics.loc[y_asset, 'Information Ratio'] * (annual_factor ** 0.5) # Annualized Information Ratio
            
            regression_statistics.loc[y_asset, 'Fitted Mean'] = ols_results.fittedvalues.mean()
            regression_statistics.loc[y_asset, 'Annualized Fitted Std Dev'] = regression_statistics.loc[y_asset, 'Fitted Mean'] * annual_factor
            if sortino_ratio:
                try:
                    regression_statistics.loc[y_asset, 'Sortino Ratio'] = regression_statistics.loc[y_asset, 'Fitted Mean'] / Y[Y < 0].std()
                except Exception as e:
                    print(f'Cannot calculate Sortino Ratio: {str(e)}. Set "calc_sortino_ratio" to False or review function')
    
    if return_model or return_fitted_values:
        return
    else:
        return filter_columns_and_indexes(
            regression_statistics,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes
        )


def calc_strategy_oos(
    y: Union[pd.Series, pd.DataFrame],
    X: Union[pd.Series, pd.DataFrame],
    intercept: bool = True,
    rolling_size: Union[None, int] = 60,
    expanding: bool = False,
    lag_number: int = 1,
    weight_multiplier: float = 100,
    weight_min: Union[None, float] = None,
    weight_max: Union[None, float] = None,
    name: str = None,
):
    """
    Calculates an out-of-sample strategy based on rolling or expanding window regression.

    Parameters:
    y (pd.Series or pd.DataFrame): Dependent variable (strategy returns).
    X (pd.Series or pd.DataFrame): Independent variable(s) (predictors).
    intercept (bool, default=True): If True, includes an intercept in the regression.
    rolling_size (int or None, default=60): Size of the rolling window for in-sample fitting.
    expanding (bool, default=False): If True, uses an expanding window instead of rolling.
    lag_number (int, default=1): Number of lags to apply to the predictors.
    weight_multiplier (float, default=100): Multiplier to adjust strategy weights.
    weight_min (float or None, default=None): Minimum allowable weight.
    weight_max (float or None, default=None): Maximum allowable weight.
    name (str, default=None): Name for labeling the strategy returns.

    Returns:
    pd.DataFrame: Time series of strategy returns.
    """
    raise Exception("Function not available - needs testing prior to use")
    try:
        y = y.copy()
        X = X.copy()
    except:
        pass
    replication_oos = calc_replication_oos(
        y=y,
        X=X,
        intercept=intercept,
        rolling_size=rolling_size,
        lag_number=lag_number,
        expanding=expanding
    )
    actual_returns = replication_oos['Actual']
    predicted_returns = replication_oos['Prediction']
    strategy_weights = predicted_returns * weight_multiplier
    weight_min = weight_min if weight_min is not None else strategy_weights.min()
    weight_max = weight_max if weight_max is not None else strategy_weights.max()
    strategy_weights = strategy_weights.clip(lower=weight_min, upper=weight_max)
    strategy_returns = (actual_returns * strategy_weights).to_frame()
    if name:
        strategy_returns.columns = [name]
    else:
        strategy_returns.columns = [f'{y.columns[0]} Strategy']
    return strategy_returns
    

def calc_iterative_regression(
    multiple_y: Union[pd.DataFrame, pd.Series],
    X: Union[pd.DataFrame, pd.Series],
    annual_factor: Union[None, int] = 12,
    intercept: bool = True,
    warnings: bool = True,
    calc_treynor_info_ratios: bool = True,
    calc_sortino_ratio: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Performs iterative regression across multiple dependent variables (assets).

    Parameters:
    multiple_y (pd.DataFrame or pd.Series): Dependent variables for multiple assets.
    X (pd.DataFrame or pd.Series): Independent variable(s) (predictors).
    annual_factor (int or None, default=12): Factor for annualizing regression statistics.
    intercept (bool, default=True): If True, includes an intercept in the regression.
    warnings (bool, default=True): If True, prints warnings about assumptions.
    calc_treynor_info_ratios (bool, default=True): If True, calculates Treynor and Information ratios.
    calc_sortino_ratio (bool, default=False): If True, calculates the Sortino ratio.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary statistics for each asset regression.
    """
    multiple_y = multiple_y.copy()
    X = X.copy()

    if 'date' in multiple_y.columns.str.lower():
        multiple_y = multiple_y.rename({'Date': 'date'}, axis=1)
        multiple_y = multiple_y.set_index('date')
    multiple_y.index.name = 'date'

    if 'date' in X.columns.str.lower():
        X = X.rename({'Date': 'date'}, axis=1)
        X = X.set_index('date')
    X.index.name = 'date'

    regressions = pd.DataFrame({})
    for asset in multiple_y.columns:
        y = multiple_y[[asset]]
        new_regression = calc_regression(
            y, X, annual_factor=annual_factor, intercept=intercept, warnings=warnings,
            calc_treynor_info_ratios=calc_treynor_info_ratios,
            calc_sortino_ratio=calc_sortino_ratio,
        )
        warnings = False
        regressions = pd.concat([regressions, new_regression], axis=0)
    
    return filter_columns_and_indexes(
        regressions,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def calc_replication_oos(
    y: Union[pd.Series, pd.DataFrame],
    X: Union[pd.Series, pd.DataFrame],
    intercept: bool = True,
    rolling_size: Union[None, int] = 60,
    expanding: bool = False,
    return_r_squared_oos: float = False,
    lag_number: int = 1,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Performs out-of-sample replication of a time series regression with rolling or expanding windows.

    Parameters:
    y (pd.Series or pd.DataFrame): Dependent variable (actual returns).
    X (pd.Series or pd.DataFrame): Independent variable(s) (predictors).
    intercept (bool, default=True): If True, includes an intercept in the regression.
    rolling_size (int or None, default=60): Size of the rolling window for in-sample fitting.
    expanding (bool, default=False): If True, uses an expanding window instead of rolling.
    return_r_squared_oos (float, default=False): If True, returns the out-of-sample R-squared.
    lag_number (int, default=1): Number of lags to apply to the predictors.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary statistics for the out-of-sample replication.
    """
    raise Exception("Function not available - needs testing prior to use")
    try:
        y = y.copy()
        X = X.copy()
    except:
        pass
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)
    if 'date' in X.columns.str.lower():
        X = X.rename({'Date': 'date'}, axis=1)
        X = X.set_index('date')
    X.index.name = 'date'

    X = X.shift(lag_number)

    df = y.join(X, how='inner')
    y = df.iloc[:, [0]].copy()
    X = df.iloc[:, 1:].copy()

    if intercept:
        X = sm.add_constant(X)

    summary_pred = pd.DataFrame({})

    for i, last_is_date in enumerate(y.index):
        if i < (rolling_size):
            continue
        y_full = y.iloc[:i].copy()
        if expanding:
            y_rolling = y_full.copy()
        else:
            y_rolling = y_full.iloc[-rolling_size:]
        X_full = X.iloc[:i].copy()
        if expanding:
            X_rolling = X_full.copy()
        else:
            X_rolling = X_full.iloc[-rolling_size:]

        reg = sm.OLS(y_rolling, X_rolling, hasconst=intercept, missing='drop').fit()
        y_pred = reg.predict(X.iloc[i, :])
        naive_y_pred = y_full.mean()
        y_actual = y.iloc[i]
        summary_line = (
            reg.params
            .to_frame()
            .transpose()
            .rename(columns=lambda c: c.replace('const', 'Alpha') if c == 'const' else c + ' Lag Beta')
        )
        summary_line['Prediction'] = y_pred[0]
        summary_line['Naive Prediction'] = naive_y_pred.squeeze()
        summary_line['Actual'] = y_actual.squeeze()
        summary_line.index = [y.index[i]]
        summary_pred = pd.concat([summary_pred, summary_line], axis=0)

    summary_pred['Prediction Error'] = summary_pred['Prediction'] - summary_pred['Actual']
    summary_pred['Naive Prediction Error'] = summary_pred['Naive Prediction'] - summary_pred['Actual']

    rss = (np.array(summary_pred['Prediction Error']) ** 2).sum()
    tss = (np.array(summary_pred['Naive Prediction Error']) ** 2).sum()

    oos_rsquared = 1 - rss / tss

    if return_r_squared_oos:
        return pd.DataFrame(
            {'R^Squared OOS': oos_rsquared},
            index=[
                y.columns[0] + " ~ " + 
                " + ".join([
                    c.replace('const', 'Alpha') if c == 'const' else c + ' Lag Beta' for c in X.columns
                ])]
        )

    print("OOS R^Squared: {:.4%}".format(oos_rsquared))

    return filter_columns_and_indexes(
        summary_pred,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def calc_replication_oos_not_lagged_features(
    y: Union[pd.Series, pd.DataFrame],
    X: Union[pd.Series, pd.DataFrame],
    intercept: bool = True,
    rolling_size: Union[None, int] = 60,
    return_r_squared_oos: float = False,
    r_squared_time_series: bool = False,
    return_parameters: bool = True,
    oos: int = 1,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Performs out-of-sample replication without lagged features.

    Parameters:
    y (pd.Series or pd.DataFrame): Dependent variable (actual returns).
    X (pd.Series or pd.DataFrame): Independent variable(s) (predictors).
    intercept (bool, default=True): If True, includes an intercept in the regression.
    rolling_size (int or None, default=60): Size of the rolling window for in-sample fitting.
    return_r_squared_oos (float, default=False): If True, returns the out-of-sample R-squared.
    r_squared_time_series (bool, default=False): If True, calculates time-series R-squared.
    return_parameters (bool, default=True): If True, returns regression parameters.
    oos (int, default=1): Number of periods for out-of-sample evaluation.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary statistics for the out-of-sample replication.
    """
    raise Exception("Function not available - needs testing prior to use")
    try:
        y = y.copy()
        X = X.copy()
    except:
        pass
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)
    if 'date' in X.columns.str.lower():
        X = X.rename({'Date': 'date'}, axis=1)
        X = X.set_index('date')
    X.index.name = 'date'

    oos_print = "In-Sample" if oos == 0 else f"{oos}OS"

    summary = defaultdict(list)

    if isinstance(y, pd.Series):
        y = pd.DataFrame(y)
    y_name = y.columns[0]
    
    for idx in range(rolling_size, len(y.index)+1-oos, 1):
        X_rolling = X.iloc[idx-rolling_size:idx].copy()
        y_rolling = y.iloc[idx-rolling_size:idx, 0].copy()

        y_oos = y.iloc[idx-1+oos, 0].copy()
        X_oos = X.iloc[idx-1+oos, :].copy()

        if intercept:
            X_rolling = sm.add_constant(X_rolling)

        try:
            regr = sm.OLS(y_rolling, X_rolling, missing="drop", hasconst=intercept).fit()
        except ValueError:
            y_rolling = y_rolling.reset_index(drop=True)
            X_rolling = X_rolling.reset_index(drop=True)
            regr = sm.OLS(y_rolling, X_rolling, missing="drop", hasconst=intercept).fit()

        for jdx, coeff in enumerate(regr.params.index):
            if coeff != 'const':
                summary[f"{coeff} Beta {oos_print}"].append(regr.params[jdx])
            else:
                summary[f"{coeff} {oos_print}"].append(regr.params[jdx])

        if intercept:
            y_pred = regr.params[0] + (regr.params[1:] @ X_oos)
        else:
            y_pred = regr.params @ X_oos

        summary[f"{y_name} Replicated"].append(y_pred)
        summary[f"{y_name} Actual"].append(y_oos)

    summary = pd.DataFrame(summary, index=X.index[rolling_size-1+oos:])

    if r_squared_time_series:
        time_series_error = pd.DataFrame({})
        for idx in range(rolling_size, len(y.index)+1-oos, 1):
            y_rolling = y.iloc[idx-rolling_size:idx, 0].copy()
            y_oos = y.iloc[idx-1+oos, 0].copy()
            time_series_error.loc[y.index[idx-1+oos], 'Naive Error'] = y_oos - y_rolling.mean()
        time_series_error['Model Error'] = summary[f"{y_name} Actual"] - summary[f"{y_name} Replicated"]
        oos_rsquared = (
            1 - time_series_error['Model Error'].apply(lambda x: x ** 2).sum()
            / time_series_error['Naive Error'].apply(lambda x: x ** 2).sum()
        )
    else:
        oos_rsquared = (
            1 - (summary[f"{y_name} Actual"] - summary[f"{y_name} Replicated"]).var()
            / summary[f"{y_name} Actual"].var()
        )

    if return_r_squared_oos:
        return oos_rsquared
    
    if not return_parameters:
        summary = summary[[f"{y_name} Actual", f"{y_name} Replicated"]]

    if not intercept:
        summary = summary.rename(columns=lambda c: c.replace(' Replicated', f' Replicated no Intercept'))

    if not intercept:
        print(f"R^Squared {oos_print} without Intercept: {oos_rsquared:.2%}")

    else:
        print(f"R^Squared {oos_print}: {oos_rsquared:.2%}")

    summary = summary.rename(columns=lambda c: (
        c.replace(' Replicated', f' Replicated {oos_print}').replace(' Actual', f' Actual {oos_print}')
    ))

    return filter_columns_and_indexes(
        summary,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def calc_portfolio_returns(
    returns: pd.DataFrame,
    weights: Union[dict, list, pd.Series, pd.DataFrame],
    port_name: Union[None, str] = None
):
    """
    Creates a portfolio by applying the specified weights to the asset returns.

    Parameters:
    returns (pd.DataFrame): Time series of asset returns.
    weights (list or pd.Series): Weights to apply to the returns. If a list or pd.Series is provided, it will be converted into a dict.
    port_name (str or None, default=None): Name for the portfolio. If None, a name will be generated based on asset weights.

    Returns:
    pd.DataFrame: The portfolio returns based on the provided weights.
    """

    returns = returns.copy()

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    # Check returns size and weight size:
    if returns.shape[1] != len(weights):
        raise Exception(f"Returns have {returns.shape[1]} assets, but {len(weights)} weights were provided")
    
    if isinstance(weights, list):
        weights = dict(zip(returns.columns, weights))
    elif isinstance(weights, pd.Series):
        weights = weights.to_dict()
    elif isinstance(weights, pd.DataFrame):
        list(weights.to_dict().values())[0]
    elif isinstance(weights, dict):
        pass
    else:
        raise Exception("Weights must be a dict, list, pd.Series, or pd.DataFrame")


    returns = returns[list(weights.keys())]
    port_returns = pd.DataFrame(returns @ list(weights.values()))

    if port_name is None:
        print("Portfolio:"+" + ".join([f"{n} ({w:.2%})" for n, w in weights.items()]))
        port_name = 'Portfolio'
    port_returns.columns = [port_name]
    return port_returns


def calc_ewma_volatility(
        returns: pd.Series,
        theta : float = 0.94,
        initial_vol : float = .2 / np.sqrt(252)
    ) -> pd.Series:
    var_t0 = initial_vol ** 2
    ewma_var = [var_t0]
    for i in range(len(returns.index)):
        new_ewma_var = ewma_var[-1] * theta + (returns.iloc[i] ** 2) * (1 - theta)
        ewma_var.append(new_ewma_var)
    ewma_var.pop(0) # Remove var_t0
    ewma_vol = [np.sqrt(v) for v in ewma_var]
    return pd.Series(ewma_vol, index=returns.index)


def calc_garch_volatility(
        returns: pd.Series,
        p: int = 1,
        q: int = 1
    ):
    model = arch_model(returns, vol='Garch', p=p, q=q)
    fitted_model = model.fit(disp='off')
    fitted_values = fitted_model.conditional_volatility
    return pd.Series(fitted_values, index=returns.index)


def calc_var_cvar_summary(
    returns: Union[pd.Series, pd.DataFrame],
    percentile: Union[None, float] = .05,
    window: Union[None, str] = None,
    return_hit_ratio: bool = False,
    filter_first_hit_ratio_date: Union[None, str, datetime.date] = None,
    z_score: float = None,
    shift: int = 1,
    std_formula: bool = False,
    ewma_theta : float = .94,
    ewma_initial_vol : float = .2 / np.sqrt(252),
    garch_p: int = 1,
    garch_q: int = 1,
    return_stats: Union[str, list] = ['Returns', 'VaR', 'CVaR', 'Vol'],
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Calculates a summary of VaR (Value at Risk) and CVaR (Conditional VaR) for the provided returns.

    Parameters:
    returns (pd.Series or pd.DataFrame): Time series of returns.
    percentile (float or None, default=0.05): Percentile to calculate the VaR and CVaR.
    window (str or None, default=None): Window size for rolling calculations.
    return_hit_ratio (bool, default=False): If True, returns the hit ratio for the VaR.
    filter_first_hit_ratio_date (str, datetime.date or None, default=None): Date to filter the hit ratio calculation from then on.
    z_score (float, default=None): Z-score for parametric VaR calculation, in case no percentile is provided.
    shift (int, default=1): Period shift for VaR/CVaR calculations.
    std_formula (bool, default=False): If True, uses the normal volatility formula with .std(). Else, use squared returns.
    return_stats (str or list, default=['Returns', 'VaR', 'CVaR', 'Vol']): Statistics to return in the summary.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary of VaR and CVaR statistics.
    """
    if window is None:
        print('Using "window" of 60 periods, since none was specified')
        window = 60
    if isinstance(returns, pd.DataFrame):
        returns_series = returns.iloc[:, 0]
        returns_series.index = returns.index
        returns = returns_series.copy()
    elif isinstance(returns, pd.Series):
        returns = returns.copy()
    else:
        raise TypeError('returns must be either a pd.DataFrame or a pd.Series')

    summary = pd.DataFrame({})

    # Returns
    summary[f'Returns'] = returns

    # VaR
    summary[f'Expanding {window} Historical VaR ({percentile:.2%})'] = returns.expanding(min_periods=window).quantile(percentile)
    summary[f'Rolling {window} Historical VaR ({percentile:.2%})'] = returns.rolling(window=window).quantile(percentile)
    if std_formula:
        summary[f'Expanding {window} Volatility'] = returns.expanding(window).std()
        summary[f'Rolling {window} Volatility'] = returns.rolling(window).std()
    else: # Volaility assuming zero mean returns
        summary[f'Expanding {window} Volatility'] = np.sqrt((returns ** 2).expanding(window).mean())
        summary[f'Rolling {window} Volatility'] = np.sqrt((returns ** 2).rolling(window).mean())
    summary[f'EWMA {ewma_theta:.2f} Volatility'] = calc_ewma_volatility(returns, theta=ewma_theta, initial_vol=ewma_initial_vol)
    summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility'] = calc_garch_volatility(returns, p=garch_p, q=garch_q)
    
    # Parametric VaR assuming zero mean returns
    z_score = norm.ppf(percentile) if z_score is None else z_score
    summary[f'Expanding {window} Parametric VaR ({percentile:.2%})'] = summary[f'Expanding {window} Volatility'] * z_score
    summary[f'Rolling {window} Parametric VaR ({percentile:.2%})'] = summary[f'Rolling {window} Volatility'] * z_score
    summary[f'EWMA {ewma_theta:.2f} Parametric VaR ({percentile:.2%})'] = summary[f'EWMA {ewma_theta:.2f} Volatility'] * z_score
    summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({percentile:.2%})'] = summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility'] * z_score

    if return_hit_ratio:
        var_stats = [
            f'Expanding {window} Historical VaR ({percentile:.2%})',
            f'Rolling {window} Historical VaR ({percentile:.2%})',
            f'Expanding {window} Parametric VaR ({percentile:.2%})',
            f'Rolling {window} Parametric VaR ({percentile:.2%})',
            f'EWMA {ewma_theta:.2f} Parametric VaR ({percentile:.2%})',
            f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({percentile:.2%})'
        ]
        
        summary_hit_ratio = summary.copy()
        summary_hit_ratio[var_stats] = summary_hit_ratio[var_stats].shift()
        if filter_first_hit_ratio_date:
            if isinstance(filter_first_hit_ratio_date, (datetime.date, datetime.datetime)):
                filter_first_hit_ratio_date = filter_first_hit_ratio_date.strftime("%Y-%m-%d")
            summary_hit_ratio = summary.loc[filter_first_hit_ratio_date:]
        summary_hit_ratio = summary_hit_ratio.dropna(axis=0)
        summary_hit_ratio[var_stats] = summary_hit_ratio[var_stats].apply(lambda x: (x - summary['Returns']) > 0)
        
        hit_ratio = pd.DataFrame(summary_hit_ratio[var_stats].mean(), columns=['Hit Ratio'])
        hit_ratio['Hit Ratio Error'] = (hit_ratio['Hit Ratio'] - percentile) / percentile
        hit_ratio['Hit Ratio Absolute Error'] = abs(hit_ratio['Hit Ratio Error'])
        hit_ratio = hit_ratio.sort_values('Hit Ratio Absolute Error')
        return filter_columns_and_indexes(
            hit_ratio,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes
        )

    # CVaR
    summary[f'Expanding {window} Historical CVaR ({percentile:.2%})'] = returns.expanding(window).apply(lambda x: x[x < x.quantile(percentile)].mean())
    summary[f'Rolling {window} Historical CVaR ({percentile:.2%})'] = returns.rolling(window).apply(lambda x: x[x < x.quantile(percentile)].mean())
    summary[f'Expanding {window} Parametrical CVaR ({percentile:.2%})'] = - norm.pdf(z_score) / percentile * summary[f'Expanding {window} Volatility']
    summary[f'Rolling {window} Parametrical CVaR ({percentile:.2%})'] = - norm.pdf(z_score) / percentile * summary[f'Rolling {window} Volatility']
    summary[f'EWMA {ewma_theta:.2f} Parametrical CVaR ({percentile:.2%})'] = - norm.pdf(z_score) / percentile * summary[f'EWMA {ewma_theta:.2f} Volatility']
    summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Parametrical CVaR ({percentile:.2%})'] = - norm.pdf(z_score) / percentile * summary[f'GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility']

    if shift > 0:
        shift_columns = [c for c in summary.columns if not bool(re.search("returns", c))]
        summary[shift_columns] = summary[shift_columns].shift(shift).dropna()
        print(f'VaR and CVaR are given shifted by {shift:0f} period(s).')
    else:
        print('VaR and CVaR are given in-sample.')

    return_stats = [return_stats.lower()] if isinstance(return_stats, str) else [s.lower() for s in return_stats]
    return_stats = list(map(lambda x: 'volatility' if x == 'vol' else x, return_stats))
    
    if return_stats == ['all'] or set(return_stats) == set(['returns', 'var', 'cvar', 'volatility']):
        summary = summary.loc[:, lambda df: df.columns.map(lambda c: bool(re.search(r"\b" + r"\b|\b".join(return_stats) + r"\b", c.lower())))]
        
    return filter_columns_and_indexes(
        summary,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )

def plot_var(
        returns: Union[pd.DataFrame, pd.Series],
        var: Union[pd.DataFrame, pd.Series],
        var_name: str = 'VaR',
        percentile: Union[None, float] = .05,
        figsize: tuple = (15, 7),
        limit = True,
        colors: Union[list, str] = ["blue", "red", "orange"]
        ):
    """
    Plots a variance graph with returns and highlights returns < VaR 

    Parameters:

    """
    if not isinstance(returns, (pd.DataFrame, pd.Series)):
       raise TypeError('returns must be either a pd.DataFrame or a pd.Series')
    if not isinstance(var, (pd.DataFrame, pd.Series)):
        raise TypeError('var must be either a pd.DataFrame or a pd.Series')
    
    returns = pd.merge(returns, var, left_index=True, right_index=True).dropna()
    asset_name = returns.columns[0]
    returns.columns = [asset_name, var_name]

    excess_returns_surpass_var = (
        returns
        .dropna()
        .loc[lambda df: df[asset_name] < df[var_name]]
    )
    plt.figure(figsize=figsize)
    plt.axhline(y=0, linestyle='--', color='black', alpha=0.5)
    plt.plot(
        returns.index,
        returns[var_name],
        color=colors[0],
        label=var_name
    )
    plt.plot(
        returns.index,
        returns[asset_name],
        color=colors[2],
        label=f"{asset_name} Returns",
        alpha=.2
    )
    plt.plot(
        excess_returns_surpass_var.index,
        excess_returns_surpass_var[asset_name],
        linestyle="",
        marker="o",
        color=colors[1],
        label=f"Return < {var_name}",
        markersize=1.5
    )
    if limit:
        plt.ylim(min(returns[asset_name]), .01)

    hit_ratio = len(excess_returns_surpass_var.index) / len(returns.index)
    hit_ratio_error = abs((hit_ratio / percentile) - 1)
    plt.title(f"{var_name} of SPY Excess Returns")
    plt.xlabel(f"Hit Ratio: {hit_ratio:.2%}; Hit Ratio Error: {hit_ratio_error:.2%}")
    plt.ylabel("Excess Returns")
    plt.legend()
    plt.show()

    return


def calc_rolling_oos_port(
    returns: pd.DataFrame,
    weights_func,
    window: Union[None, int] = None,
    weights_func_params: dict = {},
    port_name: str = 'Portfolio OOS',
    expanding: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Calculates a rolling out-of-sample portfolio based on a rolling or expanding window optimization.

    Parameters:
    returns (pd.DataFrame): Time series of asset returns.
    weights_func (function): Function to calculate the portfolio weights.
    window (int or None, default=None): Rolling window size for in-sample optimization.
    weights_func_params (dict, default={}): Additional parameters for the weights function.
    port_name (str, default='Portfolio OOS'): Name for the portfolio.
    expanding (bool, default=False): If True, uses an expanding window instead of a rolling one.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Out-of-sample portfolio returns.
    """
    raise Exception("Function not available - needs testing prior to use")
    if window is None:
        print('Using "window" of 60 periods for in-sample optimization, since none were provided.')
        window = 60
    returns = returns.copy()
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    port_returns_oos = pd.DataFrame({})

    for idx in range(0, len(returns.index)-window):
        modified_idx = 0 if expanding else idx
        weights_func_all_params = {'returns': returns.iloc[modified_idx:(window+idx), :]}
        weights_func_all_params.update(weights_func_params)
        wts = weights_func(**weights_func_all_params).iloc[:, 0]
        idx_port_return_oos = sum(returns.iloc[window, :].loc[wts.index] * wts)
        idx_port_return_oos = pd.DataFrame(
            {port_name: idx_port_return_oos},
            index=[returns.index[idx+window]]
        )
        port_returns_oos = pd.concat([port_returns_oos, idx_port_return_oos])

    return filter_columns_and_indexes(
        port_returns_oos,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def calc_fx_exc_ret(
    fx_rates: pd.DataFrame,
    rf_rates: pd.DataFrame,
    transform_to_log_fx_rates: bool = True,
    transform_to_log_rf_rates: bool = True,
    rf_to_fx: dict = None,
    base_rf: str = None,
    base_rf_series: Union[pd.Series, pd.DataFrame] = None,
    annual_factor: Union[int, None] = None,
    return_exc_ret: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Calculates foreign exchange excess returns by subtracting risk-free rates from FX rates.

    Parameters:
    fx_rates (pd.DataFrame): Time series of FX rates.
    rf_rates (pd.DataFrame): Time series of risk-free rates.
    transform_to_log_fx_rates (bool, default=True): If True, converts FX rates to log returns.
    transform_to_log_rf_rates (bool, default=True): If True, converts risk-free rates to log returns.
    rf_to_fx (dict, default=None): Mapping of risk-free rates to FX pairs.
    base_rf (str, default=None): Base risk-free rate to use for calculations.
    base_rf_series (pd.Series or pd.DataFrame, default=None): Time series of the base risk-free rate.
    annual_factor (int or None, default=None): Factor for annualizing the returns.
    return_exc_ret (bool, default=False): If True, returns the excess returns instead of summary statistics.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary statistics or excess returns based on FX rates and risk-free rates.
    """
    raise Exception("Function not available - needs testing prior to use")
    fx_rates = fx_rates.copy()
    rf_rates = rf_rates.copy()
    if isinstance(base_rf_series, (pd.Series, pd.DataFrame)):
        base_rf_series = base_rf_series.copy()

    if rf_to_fx is None:
        rf_to_fx = {
            'GBP1M': 'USUK',
            'EUR1M': 'USEU',
            'CHF1M': 'USSZ',
            'JPY1M': 'USJP'
        }

    if transform_to_log_fx_rates:
        fx_rates = fx_rates.applymap(lambda x: math.log(x))

    if transform_to_log_rf_rates:
        rf_rates = rf_rates.applymap(lambda x: math.log(x + 1))

    if base_rf is None and base_rf_series is None:
        print("No 'base_rf' or 'base_rf_series' was provided. Trying to use 'USD1M' as the base risk-free rate.")
        base_rf = 'USD1M'
    if base_rf_series is None:
        base_rf_series = rf_rates[base_rf]

    all_fx_holdings_exc_ret = pd.DataFrame({})
    for rf, fx in rf_to_fx.items():
        fx_holdings_exc_ret = fx_rates[fx] - fx_rates[fx].shift(1) + rf_rates[rf].shift(1) - base_rf_series.shift(1)
        try:
            rf_name = re.sub('[0-9]+M', '', rf)
        except:
            rf_name = rf
        fx_holdings_exc_ret = fx_holdings_exc_ret.dropna(axis=0).to_frame(rf_name)
        all_fx_holdings_exc_ret = all_fx_holdings_exc_ret.join(fx_holdings_exc_ret, how='outer')

    if not return_exc_ret:
        return filter_columns_and_indexes(
            calc_summary_statistics(all_fx_holdings_exc_ret, annual_factor=annual_factor),
            keep_columns=keep_columns, drop_columns=drop_columns,
            keep_indexes=keep_indexes, drop_indexes=drop_indexes
        )
    else:
        return filter_columns_and_indexes(
            all_fx_holdings_exc_ret,
            keep_columns=keep_columns, drop_columns=drop_columns,
            keep_indexes=keep_indexes, drop_indexes=drop_indexes
        )
    

def calc_fx_regression(
    fx_rates: pd.DataFrame,
    rf_rates: pd.DataFrame,
    transform_to_log_fx_rates: bool = True,
    transform_to_log_rf_rates: bool = True,
    rf_to_fx: dict = None,
    base_rf: str = None,
    base_rf_series: Union[pd.Series, pd.DataFrame] = None,
    annual_factor: Union[int, None] = None,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    print_analysis: bool = True
):
    """
    Calculates FX regression and provides an analysis of how the risk-free rate differentials affect FX rates.

    Parameters:
    fx_rates (pd.DataFrame): Time series of FX rates.
    rf_rates (pd.DataFrame): Time series of risk-free rates.
    transform_to_log_fx_rates (bool, default=True): If True, converts FX rates to log returns.
    transform_to_log_rf_rates (bool, default=True): If True, converts risk-free rates to log returns.
    rf_to_fx (dict, default=None): Mapping of risk-free rates to FX pairs.
    base_rf (str, default=None): Base risk-free rate to use for calculations.
    base_rf_series (pd.Series or pd.DataFrame, default=None): Time series of the base risk-free rate.
    annual_factor (int or None, default=None): Factor for annualizing returns.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    print_analysis (bool, default=True): If True, prints an analysis of the regression results.

    Returns:
    pd.DataFrame: Summary of regression statistics for the FX rates and risk-free rate differentials.
    """
    raise Exception("Function not available - needs testing prior to use")
    fx_rates = fx_rates.copy()
    rf_rates = rf_rates.copy()
    if isinstance(base_rf_series, (pd.Series, pd.DataFrame)):
        base_rf_series = base_rf_series.copy()

    if rf_to_fx is None:
        rf_to_fx = {
            'GBP1M': 'USUK',
            'EUR1M': 'USEU',
            'CHF1M': 'USSZ',
            'JPY1M': 'USJP'
        }

    if transform_to_log_fx_rates:
        fx_rates = fx_rates.applymap(lambda x: math.log(x))

    if transform_to_log_rf_rates:
        rf_rates = rf_rates.applymap(lambda x: math.log(x + 1))

    if base_rf is None and base_rf_series is None:
        print("No 'base_rf' or 'base_rf_series' was provided. Trying to use 'USD1M' as the base risk-free rate.")
        base_rf = 'USD1M'
    if base_rf_series is None:
        base_rf_series = rf_rates[base_rf]

    if annual_factor is None:
        print("Regression assumes 'annual_factor' equals to 12 since it was not provided")
        annual_factor = 12

    all_regressions_summary = pd.DataFrame({})

    for rf, fx in rf_to_fx.items():
        try:
            rf_name = re.sub('[0-9]+M', '', rf)
        except:
            rf_name = rf
        factor = (base_rf_series - rf_rates[rf]).to_frame('Base RF - Foreign RF')
        strat = fx_rates[fx].diff().to_frame(rf_name)
        regression_summary = calc_regression(strat, factor, annual_factor=annual_factor, warnings=False)
        all_regressions_summary = pd.concat([all_regressions_summary, regression_summary])

    if print_analysis:
        try:
            print('\n' * 2)
            for currency in all_regressions_summary.index:
                fx_beta = all_regressions_summary.loc[currency, 'Base RF - Foreign RF Beta']
                fx_alpha = all_regressions_summary.loc[currency, 'Alpha']
                print(f'For {currency} against the base currency, the Beta is {fx_beta:.2f}.')
                if 1.1 >= fx_beta and fx_beta >= 0.85:
                    print(
                        'which shows that, on average, the difference in risk-free rate is mainly offset by the FX rate movement.'
                    )
                elif fx_beta > 1.1:
                    print(
                        'which shows that, on average, the difference in risk-free rate is more than offset by the FX rate movement.,\n'
                        'Therefore, on average, the currency with the lower risk-free rate outperforms.'
                    )
                elif fx_beta < 0.85 and fx_beta > 0.15:
                    print(
                        'which shows that, on average, the difference in risk-free rate is only partially offset by the FX rate movement.\n'
                        'Therefore, on average, the currency with the higher risk-free rate outperforms.'
                    )
                elif fx_beta <= 0.15 and fx_beta >= -0.1:
                    print(
                        'which shows that, on average, the difference in risk-free rate is almost not offset by the FX rate movement.\n'
                        'Therefore, on average, the currency with the higher risk-free rate outperforms.'
                    )
                elif fx_beta <= 0.15 and fx_beta >= -0.1:
                    print(
                        'which shows that, on average, the difference in risk-free rate is almost not offset by the FX rate movement.\n'
                        'Therefore, on average, the currency with the higher risk-free rate outperforms.'
                    )
                else:
                    print(
                        'which shows that, on average, the change FX rate helps the currency with the highest risk-free return.\n'
                        'Therefore, the difference between returns is increased, on average, by the changes in the FX rate.'
                    )
                print('\n' * 2)
        except:
            print('Could not print analysis. Review function.')

    return filter_columns_and_indexes(
        all_regressions_summary,
        keep_columns=keep_columns, drop_columns=drop_columns,
        keep_indexes=keep_indexes, drop_indexes=drop_indexes
    )


def calc_dynamic_carry_trade(
    fx_rates: pd.DataFrame,
    rf_rates: pd.DataFrame,
    transform_to_log_fx_rates: bool = True,
    transform_to_log_rf_rates: bool = True,
    rf_to_fx: dict = None,
    base_rf: str = None,
    base_rf_series: Union[pd.Series, pd.DataFrame] = None,
    annual_factor: Union[int, None] = None,
    return_premium_series: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
):
    """
    Calculates the dynamic carry trade strategy based on FX rates and risk-free rate differentials.

    Parameters:
    fx_rates (pd.DataFrame): Time series of FX rates.
    rf_rates (pd.DataFrame): Time series of risk-free rates.
    transform_to_log_fx_rates (bool, default=True): If True, converts FX rates to log returns.
    transform_to_log_rf_rates (bool, default=True): If True, converts risk-free rates to log returns.
    rf_to_fx (dict, default=None): Mapping of risk-free rates to FX pairs.
    base_rf (str, default=None): Base risk-free rate to use for calculations.
    base_rf_series (pd.Series or pd.DataFrame, default=None): Time series of the base risk-free rate.
    annual_factor (int or None, default=None): Factor for annualizing the returns.
    return_premium_series (bool, default=False): If True, returns the premium series instead of summary statistics.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary of the carry trade strategy statistics or premium series.
    """
    raise Exception("Function not available - needs testing prior to use")
    if annual_factor is None:
        print("Regression assumes 'annual_factor' equals to 12 since it was not provided")
        annual_factor = 12
        
    fx_regressions = calc_fx_regression(
        fx_rates, rf_rates, transform_to_log_fx_rates, transform_to_log_rf_rates,
        rf_to_fx, base_rf, base_rf_series, annual_factor
    )

    fx_rates = fx_rates.copy()
    rf_rates = rf_rates.copy()
    if isinstance(base_rf_series, (pd.Series, pd.DataFrame)):
        base_rf_series = base_rf_series.copy()

    if rf_to_fx is None:
        rf_to_fx = {
            'GBP1M': 'USUK',
            'EUR1M': 'USEU',
            'CHF1M': 'USSZ',
            'JPY1M': 'USJP'
        }

    if transform_to_log_fx_rates:
        fx_rates = fx_rates.applymap(lambda x: math.log(x))

    if transform_to_log_rf_rates:
        rf_rates = rf_rates.applymap(lambda x: math.log(x + 1))

    if base_rf is None and base_rf_series is None:
        print("No 'base_rf' or 'base_rf_series' was provided. Trying to use 'USD1M' as the base risk-free rate.")
        base_rf = 'USD1M'
    if base_rf_series is None:
        base_rf_series = rf_rates[base_rf]

    all_expected_fx_premium = pd.DataFrame({})
    for rf in rf_to_fx.keys():
        try:
            rf_name = re.sub('[0-9]+M', '', rf)
        except:
            rf_name = rf
        fx_er_usd = (base_rf_series.shift(1) - rf_rates[rf].shift(1)).to_frame('ER Over USD')
        expected_fx_premium = fx_regressions.loc[rf_name, 'Alpha'] + (fx_regressions.loc[rf_name, 'Base RF - Foreign RF Beta'] - 1) * fx_er_usd
        expected_fx_premium = expected_fx_premium.rename(columns={'ER Over USD': rf_name})
        all_expected_fx_premium = all_expected_fx_premium.join(expected_fx_premium, how='outer')

    if return_premium_series:
        return filter_columns_and_indexes(
            all_expected_fx_premium,
            keep_columns=keep_columns, drop_columns=drop_columns,
            keep_indexes=keep_indexes, drop_indexes=drop_indexes
        )
    
    all_expected_fx_premium = all_expected_fx_premium.dropna(axis=0)
    summary_statistics = (
        all_expected_fx_premium
        .applymap(lambda x: 1 if x > 0 else 0)
        .agg(['mean', 'sum', 'count'])
        .set_axis(['% of Periods with Positive Premium', 'Nº of Positive Premium Periods', 'Total Number of Periods'])
    )
    summary_statistics = pd.concat([
        summary_statistics,
        (
            all_expected_fx_premium
            .agg(['mean', 'std', 'min', 'max', 'skew', 'kurtosis'])
            .set_axis(['Mean', 'Vol', 'Min', 'Max', 'Skewness', 'Kurtosis'])
        )
    ])
    summary_statistics = summary_statistics.transpose()
    summary_statistics['Annualized Mean'] = summary_statistics['Mean'] * annual_factor
    summary_statistics['Annualized Vol'] = summary_statistics['Vol'] * math.sqrt(annual_factor)
    
    return filter_columns_and_indexes(
        summary_statistics,
        keep_columns=keep_columns, drop_columns=drop_columns,
        keep_indexes=keep_indexes, drop_indexes=drop_indexes
    )
