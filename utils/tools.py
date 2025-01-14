# Title: Tools for Financial Data Analysis

import datetime
import yfinance as yf
import numpy as np
import holidays

import pandas as pd
pd.options.display.float_format = "{:,.4f}".format
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

from typing import Union, List, Dict

import re


from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


def bday(date):
    us_bus = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return bool(len(pd.bdate_range(date, date,freq=us_bus)))

def prev_bday(date: Union[str, datetime.date], force_prev: bool = False) -> Union[str, datetime.date]:

    if isinstance(date,str):
        date = datetime.datetime.strptime(date,'%Y-%m-%d')
        date2str = True
    else:
        date2str = False
        
    if force_prev:
        date += -datetime.timedelta(days=1)
    while not bday(date):
        date += -datetime.timedelta(days=1)
    
    if date2str:
        date = date.strftime('%Y-%m-%d')
        
    return date


from datetime import timedelta


def next_business_day(date: datetime.date):
    
    ONE_DAY = timedelta(days=1)
    HOLIDAYS_US = holidays.US()

    next_day = date
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day += ONE_DAY
    return next_day


def get_financial_data(output_file: str, stock: str, start_date: str, end_date: str, interval: str):
    """
    Loads financial data for a specified stock from Yahoo Finance.

    Parameters:
    stock (str): The stock ticker symbol.
    start_date (str): The start date for the data (YYYY-MM-DD).
    end_date (str): The end date for the data (YYYY-MM-DD).
    interval (str): The interval for the data ("1m", "1h", "1d").
    output_file (str): The name of the output file to save the data.

    Returns:
    pd.DataFrame: The financial data for the specified stock.
    
    """
    try:
        df = pd.read_csv(output_file)
        print(f'File data found...reading {stock} data')
    except FileNotFoundError:
        print(f'File not found...downloading the {stock} data')
        df = yf.download(stock, start=start_date, end=end_date, interval=interval)
        df.to_csv(output_file)
        df = pd.read_csv(output_file)
    
    return df


def clean_yfinance_data(df: pd.DataFrame, keep_cols: Union[List, str] = None, drop_cols: Union[List, str] = None):
    """
    Cleans the data downloaded from Yahoo Finance.

    Parameters:
    df (pd.DataFrame): The data downloaded from Yahoo Finance.
    keep_cols (list or str, default=None): Columns to keep in the DataFrame.
    drop_cols (list or str, default=None): Columns to drop from the DataFrame.

    Returns:
    pd.DataFrame: The cleaned data.
    """
    df.columns = [col.lower() for col in df.columns]
    df.set_index(df.columns[0], inplace=True)
    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[10], (datetime.date, datetime.datetime)):
        df.index.name = 'date'
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    # df = df.resample('1min').ffill()
    
    df['returns'] = df['adj close'].pct_change()

    df['volume_delta'] = df['volume'].pct_change()
    df['volume_delta'] = np.where(df['volume_delta'] == float('inf'), 1, df['volume_delta'])
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    if keep_cols:
        df = df[keep_cols]
    
    df.dropna(inplace=True)
    
    return df


def read_excel_default(excel_name: str,
                       sheet_name: str = None, 
                       index_col : int = 0,
                       parse_dates: bool =True,
                       print_sheets: bool = False,
                       **kwargs):
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
    - The function ensures that the index name is set to 'date' if the index column name is 'date', 'dates' or 'datatime', or if the index contains date-like values.
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
    df = pd.read_excel(excel_name, index_col=index_col, parse_dates=parse_dates,  sheet_name=sheet_name, **kwargs)
    df.columns = [col.lower() for col in df.columns]
    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[0], (datetime.date, datetime.datetime)):
        df.index.name = 'date'
    return df


def read_csv_default(csv_name: str,
                     index_col: int = 0,
                     parse_dates: bool = True,
                     print_data: bool = False,
                     keep_cols: Union[List, str] = None,
                     drop_cols: Union[List, str] = None,
                     **kwargs):
    """
    Reads a CSV file and returns a DataFrame with specified options.

    Parameters:
    csv_name (str): The path to the CSV file.
    index_col (int, default=0): Column to use as the row index labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_data (bool, default=False): If True, prints the first few rows of the DataFrame.
    keep_cols (list or str, default=None): Columns to read from the CSV file.
    drop_cols (list or str, default=None): Columns to drop from the DataFrame.
    **kwargs: Additional arguments passed to `pd.read_csv`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.

    Notes:
    - The function ensures that the index name is set to 'date' if the index column name is 'date', 'dates' or 'datatime', or if the index contains date-like values.
    """

    df = pd.read_csv(csv_name, index_col=index_col, parse_dates=parse_dates, **kwargs)
    df.columns = [col.lower() for col in df.columns]

    # Filter columns if keep_cols is specified
    if keep_cols is not None:
        if isinstance(keep_cols, str):
            keep_cols = [keep_cols]
        df = df[keep_cols]

    # Drop columns if drop_cols is specified
    if drop_cols is not None:
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]
        df = df.drop(columns=drop_cols, errors='ignore')

    # Print data if print_data is True
    if print_data:
        print("Columns:", ", ".join(df.columns))
        print(df.head(3))
        print('-' * 70)
    
    # Set index name to 'date' if appropriate
    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[0], (datetime.date, datetime.datetime)):
        df.index.name = 'date'
    
    return df


def read_pickle_default(pkl_name: str,
                        index_col: int = 0,
                        parse_dates: bool = True,
                        print_data: bool = False,
                        keep_cols: Union[List, str] = None,
                        drop_cols: Union[List, str] = None,
                        **kwargs):
    """
    Reads a Pickle file and returns a DataFrame with specified options.

    Parameters:
    pkl_name (str): The path to the Pickle file.
    index_col (int, default=0): Column to use as the row index labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_data (bool, default=False): If True, prints the first few rows of the DataFrame.
    keep_cols (list or str, default=None): Columns to read from the Pickle file.
    drop_cols (list or str, default=None): Columns to drop from the DataFrame.
    **kwargs: Additional arguments passed to `pd.read_pickle`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.

    Notes:
    - The function ensures that the index name is set to 'date' if the index column name is 'date', 'dates' or 'datatime', or if the index contains date-like values.
    """

    # Load the Pickle file
    df = pd.read_pickle(pkl_name, **kwargs)
    df.columns = [col.lower() for col in df.columns]

    if index_col is not None:
        df = df.set_index(df.columns[index_col])

    if parse_dates:
        try:
            df.index = pd.to_datetime(df.index, errors='ignore')
        except Exception as e:
            print(f"Warning: Failed to parse dates in index. Error: {e}")

    # Filter columns if keep_cols is specified
    if keep_cols is not None:
        if isinstance(keep_cols, str):
            keep_cols = [keep_cols]
        df = df[keep_cols]

    # Drop columns if drop_cols is specified
    if drop_cols is not None:
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]
        df = df.drop(columns=drop_cols, errors='ignore')

    # Print data if print_data is True
    if print_data:
        print("Columns:", ", ".join(df.columns))
        print(df.head(3))
        print('-' * 70)

    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[0], (datetime.date, datetime.datetime)):
        df.index.name = 'date'

    return df


def time_series_to_df(returns: Union[pd.DataFrame, pd.Series, List[pd.Series]], name: str = "Returns"):
    """
    Converts returns to a DataFrame if it is a Series or a list of Series.

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.

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
                raise TypeError(f'{name} must be either a pd.DataFrame or a list of pd.Series')
            
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print(f'Could not convert {name} to float. Check if there are any non-numeric values')
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

    # Set index name to 'date' if appropriate
    
    if returns.index.name is not None:
        if returns.index.name.lower() in ['date', 'dates', 'datetime']:
            returns.index.name = 'date'
    elif isinstance(returns.index[0], (datetime.date, datetime.datetime)):
        returns.index.name = 'date'
    elif 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    elif 'datetime' in returns.columns.str.lower():
        returns = returns.rename({'Datetime': 'date'}, axis=1)
        returns = returns.rename({'datetime': 'date'}, axis=1)
        returns = returns.set_index('date')

    # Convert dates to datetime if not already in datetime format or if minutes are 0
    try:
        returns.index = pd.to_datetime(returns.index, utc=True)
    except ValueError:
        print('Could not convert the index to datetime. Check the index format for invalid dates.')
    if not isinstance(returns.index, pd.DatetimeIndex) or (returns.index.minute == 0).all():
        returns.index = pd.to_datetime(returns.index.map(lambda x: x.date()))
        
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    return returns


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
        keep_columns = [re.escape(col) for col in keep_columns]
        keep_columns = "(?i).*(" + "|".join(keep_columns) + ").*" if isinstance(keep_columns, list) else "(?i).*" + keep_columns + ".*"
        df = df.filter(regex=keep_columns)
        if drop_columns is not None:
            print('Both "keep_columns" and "drop_columns" were specified. "drop_columns" will be ignored.')

    elif drop_columns is not None:
        drop_columns = [re.escape(col) for col in drop_columns]
        drop_columns = "(?i).*(" + "|".join(drop_columns) + ").*" if isinstance(drop_columns, list) else "(?i).*" + drop_columns + ".*"
        df = df.drop(columns=df.filter(regex=drop_columns).columns)

    # Indexes
    if keep_indexes is not None:
        keep_indexes = [re.escape(col) for col in keep_indexes]
        keep_indexes = "(?i).*(" + "|".join(keep_indexes) + ").*" if isinstance(keep_indexes, list) else "(?i).*" + keep_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
        if drop_indexes is not None:
            print('Both "keep_indexes" and "drop_indexes" were specified. "drop_indexes" will be ignored.')

    elif drop_indexes is not None:
        drop_indexes = [re.escape(col) for col in drop_indexes]
        drop_indexes = "(?i).*(" + "|".join(drop_indexes) + ").*" if isinstance(drop_indexes, list) else "(?i).*" + drop_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
    
    return df