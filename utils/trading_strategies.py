import pandas as pd
import numpy as np
pd.options.display.float_format = "{:,.4f}".format
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
from typing import Union, List

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from scipy.stats import norm

import re

from utils.tools import *

def signal_to_returns(
        
    signals: Union[pd.DataFrame, pd.Series],
    prices: Union[pd.DataFrame, pd.Series],
    strategy_name: str = None,
    is_buy_sell: bool = False,
    plot_signals: bool = False,
    plot_returns: bool = False
):
    """
    Translates trading signals and asset returns into strategy returns by multiplying the signals with the asset returns.

    Parameters:
    signals (pd.DataFrame or pd.Series): DataFrame of trading signals for the asset: either position/weights (ex. -1, 0, +1) or trade signals (ex. -1, +1).
    prices (pd.DataFrame or pd.Series): DataFrame of a single asset prices.
    strategy_name (str, default=None): Name for the strategy. If None, a name will be generated based on the signals.
    is_buy_sell (bool, default=False): If True, the signals are buy/sell signals (1, -1) instead of position weights.
    
    Returns:
    pd.DataFrame: DataFrame of strategy returns based on the signals and asset returns.
    """

    signals = time_series_to_df(signals)  # Convert returns to DataFrame if it is a Series
    fix_dates_index(signals)  # Ensure the date index is in datetime format and values are floats

    prices = time_series_to_df(prices)  # Convert returns to DataFrame if it is a Series
    fix_dates_index(prices)  # Ensure the date index is in datetime format and values are floats

    # Select the column that have 'signals' or 'Signals' in the column name:
    signals = signals.loc[:, signals.columns.str.contains(r'\bsignals?\b', case=False)]
    if signals.shape[1] == 0:
        raise Exception("No signals found in the DataFrame. Column names must contain 'signals' or 'Signals'.")
    elif signals.shape[1] > 1:
        print("Multiple signal columns found. Using the first column.")

    if prices.shape[1] > 1:
        print("Too many assets. Using the first asset only.")
        prices = prices.iloc[:, 0]
    
    asset_name = prices.columns[0]

    # Get the name in the column: it will beeverything except signals/Signals, excluding leading or trailing spaces
    if strategy_name is None:
        strategy_name = re.sub(r'([tT]rading\s*)?[sS]ignals?', '', signals.columns[0]).strip()
        if strategy_name == '':
            raise Exception("No strategy name found in the signal column. Provide a strategy name.")
            raise Exception("No strategy name found in the signal column. Provide a strategy name.")

    returns = prices.pct_change().iloc[1:]  # Calculate the asset returns

    if is_buy_sell:
        trade_signals = signals.copy()
        positions = signals.cumsum()
    else:
        trade_signals = signals.diff()
        positions = signals.copy()

    strategy_returns = pd.merge(positions.shift(), returns, left_index=True, right_index=True, how='inner') # shift to avoid look-ahead bias
    strategy_returns[f'Returns ({strategy_name})'] = strategy_returns.iloc[:, 0] * strategy_returns.iloc[:, 1]
    strategy_returns = strategy_returns.iloc[1:]
    strategy_returns = strategy_returns[[f'Returns ({strategy_name})']]

    # Plot the trading signals
    if plot_signals:

        plt.figure(figsize=(12, 5))
        plt.plot(prices.iloc[:,0], label=f'{asset_name} Adj. Prices', alpha=0.5)
        
        # Plot buy signals
        buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
        plt.plot(buy_signals, prices.iloc[:, 0][buy_signals], '^', markersize=10, color='g', lw=0, label='Buy Signal')

        # Plot sell signals
        sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
        plt.plot(sell_signals, prices.iloc[:, 0][sell_signals], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        plt.title(f'{asset_name} Adj. Prices and Trading Signals ({strategy_name} Strategy)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot the cumulative returns of the strategy
    if plot_returns:
        
        # Calculate the cumulative returns of the strategy
        strategy_cumulative = (1 + strategy_returns).cumprod() - 1
        asset_returns_cumulative = (1 + returns).cumprod() - 1
        plt.figure(figsize=(12, 5))
        plt.plot(strategy_cumulative.iloc[:,0], label=f'{strategy_name} Strategy Cumulative Return')
        
        plt.plot(asset_returns_cumulative.iloc[:,0], label=f'{asset_name} Buy and Hold Strategy Cumulative Return')
        plt.title(f'Cumulative Returns of the {strategy_name} vs. Buy and Hold Strategy')
        plt.legend()
        plt.grid(True)
        plt.show()

    return strategy_returns


def calc_buy_hold_port_returns(
    returns: Union[pd.DataFrame, List[pd.Series], pd.Series],
    weights: Union[dict, list, pd.Series, pd.DataFrame],
    return_cumulative: bool = False,
    port_name: Union[None, str] = None
):
    """
    Creates a buy-and-hold portfolio by applying initial weights to the asset returns 
    and letting them fluctuate over time without rebalancing.

    Parameters:
    returns (pd.DataFrame, pd.Series or List of pd.Series): Time series of asset returns.
    weights (list or pd.Series): Initial weights to apply to the returns It could be a portfolio or single asset. 
                                If a list or pd.Series is provided, it will be converted into a dict.
    port_name (str or None, default=None): Name for the portfolio. If None, a name will be generated based on asset weights.

    Returns:
    pd.DataFrame: The portfolio returns based on the initial weights and cumulative returns.
    """

    returns = time_series_to_df(returns)  # Convert returns to DataFrame if it is a Series or list of Series
    fix_dates_index(returns)  # Ensure the date index is in datetime format and returns are floats

    # Convert weights to dictionary format if necessary
    if isinstance(weights, list):
        print("Weights are a list. Converting to dict assuming same order as columns in returns")
        weights = dict(zip(returns.columns, weights))
    elif isinstance(weights, pd.Series):
        weights = weights.to_dict()
    elif isinstance(weights, pd.DataFrame):
        weights = list(weights.to_dict().values())[0]
    elif isinstance(weights, dict):
        pass
    else:
        raise Exception("Weights must be a dict, list, pd.Series, or pd.DataFrame")

    # Check if the number of assets in returns matches the number of weights provided
    if returns.shape[1] != len(weights):
        raise Exception(f"Returns have {returns.shape[1]} assets, but {len(weights)} weights were provided")

    # Ensure columns match weights keys
    returns = returns[list(weights.keys())]

    # Calculate the initial portfolio value based on initial weights
    weighted_returns = returns.multiply(list(weights.values()), axis=1)

    # Sum the weighted cumulative returns across assets to get portfolio value over time
    portfolio_returns = weighted_returns.sum(axis=1)
    
    # Calculate portfolio returns
    portfolio_returns_cumulative = (1 + portfolio_returns).cumprod() - 1

    # Assign name to the portfolio returns
    if port_name is None:
        print("Buy-and-Hold Portfolio: " + " + ".join([f"{n} ({w:.2%})" for n, w in weights.items()]))
        port_name = 'Buy_and_Hold_Portfolio'
    portfolio_returns = pd.DataFrame(portfolio_returns, columns=[port_name])

    if return_cumulative:
        return portfolio_returns_cumulative

    return portfolio_returns


def calc_sma_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    short_window: 50,
    long_window: 200,
    returns_benchmark: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'SMA',
):
    """
    Creates a trade strategy returns or signals using a moving average crossover strategy by adjusting weights (-100%, or 100%)
    based on short and long moving averages for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices (not returns).
    short_window (int, default=50): Window for the short moving average.
    long_window (int, default=200): Window for the long moving average.
    returns_benchmark (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    strategy_name (str, default='SMA'): Name for the strategy. If None, a name will be generated based on the moving average windows.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by moving average crossover strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if returns_benchmark is not None:
        returns_benchmark = time_series_to_df(returns_benchmark, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(returns_benchmark)  # Ensure index is datetime and data is in float format
        benchmark_cummulative = (1 + returns_benchmark).cumprod() - 1
        
    # Filter out any NaN values and sort the prices
    prices.dropna(inplace=True)
    prices.sort_index(inplace=True)

    # Calculate moving averages
    short_MA = prices.rolling(window=short_window).mean()
    long_MA = prices.rolling(window=long_window).mean()

    # Identify the crossover points
    signals = pd.DataFrame(np.where(short_MA > long_MA, 1.0, -1.0), index=long_MA.index, columns=prices.columns)

    position = signals.copy()
    trade_signals = position.diff()
    trade_signals = trade_signals.iloc[long_window:]

    # Calculate the strategy returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns = returns.iloc[long_window:].dropna()
    returns_cumulative = returns_cumulative.iloc[long_window:].dropna()

    strategy_returns = returns* position.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Calculate the cumulative returns of the strategy
    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    # Plot stretegy:
    if plot_strategy == True:
        indexes = prices.index[long_window:]
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        # Plot the closing price and the SMAs
        plt.figure(figsize=(12, 5))
        plt.plot(prices.iloc[:,0], label=f'{asset_name} Adj. Prices', alpha=0.5)
        plt.plot(short_MA.iloc[:,0], label=f'{short_window}-day MA', alpha=0.75)
        plt.plot(long_MA.iloc[:,0], label=f'{long_window}-day MA', alpha=0.75)
        
        # Plot buy signals
        buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
        plt.plot(buy_signals, prices.iloc[:, 0][buy_signals], '^', markersize=10, color='g', lw=0, label='Buy Signal')

        # Plot sell signals
        sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
        plt.plot(sell_signals, prices.iloc[:, 0][sell_signals], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        plt.title('Simple Moving Average (SMA) Crossover Strategy')
        plt.xlim(indexes[0], indexes[-1])
        plt.legend()
        plt.show()

        # Plot the cumulative returns of the strategy vs. the asset
        plt.figure(figsize=(12, 5))
        plt.plot(strategy_cumulative.iloc[:,0], label=f'MA ({short_window}, {long_window}) Strategy Cumulative Return')
        if returns_benchmark is not None:

            benchmark_cummulative = benchmark_cummulative.reindex(strategy_cumulative.index)
            benchmark_cummulative.fillna(method='ffill', inplace=True)
            benchmark_cummulative.fillna(0, inplace=True)

            plt.plot(benchmark_cummulative.iloc[:,0], label='Benchmark Cumulative Return')
            plt.title('Cumulative Returns of the Simple Moving Average (SMA) Crossover Strategy vs. Benchmark')
        else:
            plt.plot(returns_cumulative.iloc[:,0], label=f'{asset_name} Cumulative Return')
            plt.title('Cumulative Returns of the Simple Moving Average (SMA) Crossover Strategy')
        plt.xlim(indexes[0], indexes[-1])
        plt.legend()
        plt.show()
    
    # Return trading signals, and moving averages
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                ma_short = short_MA.rename(columns=lambda col: f"{col} (Short MA)")
                ma_long = long_MA.rename(columns=lambda col: f"{col} (Long MA)")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                ma_short = short_MA.rename(columns={short_MA.columns[0]: f"Short MA"})
                ma_long = long_MA.rename(columns={long_MA.columns[0]: f"Long MA"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            ma_short = short_MA.to_frame(name=f"Short MA")
            ma_long = long_MA.to_frame(name="Long MA")
        
        output_df = signals.copy()
        output_df = output_df.merge(ma_short, left_index=True, right_index=True, how='inner')
        output_df = output_df.merge(ma_long, left_index=True, right_index=True, how='inner')

        return output_df

    # Return cumulative returns of the strategy
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")

    return strategy_returns



def calc_boll_bands_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    ma_window: int = 20,
    n_std_dev: int = 2,
    returns_benchmark: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'BB',
):
    """
    Creates a trade strategy returns or signals using a Bollinger Bands strategy by adjusting weights (-100%, or 100%)
    based on short and long moving averages for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices (not returns).
    ma_window (int, default=20): Window for the moving average.
    n_std_dev (int, default=2): Number of standard deviations for the Bollinger Bands.
    returns_benchmark (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    strategy_name (str, default='SMA'): Name for the strategy. If None, a name will be generated based on the moving average windows.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by Bollinger Bands strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if returns_benchmark is not None:
        returns_benchmark = time_series_to_df(returns_benchmark, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(returns_benchmark)  # Ensure index is datetime and data is in float format
        benchmark_cummulative = (1 + returns_benchmark).cumprod() - 1
        
    # Filter out any NaN values and sort the prices
    prices.dropna(inplace=True)
    prices.sort_index(inplace=True)

    # Calculate the moving average and standard deviation
    moving_average = prices.rolling(window=ma_window).mean()
    std_dev = prices.rolling(window=ma_window).std()
    upper_band = moving_average + (std_dev * n_std_dev)
    lower_band = moving_average - (std_dev * n_std_dev)

    upper_band = upper_band.iloc[ma_window:]
    lower_band = lower_band.iloc[ma_window:]

    # Identify the crossover points
    #signals = pd.DataFrame(np.where(prices.iloc[ma_window:] < lower_band, 1.0, np.where(prices[ma_window:] > upper_band, -1.0, 0.0)),
    #                       index=prices.index[ma_window:],
    #                       columns=prices.columns)

    # Generate trade signals based on conditions
    trade_signals_bb = np.where((prices.iloc[ma_window:].shift() >= lower_band) & (prices.iloc[ma_window:] < lower_band), 1.0, 0)
    trade_signals_bb = np.where((prices.iloc[ma_window:].shift() <= upper_band) & (prices.iloc[ma_window:] > upper_band), -1.0, trade_signals_bb)
    trade_signals_ma = np.where((prices.iloc[ma_window:].shift() >= moving_average.iloc[ma_window:].shift()) & (prices.iloc[ma_window:] < moving_average.iloc[ma_window:]), 1.0, 0)
    trade_signals_ma = np.where((prices.iloc[ma_window:].shift() <= moving_average.iloc[ma_window:].shift()) & (prices.iloc[ma_window:] > moving_average.iloc[ma_window:]), -1.0, trade_signals_ma)
    
    # Convert to DataFrame and set the initial position
    trade_signals_bb = pd.DataFrame(trade_signals_bb, index=prices.index[ma_window:], columns=prices.columns)
    trade_signals_ma = pd.DataFrame(trade_signals_ma, index=prices.index[ma_window:], columns=prices.columns)

    # Calculate position based on trade signals
    position = pd.DataFrame(index=trade_signals_bb.index, columns=trade_signals_bb.columns)
    position.iloc[0] = 1 if prices.iloc[ma_window, 0] <  moving_average.iloc[ma_window, 0] else -1 if prices.iloc[ma_window, 0] >  moving_average.iloc[ma_window, 0] else 0
    
    for i in range(1, len(trade_signals_bb)):
        if trade_signals_bb.iloc[i, 0] == 1:  # Long signal
            position.iloc[i, 0] = 1
        elif trade_signals_bb.iloc[i, 0] == -1:  # Short signal
            position.iloc[i, 0] = -1
        elif trade_signals_ma.iloc[i, 0] != 0:  # Crossing mean, zero position
            position.iloc[i, 0] = 0
        else:  # Neutral signal
            position.iloc[i, 0] = position.iloc[i-1, 0]

    signals = position.copy()

    trade_signals = position.diff()
    trade_signals.fillna(position.iloc[0], inplace=True)

    # Calculate the strategy returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns = returns.iloc[ma_window:].dropna()
    returns_cumulative = returns_cumulative.iloc[ma_window:].dropna()

    strategy_returns = returns* position.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Calculate the cumulative returns of the strategy
    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    # Plot stretegy:
    if plot_strategy == True:
        indexes = prices.index[ma_window:]
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        # Plot the closing price and the SMAs
        plt.figure(figsize=(12, 5))
        plt.plot(prices.iloc[:,0], label=f'{asset_name} Adj. Prices', alpha=1)
        plt.plot(moving_average.iloc[:,0], label=f'{ma_window}-day MA', alpha=0.75)
        plt.plot(upper_band.iloc[:,0], label=f'Upper Band ({n_std_dev} Std. Dev.)', color='purple',alpha=0.5)
        plt.plot(lower_band.iloc[:,0], label=f'Lower Band ({n_std_dev} Std. Dev.)', color='purple', alpha=0.5)
        plt.fill_between(lower_band.index, lower_band.iloc[:,0], upper_band.iloc[:,0], color='purple', alpha=0.15)
        
        # Plot buy signals
        buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
        plt.plot(buy_signals, prices.iloc[:, 0][buy_signals], '^', markersize=5, color='g', lw=0, label='Buy Signal')

        # Plot sell signals
        sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
        plt.plot(sell_signals, prices.iloc[:, 0][sell_signals], 'v', markersize=5, color='r', lw=0, label='Sell Signal')

        plt.xlim(indexes[0], indexes[-1])
        plt.title('Bollinger Bands Crossover Strategy')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot the cumulative returns of the strategy vs. the asset
        plt.figure(figsize=(12, 5))
        plt.plot(strategy_cumulative.iloc[:,0], label=f'Bollinger Bands ({ma_window}) Strategy Cumulative Return')
        if returns_benchmark is not None:

            benchmark_cummulative = benchmark_cummulative.reindex(strategy_cumulative.index)
            benchmark_cummulative.fillna(method='ffill', inplace=True)
            benchmark_cummulative.fillna(0, inplace=True)

            plt.plot(benchmark_cummulative.iloc[:,0], label='Benchmark Cumulative Return')
            plt.title('Cumulative Returns of the Bollinger Bands Strategy vs. Benchmark')
        else:
            plt.plot(returns_cumulative.iloc[:,0], label=f'{asset_name} Cumulative Return')
            plt.title('Cumulative Returns of the Bollinger Bands Strategy')

        plt.xlim(indexes[0], indexes[-1])
        plt.grid(True)
        plt.legend()
        plt.show()
    
    # Return trading signals, and moving averages
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                moving_average = moving_average.rename(columns=lambda col: f"{col} (MA)")
                upper_band = upper_band.rename(columns=lambda col: f"{col} (Upper Band)")
                lower_band = lower_band.rename(columns=lambda col: f"{col} (Lower Band)")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                moving_average = moving_average.rename(columns={moving_average.columns[0]: f"MA"})
                upper_band = upper_band.rename(columns={upper_band.columns[0]: f"Upper Band"})
                lower_band = lower_band.rename(columns={lower_band.columns[0]: f"Lower Band"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            moving_average = moving_average.to_frame(name=f"Moving Average")
            upper_band = upper_band.to_frame(name=f"Upper Band")
            lower_band = lower_band.to_frame(name=f"Lower Band")
        
        output_df = signals.copy()
        output_df = output_df.merge(moving_average, left_index=True, right_index=True, how='inner', suffixes=('', '_moving_average'))
        output_df = output_df.merge(upper_band, left_index=True, right_index=True, how='inner', suffixes=('', '_upper_band'))
        output_df = output_df.merge(lower_band, left_index=True, right_index=True, how='inner', suffixes=('', '_lower_band'))

        return output_df

    # Return cumulative returns of the strategy
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")

    return strategy_returns


def calc_ema_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    short_window: 12,
    long_window: 26,
    returns_benchmark: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'EMA',
):
    """
    Creates a trade strategy returns or signals using a exponential moving average (EMA) crossover strategy by adjusting weights (-100%, or 100%)
    based on short and long moving averages for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices (not returns).
    short_window (int, default=12): Window for the short moving average.
    long_window (int, default=26): Window for the long moving average.
    returns_benchmark (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    strategy_name (str, default='EMA'): Name for the strategy. If None, a name will be generated based on the moving average windows.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by exponential moving average (EMA) crossover strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if returns_benchmark is not None:
        returns_benchmark = time_series_to_df(returns_benchmark, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(returns_benchmark)  # Ensure index is datetime and data is in float format
        benchmark_cummulative = (1 + returns_benchmark).cumprod() - 1
        
    # Filter out any NaN values and sort the prices
    prices.dropna(inplace=True)
    prices.sort_index(inplace=True)

    # Calculate moving averages
    short_EMA = prices.ewm(span=short_window, adjust=False).mean()
    long_EMA = prices.ewm(span=long_window, adjust=False).mean()

    # Identify the crossover points
    signals = pd.DataFrame(np.where(short_EMA > long_EMA, 1.0, -1.0), index=long_EMA.index, columns=prices.columns)

    position = signals.copy()
    trade_signals = position.diff()
    trade_signals = trade_signals.iloc[long_window:]


    # Calculate the strategy returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns = returns.iloc[long_window:].dropna()
    returns_cumulative = returns_cumulative.iloc[long_window:].dropna()

    strategy_returns = returns* position.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Calculate the cumulative returns of the strategy
    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    # Plot stretegy:
    if plot_strategy == True:
        indexes = prices.index[long_window:]
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        # Plot the closing price and the EMAs
        plt.figure(figsize=(12, 5))
        plt.plot(prices.iloc[:,0], label=f'{asset_name} Adj. Prices', alpha=0.5)
        plt.plot(short_EMA.iloc[:,0], label=f'{short_window}-day EMA', alpha=0.75)
        plt.plot(long_EMA.iloc[:,0], label=f'{long_window}-day EMA', alpha=0.75)
        
        # Plot buy signals
        buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
        plt.plot(buy_signals, prices.iloc[:, 0][buy_signals], '^', markersize=10, color='g', lw=0, label='Buy Signal')

        # Plot sell signals
        sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
        plt.plot(sell_signals, prices.iloc[:, 0][sell_signals], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        plt.title('Simple Moving Average (EMA) Crossover Strategy')

        plt.xlim(indexes[0], indexes[-1])
        plt.legend()
        plt.show()

        # Plot the cumulative returns of the strategy vs. the asset
        plt.figure(figsize=(12, 5))
        plt.plot(strategy_cumulative.iloc[:,0], label=f'EMA ({short_window}, {long_window}) Strategy Cumulative Return')
        if returns_benchmark is not None:

            benchmark_cummulative = benchmark_cummulative.reindex(strategy_cumulative.index)
            benchmark_cummulative.fillna(method='ffill', inplace=True)
            benchmark_cummulative.fillna(0, inplace=True)

            plt.plot(benchmark_cummulative.iloc[:,0], label='Benchmark Cumulative Return')
            plt.title('Cumulative Returns of the EMA Crossover Strategy vs. Benchmark')
        else:
            plt.plot(returns_cumulative.iloc[:,0], label=f'{asset_name} Cumulative Return')
            plt.title('Cumulative Returns of the EMA Crossover Strategy')

        plt.xlim(indexes[0], indexes[-1])
        plt.legend()
        plt.show()

    # Return trading signals, and moving averages
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                ma_short = short_EMA.rename(columns=lambda col: f"{col} (Short EMA)")
                ma_long = long_EMA.rename(columns=lambda col: f"{col} (Long EMA)")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                ma_short = short_EMA.rename(columns={short_EMA.columns[0]: f"Short EMA"})
                ma_long = long_EMA.rename(columns={long_EMA.columns[0]: f"Long EMA"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            ma_short = short_EMA.to_frame(name=f"Short EMA")
            ma_long = long_EMA.to_frame(name="Long EMA")
            
        output_df = signals.copy()
        output_df = output_df.merge(ma_short, left_index=True, right_index=True, how='inner', suffixes=('', '_ma_short'))
        output_df = output_df.merge(ma_long, left_index=True, right_index=True, how='inner', suffixes=('', '_ma_long'))

        return output_df

    # Return cumulative returns of the strategy
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")

    return strategy_returns



def calc_macd_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    short_window: 12,
    long_window: 26,
    signal_window: 9,
    returns_benchmark: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'MACD',
):
    """
    Creates a trade strategy returns or signals using a Moving Average Convergence/Divergence (MACD) strategy by adjusting weights (-100%, or 100%)
    based on short and long moving averages for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices (not returns).
    short_window (int, default=12): Window for the short moving average.
    long_window (int, default=26): Window for the long moving average.
    returns_benchmark (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    strategy_name (str, default='MACD'): Name for the strategy. If None, a name will be generated based on the moving average windows.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by Moving Average Convergence/Divergence (MACD) strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if returns_benchmark is not None:
        returns_benchmark = time_series_to_df(returns_benchmark, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(returns_benchmark)  # Ensure index is datetime and data is in float format
        benchmark_cummulative = (1 + returns_benchmark).cumprod() - 1
        
    # Filter out any NaN values and sort the prices
    prices.dropna(inplace=True)
    prices.sort_index(inplace=True)

    # Calculate the short-term and long-term EMAs and MACD
    short_EMA = prices.ewm(span=short_window, adjust=False).mean()
    long_EMA = prices.ewm(span=long_window, adjust=False).mean()
    macd = short_EMA - long_EMA
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()

    # Identify the crossover points
    signals = pd.DataFrame(np.where(macd > signal_line, 1.0, -1.0), index=long_EMA.index, columns=prices.columns)

    position = signals.copy()
    trade_signals = position.diff()
    trade_signals = trade_signals.iloc[long_window:]


    # Calculate the strategy returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns = returns.iloc[long_window:].dropna()
    returns_cumulative = returns_cumulative.iloc[long_window:].dropna()

    strategy_returns = returns* position.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Calculate the cumulative returns of the strategy
    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    # Plot stretegy:
    if plot_strategy == True:
        indexes = prices.index[long_window:]
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        # Plot the closing price and the EMAs
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot the adjusted price in the first subplot (ax1)
        ax1.plot(prices.iloc[:, 0], label=asset_name, color='blue', alpha=0.5)
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.set_title(f'{asset_name} Adj. Prices')
        ax1.legend(loc='upper left')

        # Plot the RSI in the second subplot (ax2)
        ax2.plot(macd.iloc[:, 0], label=f'MACD ({short_window}, {long_window})', color='purple', alpha=0.5)
        ax2.plot(signal_line.iloc[:, 0], label=f'Signal Line ({signal_window})', color='green', alpha=0.75)
        plt.title('Moving Average Convergence/Divergence (MACD) Indicator')

        plt.xlim(indexes[0], indexes[-1])
        ax2.set_ylabel('MACD', fontsize=10)
        ax2.grid(True)
        
        # Plot buy signals
        buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
        plt.plot(buy_signals, macd.iloc[:, 0][buy_signals], '^', markersize=10, color='g', lw=0, label='Buy Signal')

        # Plot sell signals
        sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
        plt.plot(sell_signals, macd.iloc[:, 0][sell_signals], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        plt.title('Moving Average Convergence/Divergence (MACD) Strategy')
        plt.xlim(indexes[0], indexes[-1])
        plt.legend()
        plt.show()

        # Plot the cumulative returns of the strategy vs. the asset
        plt.figure(figsize=(12, 5))
        plt.plot(strategy_cumulative.iloc[:,0], label=f'MACD ({signal_window}, {short_window}, {long_window}) Strategy Cumulative Return')
        if returns_benchmark is not None:

            benchmark_cummulative = benchmark_cummulative.reindex(strategy_cumulative.index)
            benchmark_cummulative.fillna(method='ffill', inplace=True)
            benchmark_cummulative.fillna(0, inplace=True)
            
            plt.plot(benchmark_cummulative.iloc[:, 0], label='Benchmark Cumulative Return')
            plt.title('Cumulative Returns of the MACD Strategy vs. Benchmark')
        else:
            plt.plot(returns_cumulative.iloc[:,0], label=f'{asset_name} Cumulative Return')
            plt.title('Cumulative Returns of the MACD Strategy')
            
        
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Return trading signals, and moving averages
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                ma_short = short_EMA.rename(columns=lambda col: f"{col} (Short EMA)")
                ma_long = long_EMA.rename(columns=lambda col: f"{col} (Long EMA)")
                ma_signal_line = signal_line.rename(columns=lambda col: f"{col} (Signal Line)")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                ma_short = short_EMA.rename(columns={short_EMA.columns[0]: f"Short EMA"})
                ma_long = long_EMA.rename(columns={long_EMA.columns[0]: f"Long EMA"})
                ma_signal_line = signal_line.rename(columns={signal_line.columns[0]: f"Signal Line"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            ma_short = short_EMA.to_frame(name=f"Short EMA")
            ma_long = long_EMA.to_frame(name="Long EMA")
            ma_signal_line = signal_line.to_frame(name="Signal Line")
        
        output_df = signals.copy()
        output_df = output_df.merge(ma_short, left_index=True, right_index=True, how='inner')
        output_df = output_df.merge(ma_long, left_index=True, right_index=True, how='inner')
        output_df = output_df.merge(ma_signal_line, left_index=True, right_index=True, how='inner')

        return output_df

    # Return cumulative returns of the strategy
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")

    return strategy_returns


def calc_rsi_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    rsi_period: int = 14,
    overbought: int = 70,
    oversold: int = 30,
    returns_benchmark: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'RSI',
):
    """
    Creates a trade strategy using an RSI (Relative Strength Index) strategy by adjusting weights (-100%, 0, or 100%) 
    based on RSI values for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices.
    rsi_period (int, default=14): Period for calculating RSI.
    overbought (int, default=70): RSI threshold for overbought condition.
    oversold (int, default=30): RSI threshold for oversold condition.
    returns_benchmark (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, 0, +1) and the RSI values. Else, returns the cumulative returns of the strategy.
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals, and plot the cumulative returns of the strategy.
    strategy_name (str, default='RSI'): Name for the strategy. If None, a name will be generated based on the RSI period.

    Returns:
    pd.DataFrame: Strategy returns, trading signals, or RSI values based on parameters.
    """


    prices = time_series_to_df(prices, name='Prices') # Convert benchmark to DataFrame if needed
    fix_dates_index(prices) # Ensure index is datetime and data is in float format

    if returns_benchmark is not None:
        returns_benchmark = time_series_to_df(returns_benchmark, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(returns_benchmark)  # Ensure index is datetime and data is in float format
        benchmark_cummulative = (1 + returns_benchmark).cumprod() - 1

    # Calculate RSI
    delta = prices.diff()
    gain = pd.DataFrame(np.where(delta > 0, delta, 0), index=prices.index, columns=prices.columns)
    loss = pd.DataFrame(np.where(delta < 0, -delta, 0), index=prices.index, columns=prices.columns)
    avg_gain = pd.DataFrame(gain).rolling(window=rsi_period).mean()
    avg_loss = pd.DataFrame(loss).rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.reindex(prices.index)
    rsi = rsi.iloc[rsi_period:]
    
    # Generate buy/sell signals based on RSI thresholds

    # Generate buy/sell signals based on RSI thresholds (stay sold above overbought, stay bought below oversold, and neutral in between)
    signals = pd.DataFrame(np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0)),
                           index=prices.index[rsi_period:],
                           columns=prices.columns)

    # Alternative method to generate buy/sell signals based on RSI thresholds --> does not work well
    #signals = np.where((rsi.shift(1) > oversold) & (rsi <= oversold), 1.0, np.nan)  # Buy trade signal when RSI crosses below oversold (and only sell if it crosses above overbought)
    #signals = np.where((rsi.shift(1) < overbought) & (rsi >= overbought), -1.0, signals)  # Sell trade signal when RSI crosses above overbought (and only buy if it crosses below oversold)
    #signals = pd.DataFrame(signals, index=prices.index[rsi_period:], columns=prices.columns)

    position = signals.copy()
    trade_signals = position.diff()

    # Calculate returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns = returns.iloc[rsi_period:].dropna()
    returns_cumulative = returns_cumulative.iloc[rsi_period:].dropna()
    
    strategy_returns = returns * position.shift() # shift to avoid look-ahead bias
    strategy_returns.dropna(inplace=True)

    # Cumulative returns for the strategy
    strategy_cumulative = (1 + strategy_returns).cumprod() - 1
    
    # Plot strategy
    if plot_strategy:
        indexes = prices.index[rsi_period:]
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")

        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
        # Plot the adjusted price in the first subplot (ax1)
        ax1.plot(prices.iloc[:, 0], label=asset_name, color='blue', alpha=0.5)
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.set_title(f'{asset_name} Adj. Prices')
        ax1.legend(loc='upper left')

        # Plot the RSI in the second subplot (ax2)
        ax2.plot(rsi.iloc[:, 0], label='RSI', color='purple', alpha=0.75)
        ax2.axhline(y=overbought, color='red', linestyle='--', label='Overbought Level (70)')
        ax2.axhline(y=oversold, color='green', linestyle='--', label='Oversold Level (30)')
        ax2.set_title(f'{strategy_name} ({rsi_period}) Indicator')
        ax2.set_ylim(0, 100)
        plt.xlim(indexes[0], indexes[-1])
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.grid(True)
        
        # Buy and sell signals on the RSI plot
        buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
        sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
        ax2.plot(buy_signals, rsi.iloc[:, 0][buy_signals], '^', markersize=10, color='g', lw=0, label='Buy Signal')
        ax2.plot(sell_signals, rsi.iloc[:, 0][sell_signals], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        # Display the plot
        plt.show()

        # Plot the cumulative returns of the strategy vs. the asset
        plt.figure(figsize=(12, 5))
        plt.plot(strategy_cumulative.iloc[:, 0], label=f'RSI Strategy ({rsi_period}) Cumulative Return')
        if returns_benchmark is not None:

            benchmark_cummulative = benchmark_cummulative.reindex(strategy_cumulative.index)
            benchmark_cummulative.fillna(method='ffill', inplace=True)
            benchmark_cummulative.fillna(0, inplace=True)
            
            plt.plot(benchmark_cummulative.iloc[:, 0], label='Benchmark Cumulative Return')
            plt.title('Cumulative Returns of the RSI Strategy vs. Benchmark')
        else:
            plt.plot(returns_cumulative.iloc[:,0], label=f'{asset_name} Cumulative Return')
            plt.title('Cumulative Returns of the RSI Strategy')
            
        plt.legend()
        plt.xlim(indexes[0], indexes[-1])
        plt.grid(True)
        plt.show()

    # Return trading signals, and RSI indicator
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                rsi = rsi.rename(columns=lambda col: f"{col} ({strategy_name} Indicator)")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                rsi = rsi.rename(columns={rsi.columns[0]: f"Indicator ({strategy_name})"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            rsi = rsi.to_frame(name=f"{strategy_name} Indicator")
        
        output_df = signals.copy()
        output_df = output_df.merge(rsi, left_index=True, right_index=True, how='inner')

        return output_df


    # Return cumulative returns of the strategy if not returning signals/RSI
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")

    return strategy_returns

