# Description: Multiple functions to calculate and plot trading signals and returns based on trading strategies.

import pandas as pd
import numpy as np
pd.options.display.float_format = "{:,.4f}".format
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
from typing import Union, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import warnings
warnings.filterwarnings("ignore")

from scipy.stats import norm

import re
import os

import warnings

warnings.filterwarnings('ignore')

PLOT_WIDTH, PLOT_HEIGHT = 12, 8
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
IMAGE_PATH = os.path.join(current_dir, "images")

from utils.tools import *


def signal_to_returns(
        
    signals: Union[pd.DataFrame, pd.Series],
    prices: Union[pd.DataFrame, pd.Series],
    strategy_name: str = None,
    is_buy_sell: bool = False,
    up_weight: float = 1.0,
    down_weight: float = 1.0,
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
    up_weight (float, default=1.0): Weight to apply to buy signals.
    down_weight (float, default=1.0): Weight to apply to sell signals.

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
        signals = signals.iloc[:, 0]

    if prices.shape[1] > 1:
        print("Too many assets. Using the first asset only.")
        prices = prices.iloc[:, 0]
    
    asset_name = prices.columns[0]

    # Get the name in the column: it will beeverything except signals/Signals, excluding leading or trailing spaces
    if strategy_name is None:
        strategy_name = re.sub(r'([tT]rading\s*)?[sS]ignals?', '', signals.columns[0]).strip() if isinstance(signals, pd.DataFrame) else re.sub(r'([tT]rading\s*)?[sS]ignals?', '', signals.name).strip()
        if strategy_name == '':
            raise Exception("No strategy name found in the signal column. Provide a strategy name.")

    # Calculate trade signals and positions
    if is_buy_sell:
        trade_signals = signals.copy()
        positions = signals.cumsum()
    else:
        positions = signals.copy()
        trade_signals = signals.diff()
        
    # Apply up_weight and down_weight to the trade signals:
    positions[positions > 0] *= up_weight
    positions[positions < 0] *= down_weight

    # Calculate the strategy returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)
    
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")

    # Plot the trading signals
    if plot_signals:

        plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
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
    weights: Union[dict, list, pd.Series, pd.DataFrame] = None,
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

    if returns.shape[1] == 0 and weights is None:
        portfolio_returns = returns.copy()

    else:    
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


def calc_roc(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    periods: List[int]
):
    """
    Calculates the Rate of Change (ROC) for specific periods from a time series of returns.

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    periods (List[int]): List of periods for calculating the Rate of Change.

    Returns:
    pd.DataFrame or None: Returns Rate of Change DataFrame
    """

    # Convert returns to DataFrame if it is a Series or a list of Series
    returns = time_series_to_df(returns)
    fix_dates_index(returns)  # Fix the date index of the DataFrame and convert returns to float

    roc_df = pd.DataFrame(index=returns.index)

    for period in periods:
        if len(returns.columns) > 1:
            for col in returns.columns:
                roc_col_name = f'roc_{period} ({col})'
                roc_df[roc_col_name] = returns[col].rolling(window=period).apply(lambda x: np.prod(1 + x) - 1, raw=False)
        else:
            roc_col_name = f'roc_{period}'
            roc_df[roc_col_name] = returns.iloc[:, 0].rolling(window=period).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    return roc_df

def plot_roc_strategy(
    prices: pd.DataFrame,
    roc_signals: pd.DataFrame,
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame = None,
    asset_name: str = 'Asset',
    save_path: str = None
):
    """
    Plots the ROC-based strategy and cumulative returns.

    Parameters:
    prices (pd.DataFrame): Adjusted prices of the asset.
    roc_signals (pd.DataFrame): Trading signals generated by the ROC strategy.
    strategy_returns (pd.DataFrame): Returns of the strategy.
    benchmark_returns (pd.DataFrame, optional): Benchmark returns for comparison. Default is None.
    asset_name (str): Name of the asset being plotted. Default is 'Asset'.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    None
    """

    # Calculate cumulative returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    else:
        benchmark_cumulative = None

    # Format indexes for plotting
    indexes_prices = prices.index
    if indexes_prices[0].hour == 0 and indexes_prices[0].minute == 0 and indexes_prices[0].second == 0:
        formatted_index = indexes_prices.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_prices.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index = range(len(indexes_prices))

    # Plot the asset price and trading signals
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(continuous_index, prices.iloc[:, 0], label=f'{asset_name} Adj. Prices', alpha=0.6, linewidth=1.5)

    # Plot buy signals
    buy_signals = roc_signals[roc_signals > 0].index
    buy_signals = [i for i in range(len(prices.index)) if prices.index[i] in buy_signals]
    ax.scatter(buy_signals, prices.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', lw=0, alpha=0.7)

    # Plot sell signals
    sell_signals = roc_signals[roc_signals < 0].index
    sell_signals = [i for i in range(len(prices.index)) if prices.index[i] in sell_signals]
    ax.scatter(sell_signals, prices.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', lw=0, alpha=0.7)

    ax.set_title('Rate of Change (ROC) Strategy Signals', fontsize=14)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"{save_path}/roc_strategy_signals.png", bbox_inches='tight')

    # Show the plot
    plt.show()

    # Plot the cumulative returns of the strategy vs. the asset
    indexes_strat_ret = strategy_cumulative.index
    if indexes_strat_ret[0].hour == 0 and indexes_strat_ret[0].minute == 0 and indexes_strat_ret[0].second == 0:
        formatted_index = indexes_strat_ret.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_strat_ret.strftime('%Y-%m-%d\n%H:%M:%S')
    
    continuous_index_ret = range(len(indexes_strat_ret))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(continuous_index_ret, strategy_cumulative.iloc[:, 0], label='ROC Strategy Cumulative Return', linewidth=1.5, color='blue')

    if benchmark_cumulative is not None:
        benchmark_cumulative = benchmark_cumulative.reindex(strategy_cumulative.index)
        benchmark_cumulative.fillna(method='ffill', inplace=True)
        benchmark_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_ret, benchmark_cumulative.iloc[:, 0], label='Benchmark Cumulative Return', linestyle='-', linewidth=1.2, color='orange')
        ax.set_title('Cumulative Returns of the Rate of Change (ROC) Strategy vs. Benchmark', fontsize=14)
    else:
        returns_cumulative = returns_cumulative.reindex(strategy_cumulative.index)
        returns_cumulative.fillna(method='ffill', inplace=True)
        returns_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_ret, returns_cumulative.iloc[:, 0], label=f'{asset_name} Cumulative Return', linewidth=1.5, color='orange')
        ax.set_title('Cumulative Returns of the Rate of Change (ROC) Strategy', fontsize=14)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index_ret) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax.axhline(0, color='darkgrey', linewidth=1, linestyle='-')

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"{save_path}/roc_strategy_cumulative.png", bbox_inches='tight')

    plt.show()


def calc_roc_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    periods: List[int],
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    image_save_path: str = None,
    strategy_name: str = 'ROC',
):
    """
    Calculates a trading strategy based on the Rate of Change (ROC) and generates buy/sell signals.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices.
    periods (List[int]): List of periods for calculating the Rate of Change.
    benchmark_returns (pd.Series or pd.DataFrame, default=None): Benchmark returns for comparison.
    return_signals (bool, default=False): If True, return trading signals instead of cumulative returns.
    plot_strategy (bool, default=True): If True, plot the strategy.
    image_save_path (str, default=None): Path to save plots if plot_strategy is True.
    strategy_name (str, default='ROC'): Name of the strategy.

    Returns:
    pd.DataFrame: DataFrame with ROC columns and trading signals or cumulative returns.
    """

    prices = time_series_to_df(prices)  # Ensure prices are in DataFrame format
    fix_dates_index(prices)  # Ensure index is datetime and sorted

    if benchmark_returns is not None:
        benchmark_returns = time_series_to_df(benchmark_returns)  # Ensure benchmark is in DataFrame format
        fix_dates_index(benchmark_returns)  # Ensure benchmark index is datetime and sorted

    # Filter out any NaN values and sort prices
    prices.dropna(inplace=True)
    prices.sort_index(inplace=True)

    returns = prices.pct_change()
    long_window = max(periods)

    # Calculate ROC for specified periods
    roc_df = calc_roc(returns, periods)

    # Create buy/sell signals
    signals = pd.DataFrame(index=prices.index)
    for roc in roc_df.columns:
        buy_threshold = roc_df[roc].mean() + roc_df[roc].std()  # Buy if roc is above mean + std
        sell_threshold = roc_df[roc].mean() - roc_df[roc].std()  # Sell if roc is below mean - std

        signals[roc] = np.where(roc_df[roc] < sell_threshold, -1,
                                np.where(roc_df[roc] > buy_threshold, 1, 0))

    # Combine signals into probabilities
    signal_probs = signals.mean(axis=1)  # Average probabilities across all periods
    signal_probs.name = 'Signal Probabilities'

    # Find consolidated signals
    roc_signal = pd.DataFrame(np.where(signal_probs >= 0.5, 1, np.where(signal_probs <= -0.5, -1, 0)), index=signals.index, columns=prices.columns)
    
    positions = roc_signal.copy()
    trade_signals = positions.diff()
    trade_signals = trade_signals.iloc[long_window:]
    
    # Calculate the strategy returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Plot strategy if required
    if plot_strategy:
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")

        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        plot_roc_strategy(
            prices=prices,
            roc_signals=roc_signal,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            asset_name=asset_name,
            save_path=image_save_path,
        )

    # Return trading signals, and ROC metrics
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                roc_df = roc_df.rename(columns=lambda col: f"{col} ({strategy_name})")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                roc_df = roc_df.rename(columns={roc_df.columns[0]: f"{strategy_name}"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            roc_df = roc_df.to_frame(name=f"{strategy_name}")

        output_df = signals.copy()
        output_df = output_df.merge(roc_df, left_index=True, right_index=True, how='inner')

        return output_df

    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")

    return strategy_returns



def plot_sma_strategy(prices: pd.DataFrame,
                      short_MA: pd.DataFrame,
                      long_MA: pd.DataFrame,
                      trade_signals: pd.DataFrame,
                      strategy_returns: pd.DataFrame,
                      benchmark_returns: pd.DataFrame = None, 
                      asset_name: str = 'Asset',
                      short_window: int = 50,
                      long_window: int = 200,
                      save_path: str = IMAGE_PATH):
    """
    Plots the SMA crossover strategy and cumulative returns.

    Parameters:
    prices (pd.DataFrame): Adjusted prices of the asset.
    short_MA (pd.DataFrame): Short moving average.
    long_MA (pd.DataFrame): Long moving average.
    trade_signals (pd.DataFrame): Trade signals generated by the strategy.
    strategy_cumulative (pd.DataFrame): Cumulative returns of the strategy.
    returns_cumulative (pd.DataFrame): Cumulative returns of the asset.
    benchmark_cumulative (pd.DataFrame, optional): Cumulative returns of the benchmark. Default is None.
    asset_name (str): Name of the asset being plotted. Default is 'Asset'.
    short_window (int): Window for the short moving average. Default is 50.
    long_window (int): Window for the long moving average. Default is 200.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    None
    """
    
    # Calculate cumulative returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns_cumulative = returns_cumulative.iloc[long_window:].dropna()

    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    else:
        benchmark_cumulative = None

    # Format indexes
    indexes_prices = prices.index
    if indexes_prices[0].hour == 0 and indexes_prices[0].minute == 0 and indexes_prices[0].second == 0:
        formatted_index = indexes_prices.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_prices.strftime('%Y-%m-%d\n%H:%M:%S')
    
    continuous_index = range(len(indexes_prices))

    # Plot the closing price and the SMAs
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    # Plot the closing price and the SMAs
    ax.plot(continuous_index, prices.iloc[:, 0], label=f'{asset_name} Adj. Prices', alpha=0.6, linewidth=1.5)
    ax.plot(continuous_index, short_MA.iloc[:, 0], label=f'{short_window}-day MA', alpha=0.8, linewidth=1.2, linestyle='--')
    ax.plot(continuous_index, long_MA.iloc[:, 0], label=f'{long_window}-day MA', alpha=0.8, linewidth=1.2, linestyle='--')

    # Plot buy signals
    buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
    buy_signals = [i for i in range(len(prices.index)) if prices.index[i] in buy_signals]
    ax.scatter(buy_signals, prices.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', lw=0, alpha=0.7)
    # Plot sell signals
    sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
    sell_signals = [i for i in range(len(prices.index)) if prices.index[i] in sell_signals]
    ax.scatter(sell_signals, prices.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', lw=0, alpha=0.7)

    ax.set_title('Simple Moving Average (SMA) Crossover Strategy', fontsize=14)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([long_window, len(formatted_index) - 1])

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    # Save the plot if a path is provided
    if save_path:
        print(IMAGE_PATH)
        fig.savefig(f"{save_path}/sma_strategy_signals.png", bbox_inches='tight')

    # Show the plot
    plt.show()


    # Plot the cumulative returns of the strategy vs. the asset
    indexes_strat_ret = strategy_cumulative.index
    if indexes_strat_ret[0].hour == 0 and indexes_strat_ret[0].minute == 0 and indexes_strat_ret[0].second == 0:
        formatted_index = indexes_strat_ret.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_strat_ret.strftime('%Y-%m-%d\n%H:%M:%S')
    
    continuous_index_ret = range(len(indexes_strat_ret))

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(continuous_index_ret, strategy_cumulative.iloc[:, 0], label=f'MA ({short_window}, {long_window}) Strategy Cumulative Return', linewidth=1.5)

    if benchmark_cumulative is not None:
        benchmark_cumulative = benchmark_cumulative.reindex(strategy_cumulative.index)
        benchmark_cumulative.fillna(method='ffill', inplace=True)
        benchmark_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_ret, benchmark_cumulative.iloc[:, 0], label='Benchmark Cumulative Return', linestyle='--', linewidth=1.2)
        ax.title('Cumulative Returns of the Simple Moving Average (SMA) Crossover Strategy vs. Benchmark', fontsize=14)
    else:
        returns_cumulative = returns_cumulative.reindex(strategy_cumulative.index)
        returns_cumulative.fillna(method='ffill', inplace=True)
        returns_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_ret, returns_cumulative.iloc[:, 0], label=f'{asset_name} Cumulative Return', linewidth=1.5)
        ax.set_title('Cumulative Returns of the Simple Moving Average (SMA) Crossover Strategy', fontsize=14)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index_ret[long_window:]) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax.axhline(0, color='darkgrey', linewidth=1, linestyle='-')  # Make the zero line thicker

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)
    if save_path:
        fig.savefig(f"{save_path}/sma_strategy_cumulative.png", bbox_inches='tight')
    plt.show()

    return


def calc_sma_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    short_window: 50,
    long_window: 200,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    image_save_path: str = IMAGE_PATH,
    strategy_name: str = 'SMA',
):
    """
    Creates a trade strategy returns or signals using a moving average crossover strategy by adjusting weights (-100%, or 100%)
    based on short and long moving averages for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices (not returns).
    short_window (int, default=50): Window for the short moving average.
    long_window (int, default=200): Window for the long moving average.
    benchmark_returns (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    image_save_path (str, default=None): Path to save the images. If None, images will not be saved.
    strategy_name (str, default='SMA'): Name for the strategy. If None, a name will be generated based on the moving average windows.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by moving average crossover strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if benchmark_returns is not None:
        benchmark_returns = time_series_to_df(benchmark_returns, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(benchmark_returns)  # Ensure index is datetime and data is in float format
        
    # Filter out any NaN values and sort the prices
    prices.dropna(inplace=True)
    prices.sort_index(inplace=True)

    # Calculate moving averages
    short_MA = prices.rolling(window=short_window).mean()
    long_MA = prices.rolling(window=long_window).mean()

    # Identify the crossover points
    signals = pd.DataFrame(np.where(short_MA > long_MA, 1.0, -1.0), index=long_MA.index, columns=prices.columns)

    positions = signals.copy()
    trade_signals = positions.diff()
    trade_signals = trade_signals.iloc[long_window:]

    # Calculate the strategy returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Plot strategy:
    if plot_strategy == True:
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        plot_sma_strategy(
            prices=prices,
            short_MA=short_MA,
            long_MA=long_MA,
            trade_signals=trade_signals,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            asset_name=asset_name,
            short_window=short_window,
            long_window=long_window,
            save_path=image_save_path
        )
    
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



def plot_boll_bands_strategy(prices: pd.DataFrame,
                             moving_average: pd.DataFrame,
                             upper_band: pd.DataFrame,
                             lower_band: pd.DataFrame,
                             trade_signals: pd.DataFrame,
                             strategy_returns: pd.DataFrame,
                             benchmark_returns: pd.DataFrame = None, 
                             asset_name: str = 'Asset',
                             ma_window: int = 20,
                             n_std_dev: int = 2,
                             save_path: str = None):
    """
    Plots the Bollinger Bands strategy and cumulative returns.

    Parameters:
    prices (pd.DataFrame): Adjusted prices of the asset.
    moving_average (pd.DataFrame): Moving average.
    upper_band (pd.DataFrame): Upper Bollinger Band.
    lower_band (pd.DataFrame): Lower Bollinger Band.
    trade_signals (pd.DataFrame): Trade signals generated by the strategy.
    strategy_cumulative (pd.DataFrame): Cumulative returns of the strategy.
    returns_cumulative (pd.DataFrame): Cumulative returns of the asset.
    benchmark_cumulative (pd.DataFrame, optional): Cumulative returns of the benchmark. Default is None.
    asset_name (str): Name of the asset being plotted. Default is 'Asset'.
    ma_window (int): Window for the moving average. Default is 20.
    n_std_dev (int): Number of standard deviations for the bands. Default is 2.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    None
    """
    
    # Calculate cumulative returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns_cumulative = returns_cumulative.iloc[ma_window:].dropna()

    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    else:
        benchmark_cumulative = None

    # Format indexes
    prices = prices.copy()
    moving_average = moving_average.copy()
    prices = prices.iloc[ma_window:]
    moving_average = moving_average.iloc[ma_window:]

    indexes_prices = prices.index
    if indexes_prices[0].hour == 0 and indexes_prices[0].minute == 0 and indexes_prices[0].second == 0:
        formatted_index = indexes_prices.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_prices.strftime('%Y-%m-%d\n%H:%M:%S')
    
    continuous_index = range(len(indexes_prices))

    # Plot the closing price and the Bollinger Bands
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    ax.plot(continuous_index, prices.iloc[:, 0], label=f'{asset_name} Adj. Prices', alpha=0.6, linewidth=1.5)
    ax.plot(continuous_index, moving_average.iloc[:, 0], label=f'{ma_window}-day MA', alpha=0.8, linewidth=1.2, linestyle='--')
    ax.plot(continuous_index, upper_band.iloc[:, 0], label=f'Upper Band ({n_std_dev} Std. Dev.)', color='purple', alpha=0.6, linewidth=1.2)
    ax.plot(continuous_index, lower_band.iloc[:, 0], label=f'Lower Band ({n_std_dev} Std. Dev.)', color='purple', alpha=0.6, linewidth=1.2)
    ax.fill_between(continuous_index, lower_band.iloc[:, 0], upper_band.iloc[:, 0], color='purple', alpha=0.15)

    # Plot buy signals
    buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
    buy_signals = [i for i in range(len(prices.index)) if prices.index[i] in buy_signals]
    ax.scatter(buy_signals, prices.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', lw=0, alpha=0.7)

    # Plot sell signals
    sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
    sell_signals = [i for i in range(len(prices.index)) if prices.index[i] in sell_signals]
    ax.scatter(sell_signals, prices.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', lw=0, alpha=0.7)

    ax.set_title('Bollinger Bands Crossover Strategy', fontsize=14)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([ma_window, len(formatted_index) - 1])

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"{save_path}/boll_bands_strategy_signals.png", bbox_inches='tight')

    # Show the plot
    plt.show()

    # Plot the cumulative returns of the strategy vs. the asset
    indexes_strat_ret = strategy_cumulative.index
    if indexes_strat_ret[0].hour == 0 and indexes_strat_ret[0].minute == 0 and indexes_strat_ret[0].second == 0:
        formatted_index = indexes_strat_ret.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_strat_ret.strftime('%Y-%m-%d\n%H:%M:%S')
    
    continuous_index_ret = range(len(indexes_strat_ret))

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(continuous_index_ret, strategy_cumulative.iloc[:, 0], label=f'Bollinger Bands ({ma_window}) Strategy Cumulative Return', linewidth=1.5)

    if benchmark_cumulative is not None:
        benchmark_cumulative = benchmark_cumulative.reindex(strategy_cumulative.index)
        benchmark_cumulative.fillna(method='ffill', inplace=True)
        benchmark_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_ret, benchmark_cumulative.iloc[:, 0], label='Benchmark Cumulative Return', linestyle='-', linewidth=1.2)
        ax.set_title('Cumulative Returns of the Bollinger Bands Strategy vs. Benchmark', fontsize=14)
    else:
        returns_cumulative = returns_cumulative.reindex(strategy_cumulative.index)
        returns_cumulative.fillna(method='ffill', inplace=True)
        returns_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_ret, returns_cumulative.iloc[:, 0], label=f'{asset_name} Cumulative Return', linewidth=1.5)
        ax.set_title('Cumulative Returns of the Bollinger Bands Strategy', fontsize=14)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index_ret) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax.axhline(0, color='darkgrey', linewidth=1, linestyle='-')

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    if save_path:
        fig.savefig(f"{save_path}/boll_bands_strategy_cumulative.png", bbox_inches='tight')
    plt.show()

    return

def calc_boll_bands_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    ma_window: int = 20,
    n_std_dev: int = 2,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'BB',
    image_save_path: str = IMAGE_PATH
):
    """
    Creates a trade strategy returns or signals using a Bollinger Bands strategy by adjusting weights (-100%, or 100%)
    based on short and long moving averages for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices (not returns).
    ma_window (int, default=20): Window for the moving average.
    n_std_dev (int, default=2): Number of standard deviations for the Bollinger Bands.
    benchmark_returns (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    strategy_name (str, default='SMA'): Name for the strategy. If None, a name will be generated based on the moving average windows.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by Bollinger Bands strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if benchmark_returns is not None:
        benchmark_returns = time_series_to_df(benchmark_returns, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(benchmark_returns)  # Ensure index is datetime and data is in float format
        
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

    # Calculate positions based on trade signals
    positions = pd.DataFrame(index=trade_signals_bb.index, columns=trade_signals_bb.columns)
    positions.iloc[0] = 1 if prices.iloc[ma_window, 0] <  moving_average.iloc[ma_window, 0] else -1 if prices.iloc[ma_window, 0] >  moving_average.iloc[ma_window, 0] else 0
    
    for i in range(1, len(trade_signals_bb)):
        if trade_signals_bb.iloc[i, 0] == 1:  # Long signal
            positions.iloc[i, 0] = 1
        elif trade_signals_bb.iloc[i, 0] == -1:  # Short signal
            positions.iloc[i, 0] = -1
        elif trade_signals_ma.iloc[i, 0] != 0:  # Crossing mean, zero position
            positions.iloc[i, 0] = 0
        else:  # Neutral signal
            positions.iloc[i, 0] = positions.iloc[i-1, 0]

    signals = positions.copy()

    trade_signals = positions.diff()
    trade_signals.fillna(positions.iloc[0], inplace=True)

    # Calculate the strategy returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Plot strategy:
    if plot_strategy == True:
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        plot_boll_bands_strategy(
            prices=prices,
            moving_average=moving_average,
            upper_band=upper_band,
            lower_band=lower_band,
            trade_signals=trade_signals,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            asset_name=asset_name,
            ma_window=ma_window,
            n_std_dev=n_std_dev,
            save_path=image_save_path
            )

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


def plot_ema_strategy(prices: pd.DataFrame,
                       short_EMA: pd.DataFrame,
                       long_EMA: pd.DataFrame,
                       trade_signals: pd.DataFrame,
                       strategy_returns: pd.DataFrame,
                       benchmark_returns: pd.DataFrame = None, 
                       asset_name: str = 'Asset',
                       short_window: int = 12,
                       long_window: int = 26,
                       save_path: str = None):
    """
    Plots the EWMA strategy and cumulative returns.

    Parameters:
    prices (pd.DataFrame): Adjusted prices of the asset.
    short_EMA (pd.DataFrame): Short exponential moving average.
    long_EMA (pd.DataFrame): Long exponential moving average.
    trade_signals (pd.DataFrame): Trade signals generated by the strategy.
    strategy_cumulative (pd.DataFrame): Cumulative returns of the strategy.
    returns_cumulative (pd.DataFrame): Cumulative returns of the asset.
    benchmark_cumulative (pd.DataFrame, optional): Cumulative returns of the benchmark. Default is None.
    asset_name (str): Name of the asset being plotted. Default is 'Asset'.
    short_window (int): Window for the short EMA. Default is 12.
    long_window (int): Window for the long EMA. Default is 26.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    None
    """
    
    # Calculate cumulative returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns_cumulative = returns_cumulative.iloc[long_window:].dropna()

    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    else:
        benchmark_cumulative = None

    # Format indexes
    indexes_prices = prices.index
    if indexes_prices[0].hour == 0 and indexes_prices[0].minute == 0 and indexes_prices[0].second == 0:
        formatted_index = indexes_prices.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_prices.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index = range(len(indexes_prices))

    # Plot the closing price and the EMAs
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    ax.plot(continuous_index, prices.iloc[:, 0], label=f'{asset_name} Adj. Prices', alpha=0.6, linewidth=1.5)
    ax.plot(continuous_index, short_EMA.iloc[:, 0], label=f'{short_window}-day EMA', alpha=0.8, linewidth=1.2, linestyle='--')
    ax.plot(continuous_index, long_EMA.iloc[:, 0], label=f'{long_window}-day EMA', alpha=0.8, linewidth=1.2, linestyle='--')

    # Plot buy signals
    buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
    buy_signals = [i for i in range(len(prices.index)) if prices.index[i] in buy_signals]
    ax.scatter(buy_signals, prices.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', lw=0, alpha=0.7)

    # Plot sell signals
    sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
    sell_signals = [i for i in range(len(prices.index)) if prices.index[i] in sell_signals]
    ax.scatter(sell_signals, prices.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', lw=0, alpha=0.7)

    ax.set_title('Exponential Moving Average (EMA) Crossover Strategy', fontsize=14)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"{save_path}/ewma_strategy_signals.png", bbox_inches='tight')

    plt.show()

    # Plot cumulative returns
    indexes_strategy = strategy_cumulative.index
    if indexes_strategy[0].hour == 0 and indexes_strategy[0].minute == 0 and indexes_strategy[0].second == 0:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index_cum = range(len(indexes_strategy))

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(continuous_index_cum, strategy_cumulative.iloc[:, 0], label=f'EMA ({short_window}, {long_window}) Strategy Cumulative Return', linewidth=1.5, color='blue')

    if benchmark_cumulative is not None:
        benchmark_cumulative = benchmark_cumulative.reindex(strategy_cumulative.index)
        benchmark_cumulative.fillna(method='ffill', inplace=True)
        benchmark_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, benchmark_cumulative.iloc[:, 0], label='Benchmark Cumulative Return', linestyle='--', linewidth=1.2, color='orange')
        ax.set_title('Cumulative Returns of the EMA Crossover Strategy vs. Benchmark')

    else:
        returns_cumulative = returns_cumulative.reindex(strategy_cumulative.index)
        returns_cumulative.fillna(method='ffill', inplace=True)
        returns_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, returns_cumulative.iloc[:, 0], label=f'{asset_name} Cumulative Return', linewidth=1.5, color='green')
        ax.set_title('Cumulative Returns of the EMA Crossover Strategy')

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index_cum) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))

    ax.axhline(0, color='darkgrey', linestyle='-', linewidth=1)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    if save_path:
        fig.savefig(f"{save_path}/ewma_strategy_cumulative.png", bbox_inches='tight')

    plt.show()

    return


def calc_ema_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    short_window: 12,
    long_window: 26,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'EMA',
    image_save_path: str = IMAGE_PATH
):
    """
    Creates a trade strategy returns or signals using a exponential moving average (EMA) crossover strategy by adjusting weights (-100%, or 100%)
    based on short and long moving averages for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices (not returns).
    short_window (int, default=12): Window for the short moving average.
    long_window (int, default=26): Window for the long moving average.
    benchmark_returns (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    strategy_name (str, default='EMA'): Name for the strategy. If None, a name will be generated based on the moving average windows.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by exponential moving average (EMA) crossover strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if benchmark_returns is not None:
        benchmark_returns = time_series_to_df(benchmark_returns, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(benchmark_returns)  # Ensure index is datetime and data is in float format
        
    # Filter out any NaN values and sort the prices
    prices.dropna(inplace=True)
    prices.sort_index(inplace=True)

    # Calculate moving averages
    short_EMA = prices.ewm(span=short_window, adjust=False).mean()
    long_EMA = prices.ewm(span=long_window, adjust=False).mean()

    # Identify the crossover points
    signals = pd.DataFrame(np.where(short_EMA > long_EMA, 1.0, -1.0), index=long_EMA.index, columns=prices.columns)

    positions = signals.copy()
    trade_signals = positions.diff()
    trade_signals = trade_signals.iloc[long_window:]

    # Calculate the strategy returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Plot strategy:
    if plot_strategy == True:
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        plot_ema_strategy(
            prices=prices,
            short_EMA=short_EMA,
            long_EMA=long_EMA,
            trade_signals=trade_signals,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            asset_name=asset_name,
            short_window=short_window,
            long_window=long_window,
            save_path=image_save_path
        )

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



def plot_macd_strategy(prices: pd.DataFrame,
                       macd: pd.DataFrame,
                       signal_line: pd.DataFrame,
                       trade_signals: pd.DataFrame,
                       strategy_returns: pd.DataFrame,
                       benchmark_returns: pd.DataFrame = None, 
                       asset_name: str = 'Asset',
                       short_window: int = 12,
                       long_window: int = 26,
                       signal_window: int = 9,
                       save_path: str = None):
    """
    Plots the MACD strategy and cumulative returns.

    Parameters:
    prices (pd.DataFrame): Adjusted prices of the asset.
    macd (pd.DataFrame): MACD values.
    signal_line (pd.DataFrame): Indicator Line for the MACD.
    trade_signals (pd.DataFrame): Trade signals generated by the strategy.
    strategy_cumulative (pd.DataFrame): Cumulative returns of the strategy.
    returns_cumulative (pd.DataFrame): Cumulative returns of the asset.
    benchmark_cumulative (pd.DataFrame, optional): Cumulative returns of the benchmark. Default is None.
    asset_name (str): Name of the asset being plotted. Default is 'Asset'.
    short_window (int): Window for the short EMA. Default is 12.
    long_window (int): Window for the long EMA. Default is 26.
    signal_window (int): Window for the Indicator Line EMA. Default is 9.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    None
    """

    # Calculate cumulative returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns_cumulative = returns_cumulative.iloc[long_window:].dropna()

    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    else:
        benchmark_cumulative = None

    # Format indexes
    prices = prices.iloc[long_window:].copy()
    macd = macd.iloc[long_window:].copy()
    signal_line = signal_line.iloc[long_window:].copy()
    indexes_prices = prices.index

    if indexes_prices[0].hour == 0 and indexes_prices[0].minute == 0 and indexes_prices[0].second == 0:
        formatted_index = indexes_prices.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_prices.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index = range(len(indexes_prices))

    # Plot the price and MACD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot prices
    ax1.plot(continuous_index, prices.iloc[:, 0], label=f'{asset_name} Adj. Prices', color='blue', alpha=0.6, linewidth=1.5)
    ax1.set_ylabel('Price')
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.set_title(f'{asset_name} Prices and MACD Strategy')
    ax1.legend(fontsize=10)

    # Plot buy and sell signals on the price plot
    buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
    buy_signals = [i for i in range(len(prices.index)) if prices.index[i] in buy_signals]
    ax1.scatter(buy_signals, prices.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', alpha=0.7)

    sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
    sell_signals = [i for i in range(len(prices.index)) if prices.index[i] in sell_signals]
    ax1.scatter(sell_signals, prices.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', alpha=0.7)

    # Plot MACD and Indicator Line
    ax2.plot(continuous_index, macd.iloc[:, 0], label=f'MACD ({short_window}, {long_window})', color='purple', alpha=0.75, linewidth=1.2)
    ax2.plot(continuous_index, signal_line.iloc[:, 0], label=f'Indicator Line ({signal_window})', color='green', alpha=0.75, linewidth=1.2)

    # Plot buy and sell signals on the MACD plot
    ax2.scatter(buy_signals, macd.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', alpha=0.7)
    ax2.scatter(sell_signals, macd.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', alpha=0.7)

    plt.title('Moving Average Convergence/Divergence (MACD) Indicator')
    ax2.set_ylabel('MACD')
    ax2.set_xlabel('Time')
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=10)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax2.set_xlim([0, len(formatted_index) - 1])

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"{save_path}/macd_strategy_signals.png", bbox_inches='tight')

    plt.show()

    # Plot cumulative returns
    indexes_strategy = strategy_cumulative.index
    if indexes_strategy[0].hour == 0 and indexes_strategy[0].minute == 0 and indexes_strategy[0].second == 0:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index_cum = range(len(indexes_strategy))

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(continuous_index_cum, strategy_cumulative.iloc[:, 0], label=f'MACD ({short_window}, {long_window}, {signal_window}) Strategy Cumulative Return', linewidth=1.5, color='blue')

    if benchmark_cumulative is not None:
        benchmark_cumulative = benchmark_cumulative.reindex(strategy_cumulative.index)
        benchmark_cumulative.fillna(method='ffill', inplace=True)
        benchmark_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, benchmark_cumulative.iloc[:, 0], label='Benchmark Cumulative Return', linestyle='--', linewidth=1.2, color='orange')
        ax.set_title('Cumulative Returns of MACD Strategy vs. Benchmark')
    else:
        returns_cumulative = returns_cumulative.reindex(strategy_cumulative.index)
        returns_cumulative.fillna(method='ffill', inplace=True)
        returns_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, returns_cumulative.iloc[:, 0], label=f'{asset_name} Cumulative Return', linewidth=1.5, color='green')
        ax.set_title('Cumulative Returns of MACD Strategy')

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index_cum) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax.axhline(0, color='darkgrey', linestyle='-', linewidth=1)

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    if save_path:
        fig.savefig(f"{save_path}/macd_strategy_cumulative.png", bbox_inches='tight')

    plt.show()
    return


def calc_macd_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    short_window: 12,
    long_window: 26,
    signal_window: 9,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
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
    benchmark_returns (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, +1). Else, returns the cumulative returns of the strategy
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals and plot the strategy returns.
    strategy_name (str, default='MACD'): Name for the strategy. If None, a name will be generated based on the moving average windows.

    Returns:
    pd.DataFrame: Strategy returns or trading signals by Moving Average Convergence/Divergence (MACD) strategy.
    """

    prices = time_series_to_df(prices, name='Prices')  # Convert prices to DataFrame if needed
    fix_dates_index(prices)  # Ensure index is datetime and data is in float format

    if benchmark_returns is not None:
        benchmark_returns = time_series_to_df(benchmark_returns, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(benchmark_returns)  # Ensure index is datetime and data is in float format
        
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

    positions = signals.copy()
    trade_signals = positions.diff()
    trade_signals = trade_signals.iloc[long_window:]


    # Calculate the strategy returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Plot strategy:
    if plot_strategy == True:
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")
        
        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        plot_macd_strategy(
            prices=prices,
            macd=macd,
            signal_line=signal_line,
            trade_signals=trade_signals,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            asset_name=asset_name,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
        )
    
    # Return trading signals, and moving averages
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                ma_short = short_EMA.rename(columns=lambda col: f"{col} (Short EMA)")
                ma_long = long_EMA.rename(columns=lambda col: f"{col} (Long EMA)")
                ma_signal_line = signal_line.rename(columns=lambda col: f"{col} (Indicator Line)")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                ma_short = short_EMA.rename(columns={short_EMA.columns[0]: f"Short EMA"})
                ma_long = long_EMA.rename(columns={long_EMA.columns[0]: f"Long EMA"})
                ma_signal_line = signal_line.rename(columns={signal_line.columns[0]: f"Indicator Line"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            ma_short = short_EMA.to_frame(name=f"Short EMA")
            ma_long = long_EMA.to_frame(name="Long EMA")
            ma_signal_line = signal_line.to_frame(name="Indicator Line")
        
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


def plot_rsi_strategy(prices: pd.DataFrame,
                      rsi: pd.DataFrame,
                      trade_signals: pd.DataFrame,
                      strategy_returns: pd.DataFrame,
                      benchmark_returns: pd.DataFrame = None, 
                      asset_name: str = 'Asset',
                      rsi_period: int = 14,
                      overbought: int = 70,
                      oversold: int = 30,
                      save_path: str = None):
    """
    Plots the RSI strategy and cumulative returns.

    Parameters:
    prices (pd.DataFrame): Adjusted prices of the asset.
    rsi (pd.DataFrame): RSI values.
    trade_signals (pd.DataFrame): Trade signals generated by the strategy.
    strategy_cumulative (pd.DataFrame): Cumulative returns of the strategy.
    returns_cumulative (pd.DataFrame): Cumulative returns of the asset.
    benchmark_cumulative (pd.DataFrame, optional): Cumulative returns of the benchmark. Default is None.
    asset_name (str): Name of the asset being plotted. Default is 'Asset'.
    rsi_period (int): Period for calculating RSI. Default is 14.
    overbought (int): RSI threshold for overbought condition. Default is 70.
    oversold (int): RSI threshold for oversold condition. Default is 30.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    None
    """

    # Calculate cumulative returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns_cumulative = returns_cumulative.iloc[rsi_period:].dropna()

    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    else:
        benchmark_cumulative = None

    # Format indexes
    prices = prices.iloc[rsi_period:].copy()
    indexes_prices = prices.index
    if indexes_prices[0].hour == 0 and indexes_prices[0].minute == 0 and indexes_prices[0].second == 0:
        formatted_index = indexes_prices.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_prices.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index = range(len(indexes_prices))

    # Plot the price and RSI
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot prices
    ax1.plot(continuous_index, prices.iloc[:, 0], label=f'{asset_name} Adj. Prices', color='blue', alpha=0.6, linewidth=1.5)
    ax1.set_ylabel('Price')
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.set_title(f'{asset_name} Prices and RSI Strategy')
    ax1.legend(fontsize=10)

    # Plot buy and sell signals on the price plot
    buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
    buy_signals = [i for i in range(len(prices.index)) if prices.index[i] in buy_signals]
    ax1.scatter(buy_signals, prices.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', alpha=0.7)

    sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
    sell_signals = [i for i in range(len(prices.index)) if prices.index[i] in sell_signals]
    ax1.scatter(sell_signals, prices.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', alpha=0.7)

    # Plot RSI
    ax2.plot(continuous_index, rsi.iloc[:, 0], label='RSI', color='purple', alpha=0.75, linewidth=1.2)
    ax2.axhline(y=overbought, color='red', linestyle='--', label='Overbought Level (70)')
    ax2.axhline(y=oversold, color='green', linestyle='--', label='Oversold Level (30)')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=10)

    # Plot buy and sell signals on the RSI plot
    ax2.scatter(buy_signals, rsi.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', alpha=0.7)
    ax2.scatter(sell_signals, rsi.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', alpha=0.7)


    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax2.set_xlim([0, len(formatted_index) - 1])

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"{save_path}/rsi_strategy_signals.png", bbox_inches='tight')

    plt.show()

    # Plot cumulative returns
    indexes_strategy = strategy_cumulative.index
    if indexes_strategy[0].hour == 0 and indexes_strategy[0].minute == 0 and indexes_strategy[0].second == 0:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d\n%H:%M:%S')

    continuous_index_cum = range(len(indexes_strategy))

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(continuous_index_cum, strategy_cumulative.iloc[:, 0], label=f'RSI Strategy ({rsi_period}) Cumulative Return', linewidth=1.5, color='blue')

    if benchmark_cumulative is not None:
        benchmark_cumulative = benchmark_cumulative.reindex(strategy_cumulative.index)
        benchmark_cumulative.fillna(method='ffill', inplace=True)
        benchmark_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, benchmark_cumulative.iloc[:, 0], label='Benchmark Cumulative Return', linestyle='-', linewidth=1.2, color='orange')
        ax.set_title('Cumulative Returns of the RSI Strategy vs. Benchmark')
    else:
        returns_cumulative = returns_cumulative.reindex(strategy_cumulative.index)
        returns_cumulative.fillna(method='ffill', inplace=True)
        returns_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, returns_cumulative.iloc[:, 0], label=f'{asset_name} Cumulative Return', linewidth=1.5, color='green')
        ax.set_title('Cumulative Returns of the RSI Strategy')

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index_cum) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax.axhline(0, color='darkgrey', linestyle='-', linewidth=1)

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    if save_path:
        fig.savefig(f"{save_path}/rsi_strategy_cumulative.png", bbox_inches='tight')

    plt.show()

    return

def calc_rsi_strategy(
    prices: Union[pd.DataFrame, List[pd.Series]],
    rsi_period: int = 14,
    overbought: int = 70,
    oversold: int = 30,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'RSI',
    image_save_path: str = IMAGE_PATH
):
    """
    Creates a trade strategy using an RSI (Relative Strength Index) strategy by adjusting weights (-100%, 0, or 100%) 
    based on RSI values for each asset.

    Parameters:
    prices (pd.DataFrame or List of pd.Series): Time series of asset prices.
    rsi_period (int, default=14): Period for calculating RSI.
    overbought (int, default=70): RSI threshold for overbought condition.
    oversold (int, default=30): RSI threshold for oversold condition.
    benchmark_returns (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, 0, +1) and the RSI values. Else, returns the cumulative returns of the strategy.
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals, and plot the cumulative returns of the strategy.
    strategy_name (str, default='RSI'): Name for the strategy. If None, a name will be generated based on the RSI period.
    image_save_path (str, default=None): Path to save the plots. Default is None.
    
    Returns:
    pd.DataFrame: Strategy returns, trading signals, or RSI values based on parameters.
    """

    prices = time_series_to_df(prices, name='Prices') # Convert benchmark to DataFrame if needed
    fix_dates_index(prices) # Ensure index is datetime and data is in float format

    if benchmark_returns is not None:
        benchmark_returns = time_series_to_df(benchmark_returns, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(benchmark_returns)  # Ensure index is datetime and data is in float format
    
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

    positions = signals.copy()
    trade_signals = positions.diff()

    # Calculate returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)
    
    # Plot strategy
    if plot_strategy:
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")

        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        plot_rsi_strategy(
            prices=prices,
            rsi=rsi,
            trade_signals=trade_signals,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            asset_name=asset_name,
            rsi_period=rsi_period,
            overbought=overbought,
            oversold=oversold,
            save_path=image_save_path
        )

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


def plot_mfi_strategy(prices: pd.DataFrame,
                      mfi: pd.DataFrame,
                      trade_signals: pd.DataFrame,
                      strategy_returns: pd.DataFrame,
                      benchmark_returns: pd.DataFrame = None, 
                      asset_name: str = 'Asset',
                      mfi_period: int = 14,
                      overbought: int = 80,
                      oversold: int = 20,
                      save_path: str = None):
    """
    Plots the MFI strategy and cumulative returns.

    Parameters:
    prices (pd.DataFrame): Adjusted prices of the asset.
    mfi (pd.DataFrame): MFI values.
    trade_signals (pd.DataFrame): Trade signals generated by the strategy.
    strategy_cumulative (pd.DataFrame): Cumulative returns of the strategy.
    returns_cumulative (pd.DataFrame): Cumulative returns of the asset.
    benchmark_cumulative (pd.DataFrame, optional): Cumulative returns of the benchmark. Default is None.
    asset_name (str): Name of the asset being plotted. Default is 'Asset'.
    mfi_period (int): Period for calculating MFI. Default is 14.
    overbought (int): MFI threshold for overbought condition. Default is 80.
    oversold (int): MFI threshold for oversold condition. Default is 20.
    save_path (str, optional): Path to save the plots. Default is None.

    Returns:
    None
    """

    # Calculate cumulative returns
    returns = prices.pct_change()
    returns_cumulative = (1 + returns).cumprod() - 1
    returns_cumulative = returns_cumulative.iloc[mfi_period:].dropna()

    strategy_cumulative = (1 + strategy_returns).cumprod() - 1

    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    else:
        benchmark_cumulative = None

    # Format indexes
    prices = prices.iloc[mfi_period:].copy()
    indexes_prices = prices.index
    if indexes_prices[0].hour == 0 and indexes_prices[0].minute == 0 and indexes_prices[0].second == 0:
        formatted_index = indexes_prices.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_prices.strftime('%Y-%m-%d\n%H:%M:%S')
    
    continuous_index = range(len(indexes_prices))

    # Plot the price and MFI
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot prices
    ax1.plot(continuous_index, prices.iloc[:, 0], label=f'{asset_name} Adj. Prices', color='blue', alpha=0.6, linewidth=1.5)
    ax1.set_ylabel('Price')
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.set_title(f'{asset_name} Prices and MFI Strategy')
    ax1.legend(fontsize=10)

    # Plot buy and sell signals on the price plot
    buy_signals = trade_signals[trade_signals.iloc[:, 0] > 0].index
    buy_signals = [i for i in range(len(prices.index)) if prices.index[i] in buy_signals]
    ax1.scatter(buy_signals, prices.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', alpha=0.7)

    sell_signals = trade_signals[trade_signals.iloc[:, 0] < 0].index
    sell_signals = [i for i in range(len(prices.index)) if prices.index[i] in sell_signals]

    ax1.scatter(sell_signals, prices.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', alpha=0.7)

    # Plot MFI
    ax2.plot(continuous_index, mfi.iloc[:, 0], label='MFI', color='purple', alpha=0.75, linewidth=1.2)
    ax2.axhline(y=overbought, color='red', linestyle='--', label='Overbought Level (80)')
    ax2.axhline(y=oversold, color='green', linestyle='--', label='Oversold Level (20)')
    ax2.set_ylabel('MFI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=10)

    # Plot buy and sell signals on the MFI plot
    ax2.scatter(buy_signals, mfi.iloc[:, 0][buy_signals], marker='^', color='green', label='Buy Signal', alpha=0.7)
    ax2.scatter(sell_signals, mfi.iloc[:, 0][sell_signals], marker='v', color='red', label='Sell Signal', alpha=0.7)

    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax2.set_xlim([0, len(formatted_index) - 1])

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"{save_path}/mfi_strategy_signals.png", bbox_inches='tight')
    
    plt.show()

    # Plot cumulative returns
    indexes_strategy = strategy_cumulative.index
    if indexes_strategy[0].hour == 0 and indexes_strategy[0].minute == 0 and indexes_strategy[0].second == 0:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d')
    else:
        formatted_index = indexes_strategy.strftime('%Y-%m-%d\n%H:%M:%S')
    
    continuous_index_cum = range(len(indexes_strategy))

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(continuous_index_cum, strategy_cumulative.iloc[:, 0], label=f'MFI Strategy ({mfi_period}) Cumulative Return', linewidth=1.5, color='blue')

    if benchmark_cumulative is not None:
        benchmark_cumulative = benchmark_cumulative.reindex(strategy_cumulative.index)
        benchmark_cumulative.fillna(method='ffill', inplace=True)
        benchmark_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, benchmark_cumulative.iloc[:, 0], label='Benchmark Cumulative Return', linestyle='--', linewidth=1.2, color='orange')
        ax.set_title('Cumulative Returns of the MFI Strategy vs. Benchmark')
    else:
        returns_cumulative = returns_cumulative.reindex(strategy_cumulative.index)
        returns_cumulative.fillna(method='ffill', inplace=True)
        returns_cumulative.fillna(0, inplace=True)
        ax.plot(continuous_index_cum, returns_cumulative.iloc[:, 0], label=f'{asset_name} Cumulative Return', linewidth=1.5, color='green')
        ax.set_title('Cumulative Returns of the MFI Strategy')
    
    num_ticks = 20
    tick_indices = np.linspace(0, len(continuous_index_cum) - 1, num=num_ticks, dtype=int)
    tick_labels = [formatted_index[i] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlim([0, len(formatted_index) - 1])

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax.axhline(0, color='darkgrey', linestyle='-', linewidth=1)

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)
    
    if save_path:
        fig.savefig(f"{save_path}/mfi_strategy_cumulative.png", bbox_inches='tight')
    
    plt.show()

    return


def calculate_mfi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Money Flow Index (MFI) for a given stock data DataFrame.

    Parameters:
    prices_df (pd.DataFrame): DataFrame with columns 'High', 'Low', 'Adj Close' (or 'Close'), and 'Volume'.
    period (int): Period for MFI calculation (default is 14).

    Returns:
    pd.Series: MFI values for the given DataFrame.
    """
    # Validate and prepare data
    data = prices_df.copy()
    data.columns = [col.lower() for col in data.columns]
    if 'adj close' in data.columns:
        data.dropna(subset=['close'], inplace=True)
        data.rename(columns={'adj close': 'close'}, inplace=True)

    required_columns = {'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

    # Calculate MFI values
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    raw_money_flow = typical_price * data['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_flow_sum = positive_flow.rolling(window=period).sum()
    negative_flow_sum = negative_flow.rolling(window=period).sum()

    money_flow_ratio = positive_flow_sum / negative_flow_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))

    return mfi


def calc_mfi_strategy(
    prices_df: pd.DataFrame,
    mfi_period: int = 14,
    overbought: int = 80,
    oversold: int = 20,
    benchmark_returns: Union[pd.Series, pd.DataFrame] = None,
    return_signals: bool = False,
    plot_strategy: bool = True,
    strategy_name: str = 'MFI',
    image_save_path: str = IMAGE_PATH
):
    """
    Creates a trade strategy using a Money Flow Index (MFI) strategy by adjusting weights (-100%, 0, or 100%) 
    based on MFI values for each asset.

    Parameters:
    prices_df (pd.DataFrame): DataFrame with columns 'High', 'Low', 'Adj Close' (or 'Close'), and 'Volume'.
    mfi_period (int, default=14): Period for calculating MFI.
    overbought (int, default=80): MFI threshold for overbought condition.
    oversold (int, default=20): MFI threshold for oversold condition.
    benchmark_returns (pd.Series or pd.DataFrame, default=None): Benchmark returns to compare the strategy returns to.
    return_signals (bool, default=False): If True, return the strategy signals (-1, 0, +1) and the MFI values. Else, returns the cumulative returns of the strategy.
    plot_strategy (bool, default=True): If True, plot the asset prices and trading signals, and plot the cumulative returns of the strategy.
    strategy_name (str, default='MFI'): Name for the strategy. If None, a name will be generated based on the MFI period.
    image_save_path (str, default=None): Path to save the plots. Default is None.
    
    Returns:
    pd.DataFrame: Strategy returns, trading signals, or MFI values based on parameters.
    """
    
    prices_df = time_series_to_df(prices_df, name='Prices') # Convert benchmark to DataFrame if needed
    fix_dates_index(prices_df) # Ensure index is datetime and data is in float format

    if benchmark_returns is not None:
        benchmark_returns = time_series_to_df(benchmark_returns, name='Benchmark Returns')  # Convert benchmark to DataFrame if needed
        fix_dates_index(benchmark_returns)  # Ensure index is datetime and data is in float format
    
    prices_df.columns = [col.lower() for col in prices_df.columns]
    if 'adj close' in prices_df.columns:
        prices_df.drop(columns=['close'], inplace=True)
        prices_df.rename(columns={'adj close': 'close'}, inplace=True)
    
    # Ensure prices have the necessary columns
    required_columns = {'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(prices_df.columns):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")
    
    prices = prices_df['close'].to_frame()
    
    # Calculate MFI
    mfi = calculate_mfi(prices_df, period=mfi_period).to_frame(name='MFI')

    mfi = mfi.reindex(prices.index)
    mfi = mfi.iloc[mfi_period:]

    # Generate buy/sell signals based on MFI thresholds
    signals = pd.DataFrame(np.where(mfi < oversold, 1.0, np.where(mfi > overbought, -1.0, 0.0)),
                           index=prices.index[mfi_period:],
                           columns=prices.columns)
    
    positions = signals.copy()
    trade_signals = positions.diff()

    # Calculate returns
    returns = prices.pct_change().reindex(positions.index)
    positions.columns = returns.columns
    strategy_returns = returns * positions.shift() # shift to avoid look-ahead bias 
    strategy_returns.dropna(inplace=True)

    # Plot strategy
    if plot_strategy:
        if prices.shape[1] > 1:
            print("Too many assets. Plotting strategy for the first asset only.")

        if isinstance(prices, pd.DataFrame):
            asset_name = prices.columns[0]
        elif isinstance(prices, pd.Series):
            asset_name = prices.name
        else:
            asset_name = 'Asset'

        plot_mfi_strategy(
            prices=prices,
            mfi=mfi,
            trade_signals=trade_signals,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            asset_name=asset_name,
            mfi_period=mfi_period,
            overbought=overbought,
            oversold=oversold,
            save_path=image_save_path
        )

    # Return trading signals, and MFI indicator
    if return_signals:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] > 1:
                signals = signals.rename(columns=lambda col: f"{col} ({strategy_name} Signals)")
                mfi = mfi.rename(columns=lambda col: f"{col} ({strategy_name} Indicator)")
            else:
                signals = signals.rename(columns={signals.columns[0]: f"{strategy_name} Signals"})
                mfi = mfi.rename(columns={mfi.columns[0]: f"Indicator ({strategy_name})"})
        else:
            signals = signals.to_frame(name=f"{strategy_name} Signals")
            mfi = mfi.to_frame(name=f"{strategy_name} Indicator")
        
        output_df = signals.copy()
        output_df = output_df.merge(mfi, left_index=True, right_index=True, how='inner')

        return output_df
    
    # Return cumulative returns of the strategy if not returning signals/MFI
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] > 1:
            strategy_returns = strategy_returns.rename(columns=lambda col: f"{col} ({strategy_name})")
        else:
            strategy_returns = strategy_returns.rename(columns={strategy_returns.columns[0]: f"{strategy_name} Returns"})
    else:
        strategy_returns = strategy_returns.to_frame(name=f"{strategy_name} Returns")
    
    return strategy_returns

