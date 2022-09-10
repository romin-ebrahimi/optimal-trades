import numpy as np
import pandas as pd

def back_test(df: pd.DataFrame,
              decimal_pip: int = 5,
              threshold: float = 0.5) -> pd.DataFrame:
    """
    Args: 
     df: A dataframe with timestamp, price, probability long,
         probability short, and target signal column. 
     decimal_pip: The decimal place representing 1/10 pip,
         which is used for scaling the price changes.
         e.g. EURUSD is 5 where 0.00001 is 1/10 pip.
     threshold: The probability threshold for trade entry.
     
    Returns: 
        A pandas dataframe of the simulated trading 
        entry/exit points and the expected trading returns 
        over time.
    """
    df_out = df[['time','close']].copy() # time, close price, signal, trade delta
    df_out['signal'] = 0 # signal (S,L,C) maps to (-1,1,0)
    df_out.loc[df.L > threshold,'signal'] = 1
    df_out.loc[df.S > threshold,'signal'] = -1
    df_out.at[df_out.index[-1],'signal'] = 0 # end of sample force close
    df_out['trade_delta'] = round(df['close'].diff().shift(-1)*df_out['signal'], 
                                  decimal_pip) # ignores market frictions
    df_out = df_out.fillna(0) # last element is NaN
    
    return df_out

def entry_exit(df: pd.DataFrame,
               threshold: float = 0.5) -> pd.DataFrame:
    """
    Given input data needed for back_test function, return a dataframe
    where each row is a trade entry or exit time stamp and position.
    """
    out = pd.DataFrame(columns = ['time','close','trade','open_close'])
    state = 0
    for i in range(df.shape[0]):
        if df.at[i,'L'] <=  threshold and state == 1: # close long
            temp = pd.DataFrame(df.loc[i,['time','close']].copy()).T
            temp['trade'] = 'sell'
            temp['open_close'] = 'close long'
            out = pd.concat([out, temp])
            state = 0
        elif df.at[i,'S'] <=  threshold and state == -1:
            temp = pd.DataFrame(df.loc[i,['time','close']].copy()).T
            temp['trade'] = 'buy'
            temp['open_close'] = 'close short'
            out = pd.concat([out, temp])
            state = 0
        
        if df.at[i,'L'] > threshold and state == 0: # open long
            temp = pd.DataFrame(df.loc[i,['time','close']].copy()).T
            temp['trade'] = 'buy'
            temp['open_close'] = 'open long'
            out = pd.concat([out, temp])
            state = 1
        elif df.at[i,'S'] > threshold and state == 0: # open short
            temp = pd.DataFrame(df.loc[i,['time','close']].copy()).T
            temp['trade'] = 'sell'
            temp['open_close'] = 'open short'
            out = pd.concat([out, temp])
            state = -1
    
    if state == 1:
        temp = pd.DataFrame(df.loc[(df.shape[0]-1),['time','close']].copy()).T
        temp['trade'] = 'sell'
        temp['open_close'] = 'close long'
        out = pd.concat([out, temp])
        state = 0
    elif state == -1:
        temp = pd.DataFrame(df.loc[(df.shape[0]-1),['time','close']].copy()).T
        temp['trade'] = 'buy'
        temp['open_close'] = 'close short'
        out = pd.concat([out, temp])
        state = 0
    
    assert state == 0 # no open positions at the end
    
    return out

def plot_returns(df: pd.DataFrame,
                 decimal_pip: int = 5) -> None:
    "Plot the distribution of trading returns."
    rets_l = _get_returns(df,
                          decimal_pip = decimal_pip,
                          is_long = True)
    rets_s = _get_returns(df,
                          decimal_pip = decimal_pip,
                          is_long = False)
    rets_l = [float(sum(ret)) for ret in rets_l]
    rets_s = [float(sum(ret)) for ret in rets_s]
    
    rets_l = pd.DataFrame(rets_l, columns = ['trade level returns'])
    rets_s = pd.DataFrame(rets_s, columns = ['trade level returns'])
    rets_l['trade direction'] = 'long'
    rets_s['trade_direction'] = 'short'
    
    sb.set(rc = {'figure.figsize':(13,7)})
    sb.displot(data = pd.concat([rets_l, rets_s], ignore_index = True),
               x = 'trade level returns',
               hue = 'trade direction',
               stat = 'density')
    
    return None

def _get_returns(df: pd.DataFrame,
                 threshold: float = 0.5,
                 decimal_pip: int = 5,
                 is_long = True) -> list:
    "Given trades, calculates the return series in bps."
    rets = [] # list of return series
    ee = entry_exit(df, threshold = threshold)
    
    if is_long:
        direction = 1.0
        idx_open = df[df.time.isin(ee[ee.open_close == 'open long'].time)].index.values.astype('int') 
        idx_close = df[df.time.isin(ee[ee.open_close == 'close long'].time)].index.values.astype('int')
    else:
        direction = -1.0
        idx_open = df[df.time.isin(ee[ee.open_close == 'open short'].time)].index.values.astype('int')
        idx_close = df[df.time.isin(ee[ee.open_close == 'close short'].time)].index.values.astype('int')
    
    for i in range(len(idx_open)):
        diff_series = df[idx_open[i]:(idx_close[i]+1)].close.diff().shift(-1)
        diff_series = round(diff_series*direction, decimal_pip).dropna() # trade return series rounded
       
        rets.append((diff_series / (10**-(decimal_pip-1))) / 4) # scale diff series into bps
    
    return rets

def _get_accuracy(rets: list) -> float:
    "Given trade returns, calculate trade accuracy."
    rets_sum = [sum(r) for r in rets]
    hits = sum([r > 0 for r in rets_sum])
    
    if len(rets) > 0:
        acc = round(hits / len(rets), 3)
    else:
        acc = np.nan
    
    return acc

def _get_drawdown(df_bt: pd.DataFrame) -> tuple:
    """
    Given full backtest trade deltas, calculate worst 
    drawdown percentage and the duration of the worst 
    drawdown in days.
    
    Args: 
        df_bt: A dataframe with the expected trading 
            returns over time.
    Returns:
        A tuple with the max drawdown and the number
        of days measuring the drawdown duration.
    """
    c_sum = list(df_bt.trade_delta.cumsum()) # cumulative sum of returns
    c_max = list(df_bt.trade_delta.cumsum().cummax()) # max of c_sum
    dds = [None]*len(c_sum) # list of drawdowns
    for i in range(len(dds)):
        dds[i] = c_sum[i] - c_max[i] # drawdown
        
    dd_max = min(dds) # max drawdown is the smallest drawdown
    
    idx_1 = dds.index(dd_max) # end index of drawdown
    idx_0 = c_max.index(c_max[idx_1]) # start index of drawdown
    
    n_days = (df_bt.at[idx_1,'time'] - df_bt.at[idx_0,'time']).days
    
    return (dd_max, n_days)

def trade_stats(df: pd.DataFrame,
                decimal_pip: int = 5,
                threshold: float = 0.5,
                fee_bps: int = 3,
                ANNUAL_TRADE_DAYS: float = 260) -> dict:
    """
    Calculate the trading statistics from output probabilities.
    
    Args:
        df: Dataframe containing price data and output probabilties 
            from a trading model for long and short.
        decimal_pip: The decimal place representing 1/10 pip,
            which is used for scaling the price changes.
            e.g. EURUSD is 5 where 0.00001 is 1/10 pip.
        threshold: The probability threshold for trade entry/exit.
        fee_bps: Total cost in bps of trade entry and exit.
        ANNUAL_TRADE_DAYS: FX market hours Sun 5p - Fri 5p ET,
            which translates to 260 trading days. If testing
            futures or equities, adjust this accordingly.
    
    Returns:
      A dictionary containing:
          1. E[annual return]
          2. downside standard deviation
          3. standard deviation 
          4. total sample return
          5. average return of long trades (bps)
          6. average return of short trades (bps)
          7. total sample trade count
          8. long trade accuracy
          9. short trade accuracy
          10. long trade count
          11. short trade count
          12. max drawdown (bps)
          13. max drawdown duration in days
    """
    stats = dict() # TODO: add trading friction from fee_bps
    
    bt = back_test(df, 
                   decimal_pip = decimal_pip,
                   threshold = threshold) # overall stats
    total_return = bt.trade_delta.sum() # differences are pips and thus additive
    
    n_days = (df.at[(df.shape[0] - 1),'time'] - df.at[0,'time']).days # number of days in sample
    annual_return = total_return*(ANNUAL_TRADE_DAYS/n_days) # annualized return
    std_scaler = (ANNUAL_TRADE_DAYS*(bt.shape[0]/n_days))**0.5 # annualize sdev
    std_deviation = np.std(bt.trade_delta)*std_scaler # annualized std deviation
    downside_delta = bt.trade_delta.copy()
    downside_delta[downside_delta > 0] = 0
    std_downside = np.std(downside_delta)*std_scaler # annualized downside std dev
    
    if std_deviation > 0:
        sharpe = round(annual_return / std_deviation, 2) # sharpe ratio
    else:
        sharpe = np.nan
    
    if std_downside > 0:
        sortino = round(annual_return / std_downside, 2) # sortino ratio
    else:
        sortino = np.nan
    
    rets_l = _get_returns(df,
                          threshold = threshold,
                          decimal_pip = decimal_pip,
                          is_long = True)
    rets_s = _get_returns(df,
                          threshold = threshold,
                          decimal_pip = decimal_pip,
                          is_long = False)
    trades_l = len(rets_l)
    trades_s = len(rets_s)
    trades_total = trades_s + trades_l
    
    if trades_l > 0:
        avg_l = round(sum([sum(ret) for ret in rets_l]) / float(trades_l), 0) # 1 basis point minimum
    else:
        avg_l = np.nan
    
    if trades_s > 0:
        avg_s = round(sum([sum(ret) for ret in rets_s]) / float(trades_s), 0)
    else:
        avg_s = np.nan
    
    acc_long = _get_accuracy(rets = rets_l)
    acc_short = _get_accuracy(rets = rets_s)
    acc_total = _get_accuracy(rets = rets_l + rets_s)
    
    dd = _get_drawdown(df_bt = bt)
    
    stats['annual_return'] = round(annual_return, 3)
    stats['sharpe'] = sharpe
    stats['sortino'] = sortino
    stats['std_deviation'] = round(std_deviation, 3)
    stats['total_return'] = round(total_return, 4)
    stats['trade_count_long'] = trades_l
    stats['trade_count_short'] = trades_s
    stats['trade_count_total'] = trades_total
    stats['average_return_long'] = avg_l
    stats['average_return_short'] = avg_s
    stats['accuracy_long'] = acc_long
    stats['accuracy_short'] = acc_short
    stats['accuracy_total'] = acc_total
    stats['max_drawdown'] = dd[0]
    stats['max_drawdown_days'] = dd[1]
    
    return stats

def target_optimal(df_price: pd.DataFrame, 
                   fee_bps: int = 3,
                   dd_bps: int = 0,
                   decimal_pip: int = 5) -> pd.DataFrame:
    """
    Use dynamic programming (Kadane's Algorithm) to find 
    the optimal target labels. Each trade is penalized by 
    fee_bps. Output is mapped into [0,1,2] for short, long, 
    and close (S,L,C) respectively. The drawdown constraint
    prevents any trade from having a drawdown larger than
    the given dd_bps.
    
    Args:
        df_price: Dataframe of input prices.
        fee_bps: Cost of trade entry and exit, which includes
            the expected slippage.
        dd_bps: Maximum allowable trade drawdown in bps.
        decimal_pip: The decimal place representing 1/10 pip,
            which is used for scaling the price changes.
            e.g. EURUSD is 5 where 0.00001 is 1/10 pip.
    
    Returns:
        Series containing optimal trades mapped into [0,1,2]
    """
    opt = df_price.copy() # output series
    opt.name = f"dd_{dd_bps}" 
    opt[:] = 0 # initialize all to zero
    n = df_price.shape[0]
    
    if n <= 1:
        raise ValueError("df_price requires more than 1 observation.")
    
    idx_buy = 0
    idx_max = 0
    buy_price = (df_price[0] / (10**-(decimal_pip-1))) + fee_bps # bps
    max_price = buy_price # for calculating trade max drawdown
    
    for i in range(1,n): # calculate optimal long trades
        if buy_price >= ((df_price[i] / (10**-(decimal_pip-1))) + fee_bps): # relaxed constraint
            idx_buy = i # reset trade open
            idx_max = i # reset max price
            buy_price = (df_price[i] / (10**-(decimal_pip-1))) + fee_bps
            max_price = buy_price
        elif max_price < (df_price[i] / (10**-(decimal_pip-1))):
            idx_max = i # reset max price
            max_price = (df_price[i] / (10**-(decimal_pip-1))) # higher max_price
        elif max_price - ((df_price[i] / (10**-(decimal_pip-1)))) > dd_bps: # max drawdown constraint
            if idx_buy != idx_max:
                opt[idx_buy:(idx_max + 1)] = 1 # close long trade
                
            idx_buy = i # reset trade open
            idx_max = i # reset max price
            buy_price = (df_price[i] / (10**-(decimal_pip-1))) + fee_bps
            max_price = buy_price
    
    idx_sell = 0
    idx_min = 0
    sell_price = (df_price[0] / (10**-(decimal_pip-1))) - fee_bps # bps
    min_price = sell_price # for calculating trade max drawdown
    
    for i in range(1,n): # calculate optimal short trades
        if sell_price <= ((df_price[i] / (10**-(decimal_pip-1))) - fee_bps): # relaxed constraint
            idx_sell = i # reset trade open
            idx_min = i # reset min price
            sell_price = (df_price[i] / (10**-(decimal_pip-1))) - fee_bps
            min_price = sell_price
        elif min_price > (df_price[i] / (10**-(decimal_pip-1))):
            idx_min = i # reset min price
            min_price = (df_price[i] / (10**-(decimal_pip-1))) # lower min_price
        elif max_price - ((df_price[i] / (10**-(decimal_pip-1)))) > dd_bps: # max drawdown constraint
            if idx_sell != idx_min:
                opt[idx_sell:(idx_min + 1)] = -1 # close short trade
            
            idx_sell = i # reset trade open
            idx_min = i # reset min price
            sell_price = (df_price[i] / (10**-(decimal_pip-1))) - fee_bps
            min_price = sell_price
    
    opt.loc[opt == 0] = 2 # map close
    opt.loc[opt < 0] = 0 # short
    
    return opt
