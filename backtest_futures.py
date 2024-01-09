import numpy as np
import pandas as pd


class BackTest:
    def __init__(
        self,
        data: pd.DataFrame,
        min_increment: float = 0.01,
        fee: float = 0.0,
        threshold: float = 0.5,
    ):
        self._data = data
        self._min_increment = min_increment
        self._threshold = threshold
        self._fee = fee
        self._len = len(data)

    def back_test(self) -> pd.DataFrame:
        """
        Args:
         data: A dataframe with column names time, close,
             L, and S mapping to timestamp, price, probability long,
             and probability short.
         min_increment: The minimum discrete change allowed for the
             given futures contract.
             e.g. CL - WTI Crude is 0.01 nominally.
         fee: Expected trading friction per trade. This must be in
             terms of min_increment or an error will be raised.
             e.g. for CL, the fee must be a multiple of 0.01.
         threshold: The probability threshold for trade entry.

        Returns:
            A pandas dataframe of the simulated trading
            entry/exit points and the expected trading returns
            over time.
        """
        if self._fee % self._min_increment != 0:
            raise ValueError("fee must be a multiple of min_increment.")

        _ol = "open long"
        _os = "open short"
        # Columns are: time, close price, signal, trade delta.
        df_out = self._data[["time", "close"]].copy()
        # Column signal [S, L, C] maps to [-1, 1, 0].
        df_out["signal"] = 0

        # Conditions for not long and not short are useful for the 3 class
        # example where there is a risk of both L and S being > threshold.
        condition_long = self._data.L > self._threshold
        condition_not_short = self._data.S < 1 - self._threshold
        condition_short = self._data.S > self._threshold
        condition_not_long = self._data.L < 1 - self._threshold

        df_out.loc[condition_long & condition_not_short, "signal"] = 1
        df_out.loc[condition_short & condition_not_long, "signal"] = -1

        # Force close positions at the end of the sample.
        df_out.at[df_out.index[-1], "signal"] = 0
        df_out["trade_delta"] = (
            self._data["close"].diff().shift(-1) * df_out["signal"]
        )
        # Log returns are additive and an approximation of nominal returns.
        df_out["log_returns"] = (
            np.log(self._data["close"]).diff().shift(-1) * df_out["signal"]
        )
        df_out = df_out.fillna(0)

        ee = self.entry_exit()
        idx_open_l = self._data[
            self._data.time.isin(ee[ee.open_close == _ol].time)
        ].index.values.astype("int")

        idx_open_s = self._data[
            self._data.time.isin(ee[ee.open_close == _os].time)
        ].index.values.astype("int")

        df_out.loc[idx_open_l, "trade_delta"] -= self._fee
        df_out.loc[idx_open_s, "trade_delta"] -= self._fee
        # Make sure trade deltas do not go above the minimum increment.
        df_out["trade_delta"] = round(
            df_out.trade_delta, len(str(self._min_increment)) - 2
        )

        idx_full = np.concatenate([idx_open_l, idx_open_s])
        fee_bps = np.mean(self._fee / df_out.loc[idx_full, "close"])
        df_out.loc[idx_open_l, "log_returns"] -= fee_bps
        df_out.loc[idx_open_s, "log_returns"] -= fee_bps

        return df_out

    def entry_exit(self) -> pd.DataFrame:
        """
        Given input data, return a dataframe where each row is
        a trade entry or exit time stamp and position.
        """
        _select_columns = ["time", "close", "trade", "open_close"]
        df_out = pd.DataFrame(columns=_select_columns)
        _close = "close"
        _open_long = "open long"
        _open_short = "open short"
        state = _close

        for i in range(self._len):
            if (
                self._data.at[i, "L"] <= self._threshold and state == _open_long
            ):  # Close long position.
                temp = pd.DataFrame(
                    self._data.loc[i, ["time", "close"]].copy()
                ).T
                temp["trade"] = "sell"
                temp["open_close"] = "close long"
                df_out = pd.concat([df_out, temp])
                state = _close
            elif (
                self._data.at[i, "S"] <= self._threshold
                and state == _open_short
            ):  # Close short position.
                temp = pd.DataFrame(
                    self._data.loc[i, ["time", "close"]].copy()
                ).T
                temp["trade"] = "buy"
                temp["open_close"] = "close short"
                df_out = pd.concat([df_out, temp])
                state = _close

            if (
                self._data.at[i, "L"] > self._threshold
                and self._data.at[i, "S"] < 1 - self._threshold
                and state == _close
            ):  # Open long position.
                temp = pd.DataFrame(
                    self._data.loc[i, ["time", "close"]].copy()
                ).T
                temp["trade"] = "buy"
                temp["open_close"] = "open long"
                df_out = pd.concat([df_out, temp])
                state = _open_long
            elif (
                self._data.at[i, "S"] > self._threshold
                and self._data.at[i, "L"] < 1 - self._threshold
                and state == _close
            ):  # Open short position.
                temp = pd.DataFrame(
                    self._data.loc[i, ["time", "close"]].copy()
                ).T
                temp["trade"] = "sell"
                temp["open_close"] = "open short"
                df_out = pd.concat([df_out, temp])
                state = _open_short

        if state == _open_long:
            temp = pd.DataFrame(
                self._data.loc[(self._len - 1), ["time", "close"]].copy()
            ).T
            temp["trade"] = "sell"
            temp["open_close"] = "close long"
            df_out = pd.concat([df_out, temp])
            state = _close
        elif state == _open_short:
            temp = pd.DataFrame(
                self._data.loc[(self._len - 1), ["time", "close"]].copy()
            ).T
            temp["trade"] = "buy"
            temp["open_close"] = "close short"
            df_out = pd.concat([df_out, temp])
            state = _close

        assert state == _close  # Assert that no positions are left open.

        return df_out

    def _get_returns(self, is_long=True) -> list[list]:
        "Given a list of trades, calculate the log return series."
        rets = []
        ee = self.entry_exit()
        _ol = "open long"
        _cl = "close long"
        _os = "open short"
        _cs = "close short"

        if is_long:
            direction = 1.0
            idx_open = self._data[
                self._data.time.isin(ee[ee.open_close == _ol].time)
            ].index.values.astype("int")

            idx_close = self._data[
                self._data.time.isin(ee[ee.open_close == _cl].time)
            ].index.values.astype("int")
        else:
            direction = -1.0
            idx_open = self._data[
                self._data.time.isin(ee[ee.open_close == _os].time)
            ].index.values.astype("int")

            idx_close = self._data[
                self._data.time.isin(ee[ee.open_close == _cs].time)
            ].index.values.astype("int")

        # Calculate log returns to approximate nominal returns.
        for i in range(len(idx_open)):
            start_price = self._data.at[idx_open[i], "close"]
            log_returns = (
                np.log(self._data[idx_open[i] : (idx_close[i] + 1)].close)
                .diff()
                .shift(-1)
            )
            log_returns = (
                (log_returns * direction).dropna().reset_index(drop=True)
            )
            # Subtract fee in basis points from period 0 of the trade series.
            log_returns[0] -= self._fee / start_price
            rets.append(log_returns.tolist())

        return rets

    def _get_accuracy(self, rets: list) -> float:
        "Given trade returns, calculate trade accuracy."
        rets_sum = [sum(r) for r in rets]
        hits = sum([r > 0 for r in rets_sum])

        if len(rets) > 0:
            acc = round(hits / len(rets), 3)
        else:
            acc = np.nan

        return acc

    def _get_drawdown(self, df_bt: pd.DataFrame) -> tuple[float, float]:
        """
        Given full backtest trade deltas, calculate worst
        drawdown percentage and the duration of the worst
        drawdown in days.
        """
        # Cumulative sum of log returns.
        c_sum = list(df_bt.log_returns.cumsum())
        # Running maximum of cumulative sum of log returns.
        c_max = list(df_bt.log_returns.cumsum().cummax())
        # List of drawdowns over time.
        dds = [None] * len(c_sum)
        # Iterate and calculate drawdowns over time.
        for i in range(len(dds)):
            dds[i] = c_sum[i] - c_max[i]

        dd_max = min(dds)

        idx_1 = dds.index(dd_max)  # End index of max drawdown.
        idx_0 = c_max.index(c_max[idx_1])  # Start index of max drawdown.

        n_days = (
            pd.to_datetime(df_bt.at[idx_1, "time"])
            - pd.to_datetime(df_bt.at[idx_0, "time"])
        ).days

        return (round(dd_max, 3), n_days)

    def trade_stats(self, annual_trade_days: float = 260) -> dict:
        """
        Calculate the trading statistics from output probabilities.

        Args:
            self: Contains the BackTest class parameters.
            annual_trade_days: Futures market hours Sun 5p - Fri 5p CT.

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
        stats = dict()

        bt = self.back_test()
        rets_l = self._get_returns(is_long=True)
        rets_s = self._get_returns(is_long=False)
        trades_l = len(rets_l)
        trades_s = len(rets_s)
        trades_total = trades_s + trades_l
        total_return = bt.log_returns.sum()
        n_days = (
            pd.to_datetime(self._data.at[(self._len - 1), "time"])
            - pd.to_datetime(self._data.at[0, "time"])
        ).days
        annual_return = total_return * (annual_trade_days / n_days)
        std_scaler = (annual_trade_days * (len(bt) / n_days)) ** 0.5
        annual_sdev = np.std(bt.log_returns) * std_scaler
        downside_delta = bt.log_returns.copy()
        downside_delta[downside_delta > 0] = 0
        annual_downside_sdev = np.std(downside_delta) * std_scaler

        if annual_sdev > 0:
            sharpe = round(annual_return / annual_sdev, 2)
        else:
            sharpe = np.nan

        if annual_downside_sdev > 0:
            sortino = round(annual_return / annual_downside_sdev, 2)
        else:
            sortino = np.nan

        if trades_l > 0:
            avg_l = sum([sum(ret) for ret in rets_l]) / float(trades_l)
        else:
            avg_l = np.nan

        if trades_s > 0:
            avg_s = sum([sum(ret) for ret in rets_s]) / float(trades_s)
        else:
            avg_s = np.nan

        acc_long = self._get_accuracy(rets=rets_l)
        acc_short = self._get_accuracy(rets=rets_s)
        acc_total = self._get_accuracy(rets=rets_l + rets_s)

        dd = self._get_drawdown(df_bt=bt)

        stats["annual_return"] = round(annual_return, 3)
        stats["sharpe"] = sharpe
        stats["sortino"] = sortino
        stats["std_deviation"] = round(annual_sdev, 3)
        stats["total_return"] = round(total_return, 4)
        stats["trade_count_long"] = trades_l
        stats["trade_count_short"] = trades_s
        stats["trade_count_total"] = trades_total
        stats["average_return_long"] = round(avg_l, 4)
        stats["average_return_short"] = round(avg_s, 4)
        stats["accuracy_long"] = acc_long
        stats["accuracy_short"] = acc_short
        stats["accuracy_total"] = acc_total
        stats["max_drawdown"] = dd[0]
        stats["max_drawdown_days"] = dd[1]

        return stats


def target_optimal(
    df_price: pd.DataFrame,
    min_increment: float = 0.01,
    fee: float = 0.0,
    dd_bps: int = 0,
) -> pd.DataFrame:
    """
    Use dynamic programming (Kadane's Algorithm) to find
    the optimal target labels. Each trade is penalized by
    fee_bps. Output is mapped into [0,1,2] for short, long,
    and close [S,L,C] respectively. The drawdown constraint
    prevents any trade from having a drawdown larger than
    the given dd_bps. Prices are converted to bps to allow
    for additive calculations.

    Args:
        df_price: Dataframe of input prices.
        min_increment: The minimum discrete change allowed for the
             given futures contract.
             e.g. CL - WTI Crude is 0.01 nominally.
        fee: Expected trading friction per trade. This must be in
             terms of min_increment or an error will be raised.
             e.g. for CL, the fee must be a multiple of 0.01.
        dd_bps: Maximum allowable trade drawdown in bps.

    Returns:
        Series containing optimal trades mapped into [0,1,2]
    """
    # Output series containing optimal trade mapping.
    opt = df_price.copy()
    opt.name = f"dd_{dd_bps}"
    opt[:] = 0  # Initialize all to zero.
    n = len(df_price)

    if n <= 1:
        raise ValueError("df_price requires more than 1 observation.")
    elif fee % min_increment != 0:
        raise ValueError("fee must be a multiple of min_increment.")

    idx_buy = 0
    idx_max = 0
    start_price = df_price[0]  # This is for approximating trade return.
    buy_price = start_price + fee
    max_price = buy_price  # Used to calculate trade max drawdown.

    # Calculate optimal long trades.
    for i in range(1, n):
        # Relaxed constraint.
        if buy_price >= df_price[i] + fee:
            idx_buy = i  # Reset indices of trade open and max price.
            idx_max = i
            start_price = df_price[i]  # Reset all prices.
            buy_price = start_price + fee
            max_price = buy_price
        elif max_price < df_price[i]:
            idx_max = i  # Reset max price index and the max_price.
            max_price = df_price[i]
        elif (max_price - df_price[i]) / max_price > dd_bps / 1e4:
            # If max drawdown constraint, then close long trade.
            if idx_buy != idx_max:
                opt[idx_buy : (idx_max + 1)] = 1

            idx_buy = i  # Reset indices of trade open and max price.
            idx_max = i
            start_price = df_price[i]  # Reset all prices.
            buy_price = start_price + fee
            max_price = buy_price

    idx_sell = 0
    idx_min = 0
    start_price = df_price[0]  # This is for approximating trade return.
    sell_price = start_price - fee
    min_price = sell_price

    # Calculate optimal short trades.
    for i in range(1, n):
        # Relaxed constraint.
        if sell_price <= df_price[i] - fee:
            idx_sell = i  # Reset indices of trade open and min price.
            idx_min = i
            start_price = df_price[i]  # Reset all prices.
            sell_price = start_price - fee
            min_price = sell_price
        elif min_price > df_price[i]:
            idx_min = i  # Reset min price index and the min_price.
            min_price = df_price[i]
        elif (min_price - df_price[i]) / min_price < -dd_bps / 1e4:
            # If max drawdown constraint, then close the short trade.
            if idx_sell != idx_min:
                opt[idx_sell : (idx_min + 1)] = -1

            idx_sell = i  # Reset indices of trade open and min price.
            idx_min = i
            start_price = df_price[i]  # Reset all prices.
            sell_price = start_price - fee
            min_price = sell_price

    opt.loc[opt == 0] = 2  # Closed positions.
    opt.loc[opt < 0] = 0  # Short positions.

    return opt
