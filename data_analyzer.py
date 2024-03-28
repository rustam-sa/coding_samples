import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from collections import deque

class DataAnalyzer:

    @staticmethod
    def atr(df, period=14):
        """Calculate the Average True Range (ATR) of a given DataFrame.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' price columns.
        - period (int, optional): The number of periods to calculate the ATR over. Defaults to 14.
        
        Returns:
        - pd.Series: ATR values.
        """
        high_column = df['high']
        low_column = df['low']
        prev_close_column = df['close'].shift(-1)
        high_low_arr = high_column - low_column
        high_close_arr = abs(high_column - prev_close_column)
        low_close_arr = abs(low_column - prev_close_column)
        tr = pd.concat([high_low_arr, high_close_arr, low_close_arr], axis=1).max(axis=1).iloc[::-1]
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        label = f"atr{period}"
        return atr[::-1].rename(label)
    
    @staticmethod
    def bollinger_bands(df, sma_period, distance_multiplier=2):
        """Calculate Bollinger Bands for a given DataFrame.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing the 'close' price column.
        - sma_period (int): The period over which the Simple Moving Average (SMA) is calculated.
        - distance_multiplier (int, optional): The number of standard deviations to plot the bands from the SMA. Defaults to 2.
        
        Returns:
        - pd.DataFrame: DataFrame containing the 'top_bb' and 'bottom_bb' columns for the Bollinger Bands.
        """
        closes = df['close'][::-1].reset_index(drop=True)
        sma = closes.rolling(sma_period).mean()
        std = closes.rolling(sma_period).std()
        top_band = sma + std * distance_multiplier
        bottom_band = sma - std * distance_multiplier
        bollinger_bands = pd.DataFrame({
            "top_bb": top_band, 
            "bottom_bb": bottom_band},
            columns=['top_bb', 'bottom_bb'])[::-1].reset_index(drop=True)
        return bollinger_bands
    
    def divergence(self, candles):
        """Identify and labels the type of divergence in price and RSI indicator data.
        
        Parameters:
        - candles (pd.DataFrame): DataFrame containing the 'close' price column and other data necessary for RSI calculation.
        
        Returns:
        - pd.Series: Labels indicating the type of divergence ('bullish', 'bearish', or 'no divergence') over the data set.
        """
        rsi = np.array(self.rsi(candles)[::-1].reset_index(drop=True))
        closes = np.array(candles['close'][::-1].reset_index(drop=True))
        
        def get_higher_lows(data: np.array, order=5, K=2):
          '''
          Find consecutive higher lows in price pattern.
          Must not be exceeded within the number of periods indicated by the width 
          parameter for the value to be confirmed.
          K determines how many consecutive lows need to be higher.
          '''
          # Get lows
          low_idx = argrelextrema(data, np.less, order=order)[0]
          lows = data[low_idx]
          # Ensure consecutive lows are higher than previous lows
          extrema = []
          ex_deque = deque(maxlen=K)
          for i, idx in enumerate(low_idx):
            if i == 0:
              ex_deque.append(idx)
              continue
            if lows[i] < lows[i-1]:
              ex_deque.clear()
        
            ex_deque.append(idx)
            if len(ex_deque) == K:
              extrema.append(ex_deque.copy())
        
          return extrema
        
        
        def get_lower_highs(data: np.array, order=5, K=2):
          '''
          Find consecutive lower highs in price pattern.
          Must not be exceeded within the number of periods indicated by the width 
          parameter for the value to be confirmed.
          K determines how many consecutive highs need to be lower.
          '''
          # Get highs
          high_idx = argrelextrema(data, np.greater, order=order)[0]
          highs = data[high_idx]
          # Ensure consecutive highs are lower than previous highs
          extrema = []
          ex_deque = deque(maxlen=K)
          for i, idx in enumerate(high_idx):
            if i == 0:
              ex_deque.append(idx)
              continue
            if highs[i] > highs[i-1]:
              ex_deque.clear()
        
            ex_deque.append(idx)
            if len(ex_deque) == K:
              extrema.append(ex_deque.copy())
        
          return extrema
        
        
        def get_higher_highs(data: np.array, order=5, K=2):
          '''
          Find consecutive higher highs in price pattern.
          Must not be exceeded within the number of periods indicated by the width 
          parameter for the value to be confirmed.
          K determines how many consecutive highs need to be higher.
          '''
          # Get highs
          high_idx = argrelextrema(data, np.greater, order=5)[0]
          highs = data[high_idx]
          # Ensure consecutive highs are higher than previous highs
          extrema = []
          ex_deque = deque(maxlen=K)
          for i, idx in enumerate(high_idx):
            if i == 0:
              ex_deque.append(idx)
              continue
            if highs[i] < highs[i-1]:
              ex_deque.clear()
        
            ex_deque.append(idx)
            if len(ex_deque) == K:
              extrema.append(ex_deque.copy())
        
          return extrema
        
        
        def get_lower_lows(data: np.array, order=5, K=2):
          '''
          Find consecutive lower lows in price pattern.
          Must not be exceeded within the number of periods indicated by the width 
          parameter for the value to be confirmed.
          K determines how many consecutive lows need to be lower.
          '''
          # Get lows
          low_idx = argrelextrema(data, np.less, order=order)[0]
          lows = data[low_idx]
          # Ensure consecutive lows are lower than previous lows
          extrema = []
          ex_deque = deque(maxlen=K)
          for i, idx in enumerate(low_idx):
            if i == 0:
              ex_deque.append(idx)
              continue
            if lows[i] > lows[i-1]:
              ex_deque.clear()
        
            ex_deque.append(idx)
            if len(ex_deque) == K:
              extrema.append(ex_deque.copy())
        
          return extrema
        
        
        def peaks_and_troughs(data, order, k):
            hh = get_higher_highs(data, order, k)
            hl = get_higher_lows(data, order, k)
            lh = get_lower_highs(data, order, k)
            ll = get_lower_lows(data, order, k)
            
            hh_points = np.array([(x[0], data[x[0]], x[1], data[x[1]]) for x in hh])
            hl_points = np.array([(x[0], data[x[0]], x[1], data[x[1]]) for x in hl])
            lh_points = np.array([(x[0], data[x[0]], x[1], data[x[1]]) for x in lh])
            ll_points = np.array([(x[0], data[x[0]], x[1], data[x[1]]) for x in ll])
            
            merged = np.concatenate((hh_points, hl_points, lh_points, ll_points), 
                                    axis=0)
            merged = np.concatenate((merged[:, [0, 1]], merged[:, [0, 1]]), 
                                    axis=0)
            
            return merged, hh_points, hl_points, lh_points, ll_points
        
        
        merged_closes, hh_closes, hl_closes, lh_closes, ll_closes = peaks_and_troughs(
            closes, 
            order=5, 
            k=2)
        
        merged_rsi, hh_rsi, hl_rsi, lh_rsi, ll_rsi = peaks_and_troughs(
            rsi, 
            order=5, 
            k=2)
        
        
        def attach_new_array(data, string):
            df = pd.DataFrame(data)
            df[4] = string
            return df
        
        
        hh_closes_designated = attach_new_array(hh_closes, "HH")
        hl_closes_designated = attach_new_array(hl_closes, "HL")
        ll_closes_designated = attach_new_array(ll_closes, "LL")
        lh_closes_designated = attach_new_array(lh_closes, "LH")
        closes_lows_merged_designated = pd.concat([
            ll_closes_designated,
            hl_closes_designated],
            axis=0)
        
        closes_highs_merged_designated = pd.concat([
            hh_closes_designated, 
            lh_closes_designated],
            axis=0)
        
        closes_nadirs = closes_lows_merged_designated.sort_values(0).reset_index(drop=True)
        closes_peaks = closes_highs_merged_designated.sort_values(0).reset_index(drop=True)
        
        hh_rsi_designated = attach_new_array(hh_rsi, "HH")
        hl_rsi_designated = attach_new_array(hl_rsi, "HL")
        ll_rsi_designated = attach_new_array(ll_rsi, "LL")
        lh_rsi_designated = attach_new_array(lh_rsi, "LH")
        rsi_lows_merged_designated = pd.concat([
            ll_rsi_designated,
            hl_rsi_designated],
            axis=0)
        
        rsi_highs_merged_designated = pd.concat([
            hh_rsi_designated, 
            lh_rsi_designated],
            axis=0)
        
        rsi_nadirs = rsi_lows_merged_designated.sort_values(0).reset_index(drop=True)
        rsi_peaks = rsi_highs_merged_designated.sort_values(0).reset_index(drop=True)

        
        def stitch(data, segment_length=1500):
            df = data.iloc[:, [2, 3, 4]]
            zeroes = pd.Series(np.zeros(segment_length))
            change_points = df[2].astype(int)
            zeroes[change_points] = df[4]
            out = zeroes.replace(to_replace=0, method='ffill')
            return out
        
        
        seg_length = len(candles)
        closes_peaks_stitched = stitch(closes_peaks, seg_length)
        closes_nadirs_stitched = stitch(closes_nadirs, seg_length)
        rsi_peaks_stitched = stitch(rsi_peaks, seg_length)
        rsi_nadirs_stitched = stitch(rsi_nadirs, seg_length)
        stitched_together = pd.DataFrame({'closes_peaks': closes_peaks_stitched, 
                                  'closes_nadirs': closes_nadirs_stitched,
                                  'rsi_peaks': rsi_peaks_stitched, 
                                  'rsi_nadirs': rsi_nadirs_stitched})
        conditions = [(stitched_together['closes_peaks'] == 'HH') & 
                      (stitched_together['rsi_peaks'] == 'LH'),
                      (stitched_together['closes_nadirs'] == 'LL') &
                      (stitched_together['rsi_nadirs'] == 'HL')]
        choices = ['bearish', 'bullish']
        stitched_together['divergence'] = np.select(conditions, choices, default = 'no divergence')
        return stitched_together['divergence'][::-1].reset_index(drop=True)
    
    @staticmethod
    def ema(df, span):
        """Calculate the Exponential Moving Average (EMA) of the 'close' price in a DataFrame.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing the 'close' price column.
        - span (int): The span for the EMA calculation.
        
        Returns:
        - pd.Series: EMA values.
        """
        closes = df['close']
        closes_reversed = closes.iloc[::-1]
        ema = closes_reversed.ewm(span=span).mean()
        label = f"ema{span}"
        return ema[::-1].rename(label)
    
    @staticmethod
    def obv(df, rolling_period=400):
        """Calculate the On-Balance Volume (OBV) indicator, optionally applying a rolling sum.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing 'close' and 'volume' columns.
        - rolling_period (int, optional): The window size for the rolling sum of the OBV. Defaults to 400.
        
        Returns:
        - pd.Series: OBV values, potentially smoothed with a rolling sum.
        """
        copy = df.copy()[::-1]
        obv = (np.sign(copy["close"].diff()) * copy["volume"]).fillna(0).rolling(rolling_period).sum()[::-1]
        return obv
    
    @staticmethod
    def rsi(data, span=14):
        """Calculates the Relative Strength Index (RSI) for a given set of price data.
        
        Parameters:
        - data (pd.DataFrame): DataFrame containing the 'close' price column.
        - span (int, optional): The period over which to calculate the RSI. Defaults to 14.
        Returns:
        - pd.Series: RSI values scaled between 0 to 100.
        """
        data = data['close'][::-1].reset_index(drop=True)
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=span-1, adjust=False).mean()
        ema_down = down.ewm(com=span-1, adjust=False).mean()
        rs = ema_up/ema_down
        rsi = 100 - (100/(1 + rs))
        rsi = rsi.rename("rsi")
        return rsi[::-1].reset_index(drop=True)
    
    def stochrsi(self, df, span):
        """Calculate the Stochastic Relative Strength Index (StochRSI) of a given DataFrame.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing the price data.
        - span (int): The lookback period to calculate RSI and StochRSI.
        
        Returns:
        - pd.Series: StochRSI values, smoothed and scaled to 0-100.
        """
        rsi = self.rsi(df, span)[::-1]
        stochrsi = (rsi - rsi.rolling(span).min()) / (rsi.rolling(span).max() - rsi.rolling(span).min())
        stochrsi_smooth = stochrsi.rolling(3).mean() * 100
        return stochrsi_smooth[::-1].round(2).rename("stoch_rsi")
    
    def supertrend(self, df, period=10, multiplier=3):
        """Calculate the Supertrend indicator based on Average True Range (ATR) and price data.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        - period (int): The period over which the ATR is calculated.
        - multiplier (float): The factor by which the ATR is multiplied to calculate the Supertrend.
        
        Returns:
        - pd.DataFrame: DataFrame containing 'upperband', 'lowerband', and 'in_uptrend' columns indicating the Supertrend status.
        """
        df = df.copy()
        atr = self.atr(df, period)
        hl2 = (df['high'] + df['low'])/2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        df['upperband'], df['lowerband'] = basic_upper, basic_lower
        df['upperband_previous'] = df['upperband'].shift(-1)
        df['lowerband_previous'] = df['lowerband'].shift(-1)
        df['in_uptrend'] = True
        df = df[::-1].reset_index(drop=True)

        for current in range(1, len(df.index)):
            previous = current - 1

            if df['close'][current] > df['upperband'][previous]:
                df['in_uptrend'][current] = True
            elif df['close'][current] < df['lowerband'][previous]:
                df['in_uptrend'][current] = False
            else:
                df['in_uptrend'][current] = df['in_uptrend'][previous]

                if df['in_uptrend'][current] and df['lowerband'][current] < df['lowerband'][previous]:
                    df['lowerband'][current] = df['lowerband'][previous]

                if not df['in_uptrend'][current] and df['upperband'][current] > df['upperband'][previous]:
                    df['upperband'][current] = df['upperband'][previous]
        df['timestamp'] = df['timestamp'].astype(int)
        return df[['upperband', 'lowerband', 'in_uptrend']][::-1].reset_index(drop=True)
    
    @staticmethod
    def volume_ema(df, span):
        """Calculate the Exponential Moving Average (EMA) of volume data.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing the 'volume' column.
        - span (int): The span for the EMA calculation on the volume data.
        
        Returns:
        - pd.Series: EMA of volume values.
        """
        volumes = df['volume']
        volumes_reversed = volumes.iloc[::-1]
        ema = volumes_reversed.ewm(span=span).mean()
        label = f"volume_ema{span}"
        return ema[::-1].rename(label)

