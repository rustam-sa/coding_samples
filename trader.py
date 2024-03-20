import pandas as pd
import numpy as np
import joblib
import time
import uuid
import json
import requests
import logging
import random
import csv
import os, glob

import mplfinance as mpf
import matplotlib.pyplot as plt

from pprint import pprint
from datetime import datetime
from scipy.signal import argrelextrema
from collections import deque
from decimal import Decimal as dec
from headers import ht, process_request, get_headers
from send_email import send_email

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import load_model

pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(filename="log.log",
                    encoding="utf-8",
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

url = ht['base']


def time_it(func):
    
    
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        with open("optimization_log.csv", "a", newline="") as f:
            msg = func.__name__, end-start, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), args, kwargs
            writer = csv.writer(f)
            writer.writerow(msg)
            print(msg)
        return result
    return wrap

def log(message):
    print(message)
    logging.info(message)
    
    
def dec_round(num, digits=2, rounding="ROUND_HALF_EVEN"):
    if digits == 0: 
        return dec(num).quantize(dec("1."), rounding="ROUND_HALF_EVEN")
    return dec(num).quantize(dec("." + "0" * (digits - 1) + "1"), rounding=rounding)


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class Trader:
    def __init__(self, symbol="ETH-USDT", timeframe="5min", risk=2, 
                 end_trade_multiplier=2, 
                 long_min_threshold=.72, long_max_threshold=1, 
                 short_min_threshold=.72, short_max_threshold=1,
                 distance_threshold = 0.006):
        
        self.timeframes_in_minutes = {
            "1min": 1,
            "3min": 3,
            "5min": 5, 
            "15min": 15,
            "30min": 30,
            "1hour": 60,
            "2hour": 120,
            "4hour": 240,
            "6hour": 360,
            "8hour": 480,
            "12hour": 720,
            "1day": 1440,
            "1week": 10080
            }
        self.steps = [2, 4, 6, 8, 10, 20, 30, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 400, 1000]
        self.base_borrow_precision_rounding = {
            'ADA-USDT': -1,
            'ETH-USDT': 2,
            }
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk = risk
        self.end_trade_multiplier = end_trade_multiplier
        self.set_roundings()
        self.current_position = None
        self.upper_stop = None
        self.lower_stop = None
        # self.check_for_existing_positions()
        # self.clean_up_isolated()
        self.candles = None
        self.update_candles()
        self.ema1_signal = None
        self.ema2_signal = None
        self.ema3_signal = None
        self.ema4_signal = None
        self.ema5_signal = None
        self.ema6_signal = None
        self.ema7_signal = None
        self.ema8_signal = None
        self.ema9_signal = None
        self.ema10_signal = None
        self.ema11_signal = None
        
        self.stoch_rsi_signal = None
        self.supertrend_signal_1 = None
        self.supertrend_signal_2 = None
        self.divergence_signal = None
        self.timestamp = None
        self.atr_signal = None
        self.full_processor = None
        # self.load_models()
        # self.init_message()
        
        # self.long_min_threshold = long_min_threshold
        # self.long_max_threshold = long_max_threshold
        
        # self.short_min_threshold = short_min_threshold
        # self.short_max_threshold = short_max_threshold
        
        self.distance_threshold = dec_round(distance_threshold, 4)
        
    def init_message(self):
        print(f"""Trader initiated. Params: 
              symbol: {self.symbol}, 
              timeframe: {self.timeframe}, 
              risk: {self.risk}""")
    
    def get_most_recent_file_names(self, sub_directory, last_n=2, file_type=".csv"):
        """
        return list of file names of most recent last_n files
        """
        current_dir = os.getcwd()
        all_file_names = glob.glob(f'{current_dir}/{sub_directory}/*{file_type}')
        all_file_names_sorted = sorted(all_file_names, key=os.path.getctime, reverse=False)
        target_file_names = all_file_names_sorted[-last_n:]
        
        return target_file_names
    
    def load_models(self):
        long_model_directory = "models/long"
        long_model_name = self.get_most_recent_file_names(long_model_directory, 1, ".py")[0]
        self.long_model = load_model(long_model_name)
        short_model_directory = "models/short"
        short_model_name = self.get_most_recent_file_names(short_model_directory, 1, ".py")[0]
        self.short_model = load_model(short_model_name)
        
    def set_roundings(self):
        self.base_rounding, self.quote_rounding = self.get_rounding(self.symbol)
    
    def create_rfc_pipeline(self, data):
        X = data
        numerical_features = X.select_dtypes(include='number').columns.tolist()
        categorical_features = X.select_dtypes(exclude='number').columns.tolist()
            
        numeric_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
            ])
        
        categorical_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder())
            ])
        
        full_processor = ColumnTransformer(transformers=[
            ('number', numeric_pipeline, numerical_features),
            ('category', categorical_pipeline, categorical_features)
            ])
        
        return full_processor
    
    @staticmethod
    def split_cat_and_num_cols(df):
        boolean_columns = df.select_dtypes([np.bool_]).columns
        df[boolean_columns] = df[boolean_columns].astype('int32')
        numerical_columns = df.select_dtypes([np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        numerical_df = df[numerical_columns]
        categorical_df = df[categorical_columns]

        return numerical_df, categorical_df
    
    
    @staticmethod
    def make_stationary_dataframe(dataframe):
        numerical_features = dataframe.select_dtypes(include='number').columns.tolist()
        categorical_features = dataframe.select_dtypes(exclude='number').columns.tolist()
        df = dataframe[numerical_features]
        # closing_prices = dataframe['close']
        # closing_p_value = adfuller(closing_prices)[1]
        
        # if closing_p_value <= 0.05:
        #     print(f"Closing price is stationary. p value is: {closing_p_value}.")
        #     return dataframe
        
        def make_stationary_series(series):
            series_stationary = series.diff()
            return series_stationary
        
        df = df.copy()
        column_count = len(df.columns)
        new_df = df[['timestamp']]
        for c in range(1, column_count):
            series = df.iloc[:, c]
            series_stationary = make_stationary_series(series)
            new_df[series.name] = series_stationary.copy()
        new_df = pd.concat([new_df, dataframe[categorical_features]], axis=1).dropna(axis=0)
        return new_df
    
    def data_prep(self, df):
        steps = self.steps 
        param_dict = {}
        
        param_dict['symbol'] = self.symbol
        param_dict['timeframe'] = self.timeframe
        param_dict['atr_multiplier'] = 3
        param_dict['rsi_span'] = 14
        param_dict['atr_period'] = 14
        
        for n in range(1, len(steps) + 1):
            steps_index = n - 1
            param_dict[f'supertrend_period{n}'] = steps[steps_index]
            param_dict[f'atr_period{n}'] = steps[steps_index]
            param_dict[f'rsi_span{n}'] = steps[steps_index]
            param_dict[f'ema_period_{n}'] = steps[steps_index]
            param_dict[f'bb_span_{n}'] = steps[steps_index]
        
        df = df.copy()
        df = df.iloc[1:, :].reset_index(drop=True) # gets rid of current timestamp (which is not yet finalized)
        
        # adding features
        df['atr'] = self.atr(df, param_dict['atr_period'])
        df['rsi'] = self.rsi(df, param_dict['rsi_span'])
        df['stoch_rsi'] = self.stochrsi(df, param_dict['rsi_span'])
        df['divergence'] = self.divergence(df)
        features = {}
        for n in range(1, len(steps) + 1):
            features[f'ema{n}'] = self.ema(df, param_dict[f'ema_period_{n}'])
            features[f'rsi{n}'] = self.rsi(df, param_dict[f'rsi_span{n}'])
            features[f'stoch_rsi{n}'] = self.stochrsi(df, param_dict[f'rsi_span{n}'])
            features[f'atr{n}'] = self.atr(df, param_dict[f'atr_period{n}'])
            # features[f'top_bb{n}'], features[f'bottom_bb{n}'] = self.bollinger_bands(df, param_dict[f'bb_span_{n}'])
            # features[f'upperband{n}'], features[f'lowerband{n}'], features[f'in_uptrend{n}'] = self.supertrend(
            #     df, param_dict[f'supertrend_period{n}'])
        
        features = pd.DataFrame(features)
        df = pd.concat([df, features], axis=1)
            
        columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'atr', 
                   'rsi', 'stoch_rsi', 'divergence', 'ema1', 'rsi1', 'stoch_rsi1', 
                   'atr1', 'ema2', 'rsi2', 'stoch_rsi2', 'atr2', 'ema3', 'rsi3', 
                   'stoch_rsi3', 'atr3', 'ema4', 'rsi4', 'stoch_rsi4', 'atr4', 'ema5', 
                   'rsi5', 'stoch_rsi5', 'atr5', 'ema6', 'rsi6', 'stoch_rsi6', 
                   'atr6', 'ema7', 'rsi7', 'stoch_rsi7', 'atr7', 'ema8', 'rsi8', 
                   'stoch_rsi8', 'atr8', 'ema9', 'rsi9', 'stoch_rsi9', 'atr9', 
                   'ema10', 'rsi10', 'stoch_rsi10', 'atr10', 'ema11', 'rsi11', 
                   'stoch_rsi11', 'atr11', 'ema12', 'rsi12', 'stoch_rsi12', 
                   'atr12', 'ema13', 'rsi13', 'stoch_rsi13', 'atr13', 'ema14', 
                   'rsi14', 'stoch_rsi14', 'atr14', 'ema15', 'rsi15', 
                   'stoch_rsi15', 'atr15', 'ema16', 'rsi16', 'stoch_rsi16', 
                   'atr16', 'ema17', 'rsi17', 'stoch_rsi17', 'atr17', 'ema18', 
                   'rsi18', 'stoch_rsi18', 'atr18', 'ema19', 'rsi19', 
                   'stoch_rsi19', 'atr19', 'ema20', 'rsi20', 'stoch_rsi20', 'atr20']
        df = df[columns]
        return df

    def diff(self, df, column):
        col_reversed = df[column][::-1]
        return col_reversed.diff()[::-1]

    def signal(self):
        candles = self.candles.copy()
        filename = 'candle_snapshots/' + str(candles['timestamp'][0]) + '.csv' 
        candles.to_csv(filename, index=False)
        df = self.data_prep(candles)
        latest_candle_index = 0
        latest_candle = df.iloc[[latest_candle_index]]
        self.atr_signal = latest_candle.atr[latest_candle_index]
        num_df = df.select_dtypes(include=[np.number])
        columns_to_diff = list(num_df.columns)
        for c in columns_to_diff:
            df[c] = self.diff(df, c)
        df = df.dropna().reset_index(drop=True)
        self.full_processor = self.create_rfc_pipeline(df)
        transformed_candles = self.full_processor.fit_transform(df)
        long_prob = self.long_model.predict(transformed_candles).flatten()[latest_candle_index]
        short_prob = self.short_model.predict(transformed_candles).flatten()[latest_candle_index]
        
        print(f"Long prob: {long_prob}")
        print(f"short prob: {short_prob}")
        
        signal_frame = {
            "LONG": 0, 
            "SHORT": 0
            }
        signal_frame["LONG"] = long_prob > self.long_min_threshold and long_prob < self.long_max_threshold
        signal_frame["SHORT"] = short_prob > self.short_min_threshold and short_prob < self.short_max_threshold
        
        if signal_frame["LONG"] == True and signal_frame['SHORT'] != True:
            return "LONG"
        if signal_frame["SHORT"] == True and signal_frame["LONG"] != True:
            return "SHORT"
        else:   
            return None


    @staticmethod
    def ema(df, span):
        closes = df['close']
        closes_reversed = closes.iloc[::-1]
        ema = closes_reversed.ewm(span=span).mean()
        label = f"ema{span}"
        return ema[::-1].rename(label)
    
    @staticmethod
    def volume_ema(df, span):
        volumes = df['volume']
        volumes_reversed = volumes.iloc[::-1]
        ema = volumes_reversed.ewm(span=span).mean()
        label = f"volume_ema{span}"
        return ema[::-1].rename(label)

    def _ema(self, span):
        closes = self.candles['close']
        closes_reversed = closes.iloc[::-1]
        ema = closes_reversed.ewm(span=span).mean()
        label = f"ema{span}"
        return ema[::-1].rename(label)

    @staticmethod
    def atr(df, period=14):
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

    def _atr(self, period=14):
        df = self.candles
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
    
    def _bollinger_bands(self, sma_period, distance_multiplier=2):
        candles = self.candles[::-1].reset_index(drop=True)
        closes = candles['close']
        sma = closes.rolling(sma_period).mean()
        std = closes.rolling(sma_period).std()
        top_band = sma + std * distance_multiplier
        bottom_band = sma - std * distance_multiplier
        bollinger_bands = pd.DataFrame({
            "top_bb": top_band, 
            "bottom_bb": bottom_band},
            columns=['top_bb', 'bottom_bb'])[::-1].reset_index(drop=True)
        return bollinger_bands
    
    @staticmethod
    def bollinger_bands(df, sma_period, distance_multiplier=2):
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

    def supertrend(self, df, period=10, multiplier=3):
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

    def _supertrend(self, period=10, multiplier=3):
        df = self.candles
        atr = self._atr(period)
        hl2 = (df['high'] + df['low']) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        df['upperband'], df['lowerband'] = basic_upper, basic_lower
        df['upperband_previous'] = df['upperband'].shift(-1)
        df['lowerband_previous'] = df['lowerband'].shift(-1)
        df['in_uptrend'] = True
        df = df.head(200)[::-1].reset_index(drop=True)
        
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
        return df[['upperband', 'lowerband', 'in_uptrend', 'timestamp']][::-1].reset_index(drop=True)
    
    def _rsi(self, span=14):
        data = self.candles['close'][::-1].reset_index(drop=True)
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=span-1, adjust=False).mean()
        ema_down = down.ewm(com=span-1, adjust=False).mean()
        rs = ema_up/ema_down
        rsi = 100 - (100/(1 + rs))
        rsi = rsi.rename("rsi")
        return rsi[::-1].reset_index(drop=True)
    

    @staticmethod
    def rsi(data, span=14):
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
    
    def _stochrsi(self, span):
        rsi = self._rsi(span)[::-1]
        stochrsi = (rsi - rsi.rolling(span).min()) / (rsi.rolling(span).max() - rsi.rolling(span).min())
        stochrsi_smooth = stochrsi.rolling(3).mean() * 100
        return stochrsi_smooth[::-1].round(2).rename("stoch_rsi")
    
    def stochrsi(self, df, span):
        rsi = self.rsi(df, span)[::-1]
        stochrsi = (rsi - rsi.rolling(span).min()) / (rsi.rolling(span).max() - rsi.rolling(span).min())
        stochrsi_smooth = stochrsi.rolling(3).mean() * 100
        return stochrsi_smooth[::-1].round(2).rename("stoch_rsi")
    
    def _get_candles(self):
        start, end = self.generate_start_and_end()
        payload = {
            "type": self.timeframe, 
            "symbol": self.symbol,
            "startAt": start,
            "endAt": end
            }
        response = process_request("GET", ht['candles'], **payload)
        try:
            candles = pd.DataFrame(
                response['data'],
                columns=["timestamp", "open", "close", "high", "low", "volume", "amount"]
                ).astype(float)
            candles['timestamp'] = candles['timestamp'].astype(int)
        except KeyError as e:
            time.sleep(1)
            print(f"KeyError: {e} - Retrying fetching candles...")
            return self._get_candles()
        return candles
    
    
    def get_candles(self, symbol="ETH-USDT", timeframe="5min", start_at=None, end_at=None, delay=0):
        time.sleep(delay)
        if start_at == None and end_at == None:
            start_at, end_at = self.generate_start_and_end()
        payload = {"symbol": symbol, 
                   "startAt": start_at,
                   "endAt": end_at, 
                   "type": timeframe}
        try: 
            candles = process_request("GET", ht['candles'], **payload)['data']
        except KeyError:
            print("Retrying get_candles response due to KeyError")
            time.sleep(1)
            candles = self.get_candles(symbol, timeframe, start_at, end_at)
        candles = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "close", "high", "low", "volume", "amount"]
            ).astype(float)
        candles['timestamp'] = candles['timestamp'].astype(int)
        # if candles['timestamp'].size < 1000:
        #     print("Got less than a 1000 candles, retrying...")
        #     time.sleep(3)
        #     candles = self.get_candles(symbol, timeframe, start_at, end_at)
        
        check = self.check_timestamp_sequence(candles)
        print(f"""
              FETCHING params: {symbol}, 
              {timeframe}, 
              start: {start_at}, 
              end: {end_at}
              """)
        print("Received Dataframe: ", check)
        
        return candles
    
    
    def check_timestamp_sequence(self, df):
        
        res = {
            
            "first_timestamp": 0,
            "last_timestamp": 0,
            "gaps": 0,
            "count": df['timestamp'].size,
            
            }
        
        diff = df['timestamp'].astype(int).diff().dropna().reset_index(drop=True) * - 1
        gaps_test = np.where(
            diff != self.timeframes_in_minutes[self.timeframe] * 60, 
            1, 
            0)
        res['gaps'] = any(gaps_test)
        res['first_timestamp']= df['timestamp'][0] 
        res['last_timestamp'] = df['timestamp'][df['timestamp'].size - 1]
        if res["gaps"]: 
            df.to_csv("data_to_inspect/most_recent.csv")
            print("GAP: dataframe saved to data_to_inspect inside kucoin leverage bot project")
            raise Exception("GAP detected")
        return res

    def update_candles(self):
        self.candles = self._get_candles()
    
    @time_it
    def get_mass_candles(self, span, delay=0):
        most_recent_timestamp = self.get_candles(
            self.symbol, self.timeframe, delay=delay)['timestamp'][0]
        print(most_recent_timestamp)
        timeframe = self.timeframe
        symbol = self.symbol
        subtrahend = self.timeframes_in_minutes[timeframe] * 60 * 1000
        num_of_segments = span//1000
        end = most_recent_timestamp
        ht = {}
        for x in range(num_of_segments):
            ht[end] = end - subtrahend
            end = end - subtrahend
        
        data = [self.get_candles(symbol=symbol, timeframe=timeframe, start_at=str(y), end_at=str(x)) for x, y in ht.items()]
        df = pd.concat(data).reset_index(drop=True)
        self.check_timestamp_sequence(df)
        return df
    
    @time_it
    def get_mass_candles_symbol(self, symbol, span, delay=0):
        most_recent_timestamp = self.get_candles(
            symbol, self.timeframe, delay=delay)['timestamp'][0]
        print(most_recent_timestamp)
        timeframe = self.timeframe
        symbol = symbol
        subtrahend = self.timeframes_in_minutes[timeframe] * 60 * 1000
        num_of_segments = span//1000
        end = most_recent_timestamp
        ht = {}
        for x in range(num_of_segments):
            ht[end] = end - subtrahend
            end = end - subtrahend
        
        data = [self.get_candles(symbol=symbol, timeframe=timeframe, start_at=str(y), end_at=str(x)) for x, y in ht.items()]
        df = pd.concat(data).reset_index(drop=True)
        self.check_timestamp_sequence(df)
        return df


    @staticmethod
    def get_sub_accounts():
        sub_accounts = process_request("GET", ht['sub_accounts'])
        return sub_accounts
    
    @staticmethod
    def create_account(account_type="main", currency="ETH"):
        payload = {
            "type": account_type,
            "currency": currency
                  }
        response = process_request("POST", ht['accounts'], **payload)
        return response
    
    @staticmethod
    def list_accounts(currency=None, account_type=None):
        payload = {
            "currency": currency,
            "type": account_type
                  }
        response = process_request("GET", ht['accounts'], **payload)
        return response
    
    @staticmethod
    def get_account(account_id):
        payload = {
            "accountId": account_id
            }
        endpoint = ht['accounts'] + "/"
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_ledgers(currency="ETH", days=7):
        now = int(time.time() * 1000)
        days_in_ms = days * 86400000
        start_at = str(now - days_in_ms)
        payload = {
            "currency": currency,
            "startAt": start_at
            }
        endpoint = ht['ledgers']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_sub_balance(sub_id):
        payload = {
            "subUserId": sub_id
            }
        endpoint = ht['sub_balance'] + "/"
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_all_sub_balance():
        endpoint = ht['all_sub_balance']
        response = process_request("GET", endpoint)
        return response
    
    @staticmethod
    def get_transferable(currency="USDT", account_type="MAIN"):
        payload = {
            "currency": currency,
            "type": account_type
            }
        endpoint = ht['transferable']
        response = process_request("GET", endpoint, **payload)
        return response
        
    @staticmethod
    def sub_transfer(currency, amount, direction, account_type=None, sub_account_type=None, sub_user_id=None):
        order_id = str(uuid.uuid4())
        payload = {
            "clientOid": order_id,
            "currency": currency,
            "amount": str(amount),
            "direction": direction,  # 'OUT' — master to sub | 'IN' — sub to master
            "accountType": account_type,  # 'MAIN', 'TRADE', 'MARGIN', 'CONTRACT'
            "subAccountType": sub_account_type,
            "subUserId": sub_user_id
            }
        endpoint = ht['sub_transfer']
        response = process_request("POST", endpoint, **payload)
        return response

    @staticmethod
    def inner_transfer(currency, out_of, to, amount):
        order_id = str(uuid.uuid4())
        payload = {
            "clientOid": order_id,
            "currency": currency,
            "from": out_of, 
            "to": to,
            "amount": str(amount)
            }
        endpoint = ht['inner_transfer']
        response = process_request("POST", endpoint, **payload)
        return response
    
    @staticmethod
    def create_deposit_address(currency):
        payload = {
            "currency": currency
            }
        endpoint = ht['create_deposit_address']
        response = process_request("POST", endpoint, **payload)
        return response
    
    @staticmethod
    def get_deposit_addresses(currency):
        payload = {
            "currency": currency
            }
        endpoint = ht['get_deposit_addresses']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_deposit_address(currency):
        payload = {
            "currency": currency
            }
        endpoint = ht['get_deposit_address']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_deposit_list():
        endpoint = ht['get_deposit_list']
        response = process_request("GET", endpoint)
        return response
    
    @staticmethod
    def get_withdrawals_list():
        endpoint = ht['get_withdrawals_list']
        response = process_request("GET", endpoint)
        return response
    
    @staticmethod
    def get_withdrawal_quotas(currency):
        payload = {
            "currency": currency
            }
        endpoint = ht['get_withdrawal_quotas']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def apply_withdraw(currency, address, amount):
        payload = {
            "currency": currency, 
            "address": address,
            "amount": amount
            }
        endpoint = ht['apply_withdrawal']
        response = process_request("POST", endpoint, **payload)
        return response
        
    @staticmethod
    def cancel_withdrawal(withdrawal_id):
        payload = {
            "withdrawalId": withdrawal_id
            }
        endpoint = ht['cancel_withdrawal']
        response = process_request("DELETE", endpoint, **payload)
        return response
    
    @staticmethod
    def base_fee():
        endpoint = ht['base_fee']
        response = process_request("GET", endpoint)
        return response
    
    @staticmethod
    def trade_fee(symbol):
        payload = {
            "symbols": symbol
            }  # "ETH-USDT"
        endpoint = ht['trade_fee']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def market_order(side, symbol, size=None, funds=None):
        order_id = str(uuid.uuid4())
        if size:
            payload = {
                "clientOid": order_id,
                "side": side,
                "symbol": symbol,
                "type": "market",
                "size": str(size)
                }
        if funds:
            payload = {
                "clientOid": order_id,
                "side": side,
                "symbol": symbol,
                "type": "market",
                "funds": str(funds)
                }
        endpoint = ht['new_order']
        response = process_request("POST", endpoint, **payload)
        return response
    
    @staticmethod
    def limit_order(side, symbol, price, size):
        order_id = str(uuid.uuid4())
        payload = {
            "clientOid": order_id,
            "side": side,
            "symbol": symbol,
            "type": "limit",
            "price": str(price),
            "size": str(size)
            }
        endpoint = ht['new_order']
        response = process_request("POST", endpoint, **payload)
        return response
    
    @staticmethod
    def margin_limit_order(side, symbol, price, size):
        order_id = str(uuid.uuid4())
        payload = {
            "clientOid": order_id,
            "side": side, 
            "symbol": symbol,
            "type": "limit",
            "price": str(price),
            "size": str(size)
            }
        endpoint = ht['margin_order']
        response = process_request("POST", endpoint, **payload)
        return response
        
    @staticmethod
    def margin_market_order(side, symbol, size=None, funds=None):
        order_id = str(uuid.uuid4())
        if size:
            payload = {
                "clientOid": order_id,
                "side": side,
                "symbol": symbol,
                "type": "market",
                "marginModel": "isolated",
                "size": str(size)
                }
        if funds:
            payload = {
                "clientOid": order_id,
                "side": side,
                "symbol": symbol,
                "type": "market",
                "marginModel": "isolated",
                "funds": str(funds)
                }
        endpoint = ht['margin_order']
        response = process_request("POST", endpoint, **payload)
        return response
    
    @staticmethod
    def cancel_order(order_id):
        payload = {
            "orderId": order_id
            }
        endpoint = ht['cancel_order'] + "/"
        response = process_request("DELETE", endpoint, **payload)
        return response
    
    @staticmethod
    def cancel_all_orders(symbol=None, trade_type=None):
        endpoint = ht['cancel_order']
        payload = {
            "symbol": symbol,
            "tradeType": trade_type,
            }
        if not any(payload.values()):
            response = process_request("DELETE", endpoint)
        else: 
            response = process_request("DELETE", endpoint, **payload)
        return response
    
    @staticmethod
    def list_orders(trade_type, status=None, symbol=None, side=None, order_type=None, days=7):
        now = int(time.time() * 1000)
        days_in_ms = days * 86400000
        start_at = str(now - days_in_ms)
        payload = {
            "status": status, # active or done
            "symbol": symbol, 
            "side": side, 
            "type": order_type, 
            "tradeType": trade_type, 
            "startAt": start_at
            }
        endpoint = ht['list_orders']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def historical_orders(current_page=None, page_size=None, symbol=None, days=7, side=None):
        now = int(time.time() * 1000)
        days_in_ms = days * 86400000
        start_at = str(now - days_in_ms)
        payload = {
            "currentPage": current_page, 
            "pageSize": page_size,
            "symbol": symbol, 
            "startAt": start_at,
            "side": side, 
            }
        endpoint = ht['historical_orders']
        response = process_request("GET", endpoint, **payload)
        return response
        
    @staticmethod
    def get_order(order_id):
        payload = {
            "orderId": order_id
            }
        endpoint = ht['get_order']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def list_fills(days=7):
        now = int(time.time() * 1000)
        days_in_ms = days * 86400000
        start_at = str(now - days_in_ms)
        payload = {
            "startAt": start_at
            }
        endpoint = ht['fills']
        response = process_request("GT", endpoint, **payload)
        return response
    
    @staticmethod
    def stop_order_market(side, symbol, stop, stop_price, trade_type, size=None, funds=None):
        order_id = str(uuid.uuid4())
        if size:
            payload = {
                "clientOid": order_id, 
                "side": side,  # buy or sell
                "symbol": symbol,
                "type": "market",
                "stop": stop,  # loss or entry
                "stopPrice": str(stop_price),  #
                "tradeType": trade_type,  # TRADE or MARGIN_TRADE
                "size": size,  # in base currency
                }
        if funds:
            payload = {
                "clientOid": order_id, 
                "side": side,  # buy or sell
                "symbol": symbol, 
                "type": "market",
                "stop": stop,  # loss or entry
                "stopPrice": str(stop_price), # 
                "tradeType": trade_type,  # TRADE or MARGIN_TRADE
                "funds": funds,  # in quote currency
                }
        endpoint = ht['stop_order']
        response = process_request("POST", endpoint, **payload)
        return response
    
    @staticmethod
    def stop_order_limit(side, symbol, stop, stop_price, trade_type, price, size):
        order_id = str(uuid.uuid4())
        payload = {
            "clientOid": order_id, 
            "side": side,  # buy or sell
            "symbol": symbol, 
            "type": "market",
            "stop": stop,  # loss or entry
            "stopPrice": str(stop_price),
            "tradeType": trade_type,  # TRADE or MARGIN_TRADE
            "price": str(price),
            "size": str(size)  # amount of base currency
            }
        endpoint = ht['stop_order']
        response = process_request("POST", endpoint, **payload)
        return response
    
    @staticmethod
    def cancel_stop_order(order_id):
        payload = {
            "orderId": order_id
            }
        endpoint = ht['stop_order'] + "/"
        response = process_request("DELETE", endpoint, **payload)
        return response
        
    def cancel_stop_orders(self, trade_type=None, order_ids=None):
        payload = {
            "symbol": self.symbol, 
            "tradeType": trade_type, 
            "orderIds": order_ids  # separated by commas
            }
        endpoint = ht['stop_order'] + "/cancel" 
        if not any(payload.values()):
            response = process_request("DELETE", endpoint)
        else:
            response = process_request("DELETE", endpoint, **payload)
        return response
    
    @staticmethod
    def get_stop_order_info(order_id):
        payload = {
            "orderId": order_id
            }
        endpoint = ht['stop_order'] + "/"
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def list_stop_orders(symbol=None, side=None, order_type=None, trade_type=None, days=7):
        now = int(time.time() * 1000)
        days_in_ms = days * 86400000
        start_at = str(now - days_in_ms)
        payload = {
            "symbol": symbol, 
            "side": side, 
            "type": order_type, 
            "tradeType": trade_type,
            "startAt": start_at}
        endpoint = ht['stop_order']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_stop_order_by_client_id(order_id, symbol=None):
        payload = {
            "orderId": order_id, 
            "symbol": symbol,
            }
        endpoint = ht['stop_order_client_id']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def cancel_stop_order_by_client_id(order_id, symbol=None):
        payload = {
            "clientOid": order_id, 
            "symbol": symbol,
            }
        endpoint = ht['stop_order_client_id']
        response = process_request("DELETE", endpoint, **payload)
        return response
    
    @staticmethod
    def get_symbols(market=None):
        payload = {
            "market": market
            }
        endpoint = ht['symbols']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_ticker(symbol):
        payload = {
            "symbol": symbol
            }
        endpoint = ht['ticker']
        response = process_request("GET", endpoint, **payload)
        return response
    
    @staticmethod
    def get_symbol_info(symbol):
        endpoint = ht['get_symbol_info']
        response = process_request("GET", endpoint)
        df = pd.DataFrame(response['data'])
        symbol_info = df[df['symbol'] == symbol].reset_index(drop=True)
        results = {}
        for column in symbol_info.columns:
            results[column] = symbol_info[column][0]
        return results
    
    @staticmethod
    def get_markets():
        return process_request("GET", ht['markets'])
    
    @staticmethod
    def get_currencies():
        return process_request("GET", ht['currencies'])
    
    @staticmethod
    def get_currency_detail(currency, chain=None):
        payload = {
            "currency": currency, 
            "chain": chain,
            }
        endpoint = ht['currencies'] + "/"
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_fiat_price(base=None, currencies=None):
        payload = {
            "base": base,
            "currencies": currencies
            }
        endpoint = ht['prices']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_mark_price(symbol):
        params = {
            "symbol": symbol
            }
        endpoint = f"/api/v1/mark-price/{symbol}/current"
        headers = get_headers("GET", endpoint, params)
        response = dict(requests.request(
            "GET",
            url + endpoint, 
            json=params,
            headers=headers
            ).json())
        return response
        
    @staticmethod
    def get_margin_config():
        return process_request("GET", ht['margin_config'])
    
    @staticmethod
    def get_margin_account():
        return process_request("GET", ht['margin_account'])
    
    @staticmethod
    def get_isolated_accounts(balance_currency="USDT"):
        response = process_request("GET", ht['isolated_accounts'])
        return response
    
    @staticmethod
    def post_borrow_order(currency, size, order_type="FOK", max_rate=None, term=None):
        payload = {
            "currency": currency, 
            "type": order_type,
            "size": str(size),
            "maxRate": max_rate,
            "marginModel": "isolated",
            "term": term
            }
        endpoint = ht['margin_borrow']
        return process_request("POST", endpoint, **payload)
    
    @staticmethod
    def post_borrow_order_isolated(trading_pair, currency, size, order_type="IOC", max_rate=None, term=None):
        payload = {
            "symbol": trading_pair,
            "currency": currency, 
            "size": str(size),
            "borrowStrategy": order_type, 
            "maxRate": max_rate,
            "period": term
            }
        endpoint = ht['isolated_margin_borrow']
        return process_request("POST", endpoint, **payload)
    
    @staticmethod
    def get_borrow_order(order_id):
        payload = {
            "orderId": order_id
            }
        return process_request("GET", ht['margin_borrow'], **payload)
    
    @staticmethod
    def get_repay_record(currency=None):
        payload = {
            "currency": currency
            }
        endpoint = ht['borrow_outstanding']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_repayment_record(currency=None):
        payload = {
            "currency": currency
            }
        endpoint = ht['borrow_repaid']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def one_click_repayment(currency, size, sequence="HIGHEST_RATE_FIRST"):
        payload = {
            "currency": currency, 
            "sequence": sequence, 
            "size": size
            }
        endpoint = ht['one_click_repayment']
        return process_request("POST", endpoint, **payload)
    
    @staticmethod
    def one_click_repayment_isolated(symbol, currency, size, seq_strategy='HIGHEST_RATE_FIRST'):
        endpoint = ht['isolated_repay']
        payload = {
            "symbol": symbol, 
            "currency": currency, 
            "size": size, 
            "seqStrategy": seq_strategy
            }
        return process_request("POST", endpoint, **payload)
    
    @staticmethod
    def repay_single_order(currency, trade_id, size):
        payload = {
            "currency": currency, 
            "tradeId": trade_id, 
            "size": size
            }
        endpoint = ht['repay_single_order']
        return process_request("POST", endpoint, **payload)
    
    @staticmethod
    def post_lend_order(currency, size, daily_int_rate, term):
        payload = {
            "currency": currency,
            "size": size, 
            "dailyIntRate": daily_int_rate,
            "term": term
            }
        endpoint = ht['lend_order']
        return process_request("POST", endpoint, **payload)
    
    @staticmethod
    def cancel_lend_order(order_id):
        payload = {
            "orderId": order_id
            }
        endpoint = ht['lend_order'] + "/"
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def set_auto_lend(currency, is_enable, retain_size, daily_int_rate, term):
        payload = {
            "currency": currency, 
            "isEnable": is_enable, 
            "retainSize": retain_size, 
            "dailyIntRate": daily_int_rate, 
            "term": term
            }
        endpoint = ht['toggle_auto_lend']
        return process_request("POST", endpoint, **payload)
    
    @staticmethod
    def get_active_lend_order(currency, current_page=None, page_size=None):
        payload = {
            "currency": currency, 
            "currentPage": current_page, 
            "pageSize": page_size
            }
        endpoint = ht['lend_active']
        return process_request("GET", endpoint, **payload)
        
    @staticmethod
    def get_lent_history(currency, current_page=None, page_size=None):
        payload = {
            "currency": currency, 
            "currentPage": current_page, 
            "pageSize": page_size
            }
        endpoint = ht['lent_history']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_active_lend_order_list(currency=None, current_page=None, page_size=None):
        payload = {
            "currency": currency, 
            "currentPage": current_page, 
            "pageSize": page_size
            }
        endpoint = ht['active_lend_orders']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_settled_lend_order_list(currency=None, current_page=None, page_size=None):
        payload = {
            "currency": currency, 
            "currentPage": current_page, 
            "pageSize": page_size
            }
        endpoint = ht['settled_lend_orders']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_account_lend_record(currency=None):
        payload = {
            "currency": currency
            }
        endpoint = ht["accound_lend_record"]
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_lending_market_data(currency, term=None):
        payload = {
            "currency": currency, 
            "term": term
            }
        endpoint = ht['lending_market_data']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_margin_trade_data(currency):
        payload = {
            "currency": currency
            }
        endpoint = ht['margin_trade_data']
        return process_request("GET", endpoint, **payload)
    
    @staticmethod
    def get_server_time():
        return process_request("GET", ht['server_time'])
        
    @staticmethod
    def get_server_status():
        return process_request("GET", ht['server_status'])
    
    
    def get_max_borrow_amount(self, symbol):
        accounts = pd.DataFrame(self.get_margin_account()['data']['accounts'])
        max_borrow_size = float(accounts[accounts['currency'] == symbol]['maxBorrowSize'])
        return max_borrow_size
    
    def get_max_borrow_amount_isolated(self, symbol):
        if symbol != "USDT":
            base_or_quote = "baseAsset"
        else: base_or_quote = "quoteAsset"
        return self.get_isolated_account(self.symbol)['data'][base_or_quote]['borrowableAmount']
    
    def get_position_size(self, account_size, entry, stop_loss, symbol, risk=2):
        distance_percentage = (abs(entry - stop_loss)/entry)*100
        multiplier = risk/distance_percentage
        if symbol != 'USDT':
            base_or_quote = 'combined_balance_in_base'
        else: base_or_quote = 'combined_balance_in_quote'
        combined_balance = self.get_balances(entry)
        combined_balance = combined_balance[base_or_quote]
        position_size = dec_round(multiplier) * combined_balance
        desired_borrow_amount = max(position_size - account_size, 0)
        max_borrow_amount = dec_round(self.get_max_borrow_amount_isolated(symbol))
        borrow_amount = dec_round(min(max_borrow_amount, desired_borrow_amount))
        log(f"""
            distance_percentage: {distance_percentage},
            multiplier: {multiplier},
            position_size: {position_size}, 
            desired_borrow_amount: {desired_borrow_amount}, 
            max_borrow_amount: {max_borrow_amount}, 
            borrow_amount: {borrow_amount}""")
        if desired_borrow_amount > borrow_amount:
            desired_vs_actual_borrow_amount = desired_borrow_amount - borrow_amount
            position_size = position_size - desired_vs_actual_borrow_amount
            log(f"""desired_vs_actual_borrow_amount: {desired_vs_actual_borrow_amount},
                position_size: {position_size}""")
        return position_size, borrow_amount
    
    def get_position_size_2(self, account_size, entry, stop_loss, symbol, risk=2):
        distance_percentage = (abs(entry - stop_loss)/entry)*100
        multiplier = risk/distance_percentage
        if symbol != 'USDT':
            base_or_quote = 'combined_balance_in_base'
        else: base_or_quote = 'combined_balance_in_quote'
        combined_balance = self.get_balances(entry)
        combined_balance = combined_balance[base_or_quote]
        position_size = dec_round(multiplier) * combined_balance
        desired_borrow_amount = max(position_size - account_size, 0)
        max_borrow_amount = dec_round(self.get_max_borrow_amount_isolated(symbol))
        borrow_amount = dec_round(min(max_borrow_amount, desired_borrow_amount))
        log(f"""
            distance_percentage: {distance_percentage},
            multiplier: {multiplier},
            position_size: {position_size}, 
            desired_borrow_amount: {desired_borrow_amount}, 
            max_borrow_amount: {max_borrow_amount}, 
            borrow_amount: {borrow_amount}""")
        if desired_borrow_amount > borrow_amount:
            desired_vs_actual_borrow_amount = desired_borrow_amount - borrow_amount
            position_size = position_size - desired_vs_actual_borrow_amount
            log(f"""desired_vs_actual_borrow_amount: {desired_vs_actual_borrow_amount},
                position_size: {position_size}""")
            return 0, 0
        return position_size, borrow_amount
    
    def get_debt_ratio(self):
        return self.get_margin_account()['data']['debtRatio']
    
    def _validate_then_repay(self, currency, available_balance, liability):
        bal = float(available_balance)
        debt = float(liability)
        if debt == 0:
            return
        else: 
            return self.one_click_repayment(currency, bal)

    def pay_all(self):
        margin_accounts = pd.DataFrame(self.get_margin_account()['data']['accounts'])
        margin_accounts['owes'] = np.where(margin_accounts['liability'] != "0", 1, 0)
        selected_rows = margin_accounts.loc[margin_accounts['owes']==1]
        selected_rows.apply(lambda row: self._validate_then_repay(row['currency'], row['availableBalance'], row['liability']), axis=1)    

    def pay_all_isolated(self):
        isolated_accounts = pd.DataFrame(self.get_isolated_accounts(self.symbol.split("-")[1])['data']['assets'])
        with_debt = isolated_accounts[isolated_accounts['debtRatio']!="0"]
        if len(with_debt.index):
            
            base = with_debt['baseAsset'].reset_index(drop=True)[0]
            quote = with_debt['quoteAsset'].reset_index(drop=True)[0]
            for x in (base, quote):
                if x['liability'] != "0":
                    return self.one_click_repayment_isolated(self.symbol, x['currency'], min(float(x['totalBalance']), float(x['availableBalance'])))
    
    def consider_trade(self, signal):
        self.pay_all_isolated()
        signal = signal
        if not self.current_position:
            self.cancel_stop_orders()
            current_price = dec_round(self.get_ticker(self.symbol)['data']['price'], self.quote_rounding)
            balances = self.get_balances(current_price)
            risk = dec_round(self.risk, digits=0)
            base = self.symbol.split('-')[0]
            quote = self.symbol.split('-')[1]
            if signal == "SHORT":
                log(f"  > > > signal: SHORT | current price: {current_price} | timestamp: {self.timestamp}")
                atr = dec_round(self.atr_signal, self.quote_rounding)
                # current_price_last_candle = dec_round(self.candles['close'].head(2)[1])
                # self.upper_stop = dec_round(current_price + atr * dec_round(stop_loss_multiplier), self.quote_rounding)
                self.upper_stop = dec_round(current_price * (1 + self.distance_threshold), self.quote_rounding)
                self.lower_stop = dec_round(current_price * (1 - self.distance_threshold), self.quote_rounding)
                # self.lower_stop = dec_round(current_price - atr * dec_round(take_profit_multiplier), self.quote_rounding)
                position_size, borrow_amount = self.get_position_size(
                    balances['base_balance'], 
                    current_price, 
                    self.upper_stop, 
                    symbol = base,
                    risk=risk)
                # distance_percent = abs((self.lower_stop - current_price)/current_price)
                # if distance_percent <= 0.006: 
                #     log(f'Position ignored {self.timestamp} take profit: {self.upper_stop} | stop loss: {self.lower_stop}')
                #     self.upper_stop = None
                #     self.lower_stop = None
                #     return
                log(f"stop loss: {self.upper_stop} | take profit: {self.lower_stop} | position size: {position_size} | ")
                send_email("SHORT POSITION TAKEN", f"current price: {current_price}, stop loss: {self.lower_stop} | take profit: {self.upper_stop} | position size: {position_size}")
                borrow_amount = dec_round(max(borrow_amount, 0.05), self.base_rounding, rounding="ROUND_UP")
                borrow_amount = round(borrow_amount, self.base_borrow_precision_rounding[self.symbol])
                borrow = self.post_borrow_order_isolated(self.symbol, self.symbol.split('-')[0], borrow_amount)
                log(f"borrow amount: {borrow_amount} | borrow order: {json.dumps(borrow)}")
                base_balance = balances['base_balance'] + borrow_amount
                position_size = dec_round(position_size * dec_round(0.99), self.base_rounding)
                log(f"base balance: {base_balance}")
                log("> > > placing market sell order...")
                log(f"Position size used: {position_size}")
                order = self.margin_market_order(
                    "sell", 
                    self.symbol, 
                    size=position_size)
                log(f"ORDER: {json.dumps(order)}")
                order_id = order['data']['orderId']
                order_details = self.get_order(order_id)
                log(f" > ORDER DETAILS: {order_details}")
                log("> > > setting up stops...")
                place_stop_loss = self.stop_order_market(
                    "buy",
                    symbol=self.symbol,
                    stop="entry",
                    stop_price=self.upper_stop,
                    trade_type="MARGIN_ISOLATED_TRADE",
                    size=str(position_size)
                    )
                stop_loss_id = place_stop_loss['data']['orderId']
                stop_loss_details = self.get_stop_order_info(stop_loss_id)
                log(f" > STOP LOSS DETAILS : {json.dumps(stop_loss_details)}")
                place_take_profit = self.stop_order_market(
                    "buy",
                    symbol=self.symbol, 
                    stop="loss",
                    stop_price=self.lower_stop,
                    trade_type="MARGIN_ISOLATED_TRADE",
                    size=str(position_size)
                    )
                take_profit_id = place_take_profit['data']['orderId']
                take_profit_details = self.get_stop_order_info(take_profit_id)
                log(f" > TAKE PROFIT DETAILS: {json.dumps(take_profit_details)}")
                self.current_position = "SHORT"
            
            if signal == "LONG":
                log(f"  > > > signal: LONG | current price: {current_price} | timestamp: {self.timestamp}")
                atr = dec_round(self.atr_signal, self.quote_rounding)
                # current_price_last_candle = dec_round(self.candles['close'].head(2)[1])
                # self.upper_stop = dec_round(current_price + atr * dec_round(take_profit_multiplier), self.quote_rounding)
                # self.lower_stop = dec_round(current_price - atr * dec_round(stop_loss_multiplier), self.quote_rounding)
                self.upper_stop = dec_round(current_price * (1 + self.distance_threshold), self.quote_rounding)
                self.lower_stop = dec_round(current_price * (1 - self.distance_threshold), self.quote_rounding)
                position_size, borrow_amount = self.get_position_size(
                    balances['quote_balance'], 
                    current_price, 
                    self.lower_stop, 
                    symbol = quote,
                    risk=risk)
                # distance_percent = abs((self.upper_stop - current_price)/current_price)
                # if distance_percent <= 0.006: 
                #     log(f'Position ignored {self.timestamp} take profit: {self.upper_stop} | stop loss: {self.lower_stop}')
                #     self.upper_stop = None
                #     self.lower_stop = None
                #     return
                log(f"stop loss: {self.lower_stop} | take profit: {self.upper_stop} | position size: {position_size}")
                send_email("LONG POSITION TAKEN", f"current price: {current_price}, stop loss: {self.lower_stop} | take profit: {self.upper_stop} | position size: {position_size}")
                borrow_amount = dec_round(max(borrow_amount, 0), digits=0, rounding="ROUND_UP")
                borrow = self.post_borrow_order_isolated(self.symbol, self.symbol.split('-')[1], borrow_amount)
                log(f"borrow amount: {borrow_amount} | borrow order: {json.dumps(borrow)}")
                quote_balance = balances['quote_balance'] + borrow_amount
                position_size = position_size * dec_round(0.99, digits=self.quote_rounding)
                log(f"quote balance: {quote_balance}")
                log("> > > placing market buy order...")
                log(f"Position size used: {position_size}")
                order = self.margin_market_order(
                    "buy", 
                    self.symbol, 
                    funds=position_size)
                log(f"ORDER: {json.dumps(order)}")
                order_id = order['data']['orderId']
                order_details = self.get_order(order_id)
                log(f" > ORDER DETAILS: {order_details}")
                log("> > > setting up stops...")
                place_stop_loss = self.stop_order_market(
                    "sell",
                    symbol=self.symbol,
                    stop="loss",
                    stop_price=self.lower_stop,
                    trade_type="MARGIN_ISOLATED_TRADE",
                    size=str(dec_round((position_size * dec_round(1.01))/self.lower_stop, digits=self.base_rounding))
                    )
                stop_loss_id = place_stop_loss['data']['orderId']
                stop_loss_details = self.get_stop_order_info(stop_loss_id)
                log(f" > STOP LOSS DETAILS : {json.dumps(stop_loss_details)}")
                place_take_profit = self.stop_order_market(
                    "sell",
                    symbol=self.symbol, 
                    stop="entry",
                    stop_price=self.upper_stop,
                    trade_type="MARGIN_ISOLATED_TRADE",
                    size=str(dec_round((position_size * dec_round(1.01))/self.upper_stop, digits=self.base_rounding))
                    )
                take_profit_id = place_take_profit['data']['orderId']
                take_profit_details = self.get_stop_order_info(take_profit_id)
                log(f" > TAKE PROFIT DETAILS: {json.dumps(take_profit_details)}")
                self.current_position = "LONG"
            
    def stop_notice(self, price):
        
        if self.current_position == "LONG":
            if price >= self.upper_stop:
                message = f"Took profit at {self.upper_stop}|{self.timestamp}."
                
            if price <= self.lower_stop:
                message = f"Stopped loss at {self.lower_stop}|{self.timestamp}."
        if self.current_position == "SHORT":
            if price >= self.upper_stop:
                message = f"Stopped loss at {self.upper_stop}|{self.timestamp}."
            if price <= self.lower_stop:
                message = f"Took profit at {self.lower_stop}|{self.timestamp}."
        self.current_position = None
        self.upper_stop = None
        self.lower_stop = None
        log(message)
        send_email("STOP NOTICE", message)
        time.sleep(1)
        self.pay_all_isolated()
        
    def clean_up(self):
        self.pay_all()
        with_balance = self.get_base_balances()
        try:
            with_balance.apply(lambda row: self.market_order(
                side="sell", 
                symbol=row['currency']+"-USDT", 
                size=dec_round(row['available'], self.base_rounding, rounding="ROUND_DOWN"), 
                axis=1))
            log("cleaned up (hopefully)")
        except KeyError: 
            pass
        
    
    def clean_up_isolated(self):
        if not self.current_position:
            self.pay_all_isolated()
            account = self.get_isolated_account(self.symbol)['data']['baseAsset']
            available_balance = float(account['availableBalance'])
            if available_balance >= 1:
                sell = self.margin_market_order(
                    side="sell", 
                    symbol=self.symbol, 
                    size=dec_round(available_balance, self.base_rounding)
                    )
                return sell
        
    
    def get_balances(self, current_price, trade_type=None):
        base = self.symbol.split("-")[0]
        quote = self.symbol.split("-")[1]
        base_account_id = self.list_accounts(base, trade_type)['data'][0]['id']
        quote_account_id = self.list_accounts(quote, trade_type)['data'][0]['id']
        quote_balance = dec_round(self.get_account(quote_account_id)['data']['available'])
        quote_balance_in_base = dec_round(quote_balance/current_price, self.base_rounding)
        base_balance = dec_round(self.get_account(base_account_id)['data']['available'], self.base_rounding)
        base_balance_in_quote = dec_round(base_balance * current_price)
        combined_balance_in_base = base_balance + quote_balance_in_base
        combined_balance_in_quote = dec_round(quote_balance + base_balance_in_quote)
        balances = (quote_balance, 
                base_balance_in_quote, 
                quote_balance_in_base, 
                base_balance, 
                combined_balance_in_base, 
                combined_balance_in_quote)
        columns = ('quote_balance',
                   'base_balance_in_quote',
                   'quote_balance_in_base', 
                   'base_balance', 
                   'combined_balance_in_base', 
                   'combined_balance_in_quote')
        balances_dict = dict(zip(columns, balances))
        return balances_dict
    
    def get_old_candles_with_indicators(self, span=10000, rsi_span=14):
        df = self.get_mass_candles(span)
        df['rsi'] = self.rsi(df, rsi_span)
        df['stoch_rsi'] = self.stochrsi(df, rsi_span)
        df[['upperband', 'lowerband', 'in_uptrend']] = self.supertrend(df)
        df['ema50'] = self.ema(df, 50)
        df['ema200'] = self.ema(df, 200)
        df['atr'] = self.atr(df)
        return df
    

    def divergence(self, candles):
        rsi = np.array(self.rsi(candles)[::-1].reset_index(drop=True))
        closes = np.array(candles['close'][::-1].reset_index(drop=True))
        
        def get_higher_lows(data: np.array, order=5, K=2):
          '''
          Finds consecutive higher lows in price pattern.
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
          Finds consecutive lower highs in price pattern.
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
          Finds consecutive higher highs in price pattern.
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
          Finds consecutive lower lows in price pattern.
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
    
    def positions_report(self):
        with open('log.log') as f:
            lines = f.readlines()
        losses = [x for x in lines if "Stopped loss at" in x]
        wins = [x for x in lines if "Took profit at" in x]
        win_percentage = f"{round((len(wins)/(len(losses) + len(wins))) * 100, 2)}%"
        last_candle_close = dec_round(self.get_ticker(self.symbol)['data']['price'])
        balances_total = self.get_balances(last_candle_close)['combined_balance_in_quote']
        debt_ratio = dec_round(self.get_debt_ratio())
        liability = balances_total * debt_ratio
        true_balance = balances_total - liability
        now = datetime.now()
        log(f'WLC {now}')
        log(f'WLC {win_percentage}')
        log(f"WLC Trades completed: {len(wins) + len(losses)}")
        log(f"WLC Wins: {len(wins)} | Losses: {len(losses)}")
        log(f"WLC Balance Estimate: {true_balance}")
    
    def generate_start_and_end(self):
        now = datetime.now()
        now_as_unixtimestamp = int(time.mktime(now.timetuple()))
        timeframe_in_minutes = self.timeframes_in_minutes[self.timeframe]
        timeframe_in_seconds = timeframe_in_minutes * 60
        segment_length = timeframe_in_seconds * 1500
        start = str(now_as_unixtimestamp - segment_length)
        end = str(now_as_unixtimestamp)
        return start, end
        
    def get_base_balances(self):
        accounts = pd.DataFrame(self.list_accounts()['data'])
        no_usdt = accounts[accounts['currency'] != 'USDT']
        with_balance = no_usdt[no_usdt['balance'] != '0']
        return with_balance
    
    def print_balances(self):
        accounts = pd.DataFrame(self.list_accounts()['data'])
        with_balance = accounts[accounts['balance'] != '0']
        with_balance.apply(lambda row: print(row['currency'], "balance: ", row['balance']), axis=1)

    def start_up(self):
        self.pay_all()
        print('FETCHING BALANCES...')
        self.print_balances()
        clean_up = input('Run clean up? (y/n): ')
        if clean_up == 'y':
            self.clean_up()
        x = random.randint(1, 9)
        if x == 1:
            print('May God have mercy on your soul.')
        time.sleep(0.5)
        
    @staticmethod
    def data_write(file, data):
        with open(file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
        
            # write the data
            writer.writerow(data)
            
    @staticmethod
    def truncate_candles(data, start_at=None, end_at=None):
        data = data[::-1].reset_index(drop=True)
        if not start_at: 
            start_at = data['timestamp'].head(1).index[0]
        else:
            start_at = data[data['timestamp'] == start_at].index[0]
        if not end_at:
            end_at = data['timestamp'].tail(1).index[0]
        else:
            end_at = data[data['timestamp'] == end_at].index[0]
        data = data.truncate(before=start_at, after=end_at)
        data = data[::-1].reset_index(drop=True)
        return data
    
    def get_rounding(self, symbol):
        response = self.get_symbol_info(symbol)
        
        base_increment = response['baseIncrement']
        quote_increment = response['quoteIncrement']
        
        def count_dec_spaces(num):
            d = dec(num)
            return abs(d.as_tuple().exponent)
        
        base_rounding = count_dec_spaces(base_increment)
        quote_rounding = count_dec_spaces(quote_increment)
        return base_rounding, quote_rounding
    
    
    @staticmethod
    def get_isolated_account(symbol):
        response = process_request("GET", ht['isolated_account'] + "/" + symbol)
        return response
    
    def check_for_existing_positions(self):
        response = self.list_stop_orders()    
        stop_orders = response['data']['items']
        count_of_stop_orders = len(stop_orders)
        order_types = []
        stop_prices = []
        for order in stop_orders:
            side = order['side']
            order_types.append(side)
            stop_price = order['stopPrice']
            stop_prices.append(round(float(stop_price), 2))
        
        if count_of_stop_orders == 2 and order_types[0] == order_types[1]:
            if order_types[0] == "buy":
                self.current_position = "SHORT"
                self.lower_stop = min(stop_prices)
                self.upper_stop = max(stop_prices)
                
            if order_types[0] == "sell":
                self.current_position = "LONG"
                self.lower_stop = min(stop_prices)
                self.upper_stop = max(stop_prices)

    @staticmethod
    def get_most_recent_csv_files(sub_directory, last_n=2):
        
        """
        return list of pandas dataframes of most recent last_n files
        """
        current_dir = os.getcwd()
        all_file_names = glob.glob(f'{current_dir}/{sub_directory}/*.csv')
        all_file_names_sorted = sorted(all_file_names, key=os.path.getctime, reverse=False)
        target_file_names = all_file_names_sorted[-last_n:]
        pprint(f'Accessing files {target_file_names}')
        data_frames = [pd.read_csv(x, index_col=0) for x in target_file_names]
        
        return data_frames
    
    @staticmethod
    def obv(df):
        copy = df.copy()[::-1]
        obv = (np.sign(copy["close"].diff()) * copy["volume"]).fillna(0).rolling(400).sum()[::-1]
        return obv
    
    @staticmethod
    def mfi(df, period):
        typical_price = (df['close'] + df['high'] + df['low']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = []
        negative_flow = []
        
        # Loop through the typical price
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow.append(money_flow[i-1])
                negative_flow.append(0)
                
            elif typical_price[i] < typical_price[i-1]:
                negative_flow.append(money_flow[i-1])
                positive_flow.append(0)
                
            else:
                positive_flow.append(0)
                negative_flow.append(0)
                
        positive_mf = []
        negative_mf = []
        
        for i in range(period-1, len(positive_flow)):
            positive_mf.append( sum(positive_flow[i + 1- period : i+1]))
            
        for i in range(period-1, len(negative_flow)):
            negative_mf.append( sum(negative_flow[i + 1- period : i+1]))
            
        MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf) ))
        MFI = pd.Series(MFI)
        label = f"mfi{period}"
        return MFI.rename(label)
   
    @staticmethod
    def accumulation_distribution(data):
        df = data.copy()
        df = df[::-1].reset_index(drop=True)
        adl = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if df['close'][i] > df['close'][i-1]:
                mfm = ((df['close'][i] - df['low'][i]) - (df['high'][i] - df['close'][i])) / (df['high'][i] - df['low'][i])
            elif df['close'][i] < df['close'][i-1]:
                mfm = ((df['close'][i] - df['low'][i]) - (df['high'][i] - df['close'][i])) / (df['high'][i] - df['low'][i])
            else:
                mfm = 0.0
            adl[i] = adl[i-1] + (mfm * df['volume'][i])
        adl = adl[::-1].reset_index(drop=True)
        return adl
    
    def save_candlestick_chart_with_tp_and_sl(self, df, trade_timestamp, trade_direction, lower_stop, upper_stop):
        df = df.copy()[::-1]
        df['date'] = pd.to_datetime(df['timestamp'], unit="s")
        df = df.drop('timestamp', axis=1).head(160)
        df.set_index('date', inplace=True)
        
        if trade_direction == "SHORT":
            upper_stop_color = "red"
            lower_stop_color = "green"
        else:
            upper_stop_color = "green"
            lower_stop_color = "red"
        addplot_series = [
            mpf.make_addplot([upper_stop] * len(df), color=upper_stop_color, scatter=True, marker='_'),
            mpf.make_addplot([lower_stop] * len(df), color=lower_stop_color, scatter=True, marker='_')
        ]
        
        mpf.plot(df, type='candle', axtitle=f'{trade_direction} - {trade_timestamp}', 
                 ylabel='Price', addplot=addplot_series,
                  savefig=f'trade_plots//{trade_direction}//{trade_timestamp}_candlestick_chart.png')

    def get_made_trade_data(self, trade_timestamp, lookback_period, candle_span):
        first_candle = trade_timestamp - lookback_period * self.timeframes_in_minutes[self.timeframe] * 60
        end_of_data = first_candle + candle_span * self.timeframes_in_minutes[self.timeframe] * 60
        candles = self.get_candles(self.symbol, self.timeframe, str(first_candle), str(end_of_data))
        trade_opener = candles[candles['timestamp'] <= trade_timestamp + 180].head(1).reset_index(drop=True)
        trade_opener_price = trade_opener['close'][0]
        upper_stop = round(trade_opener_price * (1 + float(self.distance_threshold)), 4)
        lower_stop = round(trade_opener_price * (1 - float(self.distance_threshold)), 4)
        
        no_prior_candles = candles[:-lookback_period]
        lower_stop_triggers = list(no_prior_candles[no_prior_candles['low'] <= lower_stop].index)
        upper_stop_triggers = list(no_prior_candles[no_prior_candles['high'] >= upper_stop].index)
        
        if len(lower_stop_triggers):
            lower_stop_trigger_index = max(lower_stop_triggers)
        else:
            lower_stop_trigger_index = 0
        
        if len(upper_stop_triggers):
            upper_stop_trigger_index = max(upper_stop_triggers)
        else:
            upper_stop_trigger_index = 0
        
        first_trigger_index = max(lower_stop_trigger_index, upper_stop_trigger_index)
        candles_segment = candles[first_trigger_index:]
        candles_segment = candles_segment[::-1].reset_index(drop=True)
        trade_opener_index = lookback_period
        return candles_segment, trade_opener_index, lower_stop, upper_stop, trade_opener_price
    
    @staticmethod
    def temp_object(temp_object=None, save_or_load="load"):
        if save_or_load == "load":
            return joblib.load("temp_object.pkl")
        else:
            joblib.dump(temp_object, "temp_object.pkl")
            
    # add to live trader (note date 5/19/2023)
    
    def get_precalculated_candle(self, recent_candles, precalculated_candles):
            oldest_timestamp = recent_candles.sort_values('timestamp', ascending=True).reset_index(drop=True)['timestamp'][0]
            target_timestamp = oldest_timestamp - self.timeframes_in_minutes[self.timeframe] * 60
            target_candle = precalculated_candles[precalculated_candles['timestamp'] == target_timestamp]
            return target_candle
        
    def ema_with_lookback(self, recent_candles, precalculated_candles, span):
        
        def get_ema(value, previous_ema_value, span):
            a = 2/(span+1)
            return value * a + (previous_ema_value * (1-a))
        
        precalculated_candle = self.get_precalculated_candle(recent_candles, precalculated_candles)
        recent_candles['ema'] = 0
        recent_candles = pd.concat([recent_candles, precalculated_candle], axis=0).sort_values('timestamp', ascending=True).reset_index(drop=True)
        for n in range(len(recent_candles) - 1):
            n += 1
            recent_candles['ema'].iloc[n] = get_ema(recent_candles['close'][n], recent_candles['ema'][n-1], 1000)
        return recent_candles
    
    