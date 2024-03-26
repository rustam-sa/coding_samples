import time
import csv
import pandas as pd
import numpy as np 
from datetime import datetime
from logging_tools import time_it
from headers import ht, process_request




class DataFetcher:
    """Fetch and process financial data for a given symbol and timeframe.
    
    Attributes:
        symbol (str): The financial instrument symbol.
        timeframe (str): candle length.
    """

    def __init__(self, symbol, timeframe):
        """Initialize the DataFetcher with a symbol and timeframe.
        
        Args:
            symbol (str): The symbol for the financial instrument.
            timeframe (str): The timeframe for data aggregation.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        # Timeframes expressed in minutes
        self.timeframes_in_minutes = {
            "1min": 1, "3min": 3, "5min": 5, "15min": 15, "30min": 30, "1hour": 60,
            "2hour": 120, "4hour": 240, "6hour": 360, "8hour": 480, "12hour": 720,
            "1day": 1440, "1week": 10080
        }

    def check_timestamp_sequence(self, df):
        """Check the sequence of timestamps in the dataframe for any gaps relative to the expected timeframe.
        
        Args:
            df (pandas.DataFrame): The dataframe containing timestamp data to check.
            
        Returns:
            dict: A summary including first timestamp, last timestamp, presence of gaps, and count of timestamps.
        """
        res = {
            
            "first_timestamp": 0,
            "last_timestamp": 0,
            "gaps": 0,
            "count": df['timestamp'].size,
            
            }
        
        diff = df['timestamp'].astype(int).diff().dropna().reset_index(drop=True) * - 1
        gaps_test = np.where(
            diff != self.timeframes_in_minutes[self.timeframe] * 60, 1, 0)
        res['gaps'] = any(gaps_test)
        res['first_timestamp']= df['timestamp'][0] 
        res['last_timestamp'] = df['timestamp'][df['timestamp'].size - 1]
        if res["gaps"]: 
            df.to_csv("data_to_inspect/most_recent.csv")
            print("GAP: dataframe saved to data_to_inspect inside kucoin leverage bot project")
            raise Exception("GAP detected")
        return res
    
    def get_candles(self, symbol="ETH-USDT", timeframe="5min", start_at=None, end_at=None, delay=0):
        """Fetch candlestick data for a given symbol and timeframe, with optional start and end timestamps.
        
        Args:
            symbol (str): The symbol for the financial instrument. Default is "ETH-USDT".
            timeframe (str): The timeframe for data aggregation. Default is "5min".
            start_at (str): Optional. The start timestamp for the data fetch.
            end_at (str): Optional. The end timestamp for the data fetch.
            delay (int): Optional. A delay in seconds before executing the request. Default is 0.
            
        Returns:
            pandas.DataFrame: A dataframe containing the fetched candlestick data.
        """
        time.sleep(delay)
        if start_at == None and end_at == None:
            start_at, end_at = self._generate_start_and_end()
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
        check = self.check_timestamp_sequence(candles)
        print(f"""
              FETCHING params: {symbol}, 
              {timeframe}, 
              start: {start_at}, 
              end: {end_at}
              """)
        print("Received Dataframe: ", check)
        
        return candles

    @time_it
    def get_mass_candles(self, span, delay=0):
        """Fetch a large dataset of candlestick data within a specified span.

        This method aggregates multiple candlestick data fetches to compile a large dataset over a given time span for the initialized symbol and timeframe.

        Args:
            span (int): The time span (in milliseconds) over which to collect candlestick data.
            delay (int): A delay (in seconds) to wait before each fetch. Useful for rate-limited APIs.

        Returns:
            pandas.DataFrame: A dataframe containing the aggregated candlestick data.
        """
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
    
    def _generate_start_and_end(self):
        """Generate start and end timestamps for fetching data based on the current time and configured timeframe.

        Calculates start and end timestamps to fetch a segment of candlestick data ending at the current time and covering a default span based on the configured timeframe.

        Returns:
            tuple: A tuple containing the start and end timestamps as strings.
        """
        now = datetime.now()
        now_as_unixtimestamp = int(time.mktime(now.timetuple()))
        timeframe_in_minutes = self.timeframes_in_minutes[self.timeframe]
        timeframe_in_seconds = timeframe_in_minutes * 60
        segment_length = timeframe_in_seconds * 1500
        start = str(now_as_unixtimestamp - segment_length)
        end = str(now_as_unixtimestamp)
        return start, end