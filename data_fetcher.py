import time
import csv
import pandas as pd

from datetime import datetime

from headers import ht, process_request, get_headers


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


class DataFetcher:
    def __init__(self, symbol, timeframe):

        pass

    def check_timestamp_sequence(self, df):
        
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