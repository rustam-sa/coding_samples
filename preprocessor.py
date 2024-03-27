import numpy as np
import pandas as pd
import sys, os, glob
import joblib
import time
from arch import arch_model
from scipy.stats import iqr, boxcox
from pathlib import Path
from pprint import pprint
from datetime import datetime
from sklearn.impute import SimpleImputer	
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, Normalizer, FunctionTransformer

from logging_tools import time_it
from data_analyzer import DataAnalyzer

pd.set_option('display.max_columns', None)
cachedir = 'cachedir'
memory = joblib.Memory(cachedir, verbose=0)


class Preprocessor:

    def __init__(self, params, dataframe, cols, steps, n_steps, save_directory):
        """Initialize a Preprocessor instance with configuration parameters, input data, and processing details.
        
        This constructor sets up the preprocessing environment with specified parameters, data, and file paths
        for saving results. It prepares the instance for performing a variety of preprocessing tasks on
        financial time series data, including scaling, transforming, segmenting, and generating technical indicators.

        Parameters:
        - params (dict): A dictionary containing preprocessing parameters such as 'long_or_short', 'candle_span', 
        'atr_multiplier', 'distance_threshold', and 'model_id'. These parameters are used for defining
        the trading strategy, indicator calculations, and model identification.
        - dataframe (pd.DataFrame): The input data to be preprocessed, expected to be a pandas DataFrame
        containing time series data of financial markets.
        - cols (list of str): A list of column names from 'dataframe' that will be specifically processed.
        - steps (list of int): A list defining specific steps or intervals to be used in certain preprocessing
        tasks, such as calculating moving averages or segmenting data.
        - n_steps (int): The number of steps to consider for calculations that require a dynamic range,
        like generating multiple moving averages.
        - save_directory (str or Path): The directory path where processed data and other outputs will be saved.
        
        Attributes:
        - Initializes various instance attributes based on the parameters provided, including settings
        for indicators, a DataAnalyzer instance for calculating technical indicators, and paths for saving outputs.
        """
        self.long_or_short = params['long_or_short']
        self.candle_span = params['candle_span']
        self.atr_multiplier = params['atr_multiplier']
        self.distance_threshold = params['distance_threshold']
        self.params = params
        self.cols = cols
        self.df = dataframe
        self.data_analyzer = DataAnalyzer()
        self.find_wins = memory.cache(self.find_wins)
        self.model_id = params['model_id']
        self.n_steps = n_steps
        self.save_directory = save_directory
        self.scaler_directory = self.get_scaling_directory()
        self.steps = steps
    
    def _return_index_of_take_profit(self, index):
        """Determine the index where a take-profit or stop-loss condition is triggered.

        This method iterates through a specified segment of the DataFrame starting from the given index
        and checks if the price hits the take-profit or stop-loss levels defined by 'upper_stop' and 'lower_stop'.
        The method returns the index of the first occurrence where the condition is met. If the conditions
        are not met within the span of 'candle_span' periods, it returns np.Inf to indicate that the take-profit
        or stop-loss was not triggered.

        Parameters:
        - index (int): The starting index from which to check for take-profit or stop-loss conditions.

        Returns:
        - float: The index at which the take-profit or stop-loss condition is first met. Returns np.Inf if the
        conditions are not met within the 'candle_span' periods from the starting index.

        Notes:
        - This method relies on 'upper_stop' and 'lower_stop' columns in the DataFrame, which should be set
        prior to calling this method. These columns represent the take-profit and stop-loss price levels,
        respectively.
        - The 'candle_span' attribute of the class determines how far ahead (in terms of index) this method
        checks for a condition trigger. The check is inclusive of the start index and goes up to
        'index + candle_span'.
        - The method supports both long and short positions as specified by the 'long_or_short' attribute
        of the class. For long positions, it checks if the 'high' price crosses 'upper_stop'. For short
        positions, it checks if the 'low' price crosses 'lower_stop'.
        """
        df_len = len(self.df)
        if index + self.candle_span >= df_len:
            return np.Inf
        
        upper_stop = self.df.at[index, "upper_stop"]
        lower_stop = self.df.at[index, "lower_stop"]
        frame_segment = self.df.iloc[index+1:index+self.candle_span, :]
        upper_triggers = frame_segment.index[frame_segment['high'] >= upper_stop].tolist()
        lower_triggers = frame_segment.index[frame_segment['low'] <= lower_stop].tolist()
        
        if not (lower_triggers + upper_triggers):
            return np.Inf
        
        if lower_triggers and not upper_triggers:
            if self.long_or_short == "short":
                return min(lower_triggers)
            else:
                return np.Inf
            
        if not lower_triggers and upper_triggers:
            if self.long_or_short == "long":
                return min(upper_triggers)
            else: 
                return np.Inf
        
        if lower_triggers and upper_triggers:
            if self.long_or_short == "long":
                if min(lower_triggers) > min(upper_triggers):
                    return min(upper_triggers)
            
                else:
                    return np.Inf
                
            if self.long_or_short == "short":
                if min(lower_triggers) < min(upper_triggers):
                    return min(lower_triggers)
                else: 
                    return np.Inf
    
    def save_weights(self, segments, column_name="win"):
        """Save the distribution of win/loss or other categorical outcomes as weights into a file.

        This method takes segmented data and calculates the distribution of a specified categorical column
        (e.g., win/loss). It then saves this distribution as weights in a pickle file for later use, such as
        sample weighting in machine learning models. The weights represent the normalized frequency of each
        category within the specified column across all segments.

        Parameters:
        - segments (list of pd.DataFrame): A list of pandas DataFrame segments from the preprocessed data.
        Each segment is expected to contain at least the column specified by `column_name`.
        - column_name (str, optional): The name of the column for which to calculate the weights. This column
        should contain categorical data (e.g., win/loss indicators). Defaults to "win".
        """
        labels = [s[column_name] for s in segments]
        weights = [l.value_counts(normalize=True)*100 for l in labels]
        temp_objects_directory = f"G:/coding_projects/model_creation_for_klb/labeled_data/{self.trade_direction}/temp_objects/weights.pkl"
        temp_objects_directory = Path(temp_objects_directory)
        joblib.dump(weights, temp_objects_directory)
        print("Saved weights from preprocessor.")
        print(f"Path: {temp_objects_directory}.")
    
    def load_weights(self, num_of_segments):
        """Load the weights from a previously saved file and selects weights for a specified number of segments.

        This method reads a pickle file containing saved weights, which represent the distribution of outcomes
        (e.g., win/loss) across various data segments. It then selects a subset of these weights corresponding
        to the specified number of segments, normalizes these selected weights to sum up to 100, and returns
        them.

        Parameters:
        - num_of_segments (int): The number of segments for which to load and normalize weights. This determines
        how many of the most recently saved weights to include in the output.

        Returns:
        - pd.Series: A pandas Series containing the normalized weights for the specified number of segments.
        The indices correspond to the categories (e.g., win/loss), and the values represent the percentage
        weight of each category.
        """
        weights_dir = f"G:/coding_projects/model_creation_for_klb/labeled_data/{self.trade_direction}/temp_objects/weights.pkl"
        weights_dir = Path(weights_dir)
        raw_weights = joblib.load(weights_dir)
        selected_weights = raw_weights[:num_of_segments]
        print("Selecting weights:")
        print(selected_weights)
        weights = pd.DataFrame(selected_weights).sum()
        total = weights.sum()
        weights = weights.apply(lambda x: (x / total) * 100)
        return weights
    
    @staticmethod
    def profit_check(df, long_or_short, distance_percentage_threshold):
        """Check whether each trade in the DataFrame could be potentially profitable based on 
        the distance percentage threshold from the entry price to the stop levels.

        This method calculates whether the potential profit (for a long trade) or loss (for a short trade)
        exceeds a specified threshold. 

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the trade data. Must include 'close', 'upper_stop', 
        and 'lower_stop' columns.
        - long_or_short (str): A string indicating the type of trade. "long" or "short". 
        - distance_percentage_threshold (float): The minimum required percentage distance between the entry price 
        ('close') and the target price.

        Returns:
        - pd.Series: A series of binary indicators (1 or 0) for each trade in the DataFrame, where 1 indicates that 
        the trade is potentially profitable based on the distance percentage threshold.

        Raises:
        - ValueError: If `long_or_short` is not one of the acceptable values ("long" or "short").
        """
        if long_or_short == "short":
            target_price = df['lower_stop']
        else: 
            target_price = df['upper_stop']
        price = df['close']
        profitable = abs((target_price - price)/price) > distance_percentage_threshold
        return profitable.astype(int)
        
    def find_wins(self):
        """Identify winning trades based on take-profit criteria and update the DataFrame.

        Computes 'upper_stop' and 'lower_stop' for each row, determines if these levels are reached
        within 'candle_span' periods, and marks trades as wins (1) if conditions are met, otherwise as losses (0).
        Updates the DataFrame in place by adding a 'win' column with these indicators and removes temporary columns.
        
        Returns:
        - pd.Series: Series of win indicators (1 for win, 0 for loss) for all trades in the DataFrame.
        """
        self.df = self.df[::-1].reset_index(drop=True)
        self.df['upper_stop'] = self.df['close'] * (1 + self.distance_threshold)
        self.df['lower_stop'] = self.df['close'] * (1 - self.distance_threshold)
        self.df['took_profit_index'] = self.df.index.map(self._return_index_of_take_profit)
        self.df['win'] = np.where(self.df['took_profit_index'] != np.Inf, 1, 0)
        self.df = self.df.drop(['upper_stop', 'lower_stop', 'took_profit_index'], axis=1)
        self.df = self.df[::-1].reset_index(drop=True)
        self.df.to_csv("find_wins_test.csv")
        return self.df['win']

    def diff(self, column):
        """Calculate the difference between consecutive elements in a given column.

        This method reverses the order of the DataFrame, computes the difference between
        consecutive elements in the specified column, and then reverses the order back to
        the original. It's useful for identifying changes between periods in time series data.

        Parameters:
        - column (pd.Series): A pandas Series for which the difference between consecutive
        elements will be calculated.

        Returns:
        - pd.Series: A Series containing the differences, with the same index as the input column.
        """
        col_reversed = column[::-1]
        return col_reversed.diff()[::-1]
        
    def segment(self):
        """Divide the DataFrame into segments and save each as a CSV file.
    
        Segments the data based on 'num_of_segments' defined in the class parameters, calculates the
        size of each segment, and iterates over the DataFrame to split it accordingly. Each segment is
        saved as a separate CSV file in the specified save directory, including any sample weights if present.

        Note: Assumes the DataFrame is already sorted in descending order by timestamp.
        """
        line_count = len(self.df)
        segment_size = int(line_count/self.params['num_of_segments'])
        pprint(f"Segment size: {segment_size}")
        segments = []
        split_start = 0
        for n in range(self.params['num_of_segments']):
            split_end = split_start + segment_size
            segment = self.df.iloc[split_start:split_end, :]
            segments.append(segment)
            split_start = split_end
        # self.save_weights(segments)
        dataframe_count = self.params['num_of_segments']
        with open(f'{self.save_directory}/dataframe_count.txt', 'w') as f:
            f.write(str(dataframe_count))
        for i, segment in enumerate(segments[::-1]):
            time.sleep(0.5)
            end_timestamp = int(segment['timestamp'].iloc[0])
            start_timestamp = int(segment['timestamp'].iloc[-1])
            segment_filename = f"{self.save_directory}/segment-{i}_{segment_size}_{start_timestamp}-{end_timestamp}.csv"
            segment = segment.drop(['timestamp'], axis=1)
            if 'sample_weights' in segment.columns.to_list():
                weights = segment['sample_weights']
                joblib.dump(weights, f"{self.save_directory}/train_data_sample_weights_{i}.pkl")
                segment = segment.drop(['sample_weights'], axis=1)
            segment.to_csv(segment_filename)
            pprint(f"{segment_filename} saved")
            
    def dataframe_count_override(self, num):
        """Override the count of dataframes in the specified save directory with a new value.
    
        Write the given number to a text file named 'preprocessed_dataframe_count.txt' within the save directory.
        This action updates the recorded count of segmented dataframes, useful for manual adjustments or
        corrections to the number of data segments after preprocessing.

        Parameters:
        - num (int): The new count of dataframes to record.
        """
        with open(f'labeled_data/{self.long_or_short}/preprocessed_dataframe_count.txt', 'w') as f:
            f.write(str(num))
            
    def create_bagging_segments_with_replacement(self, segment_size):
        """Create bagging segments from the dataset with replacement and save each as a CSV file.

        Randomly selects rows to form segments of a specified size, with replacement, to create bagged versions
        of the original data. This technique is useful for ensemble methods that benefit from bootstrapping.
        Each segment is saved as a separate CSV file in the 'save_directory'. Sample weights, if present, are
        also handled and saved in a corresponding pickle file.

        Parameters:
        - segment_size (int): The number of rows each segment should contain.
        """

        def save_segment(segment, segment_number):
            """Save a given data segment to a CSV file, including sample weights if present.

            Serializes the segment as a CSV file named according to its segment number and other parameters defined
            in the class. If 'sample_weights' are included in the segment columns, they are extracted and saved separately
            in a pickle file. 

            Parameters:
            - segment (pd.DataFrame): The data segment to be saved.
            - segment_number (int): The identifier for the segment, used in naming the output file.
            """
            segment = pd.DataFrame(segment, columns = self.df.columns)
            segment = segment.drop(['timestamp'], axis=1)
            if 'sample_weights' in segment.columns.to_list():
                weights = segment['sample_weights']
                joblib.dump(weights, f"{self.save_directory}/train_data_sample_weights_{segment_number}.pkl")
                segment = segment.drop(['sample_weights'], axis=1)
                
            time.sleep(0.5)
            segment_filename = f"labeled_data/{self.long_or_short}/segment_{segment_number}_{self.params['symbol']}-{self.params['timeframe']}-{suffix}-distance_percentage_{self.distance_threshold}_{self.long_or_short}.csv"
            segment.to_csv(segment_filename)
            pprint(f"{segment_filename} saved")
            
        suffix = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        dataset = np.array(self.df.copy())
        indexes = np.array(np.where(dataset.T[0])).flatten()
        segments_indexes = [np.random.choice(indexes, size=segment_size, replace=True) for n in range(self.params['num_of_segments'])]
        # segments = [dataset[segment_index] for segment_index in segments_indexes]
        counter = 0
        for segment_index in segments_indexes:
            segment = dataset[segment_index]
            save_segment(segment, counter)
            counter += 1
            
        # self.save_weights(segments, column_name="win")
        save_directory = "labeled_data"    
        dataframe_count = self.params['num_of_segments']
        with open(f'{save_directory}/{self.long_or_short}/preprocessed_dataframe_count.txt', 'w') as f:
            f.write(str(dataframe_count))
        
    def add_preprocessing_row(self):
        """Append a row with the current preprocessing parameters to a tracking DataFrame and save it.

        This method creates or updates a CSV file that logs the preprocessing parameters used in each run.
        It ensures that each set of parameters is recorded for reference. The method handles the creation 
        of a new row in the log DataFrame with the current parameters and saves this updated log back to 
        a CSV file.
        """
        
        def insert_missing_columns(columns:list, df):
            """Insert missing columns into a DataFrame with default values.

            Iterates through a list of column names and adds any that are missing from the DataFrame. Newly added
            columns are initialized with default values. This ensures the DataFrame has a consistent structure,
            especially before processing steps that require specific columns.

            Parameters:
            - columns (list): A list of column names to ensure are present in the DataFrame.
            - df (pd.DataFrame): The DataFrame to check and modify.

            Returns:
            - pd.DataFrame: The updated DataFrame with all specified columns present.
            """
            columns = np.array(columns)
            csv_columns = np.array(list(df.columns))
            cols_to_insert = np.setdiff1d(columns, csv_columns)
            for col in cols_to_insert:
                df[col] = 0
        
            return df
        
        filename = f"{self.long_or_short}_models_preprocessed.csv"
        preprocessed_models = pd.read_csv(filename).reset_index(drop=True)
        preprocessed_models['model_id'] = preprocessed_models['model_id'].astype(str).str.slice(0,12)
        model_row = pd.DataFrame(self.params, index = [len(preprocessed_models)])
        columns = list(model_row.columns)
        preprocessed_models = insert_missing_columns(columns, preprocessed_models)
        preprocessed_models = pd.concat([preprocessed_models, model_row], axis=0).set_index("model_id")
        preprocessed_models.to_csv(filename)
        return preprocessed_models
    
    @staticmethod
    def apply_rolling_window_scaling(segment, window_size):
        """Apply min-max scaling to a segment using a rolling window approach.

        Scales each column in the segment based on minimum and maximum values within a rolling window. This method
        is designed for time-series data where local context (defined by the window size) is important for scaling.
        The scaling formula used is (value - min) / (max - min), applied within each window for each column.

        Parameters:
        - segment (pd.DataFrame): The data segment to scale. Expected to have numerical columns.
        - window_size (int): The size of the rolling window to use for calculating min and max values for scaling.

        Returns:
        - pd.DataFrame: The scaled segment, with values transformed to a 0-1 scale based on local window min and max.
        """
        def rolling_window_scaler(arr, window_size):
            """Apply rolling window scaling on a 2D numpy array where data is ordered 
            in descending time (top row is newest). Drops rows that don't have a
            full rolling window.
            
            Args:
            - arr (np.ndarray): 2D numpy array
            - window_size (int): Size of the rolling window.
            
            Returns:
            - np.ndarray: Scaled array.
            """
            n_rows, n_cols = arr.shape
            rows_to_keep = n_rows - window_size + 1
            scaled_arr = np.empty((rows_to_keep, n_cols), dtype=np.float64)
            
            for i in range(rows_to_keep):
                # Determine the start and end indices for the rolling window
                start = i
                end = i + window_size
                
                # Extract the window
                window = arr[start:end, :]
                
                # Find min and max values in the window for each column
                min_vals = np.min(window, axis=0)
                max_vals = np.max(window, axis=0)
                
                # Scale the current row using min-max scaling of the window
                scaled_row = (arr[i, :] - min_vals) / (max_vals - min_vals + 1e-10)
                
                scaled_arr[i, :] = scaled_row
            
            return scaled_arr
        
        segment = segment.dropna()
        segment_parts = []
        segment_wins_timestamps = segment[['timestamp', 'win']]
        segment_wins_timestamps = segment_wins_timestamps.iloc[:-window_size + 1]
        segment_parts.append(segment_wins_timestamps)
        segment = segment.drop(['timestamp', 'win'], axis=1)
        numerical_segment = segment.copy().select_dtypes(include=['number'])
        non_numerical_df = segment.select_dtypes(exclude=['number'])
        non_numerical_df = non_numerical_df.iloc[:-window_size + 1]
        segment_parts.append(non_numerical_df)
        saved_numerical_columns = numerical_segment.columns
        segment_np = np.array(numerical_segment)
        scaled_segment_np = rolling_window_scaler(segment_np, window_size)
        scaled_segment_pd = pd.DataFrame(scaled_segment_np, columns=saved_numerical_columns)
        segment_parts.append(scaled_segment_pd)
        scaled_segment_pd = pd.concat(segment_parts, axis=1)
        return scaled_segment_pd
    
    @staticmethod
    def apply_rolling_window_standardization(segment, window_size):
        """Standardize a segment using a rolling window approach.

        This method computes the z-score for each value in the segment based on the mean and standard deviation
        within a rolling window. This approach is useful for time-series data where the statistical properties
        can vary over time. Standardization formula used is (value - mean) / std within each window for each column.

        Parameters:
        - segment (pd.DataFrame): The data segment to standardize. Expected to have numerical columns.
        - window_size (int): The size of the rolling window to use for calculating mean and standard deviation.

        Returns:
        - pd.DataFrame: The standardized segment, with each value transformed to a z-score based on local window statistics.
        """

        def rolling_window_standardize(arr, window_size):
            """Apply a rolling window standardization on a numpy array.
            
            Parameters:
            - arr: A 2D numpy array where rows are in descending order.
            - window_size: The size of the rolling window.
            
            Returns:
            - A 2D numpy array with standardized values.
            """
            # Reverse the array since the rows are in descending order
            arr = arr[::-1]
            
            # Compute rolling mean and standard deviation
            shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
            strides = arr.strides + (arr.strides[-1],)
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            means = np.mean(windows, axis=-1)
            stds = np.std(windows, axis=-1)
            
            # Create an empty result array
            standardized = np.empty_like(arr)
            
            # Apply standardization
            for i in range(1, window_size):
                standardized[i-1] = (arr[i-1] - means[i-1]) / (stds[i-1] + 1e-8)  # Add a small value to prevent division by zero
            for i in range(window_size - 1, arr.shape[0]):
                standardized[i] = (arr[i] - means[i - window_size + 1]) / (stds[i - window_size + 1] + 1e-8)
                
            # Reverse the standardized array back to original order
            return standardized[::-1]
        
        
        segment = segment.dropna()
        segment_parts = []
        segment_wins_timestamps = segment[['timestamp', 'win']]
        segment_wins_timestamps = segment_wins_timestamps.iloc[:-window_size + 1]
        segment_parts.append(segment_wins_timestamps)
        segment = segment.drop(['timestamp', 'win'], axis=1)
        numerical_segment = segment.copy().select_dtypes(include=['number'])
        non_numerical_df = segment.select_dtypes(exclude=['number'])
        non_numerical_df = non_numerical_df.iloc[:-window_size + 1]
        segment_parts.append(non_numerical_df)
        saved_numerical_columns = numerical_segment.columns
        segment_np = np.array(numerical_segment)
        scaled_segment_np = rolling_window_standardize(segment_np, window_size)
        scaled_segment_pd = pd.DataFrame(scaled_segment_np, columns=saved_numerical_columns)
        segment_parts.append(scaled_segment_pd)
        scaled_segment_pd = pd.concat(segment_parts, axis=1)
        return scaled_segment_pd
    
    @staticmethod
    def apply_onehot(segment):
        """Convert categorical column 'divergence' in the segment to one-hot encoded columns.

        Parameters:
        - segment (pd.DataFrame): The data segment containing the 'divergence' column.

        Returns:
        - pd.DataFrame: The segment with 'divergence' replaced by one-hot encoded columns.
        """
        divergence = segment[['divergence']]
        print(divergence)
        categories = [['no divergence', 'bullish', 'bearish']]
        categorical_transformer = ColumnTransformer(transformers=[
            ('one-hot', OneHotEncoder(categories=categories), divergence.columns)
            ])
        divergence = categorical_transformer.fit_transform(divergence)
        print(divergence)
        print(categories[0])
        divergence = pd.DataFrame(divergence, columns=categories[0])
        segment = segment.drop(['divergence'], axis=1)
        segment = pd.concat([segment, divergence], axis=1)
        print(segment)
        return segment
    
    def custom_preprocess(self, scaling_window_size, scramble, bagging):
        """Perform custom preprocessing on the DataFrame including scaling, indicator addition, and optional scrambling or bagging.

        Applies a series of preprocessing steps to the DataFrame stored in this instance. These steps include adding technical
        indicators, scaling numerical features using a specified window size, optionally scrambling the data for randomization,
        and optionally creating bagging segments with replacement. Finalizes by saving the preprocessed data and
        updating a log with preprocessing parameters.

        Parameters:
        - scaling_window_size (int): The size of the window to use for rolling window scaling operations.
        - scramble (bool): If True, randomize the order of the DataFrame rows.
        - bagging (bool): If True, create bagging segments from the data with replacement.

        Returns:
        - pd.DataFrame: The preprocessed DataFrame after all operations have been applied.
        """
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.0001
        self.find_wins()
        self.add_indicators()  
        with_nas = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        without_nas = len(self.df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        diff = ['close', 'volume', 'amount', 'obv']
        print(self.df)
        no_log = ['obv', 'divergence', 'timestamp', 'win']
        # cols_to_log = [col for col in self.df.columns if col not in no_log]
        # for col in cols_to_log: 
        #     if "rsi" not in col:
        #         self.df[col] = np.log(self.df[col])
        
        # self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # self.df = self.df.dropna().reset_index(drop=True)
        
        # for col in diff:
        #     self.df[f'{col}_diffed'] = self.diff(self.df[col])
            
        emas = [col for col in self.df.columns if 'ema' in col]
        volume_emas = [col for col in emas if "volume" in col]
        emas = [col for col in emas if "volume" not in col]
        
        for col in emas:
            self.df[f'{col} difference'] = self.df[col] - self.df['close']
            
        for col in volume_emas:
            self.df[f'{col} difference'] = self.df[col] - self.df['volume']
            
            
        drop_cols = ['high', 'low', 'open'] + emas + volume_emas
        self.df = self.df.drop(drop_cols, axis=1)
        # self.df = self.apply_rolling_window_scaling(self.df, scaling_window_size)
        # self.df = self.apply_rolling_window_standardization(self.df, scaling_window_size)
        self.df = self.apply_onehot(self.df)
        labels = self.df['win'] # moves columns to right most spot
        self.df = self.df.drop(['win'], axis=1)
        self.df['win'] = labels
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv("preprocess_test.csv", index=False)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments
        self.add_preprocessing_row()
        return self.df
    
    def custom_preprocess2(self, scaling_window_size, scramble, bagging):
        """Apply an alternative custom preprocessing sequence to the DataFrame.

        Performs a series of preprocessing operations tailored to financial time series data. 
        Includes scaling based on a rolling window, computing differences, logging transformations, 
        and optionally scrambling the dataset or creating bagged segments.

        Parameters:
        - scaling_window_size (int): Defines the window size for rolling scaling operations.
        - scramble (bool): If True, the dataset rows are randomized to remove temporal ordering.
        - bagging (bool): If True, generates bagged segments of the dataset with replacement for ensemble training.

        Returns:
        - pd.DataFrame: The DataFrame after applying the specified preprocessing operations.
        """
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.0001
        self.find_wins()
        self.add_indicators()  
        with_nas = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        without_nas = len(self.df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        diff = ['close', 'volume', 'amount', 'obv']
        
        print(self.df)
        no_log = ['obv', 'timestamp', 'win']
        cols_to_log = [col for col in self.df.columns if col not in no_log]
        for col in cols_to_log: 
            if "rsi" not in col:
                self.df[col] = np.log(self.df[col])
        
        # self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # self.df = self.df.dropna().reset_index(drop=True)
        
        for col in diff:
            self.df[f'{col}_diffed'] = self.diff(self.df[col])
            
        emas = [col for col in self.df.columns if 'ema' in col]
        volume_emas = [col for col in emas if "volume" in col]
        emas = [col for col in emas if "volume" not in col]
        
        for col in emas:
            self.df[f'{col} difference'] = self.df[col] - self.df['close']
            
        for col in volume_emas:
            self.df[f'{col} difference'] = self.df[col] - self.df['volume']
            
            
        drop_cols = ['high', 'low', 'open'] + emas + volume_emas
        # self.df = self.df.drop(drop_cols, axis=1)
        self.df = self.apply_rolling_window_scaling(self.df, scaling_window_size)
        # self.df = self.apply_rolling_window_standardization(self.df, scaling_window_size)
        self.df = self.apply_onehot(self.df)
        labels = self.df['win'] # moves columns to right most spot
        self.df = self.df.drop(['win'], axis=1)
        self.df['win'] = labels
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv("preprocess_test.csv", index=False)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments
        self.add_preprocessing_row()
        return self.df
    
    @staticmethod
    def rolling_window_robust_scaler_ascending(arr, window_size):
        """Scale an array using a robust method with a rolling window, assuming ascending order.

        Applies a robust scaling technique to a numpy array based on a rolling window approach. 
        The array is assumed to be in ascending order with the most recent data at the end.

        Parameters:
        - arr (np.array): The input array to be scaled. Expected to be a 2D array where scaling is applied column-wise.
        - window_size (int): The size of the rolling window used for calculating the median and IQR.

        Returns:
        - np.array: The scaled array, with each window's values transformed based on its local median and IQR.
        """
    # Ensure input is a numpy array
        arr = np.array(arr)
        
        # Create a copy to store scaled values
        scaled_arr = arr.copy()
        
        # Number of rows and columns
        rows, cols = arr.shape
        
        # Calculate leftover rows at the bottom (newest data)
        leftover_rows = rows % window_size
        
        # Scale the newest rows that don't fit into a complete window
        for col in range(cols):
            if leftover_rows:
                window_data = arr[-leftover_rows:, col]
                med = np.median(window_data)
                range_iqr = iqr(window_data)
                if range_iqr == 0:
                    range_iqr = 1
                scaled_window_data = (window_data - med) / range_iqr
                scaled_arr[-leftover_rows:, col] = scaled_window_data
        
        # Iterate over each column
        for col in range(cols):
            # For each window in the column, starting after the newest rows
            for start in range(rows - leftover_rows - window_size, -1, -window_size):
                end = start + window_size
                window_data = arr[start:end, col]
                
                # Compute median and IQR for the window
                med = np.median(window_data)
                range_iqr = iqr(window_data)
                
                # Handle case where IQR is 0 to avoid division by zero
                if range_iqr == 0:
                    range_iqr = 1
                
                # Scale the window data
                scaled_window_data = (window_data - med) / range_iqr
                
                # Replace the original data with scaled data
                scaled_arr[start:end, col] = scaled_window_data
            
        return scaled_arr
    
    def preprocess_percent_change_robust_rolling_scaling(self, scaling_window_size, scramble, bagging):
        """Apply percent change and robust rolling scaling to the DataFrame.

        Transforms the DataFrame by first calculating the percentage change for relevant features, then applies
        robust scaling using a rolling window. This method is designed to normalize feature values and reduce
        the impact of outliers. Optionally, the method supports scrambling the data to randomize the order 
        and creating bagged segments for ensemble model training.

        Parameters:
        - scaling_window_size (int): The size of the rolling window for robust scaling.
        - scramble (bool): If set to True, randomizes the order of the DataFrame rows.
        - bagging (bool): If set to True, creates bagged segments from the dataset with replacement.

        Returns:
        - pd.DataFrame: The DataFrame after applying percentage change transformation and robust rolling scaling.
        """
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.0001
        self.find_wins()
        self.add_indicators()  
        with_nas = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        without_nas = len(self.df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        self.df = self.percentage_change_scaled(scaling_window_size)
        self.df = self.df.dropna().reset_index(drop=True)
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv("preprocess_test.csv", index=False)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments
        self.add_preprocessing_row()
        return self.df
    
    def preprocess_percent_change(self, scramble, bagging):
        """Calculate percent changes for specified columns and optionally scramble or create bagging segments.

        Computes the percentage change for columns relevant to financial analysis (e.g., close prices, volumes)
        to capture relative changes over time. Can optionally scramble the dataset to randomize the order of rows 
        or create bagging segments with replacement.

        Parameters:
        - scramble (bool): If True, randomizes the order of rows in the DataFrame.
        - bagging (bool): If True, generates bagged segments from the DataFrame with replacement for use in ensemble modeling.

        Returns:
        - pd.DataFrame: The DataFrame with percentage changes applied to specified columns and any additional requested preprocessing.
        """
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.0001
        self.find_wins()
        self.add_indicators()  
        with_nas = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        without_nas = len(self.df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        self.df = self.apply_percentage_change()
        self.df = self.df.dropna().reset_index(drop=True)
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv("preprocess_test.csv", index=False)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments
        self.add_preprocessing_row()
        return self.df
    
    def preprocess_percent_change_keep_features(self, scramble, bagging):
        """Compute percent changes while retaining original features, with options for scrambling and bagging.

        Calculates the percent change for each column specified for transformation, adding these as new columns alongside the original 
        features in the DataFrame. This approach maintains the raw data while providing transformed features for model training. 
        
        Parameters:
        - scramble (bool): If True, shuffles the order of the DataFrame rows to randomize data presentation.
        - bagging (bool): If True, creates multiple bagged subsets of the data with replacement, suitable for training ensemble models.

        Returns:
        - pd.DataFrame: The enhanced DataFrame containing both original and percent change features, after applying optional scrambling or bagging.
        """
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.0001
        self.find_wins()
        self.add_indicators()  
        with_nas = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        without_nas = len(self.df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        self.df = self.apply_percentage_change_keep_features()
        self.df = self.df.dropna().reset_index(drop=True)
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv("preprocess_test.csv", index=False)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments
        self.add_preprocessing_row()
        return self.df
    
    def preprocess(self, scramble, bagging):
        """Preprocess the DataFrame with standard transformations, optionally scrambling or bagging the data.

        Executes a predefined sequence of preprocessing steps on the DataFrame. Includes cleaning, normalization, 
        and feature engineering. Can optionally randomize the data order or generate bagged data segments to support 
        diverse modeling approaches like ensemble methods.

        Parameters:
        - scramble (bool): If set to True, randomizes the order of the DataFrame rows to mitigate sequential bias.
        - bagging (bool): If set to True, produces bagged versions of the dataset with replacement.

        Returns:
        - pd.DataFrame: The DataFrame after applying all preprocessing steps, ready for analysis or modeling.
        """
        pprint(f"Rows in data: {len(self.df)}")
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.0001
        self.find_wins()
        self.add_indicators()  
        with_nas = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        self.df = self.log_diff(self.df)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna().reset_index(drop=True)
        without_nas = len(self.df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv("preprocess_test.csv", index=False)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments
        self.add_preprocessing_row()
        return self.df
    
    def preprocess_log(self, scramble, bagging):
        """Apply logarithmic transformation to the DataFrame, with options for scrambling and bagging.

        This method transforms the numerical columns in the DataFrame by applying a logarithm.

        Parameters:
        - scramble (bool): If True, the DataFrame rows are randomized to remove any order bias.
        - bagging (bool): If True, creates bagged segments from the transformed data with replacement.

        Returns:
        - pd.DataFrame: The log-transformed DataFrame, potentially scrambled or segmented for bagging, depending on the options selected.
        """
        pprint(f"Rows in data: {len(self.df)}")
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.0001
        self.find_wins()
        self.add_indicators()  
        with_nas = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        self.df = self.log_no_diff(self.df)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df = self.df.dropna().reset_index(drop=True)
        without_nas = len(self.df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv("preprocess_test.csv", index=False)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments
        self.add_preprocessing_row()
        return self.df
    
    def add_indicators(self):
        """Add technical indicators to the DataFrame as new columns.

        Utilizes the DataAnalyzer instance to calculate and append a variety of technical indicators specified 
        in the class parameters to the DataFrame. This can include, but is not limited to, the Relative Strength 
        Index (RSI), moving averages (such as EMA), and others based on the columns indicated for processing.

        Indicators to be added are determined by the 'cols' attribute of the class. The parameters for these calculations, 
        such as periods for moving averages, are expected to be found in the 'params' attribute.

        Updates the DataFrame in place by adding each indicator as a new column with a prefixed name indicating 
        the type of indicator (e.g., 'ema', 'rsi').
        """
        if "rsi" in self.cols:
            self.df['rsi'] = self.data_analyzer.rsi(self.df, self.params['rsi_span'])
        if "stoch_rsi" in self.cols:
            self.df['stoch_rsi'] = self.data_analyzer.stochrsi(self.df, self.params['rsi_span'])
        if "atr" in self.cols:
            self.df['atr'] = self.data_analyzer.atr(self.df, self.params['atr_period'])
        if "obv" in self.cols:
            self.df['obv'] = self.data_analyzer.obv(self.df)
    
        indicators = {}
        indicators['divergence'] = self.data_analyzer.divergence(self.df)
        for n in range(1, self.n_steps + 1):
            if f"ema{n}" in self.cols:
                indicators[f'ema{n}'] = self.data_analyzer.ema(self.df, self.params[f'ema_period_{n}'])
            if f"volume_ema{n}" in self.cols:
                indicators[f'volume_ema{n}'] = self.data_analyzer.volume_ema(self.df, self.params[f'volume_ema_period_{n}'])
            if f"rsi{n}" in self.cols:
                indicators[f'rsi{n}'] = self.data_analyzer.rsi(self.df, self.params[f'rsi_span{n}'])
            if f"stoch_rsi{n}" in self.cols:
                indicators[f'stoch_rsi{n}'] = self.data_analyzer.stochrsi(self.df, self.params[f'rsi_span{n}'])
            if f"atr{n}" in self.cols:
                indicators[f'atr{n}'] = self.data_analyzer.atr(self.df, self.params[f'atr_period{n}'])
            # indicators[f'mfi{n}'] = self.data_analyzer.mfi(self.df, self.params[f'mfi_period{n}'])
            # indicators[f'top_bb{n}'], indicators[f'bottom_bb{n}'] = self.data_analyzer.bollinger_bands(self.df, self.params[f'bb_span_{n}'])
            # indicators[f'upperband{n}'], indicators[f'lowerband{n}'], indicators[f'in_uptrend{n}'] = self.data_analyzer.supertrend(
            #     self.df, params[f'supertrend_period{n}'])
        
        indicators = pd.DataFrame(indicators)
        self.df = pd.concat([self.df, indicators], axis=1)
        return self.df
        
    @staticmethod
    def dynamic_fit_transform(dataframe):
        """Dynamically fit and transform features of a DataFrame using rolling windows.

        Applies a dynamic transformation process to the DataFrame, fitting and transforming features
        based on rolling windows of data. Handles time-series data where the statistical properties
        may vary over time. Each window's data is used to fit a transformation model (e.g., scaling or normalization),
        which is then applied to transform the data within that window.

        - Creates a preprocessing model based on the data within each rolling window.
        - Applies this model to transform the data in the corresponding window.
        - Concatenates the transformed segments to form the full transformed DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame containing the time-series data to be transformed.

        Returns:
        - pd.DataFrame: A new DataFrame where each feature has been dynamically fitted and transformed based on rolling windows.
        """
        def create_preprocessor(reference_data):
            categorical_features = reference_data.select_dtypes(exclude='number').columns.tolist()
            numerical_features = reference_data.select_dtypes(include='number').columns.tolist()
            numeric_pipeline = Pipeline(steps=[
                # ('impute', SimpleImputer(strategy='median')),
                # ('minmax_scaler', MinMaxScaler()),    # Min-max scaling step
                # ('robust_scaler', RobustScaler()),                
                ('scaler', StandardScaler()),
            
                ])
            categories = [['no divergence', 'bullish', 'bearish']]
            categorical_pipeline = Pipeline(steps=[
                # ('impute', SimpleImputer(strategy='most_frequent')),
                ('one-hot', OneHotEncoder(categories=categories))
                ])
        
            full_processor = ColumnTransformer(transformers=[
                ('number', numeric_pipeline, numerical_features),
                ('category', categorical_pipeline, categorical_features)
                ])
            full_processor = full_processor.fit(reference_data)
            return full_processor
        
        
        def _main(data, n):
            if n < 500:
                return np.array([np.NaN for c in data.columns]).flatten()
            else:
                fit_data = data[n-500:n]
                preprocessor = create_preprocessor(fit_data)
                sample = data.iloc[[n]]
                transformed_sample = preprocessor.transform(sample).flatten()
                return transformed_sample
            
            
        data = dataframe.copy()[::-1].reset_index(drop=True)
        timestamps_and_labels = data[['timestamp', 'win']]
        data = data.drop(['timestamp', 'win'], axis=1)
        transformed_data = [_main(data, n) for n in range(len(data))]
        transformed_data = pd.DataFrame(transformed_data)
        transformed_data['timestamp'] = timestamps_and_labels['timestamp']
        transformed_data['win'] = timestamps_and_labels['win']
        return transformed_data[::-1].reset_index(drop=True)
    
        
    def log_diff(self, original_df):
        """Apply logarithmic difference transformation to selected columns of a DataFrame.

        Transforms the specified columns of the DataFrame by first taking the natural logarithm of each value, 
        then calculating the difference between consecutive log-transformed values.

        Parameters:
        - original_df (pd.DataFrame): The original DataFrame to transform.

        Returns:
        - pd.DataFrame: A DataFrame with the log-difference transformation applied to the specified columns.
        """
        df = original_df.copy()
        no_diff = []
        #no_diff = ['obv', 'divergence', 'timestamp', 'win']
        for col in df.columns:
            if "rsi" in col:
                no_diff.append(col)
            if "ema" in col:
                no_diff.append(col)
            if "atr" in col:
                no_diff.append(col)
                
        df_no_diff = df[no_diff]
        diff = [item for item in df.columns if item not in df_no_diff]
        df_diff = df[diff]
        df_flipped = df_diff.copy()[::-1]
        df_log = np.log(df_flipped)
        df_log.columns = df_diff.columns
        df_log_diff = df_log.diff()                                                                              
        df_log_diff = df_log_diff[::-1] # reflippening
        new_df = pd.concat([df_log_diff, df_no_diff], axis=1)
        # df_transformed = self.dynamic_fit_transform(new_df)
        return new_df
    
    def log_no_diff(self, original_df):
        """Apply logarithmic transformation to the specified columns of a DataFrame without differencing.

        Transforms selected columns of the DataFrame by taking the natural logarithm of each value. 

        Parameters:
        - original_df (pd.DataFrame): The original DataFrame to which the logarithmic transformation will be applied.

        Returns:
        - pd.DataFrame: The DataFrame after applying the logarithmic transformation to the specified columns.
        """
        df = original_df.copy()
        no_log = ['obv', 'divergence', 'timestamp', 'win', 'atr']
            
        for col in df.columns:
            if col not in no_log:
                df[col] = np.log(df[col])
        # df_transformed = self.dynamic_fit_transform(new_df)
        return df
    
    @staticmethod
    def rolling_window_robust_scaler(arr, window_size):
        """Scale an array using robust scaling within a rolling window.

        Applies a robust scaling method to a numpy array based on a rolling window. Each window's data is scaled 
        independently, using the median and interquartile range (IQR) to reduce the impact of outliers. 

        Parameters:
        - arr (np.array): The input array to be scaled. Expected to be a 2D array with scaling applied column-wise.
        - window_size (int): The size of the rolling window used to calculate the median and IQR for scaling.

        Returns:
        - np.array: The scaled array, with each window's values transformed based on its local median and IQR.
        """
        # Ensure input is a numpy array
        arr = np.array(arr)
        
        # Create a copy to store scaled values
        scaled_arr = arr.copy()
        
        # Number of rows and columns
        rows, cols = arr.shape
        
        # Iterate over each column
        for col in range(cols):
            # For each window in the column, starting from the newest data
            for end in range(window_size, rows + 1):
                start = end - window_size
                window_data = arr[start:end, col]
                
                # Compute median and IQR for the window
                med = np.median(window_data)
                range_iqr = iqr(window_data)
                
                # Handle case where IQR is 0 to avoid division by zero
                if range_iqr == 0:
                    range_iqr = 1
                
                # Scale the entire window data using its statistics
                scaled_window_data = (window_data - med) / range_iqr
                
                # Replace the original window data with scaled data
                scaled_arr[start:end, col] = scaled_window_data
        
        # Remove the oldest rows that didn't get scaled with a full window
        scaled_arr = scaled_arr[:-(window_size-1)]
        
        return scaled_arr
        
    @staticmethod
    def percentage_change(series):
        """Calculate the percentage change between consecutive elements in a series.

        This static method computes the percentage change from one element to the next in the input series.

        Parameters:
        - series (pd.Series): A pandas Series for which the percentage change will be calculated.

        Returns:
        - pd.Series: A Series containing the percentage changes between consecutive elements.
        """
        series = series[::-1].pct_change().mul(100)[::-1]
        return series
    
    def percentage_change_scaled(self, scaling_window_size):
        """Scale the DataFrame by applying percentage change followed by robust scaling with a rolling window.

        First calculates the percentage change for each column in the DataFrame to capture relative changes over time.
        Then, applies robust scaling using a rolling window approach to reduce the influence of outliers and scale features 
        to a more uniform range.

        Parameters:
        - scaling_window_size (int): The size of the rolling window for the robust scaling operation.

        Returns:
        - pd.DataFrame: The DataFrame after applying percentage change and robust rolling window scaling.
        """
        df = self.df
        df_cols = df.columns.to_list()
        cols_to_derive_percentage_change = ["open", "close", "high", "low", "volume", "amount", "obv"]
        emas = [col for col in df_cols if "ema" in col]
        cols_to_derive_percentage_change += emas
        for col in cols_to_derive_percentage_change:
            df[f"{col}_pct_chg"] = self.percentage_change(df[col])
        timestamps_and_labels = df[['timestamp', 'win']]
        percent_change_columns = [col for col in df.columns.to_list() if "pct_chg" in col]
        pct_chg_df = df[percent_change_columns]
        pct_chg_df_saved_columns = pct_chg_df.columns.to_list()
        pct_chg_df_array = np.array(pct_chg_df)
        pct_chg_df_array_scaled = pd.DataFrame(self.rolling_window_robust_scaler(pct_chg_df_array, window_size=scaling_window_size), columns=pct_chg_df_saved_columns)
        pct_chg_df = pd.concat([pct_chg_df_array_scaled, timestamps_and_labels], axis=1)
        return pct_chg_df
    
    def apply_percentage_change(self):
        """Apply percentage change transformation to the DataFrame for specified columns.

        Calculates the percentage change for columns that are relevant to the analysis.

        The method updates the DataFrame in place, adding new columns for each specified feature that represent the percentage 
        change from the previous row to the current row.

        Returns:
        - pd.DataFrame: The DataFrame with additional columns representing the percentage change for specified features.
        """
        df = self.df
        df_cols = df.columns.to_list()
        timestamps_labels = df[['timestamp', 'win']]
        cols_to_derive_percentage_change = ["open", "close", "high", "low", "volume", "amount", "obv"]
        emas = [col for col in df_cols if "ema" in col]
        cols_to_derive_percentage_change += emas
        for col in cols_to_derive_percentage_change:
            df[f"{col}_pct_chg"] = self.percentage_change(df[col])
        percent_change_columns = [col for col in df.columns.to_list() if "pct_chg" in col]
        pct_chg_df = df[percent_change_columns]
        pct_chg_df = pd.concat([pct_chg_df, timestamps_labels], axis=1)
        return pct_chg_df
    
    def apply_percentage_change_keep_features(self):
        """Apply percentage change to selected features while retaining the original columns in the DataFrame.

        This method enhances the DataFrame by calculating percentage changes for specified features, such as price or volume,
        without removing the original columns.

        Returns:
        - pd.DataFrame: An updated DataFrame including both the original features and their respective percentage changes.
        """
        df = self.df
        df_cols = df.columns.to_list()
        cols_to_derive_percentage_change = ["open", "close", "high", "low", "volume", "amount", "obv"]
        emas = [col for col in df_cols if "ema" in col]
        cols_to_derive_percentage_change += emas
        for col in cols_to_derive_percentage_change:
            df[f"{col}_pct_chg"] = self.percentage_change(df[col])
        percent_change_columns = [col for col in df.columns.to_list() if "pct_chg" in col]
        pct_chg_df = df[percent_change_columns]
        pct_chg_df = pd.concat([pct_chg_df, df], axis=1)
        return pct_chg_df
    
    @staticmethod
    def combine_and_filter_features(df, pct_chg_df):
        """Combine the original DataFrame with percentage change features and filter based on specified criteria.

        This method merges a DataFrame of percentage changes with the original DataFrame to enrich the data with both
        absolute and relative changes. Post-merging, it applies filtering to remove or select specific features based on
        predefined criteria.

        Parameters:
        - df (pd.DataFrame): The original DataFrame containing the full set of features.
        - pct_chg_df (pd.DataFrame): A DataFrame containing percentage change features for a subset of the original columns.

        Returns:
        - pd.DataFrame: The combined DataFrame after applying the merge and filter operation, ready for further analysis or modeling.
        """
        def remove_substring_containing(listA, listB):
            return [string for string in listA if not any(sub in string for sub in listB)]

        cols_to_drop = ['open', 'high', 'low', 'stoch_rsi']
        pct_chg_df = pct_chg_df.drop(['timestamp', 'win'], axis=1)
        df = pd.concat([pct_chg_df, df], axis=1)
        cols = df.columns.to_list()
        new_cols = remove_substring_containing(cols, cols_to_drop)
        df = df[new_cols]
        return df 
    
    @staticmethod
    def drop_nas(df):
        """Drop rows with NA values from the DataFrame and reset the index.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which NA values will be removed.

        Returns:
        - pd.DataFrame: The DataFrame after removing rows with NA values and resetting the index.
        """
        with_nas = len(df)
        df = df.dropna().reset_index(drop=True)
        without_nas = len(df)
        dropped_count = with_nas - without_nas
        pprint(f"Dropped {dropped_count} NaN rows. Index reset.")
        return df        
    
    @staticmethod
    def check_if_float64_compatible(data):
        """Check if the data values are compatible with the float64 type and identify any issues.

        Evaluates a pandas Series or DataFrame to identify values that are infinite, out of the float64 range,
        or have excessive precision that might cause issues with float64 representation. Reports the problematic values
        by printing them. Does not modify the data; it only reports potential issues.

        Parameters:
        - data (pd.Series or pd.DataFrame): The data to check for float64 compatibility.
        """
        inf_indices = np.isinf(data)
        print(data[inf_indices])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        out_of_range = (data.abs() > 1.8e308) | (data.abs() < 2.2e-308) & (data != 0)
        if out_of_range.any():
            print("Numbers out of float64 range:", data[out_of_range])
        
        # Check for excessive precision (a simplistic check)
        excessive_precision = data.apply(lambda x: len(str(x).split('.')[-1])) > 17
        if excessive_precision.any():
            print("Numbers with excessive precision:", data[excessive_precision])
    
    def diff_df(self, data):
        """Calculate the difference between consecutive rows for numerical columns in the DataFrame.

        Applies a differencing operation to each numerical column in the DataFrame to highlight changes
        between consecutive rows.

        Parameters:
        - data (pd.DataFrame): The DataFrame on which to apply the differencing operation.

        Returns:
        - pd.DataFrame: A DataFrame with the differenced values for each numerical column.
        """
        df = data.copy()
        numerical_features = df.select_dtypes(include='number').columns.tolist()
        selected_cols = [col for col in numerical_features if col not in ['timestamp', 'win']]
        for col in selected_cols:
            df[col] = self.diff(df[col])
        return df

    @staticmethod
    def move_win_to_end(df):
        """Reorder the DataFrame columns to move the 'win' column to the end.

        Parameters:
        - df (pd.DataFrame): The DataFrame whose columns are to be reordered.

        Returns:
        - pd.DataFrame: The DataFrame with the 'win' column moved to the last position.
        """
        # Reorder columns
        cols = [col for col in df if col != 'win'] + ['win']
        return df[cols]

    @staticmethod
    def create_sample_weights(data, decay_factor): # descending order
        """Generate sample weights for the data, optionally using exponential decay.

        Produces a series of sample weights for the dataset. If a decay factor is provided, the weights are generated
        using exponential decay, giving more weight to more recent samples. If no decay factor is provided, linear
        weights are used, gradually increasing from the oldest to the most recent sample.

        Parameters:
        - data (pd.DataFrame or pd.Series): The data for which sample weights are to be generated.
        - decay_factor (float, optional): The factor used for exponential decay weighting. If None, linear weights are generated.

        Returns:
        - list: A list of sample weights corresponding to each row in the data, with the most recent samples having higher weights.
        """
        n = len(data)
        print(f"LEN OF DATA: {n}")
        if not decay_factor:
            weights = [i/n for i in range(n)][::-1]
            print("Using linear weight decay for sample weights.")
            
        else:
            weights = [decay_factor**(n - 1 - i) for i in range(n)][::-1]
            print("Using exponential weight decay for sample weights.")
                
        return weights
    
    def get_percentage_change(self, cols_list, df):
        """Calculate percentage changes for specified columns in the DataFrame.

        Computes the percentage change between consecutive rows for each column specified in `cols_list`.

        Parameters:
        - cols_list (list): List of column names in `df` for which to calculate percentage changes.
        - df (pd.DataFrame): The DataFrame containing the data.

        Returns:
        - pd.DataFrame: A DataFrame with new columns representing the percentage change for each specified column. The new columns
        are named with the original column name suffixed by '_pct_chg'.
        """
        cols_list = [col for col in cols_list if col not in ['timestamp', 'win']]
        work_df = df.copy()[cols_list]
        for col in work_df.columns:
            work_df[col] = self.percentage_change(work_df[col])
        work_df.columns = [col + "_pct_chg" for col in work_df.columns]
        return work_df
    
    @staticmethod
    def get_lagged_percent_change(cols_list, df, lag_num):
        dataframe = df.copy()
        for col in cols_list:
            for n in range(1, lag_num + 1):
                dataframe[col + f"lag_{n}"] = dataframe[col].shift(-n)
        return dataframe
    
    @staticmethod
    def fibonacci_iterative(n):
        a, b = 0, 1
        result = []
        for _ in range(n):
            result.append(a)
            a, b = b, a + b
        return result
        
    def get_lagged_percent_change_fibonacci(self, cols_list, df, lag_num):
        
        steps = self.fibonacci_iterative(lag_num + 2)[2:]
        dataframe = df.copy()	
        for col in cols_list:
            for n in range(0, lag_num):
                dataframe[col + f"lag_{steps[n]}"] = dataframe[col].shift(-1 * steps[n])
        return dataframe
    
    def get_lagged_percent_change_from_steps(self, cols_list, df):
        
        steps = self.steps
        dataframe = df.copy()	
        for col in cols_list:
            for n in range(len(steps)):
                dataframe[col + f"lag_{steps[n]}"] = dataframe[col].shift(-1 * steps[n])
        return dataframe
    
    def get_lagged_cols(self, cols_list, df, lag_num):
        
        steps = self.fibonacci_iterative(lag_num + 2)[2:]
        dataframe = df.copy()	
        for col in cols_list:
            for n in range(0, lag_num):
                dataframe[col + f"lag_{steps[n]}"] = dataframe[col].shift(-1 * steps[n])
        return dataframe
    
    def get_lagged_cols_by_steps(self, cols_list, df, lag_num):
        
        steps = self.steps
        dataframe = df.copy()	
        for col in cols_list:
            for n in range(0, lag_num):
                dataframe[col + f"lag_{steps[n]}"] = dataframe[col].shift(-1 * steps[n])
        return dataframe
    
    @staticmethod
    def col_diff(df, col_a: str, col_b: str):
        a = np.array(df[col_a])
        b = np.array(df[col_b])
        df[f"{col_a}-{col_b}"] = a - b
        return df
    
    @staticmethod
    def ema_cross_overs(df):
        
        def get_crossovers(work_df):
            work_df = work_df.copy()
            work_df_lag = work_df.shift(-1)
            work_df_lag.columns = [f"{col}_lag_1" for col in work_df]
            
            non_lag = work_df
            lag = work_df_lag 
            crossovers_all = []
            for col in non_lag.columns:
                ref = np.array(non_lag[col])[:, np.newaxis]
                others = [c for c in non_lag.columns if col != c]
                others = np.array(non_lag[others])
                others_lag = [c for c in lag.columns if col not in c]
                others_lag = np.array(lag[others_lag])
                
                others_diff = others - ref
                others_lag_diff = others_lag - ref
                
                others_diff = (others_diff > 0).astype(int)
                others_lag_diff = (others_lag_diff > 0).astype(int)
                
                crossovers = ~np.all(others_diff == others_lag_diff, axis=1)
                crossovers = crossovers.astype(int)
                crossovers_all.append(crossovers)
            new_columns = [f"{col}_crossover" for col in non_lag.columns]
            return pd.DataFrame(np.array(crossovers_all).T, columns=new_columns)
        
        df = df.copy()
        emas = [col for col in df.columns if "ema" in col]
        volume_emas = [col for col in emas if "volume" in col]
        emas = [col for col in emas if "volume" not in col]
        
        emas_df = df[emas]
        volume_emas_df = df[volume_emas]
        
        ema_crosses = get_crossovers(emas_df)
        volume_emas_crosses = get_crossovers(volume_emas_df)
        df = pd.concat([df, ema_crosses, volume_emas_crosses], axis=1)
        return df 
    
    def get_scaling_directory(self):
        os = self.check_os()
        if "Windows" not in os:
            separator = "/"
        else:
            separator = "\\"
        save_directory = str(self.save_directory)
        print(f"save directory: {save_directory}")
        scaler_directory = str(self.save_directory).split(separator)[:-1]
        print('scaler_directory:')
        print(scaler_directory)
        scaler_directory = Path(*scaler_directory)
        return scaler_directory
    
    def load_scaling_transformer_and_columns(self, suffix="base"):
        transformer_name = f"most_recent_transformer_{suffix}.joblib"
        transformer_save_path = Path(self.scaler_directory, transformer_name)
        if transformer_save_path.exists():
            scaling_transformer_and_columns = joblib.load(transformer_save_path)
            print(f"{suffix} scaling_transformer_and_columns loaded successfully!")
            return scaling_transformer_and_columns
        else:
            print(f"No scaling_transformer_and_columns found at {transformer_save_path}")
            return None
    
    @staticmethod
    def create_preprocessor(reference_data): # no cats
        saved_cols = reference_data.columns
        numerical_features = reference_data.select_dtypes(include='number').columns.tolist()
        numeric_pipeline = Pipeline(steps=[
            # ('impute', SimpleImputer(strategy='median')),
            # ('minmax_scaler', MinMaxScaler()),    # Min-max scaling step
            ('robust_scaler', RobustScaler(quantile_range=(25., 75.))),                
            # ('scaler', StandardScaler()),
        
            ])

        full_processor = ColumnTransformer(transformers=[
            ('number', numeric_pipeline, numerical_features),
            ])
        full_processor = full_processor.fit(reference_data)
        
        return full_processor, saved_cols
    
    def make_scaling_transformer(self, work_df, suffix="base"):
        scaling_transformer, new_columns = self.create_preprocessor(work_df)
        transformer_name = f"most_recent_transformer_{suffix}.joblib"
        transformer_save_path = Path(self.scaler_directory, transformer_name)
        joblib.dump((scaling_transformer, new_columns), transformer_save_path)
        print(f'scaling transformer and columns saved to {self.save_directory}.')
        return scaling_transformer, new_columns
    
    def scale(self, work_df, suffix="base", update_scaler=False):
        wins_and_timestamps_exist = 0
        if 'win' in work_df.columns and 'timestamp' in work_df.columns:
            wins_and_timestamps_exist = 1
            timestamps_and_wins = work_df[['timestamp', 'win']]
            work_df = work_df.drop(['timestamp', 'win'], axis=1)
        scaling_transformer_and_columns = self.load_scaling_transformer_and_columns(suffix)
        if not scaling_transformer_and_columns:
            print(f"attempting to create {suffix} scaler")
            scaling_transformer_and_columns = self.make_scaling_transformer(work_df, suffix)
            print(f"{suffix} scaler created.")
        if update_scaler:
            print('Attempting to update existing scaler.')
            scaling_transformer, new_columns = scaling_transformer_and_columns
            scaling_transformer = scaling_transformer.fit(work_df)
            print('Scaler updated with new data')
            scaling_transformer_and_columns = (scaling_transformer, new_columns)
            transformer_name = f"most_recent_transformer_{suffix}.joblib"
            transformer_save_path = Path(self.scaler_directory, transformer_name)
            joblib.dump(scaling_transformer_and_columns, transformer_save_path)
            print(f'Updated scaling transformer and columns saved to {transformer_save_path}')
            
        scaling_transformer, new_columns = scaling_transformer_and_columns
        work_df = pd.DataFrame(scaling_transformer.transform(work_df), 
                               columns = new_columns)
        if wins_and_timestamps_exist:
            work_df = pd.concat([work_df, timestamps_and_wins], axis=1)
        return work_df
    
    
    
    def rolling_scale(self, work_df, window_size=160, suffix="base"):
        wins_and_timestamps_exist = 0
        if 'win' in work_df.columns and 'timestamp' in work_df.columns:
            wins_and_timestamps_exist = 1
            timestamps_and_wins = work_df[['timestamp', 'win']]
            work_df = work_df.drop(['timestamp', 'win'], axis=1)
       
        if wins_and_timestamps_exist:
            work_df = pd.concat([work_df, timestamps_and_wins], axis=1)
        return work_df
    
    @staticmethod
    def nan_outliers(df):
        
        def nan_outliers_series(series):
        
            # Calculate Q1 and Q3
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            
            # Calculate IQR
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with NaN
            series[(series < lower_bound) | (series > upper_bound)] = np.nan
            
            return series
        
        df = df.copy()
        for col in df.columns.to_list():
            df[col] = nan_outliers_series(df[col])
        return df
    
    def box_cox_new_lambda(self, df):
        for col in df.columns:
            df[col], lambda_best_fit = boxcox(df[col])
        lambda_name = "most_recent_boxcox_lambda.joblib"
        lambda_save_path = Path(self.scaler_directory, lambda_name)
        joblib.dump(lambda_best_fit, lambda_save_path)
        return df 

    def box_cox_existing_lambda(self, df, l):
        for col in df.columns:
            df[col] = boxcox(df[col], l)
        return df 

    def box_cox(self, df):
        lambda_name = "most_recent_boxcox_lambda.joblib"
        lambda_save_path = Path(self.scaler_directory, lambda_name)
        if lambda_save_path.exists():
            print("loading existing lambda value for box cox transform")
            l = joblib.load(lambda_save_path)
            df = self.box_cox_existing_lambda(df, l)
        else:
            df = self.box_cox_new_lambda(df)
        return df 
    
    def get_ema_slopes(self, df, append_new_columns=True):
        slopes = pd.DataFrame([])
        for s in range(len(self.steps) - 1):
            t_1 = self.steps[s]
            t_2 = self.steps[s+1]
            denominator = (t_2 - t_1)
            
            numerator = df[f'ema{s+2}'] - df[f'ema{s+1}']
            slopes[f"ema_slope{s+1}"] = numerator/denominator
            
            numerator = df[f'volume_ema{s+2}'] - df[f'volume_ema{s+1}']
            slopes[f"volume_ema_slope{s+1}"] = numerator/denominator
        
        if not append_new_columns:
            return slopes
        else:
            concated = pd.concat([df, slopes], axis=1)
            return concated
        
    
    def get_selected_col_slopes(self, df, selected_col, lag_count=3):
        work_df = df.copy()[[selected_col]]
        work_df_scaled = self.scale(work_df, selected_col + "_slope")
        lag = self.get_lagged_percent_change_fibonacci([selected_col], work_df_scaled, lag_count)
        work_df = lag
        columns = work_df.columns.to_list()
        
        slopes_to_calculate = work_df.shape[1] - 1
        slope_steps = self.fibonacci_iterative(lag_count+2)[2:]
        slopes = pd.DataFrame([])
        for s in range(slopes_to_calculate):
            x_1 = work_df[columns[s]]
            x_2 = work_df[columns[s+1]]
            numerator = x_2 - x_1
            denominator = slope_steps[s]
            slopes[f"{selected_col} slope{s+1}"] = numerator/denominator
            
        
        return slopes
    
    def get_selected_col_slopes_by_steps(self, df, selected_col):
        work_df = df.copy()[[selected_col]]
        work_df_scaled = self.scale(work_df, selected_col + "_slope")
        lag = self.get_lagged_percent_change_from_steps([selected_col], work_df_scaled)
        work_df = lag
        columns = work_df.columns.to_list()
        
        slopes_to_calculate = work_df.shape[1] - 1
        slope_steps = self.steps
        slopes = pd.DataFrame([])
        for s in range(slopes_to_calculate):
            x_1 = work_df[columns[s]]
            x_2 = work_df[columns[s+1]]
            numerator = x_2 - x_1
            denominator = slope_steps[s]
            slopes[f"{selected_col}_slope{s+1}"] = numerator/denominator
            
    
    @staticmethod
    def rolling_window_dataframe(dataframe, window_size):
        
        def rolling_window_scale_series(data, window_size):
            series = data.copy()[::-1].reset_index(drop=True)
            series = np.array(series)
            scaled_values = []
            for i in range(len(series)):
                    # Get the window data
                    if i - window_size + 1 < 0:
                        # print('Window too small.')
                        scaled_values.append(np.NaN)
                    else:
                        window_data = series[i-window_size+1:i+1]
                        median = np.median(window_data)
                        iqr = np.percentile(window_data, 75) - np.percentile(window_data, 25)
                        
                        # Perform robust scaling
                        if iqr == 0:  # To avoid division by zero
                            scaled_value = 0
                        else:
                            scaled_value = (series[i] - median) / iqr
                            # print(f"Scaling value {series[i]}, changed to {scaled_value}.")
                
                        scaled_values.append(scaled_value)
                        
            scaled_values = pd.Series(scaled_values)[::-1]
            return scaled_values
        
        work_df = dataframe.copy().drop(['timestamp', 'win'], axis=1)
        numerical_features = work_df.select_dtypes(include='number').columns.tolist()
        work_df = work_df[numerical_features]
        for col in work_df.columns:	
            dataframe[col] = rolling_window_scale_series(work_df[col], window_size)
        return dataframe
    
    
    @staticmethod
    def binarize(df):
        
        def positive_or_negative_change(df):
            df = df.copy()
            df = df.iloc[:, ::-1].diff(axis=1).iloc[:, ::-1]
            df = df.iloc[:,:-1]
            for col in df.columns:
                df[col] = np.where(df[col] > 0, 1, 0)
                
            new_cols = [f"{col}_change" for col in df.columns]
            df.columns = new_cols
            return df
        
        df = df.copy()    
        
        # getting emas
        emas = [col for col in df.columns if "ema" in col]
        volume_emas = [col for col in emas if "volume" in col]
        emas = [col for col in emas if "volume" not in col]
        volume_emas = df[volume_emas]
        emas = df[emas]
        
        
        emas = positive_or_negative_change(emas)
        volume_emas = positive_or_negative_change(volume_emas)
        
        # getting pct_chg
        pct_chg = [col for col in df.columns if "pct_chg" in col]
        
        close_pct_chg = [col for col in pct_chg if "close" in col]
        volume_pct_chg = [col for col in pct_chg if "volume" in col]
        close_pct_chg = df[close_pct_chg]
        volume_pct_chg = df[volume_pct_chg]
        
        close_pct_chg = positive_or_negative_change(close_pct_chg)
        volume_pct_chg = positive_or_negative_change(volume_pct_chg)
        df = pd.concat([
            emas, volume_emas, close_pct_chg, volume_pct_chg
            ], axis=1)
        return df
    
    
    @staticmethod
    def get_fractal_levels(data):
        df = data.copy()[::-1].reset_index(drop=True)
        # determine bullish fractal 
        def is_support(df,i):  
          cond1 = df['low'][i] < df['low'][i-1]   
          cond2 = df['low'][i] < df['low'][i+1]   
          cond3 = df['low'][i+1] < df['low'][i+2]   
          cond4 = df['low'][i-1] < df['low'][i-2]  
          return (cond1 and cond2 and cond3 and cond4) 
        # determine bearish fractal
        
        
        def is_resistance(df,i):  
          cond1 = df['high'][i] > df['high'][i-1]   
          cond2 = df['high'][i] > df['high'][i+1]   
          cond3 = df['high'][i+1] > df['high'][i+2]   
          cond4 = df['high'][i-1] > df['high'][i-2]  
          return (cond1 and cond2 and cond3 and cond4)
        
        
        # to make sure the new level area does not exist already
        def is_far_from_level(value, levels, df):    
          ave =  np.mean(df['high'] - df['low'])    
          return np.sum([abs(value-level)<ave for _,level in levels])==0
        
        
        # a list to store resistance and support levels
        levels = []
        for i in range(2, df.shape[0] - 2):  
          if is_support(df, i):    
            low = df['low'][i]    
            if is_far_from_level(low, levels, df):      
              levels.append((i, low))  
          elif is_resistance(df, i):    
            high = df['high'][i]    
            if is_far_from_level(high, levels, df):      
              levels.append((i, high))
              
              
        index, values = zip(*levels)
        series = pd.Series(values, index=index)
        
        n = len(df) - 1
        full_index = range(n+1)
        series = series.reindex(full_index)
        series = series.fillna(method='ffill')
        
        res = series[::-1].reset_index(drop=True)
        
        return res


    @staticmethod
    def custom_rolling_min_max_scaling(data_to_scale, reference_data, window_size):
        """
        input: 
            data_to_scale = data you wanna scale
            reference_data = data you're using as reference, normally the 'close' column

            both are accepted in ASCENDING ORDER

            window_size = lookback window length
        output: 
            scaled data = returns scaled data in ASCENDING ORDER
        
        """
        data_to_scale, reference_data = np.array(data_to_scale), np.array(reference_data)
        assert len(reference_data) >= len(data_to_scale)
        n = len(data_to_scale)
        scaled_data = np.zeros(n)  # Initialize an array to hold the scaled values

        # Iterate over the array to apply scaling within each window
        for i in range(n):
            # Determine the start and end of the current window
            start = i - window_size + 1
            end = i + 1
            if start < 0:
                scaled_data[i] = np.nan
            
            else:


                # Extract the current window
                window = reference_data[start:end]
                
                # Calculate the min and max values within the window
                min_val = np.min(window)
                max_val = np.max(window)
                
                # Scale the current value within the window
                if max_val != min_val:  # Avoid division by zero
                    scaled_data[i] = (data_to_scale[i] - min_val) / (max_val - min_val)
                else:  # Handle case where all values in the window are the same
                    scaled_data[i] = 0  # or any appropriate value, e.g., 0.5 or 1
        scaled_data = pd.Series(scaled_data)
        return scaled_data

    def preprocess(self, window_size, scramble, bagging, weight_decay=None, 
                                     lag_count=3, remove_outliers=False,
                                     scale_suffix="base", update_scaler=False):
        self.df = self.df.copy()
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.000001
        self.find_wins()
        self.add_indicators()        
        levels = self.get_fractal_levels(self.df)
        self.df['levels'] = levels

        # arch models
        arch_model_closes = arch_model(self.df['close'].copy()[::-1].reset_index(drop=True), mean="Zero", vol="GARCH", p=1, q=1)
        arch_model_closes_fitted = arch_model_closes.fit()

        closes_copy_flipped = self.df['close'].copy()[::-1].reset_index(drop=True)
        mad_df= pd.DataFrame([])
        for step in self.steps:
            medians = closes_copy_flipped.rolling(step).median()
            deviations_from_median = closes_copy_flipped - medians
            absolute_deviations_from_median = abs(deviations_from_median)
            mad_df[f"MAD_{step}"] = absolute_deviations_from_median.rolling(step).median()[::-1].reset_index(drop=True)
        
        volume_copy_flipped = self.df['volume'].copy()[::-1].reset_index(drop=True)

        for step in self.steps:
            medians = volume_copy_flipped.rolling(step).median()
            deviations_from_median = volume_copy_flipped - medians
            absolute_deviations_from_median = abs(deviations_from_median)
            mad_df[f"MAD_volume_{step}"] = absolute_deviations_from_median.rolling(step).median()[::-1].reset_index(drop=True)


        self.df = self.df[self.cols]
        new_data = []
        new_data.append(mad_df)

        # min max normalization 
        emas = [col for col in self.df.columns if "ema" in col]
        volume_emas = [col for col in emas if "volume" in col]

        emas = [col for col in emas if "volume" not in col]


        cols_to_scale_based_on_close = ["close", "levels"] + emas
        data_scaled_on_close = pd.DataFrame([])
        for col in cols_to_scale_based_on_close:

            reference_data = self.df['close'].copy()[::-1].reset_index(drop=True)
            scaled_col = self.custom_rolling_min_max_scaling(reference_data, self.df[col].copy()[::-1].reset_index(drop=True), window_size)
            data_scaled_on_close[col] = scaled_col[::-1].reset_index(drop=True)

        new_data.append(data_scaled_on_close)

        cols_to_scale_based_on_volume = ["volume"] + volume_emas
        data_scaled_on_volume = pd.DataFrame([])
        for col in cols_to_scale_based_on_volume:

            reference_data = self.df['volume'].copy()[::-1].reset_index(drop=True)
            scaled_col = self.custom_rolling_min_max_scaling(reference_data, self.df[col].copy()[::-1].reset_index(drop=True), window_size)
            data_scaled_on_volume[col] = scaled_col[::-1].reset_index(drop=True)

        new_data.append(data_scaled_on_volume)

        cols_to_scale_on_self = ["atr", "rsi", "amount"]
        data_scaled_on_self = pd.DataFrame([])
        for col in cols_to_scale_on_self:
            col_to_scale = self.df[col].copy()[::-1].reset_index(drop=True)
            reference_col = col_to_scale.copy()
            scaled_col = self.custom_rolling_min_max_scaling(reference_col, col_to_scale, window_size)
            data_scaled_on_self[col] = scaled_col[::-1].reset_index(drop=True)

        new_data.append(data_scaled_on_self)


        for dataframe in new_data:
            for col in dataframe.columns:
               self.df[col] = dataframe[col]
        self.df = self.df.dropna().reset_index(drop=True) 
 
        self.df['closes_cond_vol'] = arch_model_closes_fitted.conditional_volatility[::-1].reset_index(drop=True)

        arch_model_volume = arch_model(self.df['volume'].copy()[::-1].reset_index(drop=True), mean="Zero", vol="GARCH", p=1, q=1)
        arch_model_volume_fitted = arch_model_volume.fit()
        self.df['volume_cond_vol'] = arch_model_volume_fitted.conditional_volatility[::-1].reset_index(drop=True)
        self.df = self.df.dropna().reset_index(drop=True) 
        cols_to_drop = []
        self.df = self.df.drop(cols_to_drop, axis=1)
        self.df = self.drop_nas(self.df)
        self.df = self.move_win_to_end(self.df)
        
        
        if weight_decay:
            self.df['sample_weights'] = self.create_sample_weights(self.df, decay_factor=0)
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments

        return self.df  
    

    def preprocess_experimental(self, scramble, bagging, weight_decay=None, 
                                    lag_count=3, remove_outliers=False,
                                    scale_suffix="base", update_scaler=False):
        self.df = self.df.copy()
        numerical_features = self.df.select_dtypes(include='number').columns.tolist()
        self.df[numerical_features] = self.df[numerical_features] + 0.000001
        original_data = self.df.copy()
        self.find_wins()
        self.add_indicators()        
        levels = self.get_fractal_levels(self.df)
        self.df['levels'] = levels
        self.df['levels_diff'] = levels - self.df['close']
        closes_copy_flipped = self.df['close'].copy()[::-1].reset_index(drop=True)

        mad_df= pd.DataFrame([])
        for step in self.steps:
            medians = closes_copy_flipped.rolling(step).median()
            deviations_from_median = closes_copy_flipped - medians
            absolute_deviations_from_median = abs(deviations_from_median)

            mad_df[f"MAD_{step}"] = absolute_deviations_from_median.rolling(step).median()[::-1].reset_index(drop=True)
        
        volume_copy_flipped = self.df['volume'].copy()[::-1].reset_index(drop=True)

        for step in self.steps:
            medians = volume_copy_flipped.rolling(step).median()
            deviations_from_median = volume_copy_flipped - medians
            absolute_deviations_from_median = abs(deviations_from_median)

            mad_df[f"MAD_volume_{step}"] = absolute_deviations_from_median.rolling(step).median()[::-1].reset_index(drop=True)


        self.df = self.df[self.cols]
        new_data = []
        new_data.append(mad_df)

        columns = ['close', 'volume', 'amount']
        log_diff_columns = []
        for column_name in columns:
            column = self.df.copy()[column_name][::-1] # flipped
            column = np.log(column)
            column = column.diff()
            column = column[::-1]
            log_diff_columns.append(column)
        log_diff_dataframe = pd.concat(log_diff_columns, axis=1)
        new_data.append(log_diff_dataframe)


    
        # getting slopes
        emas = pd.DataFrame([])
        for col in self.df.columns:
            if "ema" in col:
                emas[col] = self.df[col] 
        slopes = self.get_ema_slopes(emas, False)
        new_data.append(slopes)
    
        # percent change
        percent_change_cols = ['close', 'volume', 'amount']
        percent_change = self.get_percentage_change(percent_change_cols, original_data)
        new_data.append(percent_change)
       
        # percent change lag
        percent_change_lag = self.get_lagged_percent_change_fibonacci(percent_change.columns, percent_change, lag_count)
        percent_change_lag = self.get_lagged_percent_change_from_steps(percent_change.columns, percent_change)
        new_data.append(percent_change_lag)
        
                
    
        for dataframe in new_data:
            for col in dataframe.columns:
                self.df[col] = dataframe[col]
        self.df = self.df.dropna().reset_index(drop=True) 
        # arch models
        arch_model_closes = arch_model(self.df['close'].copy()[::-1].reset_index(drop=True), mean="Zero", vol="GARCH", p=1, q=1)
        arch_model_closes_fitted = arch_model_closes.fit()
        self.df['closes_cond_vol'] = arch_model_closes_fitted.conditional_volatility[::-1].reset_index(drop=True)

        arch_model_volume = arch_model(self.df['volume'].copy()[::-1].reset_index(drop=True), mean="Zero", vol="GARCH", p=1, q=1)
        arch_model_volume_fitted = arch_model_volume.fit()
        self.df['volume_cond_vol'] = arch_model_volume_fitted.conditional_volatility[::-1].reset_index(drop=True)
        self.df = self.df.dropna().reset_index(drop=True) 

        cols_to_drop = []
        self.df = self.df.drop(cols_to_drop, axis=1)
        self.df = self.drop_nas(self.df)
        self.df = self.move_win_to_end(self.df)
        cols_not_to_scale = []
        cols_to_scale = [col for col in self.df.columns if col not in cols_not_to_scale]
        df_to_scale = self.df.copy()[cols_to_scale]
        scaled_df = self.scale(df_to_scale)
        for col in cols_to_scale:
            self.df[col] = scaled_df[col]
        
        
        if weight_decay:
            self.df['sample_weights'] = self.create_sample_weights(self.df, decay_factor=0)
        if scramble:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        if bagging:
            self.create_bagging_segments_with_replacement(1000000) # also saves the segments
        else: 
            self.segment() # also saves the segments

        return self.df  
    
    @staticmethod
    def check_os():
        if os.name == 'nt':
            return "Windows"
        elif os.name == 'posix':
            # For more detail, you can distinguish between Unix-like systems
            system_name = platform.system()
            if system_name == "Linux":
                return "Linux (Unix-like)"
            elif system_name == "Darwin":
                return "macOS (Unix-like)"
            else:
                return f"{system_name} (Unix-like)"
        else:
            return "Unknown OS"