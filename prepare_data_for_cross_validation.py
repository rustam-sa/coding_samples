"""
Cross-Validation Data Segmentation and Preprocessing for training ML models and backtesting.

This script acquires OHLCV data for selected crypto currency via kucoing exchange API
segments trading data for use in cross-validation, 
then preprocesses (scaling, financial indicators, etc.),
enabling ML model training and backtesting over multiple temporal segments. 
It includes functionality for dynamic segmentation based on specified sample counts and 
ensures that data used for scaling features are appropriately isolated to prevent lookahead bias.

Usage:
    python prepare_data_for_cross_validation.py --cross_val_count=3 --train_segment_size=700000 --scaler_size=20000 --val_segment_size=40000 --test_segment_size=40000

Parameters:
    --cross_val_count: Number of cross-validation folds
    --train_segment_size: Number of samples in each training segment
    --scaler_size: Size of the dataset used for feature scaling
    --val_segment_size: Number of samples in each validation segment
    --test_segment_size: Number of samples in each test segment


"""

import glob
import os
import joblib
import argparse
from datetime import datetime
from pathlib import Path
from pprint import pprint


# Custom module imports for trading data handling and preprocessing
from trader import Trader # custom class for interacting with exchange API (kucoin exchange used in this case)
from directory_manager import DirectoryManager # custom class for easier directory management
from preprocessor import Preprocessor # custom class for preprocessing financial data


def top_slicer_simple(dataframe, indexes:tuple):
    """
    Removes a slice from the dataframe based on provided start and end indexes.
    
    Parameters:
    - dataframe (pd.DataFrame): DataFrame to slice.
    - indexes (tuple): Start and end index for the slice to be removed.
    
    Returns:
    - pd.DataFrame: DataFrame after removing the specified slice.
    """


    begin = indexes[0]
    end = indexes[1]
    indexes_to_remove = [n for n in range(begin, end)]
    
    df = dataframe.copy().reset_index(drop=True)
    df = df.drop(indexes_to_remove).set_index(keys=['timestamp'], drop=True)
    return df


def get_all_csv_file_names(sub_directory):
    """
    Retrieves all CSV file names in a specified sub-directory, sorted by creation time.
    
    Parameters:
    - sub_directory (str): Sub-directory to search for CSV files.
    
    Returns:
    - list: Sorted list of all CSV file names in the specified sub-directory.
    """


    current_dir = os.getcwd()
    all_file_names = glob.glob(f'{current_dir}/{sub_directory}/*.csv')
    all_file_names_sorted = sorted(all_file_names, key=os.path.getctime, reverse=False)
    return all_file_names_sorted


def get_segments(df, train_sample_count, scaler_sample_count, validation_sample_count,
                 test_sample_count):
    """
    Segments the provided DataFrame into multiple parts for training, scaling, validation,
    testing, and evaluation, intended for use in machine learning models.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame to be segmented.
    - train_sample_count (int): The number of samples for the training segment.
    - scaler_sample_count (int): The number of samples for the scaler segment.
    - validation_sample_count (int): The number of samples for the validation segment.
    - test_sample_count (int): The number of samples for the test segment.
    
    Returns:
    - list: A list containing the segmented parts of the DataFrame, each reversed and reset.
    """


    work_df = df.copy()[::-1].reset_index(drop=True)

    train_start = 0
    train_end = train_sample_count
    train_segment = work_df.copy().iloc[train_start:train_end]
    
    scaler_start = train_sample_count - scaler_sample_count
    scaler_end = train_end
    scaler_segment = work_df.copy().iloc[scaler_start:scaler_end]
    
    validation_start = scaler_end
    validation_end = validation_start + validation_sample_count
    validation_segment = work_df.copy().iloc[validation_start:validation_end]
    
    test_start = validation_end
    test_end = test_start + test_sample_count
    test_segment = work_df.copy().iloc[test_start:test_end]
    
    evaluation_start = validation_start
    evaluation_end = test_end
    evaluation_segment = work_df.copy().iloc[evaluation_start:evaluation_end]
    
    segments = [train_segment, scaler_segment, validation_segment, test_segment, evaluation_segment]
    segments = [segment[::-1].reset_index(drop=True) for segment in segments]
    
    return segments



def create_segmentation_table_scale_on_val(cross_val_count, train_segment_size, scaler_size, val_segment_size, test_segment_size):
    """
    Create a segmentation table for organizing data into different sets for cross-validation.
    This function assumes data has a temporally ascending index.
    
    Parameters:
    - cross_val_count (int): The number of cross-validation folds.
    - train_segment_size (int): The size of each training segment.
    - scaler_size (int): The size of the segment used for scaling features.
    - val_segment_size (int): The size of each validation segment.
    - test_segment_size (int): The size of each test segment.
    
    Return:
    - dict: A dictionary where keys are segment names and values are tuples indicating the
            start and end indices of each segment.
    """


    segmentation_table = {}
    segmentation_table['total_data_needed'] = train_segment_size + val_segment_size + test_segment_size * 3
    for n in range(cross_val_count):
        moving_window_size = n * test_segment_size
        train_segment_start = 0 + moving_window_size
        train_segment_end = train_segment_size + moving_window_size
        segmentation_table[f'train_segment_{n}'] = train_segment_start, train_segment_end
        

        
        val_segment_start = train_segment_end
        val_segment_end = train_segment_end + val_segment_size
        segmentation_table[f'val_segment_{n}'] = val_segment_start, val_segment_end
        
        test_segment_start = val_segment_end
        test_segment_end = val_segment_end + test_segment_size
        segmentation_table[f'test_segment_{n}'] = test_segment_start, test_segment_end
        
        evaluation_segment_start = val_segment_start
        evaluation_segment_end = test_segment_end
        segmentation_table[f'evaluation_segment_{n}'] = evaluation_segment_start, evaluation_segment_end
        
        scaler_segment_start = train_segment_end - scaler_size
        scaler_segment_end = train_segment_end
        segmentation_table[f'scaler_segment_{n}'] = scaler_segment_start, scaler_segment_end
    pprint(segmentation_table)
    return segmentation_table


def get_segment(segment, start, end):
    """
    Extract a sub-segment from the given segment based on start and end indices.
    
    Parameters:
    - segment (pd.DataFrame): The input segment from which to extract the sub-segment.
    - start (int): The start index for the sub-segment extraction.
    - end (int): The end index for the sub-segment extraction.
    
    Return:
    - pd.DataFrame: The extracted sub-segment, reversed and reset.
    """


    sub_segment = segment.copy()[::-1].reset_index(drop=True)
    sub_segment = sub_segment.iloc[start:end][::-1].reset_index(drop=True)
    return sub_segment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # count of separate cross validation sets 
    parser.add_argument("--cross_val_count", type=int, default=3) 
    # for each cycle:
    parser.add_argument("--train_segment_size", type=int, default=6000) 
    parser.add_argument("--scaler_size", type=int, default=2000)
    parser.add_argument("--val_segment_size", type=int, default=10000)
    parser.add_argument("--test_segment_size", type=int, default=10000)
    parser.add_argument("--num_of_ensemble_evaluation_segments", type=int, default=20)

    args = parser.parse_args()

    cross_val_count = args.cross_val_count
    train_segment_size = args.train_segment_size
    scaler_size = args.scaler_size
    val_segment_size = args.val_segment_size
    test_segment_size = args.test_segment_size
    num_of_ensemble_evaluation_segments = args.num_of_ensemble_evaluation_segments


    segmentation_table = create_segmentation_table_scale_on_val(cross_val_count, train_segment_size, 
                                                    scaler_size, val_segment_size, 
                                                    test_segment_size)


    timeframe, symbol = "3min", "ETH-USDT"
    tag = f"cross_val_count_{cross_val_count}_{timeframe}_{symbol}"
    trade_direction = "short"

    if len(tag):
        tag = "_" + tag
    data_params = {
        "timeframe": timeframe,
        "span": segmentation_table['total_data_needed'],
        "delay": 1
        }


    timestamp = datetime.now().strftime('%B-%d-%Y_%H-%M')
    work_dir = f"work_dir_{timestamp}_{symbol}_{data_params['span']}_scaler_size_{scaler_size}{tag}"
    dm = DirectoryManager(work_dir)


    # get unprocessed data
    client = Trader(symbol, data_params['timeframe'])
    df = client.get_mass_candles(data_params['span'], delay=data_params['delay'])   
    client.check_timestamp_sequence(df)
    original_data_dir = dm.create_new_dir("original_data")
    # save unprocessed data
    df.to_csv(f"{original_data_dir}/{symbol}_{data_params['timeframe']}_{data_params['span']}.csv", 
            index=False)
    print(f"data_getter finished {datetime.now()}")


    # create preprocessing params
    params = {}
    params['long_or_short'] = trade_direction
    params['symbol'] = symbol
    params['timeframe'] = data_params['timeframe']
    params['model_id'] = datetime.now().strftime("%Y%m%d%H%M")
    params['candle_span'] = 500
    params['atr_multiplier'] = 2.5
    params['distance_threshold'] = .01
    params['rsi_span'] = 14
    params['atr_period'] = 14

    # the span parameters for indicators 
    steps = [3, 50, 100, 1000]
    lag_count = 3
    window_size = 500

    cols =  ['close', 'volume', "rsi", "atr", 'amount', 'levels', "levels_diff"]
    cols += [f'ema{n}' for n in range(1, len(steps) + 1)]
    cols += [f'volume_ema{n}' for n in range(1, len(steps)+1)]
    cols += ['timestamp', 'win']



    n_steps = len(steps)
    for n in range(1, n_steps + 1):
        steps_index = n - 1
        params[f'supertrend_period{n}'] = steps[steps_index]
        params[f'atr_period{n}'] = steps[steps_index]
        params[f'rsi_span{n}'] = steps[steps_index]
        params[f'mfi_period{n}'] = steps[steps_index]
        params[f'ema_period_{n}'] = steps[steps_index]
        params[f'volume_ema_period_{n}'] = steps[steps_index]
        params[f'bb_span_{n}'] = steps[steps_index]


    for n in range(cross_val_count):
        cross_val_cycle_keys = [key for key in segmentation_table.keys() if str(n) in key]
        
        
        # move scaler_segment to front of list to create scaler first
        scaler_segment_key = f"scaler_segment_{n}"    
        scaler_segment_key_index = cross_val_cycle_keys.index(scaler_segment_key)
        cross_val_cycle_keys.pop(scaler_segment_key_index)
        cross_val_cycle_keys.insert(0, scaler_segment_key) 
        
        for key in cross_val_cycle_keys:
            params['num_of_segments'] = 1 
            if 'evaluation_segment' in key:
                params['num_of_segments'] = num_of_ensemble_evaluation_segments
            update_scaler = False
            if "scaler" in key:
                update_scaler = True
                f"key: {key}. update_scaler variable set to True"
            # extract segments
            start_end_tuple = segmentation_table[key]
            segment = get_segment(df, *start_end_tuple)
            
            # create dir for segments
            segment_dir = dm.create_new_dir(key + "_dir")
            
            # preprocess segment (preprocessor also saves)
            preprocessor = Preprocessor(params, segment, cols, steps, n_steps, segment_dir)
            preprocessor.preprocess_experimental(scramble=False, bagging=False, weight_decay=False, 
                                                    lag_count=lag_count, remove_outliers=False,
                                                    scale_suffix="base", update_scaler=update_scaler)

    joblib.dump(segmentation_table, Path(work_dir, "segmentation_table.json"))
    print(work_dir)