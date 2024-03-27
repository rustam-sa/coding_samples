"""Encapsulate functions that create logs
"""
import time
from datetime import datetime

def time_it(func):
    """Measure the execution time of a function, logging the result to a CSV file.
    
    Args:
        func (callable): The function to be wrapped and timed.
    
    Returns:
        callable: The wrapped function with timing and logging functionality.
    """
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
