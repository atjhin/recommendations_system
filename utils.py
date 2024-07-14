import time as time
import pandas as pd

#################################### Common functions ####################################

def calculate_time(func, *args, **kwargs):
    start = time.time()
    output = func(*args, **kwargs)
    total_time = time.time() - start
    if total_time > 200:
        minute = np.floor(total_time/60)
        second = round(total_time%60, 2)
        print(f"Time taken is {minute} minutes and {second} seconds")
    else:
        print(f"Time taken is {round(total_time, 2)} seconds")
    return output


##########################################################################################

#################################### Extract data  ####################################