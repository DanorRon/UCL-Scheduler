import numpy as np
import pandas as pd

# Import the database (availability DataFrame) from access_availability.py
from .access_availability import availability_df

num_days = 2
num_shifts = 12

missing_data_functionality = True #if true, will fill in missing data with "Available;" otherwise will fill in with "Not available"

#convert the availability dataframe to a boolean array
def to_bool_array(df):
    if len(df.shape) == 1:
        result = np.empty((df.shape[0]))
        for i in range(df.shape[0]):
            if df.iloc[i] == 'Available':
                result[i] = 1
            elif df.iloc[i] == 'Not available':
                result[i] = 0
            else:
                result[i] = 1 if missing_data_functionality else 0
    else:
        result = np.empty((df.shape[0], df.shape[1]))
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if df.iloc[i][j] == 'Available':
                    result[i][j] = 1
                elif df.iloc[i][j] == 'Not available':
                    result[i][j] = 0
                else:
                    result[i][j] = 1 if missing_data_functionality else 0
    return result

availability_df = availability_df.drop('', axis=1) #drop empty columns

all_members = availability_df.iloc[:, 0].tolist()
cast_members = all_members[3:]

cast_availability = to_bool_array(availability_df.iloc[3:, 1:])
cast_availability = np.reshape(cast_availability, (len(cast_members), num_days, num_shifts))

leader_availability = to_bool_array(availability_df.iloc[1, 1:])
leader_availability = np.reshape(leader_availability, (1, num_days, num_shifts))