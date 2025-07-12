import numpy as np
import pandas as pd

# Import the database (availability DataFrame) from access_availability.py
from .access_availability import availability_df

#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_columns', 100)

num_days = 7
num_shifts = 13

missing_data_functionality = True #if true, will fill in missing data with "Available;" otherwise will fill in with "Not available"

# convert the availability dataframe to a boolean array
# 'False' is available, 'True' is not available
def to_bool_array(df):
    if len(df.shape) == 1:
        result = np.empty((df.shape[0]))
        for i in range(df.shape[0]):
            if df.iloc[i] == 'FALSE':
                result[i] = 1
            elif df.iloc[i] == 'TRUE':
                result[i] = 0
            else:
                result[i] = 1 if missing_data_functionality else 0
    else:
        result = np.empty((df.shape[0], df.shape[1]))
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if df.iloc[i, j] == 'FALSE':
                    result[i][j] = 1
                elif df.iloc[i, j] == 'TRUE':
                    result[i][j] = 0
                else:
                    result[i][j] = 1 if missing_data_functionality else 0
    return result

# Note: We consider the leaders to be part of the cast, so we include them in the cast members list

# parse the names of the leaders and cast members
names_column = availability_df.iloc[:, 0]
leader_index = names_column[names_column == 'Production Team'].index[0]
cast_index = names_column[names_column == 'Cast'].index[0]
leaders = names_column[leader_index + 1:cast_index]
cast_members = names_column[cast_index + 1:]
cast_members = pd.concat([leaders, cast_members]).tolist() #include leaders in all members
leaders = leaders.tolist()

num_members = len(cast_members)
num_leaders = len(leaders)

# Note: The order of removal matters! Removing rows/columns changes indices

# trim unneeded rows: all rows before leader names, and 'Cast' row
availability_df = availability_df.iloc[leader_index + 1:]
availability_df = availability_df.drop(availability_df.iloc[:, 0][availability_df.iloc[:, 0] == 'Cast'].index[0], axis=0) #remove 'Cast' row

# trim unneeded columns: only the first column (containing names)
availability_df = availability_df.drop(0, axis=1)

# Remove empty columns (columns between days)
signal_row = availability_df.iloc[2, :] # if elt in this row is '', then the column is empty
empty_columns = signal_row[signal_row == ''].index.tolist()
availability_df = availability_df.drop(empty_columns, axis=1) #drop empty columns

# Remove empty rows (probably not any)
signal_column = availability_df.iloc[:, 0] # if elt in this column is '', then the row is empty
empty_rows = signal_column[signal_column == ''].index.tolist()
availability_df = availability_df.drop(empty_rows, axis=0) #drop empty rows

# convert the availability dataframe to a boolean array
cast_availability = to_bool_array(availability_df)
cast_availability = np.reshape(cast_availability, (len(cast_members), num_days, num_shifts))

leader_availability = to_bool_array(availability_df.iloc[0 : num_leaders, :])
leader_availability = np.reshape(leader_availability, (num_leaders, num_days, num_shifts))