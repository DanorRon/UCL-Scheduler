"""
Availability Manager - Combines Google Sheets access and data parsing functionality.
Provides a unified interface for getting and parsing availability data.
"""

import numpy as np
import pandas as pd
import gspread
from gspread.auth import service_account
import os
import re
from pathlib import Path
from typing import Tuple, List, Optional

num_days = 7
num_shifts = 13

def parse_spreadsheet_url(url: str) -> str:
    """
    Extract the spreadsheet key from a Google Sheets URL.
    
    Args:
        url: Google Sheets URL (e.g., https://docs.google.com/spreadsheets/d/SPREADSHEET_KEY/edit)
        
    Returns:
        The spreadsheet key
        
    Raises:
        ValueError: If the URL doesn't contain a valid spreadsheet key
    """
    # Pattern to match Google Sheets URLs
    # Matches URLs like:
    # https://docs.google.com/spreadsheets/d/SPREADSHEET_KEY/edit
    # https://docs.google.com/spreadsheets/d/SPREADSHEET_KEY/edit#gid=0
    # https://docs.google.com/spreadsheets/d/SPREADSHEET_KEY/
    pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Could not extract spreadsheet key from URL: {url}")
    
    return match.group(1)

def get_availability_data(spreadsheet_key: str) -> pd.DataFrame:
    """
    Get availability data from Google Sheets.
    
    Args:
        spreadsheet_key: The Google Sheets spreadsheet key
        
    Returns:
        pandas DataFrame containing the availability data
        
    Raises:
        FileNotFoundError: If credentials file is not found and environment variable is not set
        Exception: If data cannot be retrieved from Google Sheets
    """
    import os
    import json
    
    # Get the credentials file path relative to the package
    package_dir = Path(__file__).parent.parent
    credentials_path = package_dir / "credentials" / "ucl-scheduler-866343adad65.json"
    
    # Try to use credentials file first (for local development)
    if credentials_path.exists():
        print("Using credentials file for local development")
        gc = service_account(filename=credentials_path)
    else:
        # Fall back to environment variable (for deployment)
        print("Credentials file not found, checking environment variable")
        credentials_json = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
        
        if not credentials_json:
            raise FileNotFoundError(
                f"Credentials file not found at {credentials_path} and "
                "GOOGLE_SHEETS_CREDENTIALS environment variable not set"
            )
        
        try:
            # Parse the JSON credentials from environment variable
            credentials_dict = json.loads(credentials_json)
            from gspread import service_account_from_dict
            gc = service_account_from_dict(credentials_dict)
            print("Using credentials from environment variable")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in GOOGLE_SHEETS_CREDENTIALS environment variable")
        except Exception as e:
            raise ValueError(f"Failed to create Google Sheets client from environment: {str(e)}")
    
    # Open the spreadsheet by key
    try:
        spreadsheet = gc.open_by_key(spreadsheet_key)
    except Exception as e:
        raise Exception(f"Failed to open spreadsheet with key {spreadsheet_key}: {str(e)}")
    
    # Get worksheet data (assuming first worksheet)
    try:
        worksheet = spreadsheet.get_worksheet(0)
        data = worksheet.get_all_values()
    except Exception as e:
        raise Exception(f"Failed to retrieve data from spreadsheet: {str(e)}")
    
    # Convert to pandas DataFrame
    availability_df = pd.DataFrame(data)
    
    if availability_df.empty:
        raise Exception("No data retrieved from spreadsheet")
    
    return availability_df

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

def parse_availability_data(availability_df: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Parse availability data from a Google Sheets DataFrame.
    
    Args:
        availability_df: Raw DataFrame from Google Sheets
        
    Returns:
        Tuple of (cast_members, leaders, cast_availability, leader_availability)
        - cast_members: List of all member names (including leaders)
        - leaders: List of leader names only
        - cast_availability: Boolean array of shape (num_members, num_days, num_shifts)
        - leader_availability: Boolean array of shape (num_leaders, num_days, num_shifts)
        
    Raises:
        ValueError: If the DataFrame structure is invalid or required sections are missing
    """

    #pd.set_option('display.max_rows', 100)
    #pd.set_option('display.max_columns', 100)

    # Note: We consider the leaders to be part of the cast, so we include them in the cast members list

    # Validate DataFrame structure
    if availability_df.empty:
        raise ValueError("Availability DataFrame is empty")
    
    if availability_df.shape[1] < 1:
        raise ValueError("Availability DataFrame has no columns")
    
    # parse the names of the leaders and cast members
    names_column = availability_df.iloc[:, 0]
    
    # Check for required sections
    if 'Production Team' not in names_column.values:
        raise ValueError("Required 'Production Team' section not found in availability data")
    
    if 'Cast' not in names_column.values:
        raise ValueError("Required 'Cast' section not found in availability data")
    
    leader_index = names_column[names_column == 'Production Team'].index[0]
    cast_index = names_column[names_column == 'Cast'].index[0]
    leaders = names_column[leader_index + 1:cast_index]
    cast_members = names_column[cast_index + 1:]
    
    # Validate that we have data
    if len(leaders) == 0:
        raise ValueError("No leaders found in availability data")
    
    if len(cast_members) == 0:
        raise ValueError("No cast members found in availability data")
    
    cast_members = pd.concat([leaders, cast_members]).tolist() #include leaders in cast members, so this is really all members
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
    
    return cast_members, leaders, cast_availability, leader_availability

# Default spreadsheet key for backward compatibility
DEFAULT_SPREADSHEET_KEY = "1gQgJI0Ar2K9pkKZkmZxzxPTR95355VsVShk5_aMvxIo"

def get_parsed_availability(spreadsheet_key: Optional[str] = None) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Get and parse availability data from Google Sheets.
    
    Args:
        spreadsheet_key: The Google Sheets spreadsheet key. If None, uses default.
        
    Returns:
        Tuple of (cast_members, leaders, cast_availability, leader_availability)
        
    Raises:
        ValueError: If spreadsheet_key is not provided and no default is available
        Exception: If data cannot be retrieved or parsed
    """
    if spreadsheet_key is None:
        raise ValueError("spreadsheet_key is required but not provided")
    
    # Get raw data from Google Sheets
    availability_df = get_availability_data(spreadsheet_key)
    
    # Parse the data
    return parse_availability_data(availability_df)

# Note: Data is now loaded on-demand when needed, not at module import time