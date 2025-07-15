
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .availability_manager import parse_spreadsheet_url

room_spreadsheet_url = "https://docs.google.com/spreadsheets/d/1kVTIMS_t30-R_eOP-qvCxK9HbOpXkOfSicA3Z1oAaHI/edit?gid=652842963#gid=652842963"
worksheet_index = 1 # Which worksheet on the spreadsheet to use

spreadsheet_key = parse_spreadsheet_url(room_spreadsheet_url)


def get_room_data(spreadsheet_key: str, worksheet_index: int = 0) -> Dict:
    """
    Get room data from Google Sheets using Google Sheets API v4.
    
    Args:
        spreadsheet_key: The Google Sheets spreadsheet key
        worksheet_index: Index of the worksheet to access
        
    Returns:
        Dictionary containing worksheet data and metadata
        
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
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        
        # Set up credentials
        scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
        credentials = Credentials.from_service_account_file(str(credentials_path), scopes=scopes)
        
        # Build the service
        service = build('sheets', 'v4', credentials=credentials)
        
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
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
            
            # Set up credentials
            scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
            credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
            
            # Build the service
            service = build('sheets', 'v4', credentials=credentials)
            print("Using credentials from environment variable")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in GOOGLE_SHEETS_CREDENTIALS environment variable")
        except Exception as e:
            raise ValueError(f"Failed to create Google Sheets client from environment: {str(e)}")
    
    # Get spreadsheet metadata
    try:
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_key).execute()
    except Exception as e:
        raise Exception(f"Failed to open spreadsheet with key {spreadsheet_key}: {str(e)}")
    
    # Get worksheet data with formatting
    try:
        worksheet_title = spreadsheet['sheets'][worksheet_index]['properties']['title']
        
        # Get all data with formatting
        request = service.spreadsheets().get(
            spreadsheetId=spreadsheet_key,
            ranges=[worksheet_title],
            includeGridData=True
        )
        response = request.execute()
        
        return {
            'spreadsheet_id': spreadsheet_key,
            'worksheet_title': worksheet_title,
            'worksheet_index': worksheet_index,
            'service': service,
            'response': response
        }
        
    except Exception as e:
        raise Exception(f"Failed to retrieve data from room spreadsheet: {str(e)}")


def get_worksheet_values_and_colors(worksheet: Dict) -> pd.DataFrame:
    """
    Get worksheet data values and background colors at the same time.
    
    Args:
        worksheet: Dictionary containing worksheet data from get_room_data()
        
    Returns:
        DataFrame with cell values and background colors combined
    """
    response = worksheet['response']
    
    # Extract data values and colors from the response
    all_values = []
    all_colors = []
    
    if 'sheets' in response and len(response['sheets']) > 0:
        sheet = response['sheets'][0]
        if 'data' in sheet and len(sheet['data']) > 0:
            grid_data = sheet['data'][0]
            if 'rowData' in grid_data:
                for row_data in grid_data['rowData']:
                    row_values = []
                    row_colors = []
                    if 'values' in row_data:
                        for cell_data in row_data['values']:
                            # Get the cell value
                            cell_value = cell_data.get('formattedValue', '')
                            row_values.append(cell_value)
                            
                            # Get the cell background color
                            if 'userEnteredFormat' in cell_data:
                                format_info = cell_data['userEnteredFormat']
                                if 'backgroundColor' in format_info:
                                    bg_color = format_info['backgroundColor']
                                    # Convert RGB values to hex
                                    red = bg_color.get('red', 0)
                                    green = bg_color.get('green', 0)
                                    blue = bg_color.get('blue', 0)
                                    
                                    # Convert to hex format
                                    hex_color = f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
                                    row_colors.append(hex_color)
                                else:
                                    row_colors.append(None)
                            else:
                                row_colors.append(None)
                    all_values.append(row_values)
                    all_colors.append(row_colors)
    
    rooms_df = pd.DataFrame(all_values)
    
    # Add background color data to the dataframe
    for i in range(len(rooms_df)):
        for j in range(len(rooms_df.columns)):
            cell_value = rooms_df.iloc[i, j]
            cell_color = all_colors[i][j] if i < len(all_colors) and j < len(all_colors[i]) else None
            rooms_df.iloc[i, j] = (cell_value, cell_color)
    
    return rooms_df

def is_merged_cell(worksheet: Dict, row: int, col: int) -> bool:
    """
    Check if a cell is merged in the worksheet.
    Args:
        worksheet: Dictionary from get_room_data (Google Sheets API response)
        row: 0-indexed row
        col: 0-indexed column
    Returns:
        True if the cell is merged, False otherwise
    """
    response = worksheet['response']
    # Find the correct sheet (assume first sheet for now)
    if 'sheets' in response and len(response['sheets']) > 0:
        sheet = response['sheets'][0]
        merges = sheet.get('merges', [])
        for merge in merges:
            start_row = merge['startRowIndex']
            end_row = merge['endRowIndex']
            start_col = merge['startColumnIndex']
            end_col = merge['endColumnIndex']
            if start_row <= row < end_row and start_col <= col < end_col:
                return True
    return False

def parse_room_data(worksheet: Dict) -> List[Dict[str, List[bool]]]:
    """
    Parse room data from a worksheet.

    Args:
        worksheet: Dictionary containing room data

    Returns:
        List of lists of lists of booleans, where the first list is for days, the second list is for rooms, and the third list is for hours
    """    

    # None or #ffffff means the room is available

    rooms_df = get_worksheet_values_and_colors(worksheet)

    # Color all merged cells so they will be marked as unavailable
    for i in range(len(rooms_df)):
        for j in range(len(rooms_df.columns)):
            if is_merged_cell(worksheet, i, j):
                rooms_df.iloc[i, j] = (rooms_df.iloc[i, j][0], "#ff0000")

    room_names = [entry[0] for entry in rooms_df.iloc[2:30, 2].tolist()]

    # Split days up by iterating through first column
    day_indices = []
    for i in range(len(rooms_df)):
        if rooms_df.iloc[i, 0][0].lower() in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            day_indices.append(i)
    
    parsed_room_data = []

    # Split up the data by days and convert to boolean values
    for i in range(len(day_indices)):
        current_day_data = rooms_df.iloc[day_indices[i]:day_indices[i] + len(room_names), 5:]
        for j in range(len(current_day_data)):
            for k in range(len(current_day_data.columns)):
                if current_day_data.iloc[j, k][1] in ["#ffffff", None]: # Replace elements with a room availability flag
                    current_day_data.iloc[j, k] = True
                else:
                    current_day_data.iloc[j, k] = False
        current_day_data = current_day_data.values.tolist()

        hourly_current_day_data = {}
        for row_index, row in enumerate(current_day_data):
            hourly_row = []
            for index in range(len(row)):
                if index % 2 == 0: # 1st, 3rd, 5th, etc.
                    hourly_row.append(row[index] and row[index + 1])
            hourly_current_day_data[room_names[row_index]] = hourly_row
        
        parsed_room_data.append(hourly_current_day_data)
    
    return parsed_room_data

def get_parsed_room_data(spreadsheet_key: str = spreadsheet_key, worksheet_index: int = worksheet_index) -> List[Dict[str, List[bool]]]:
    room_data = get_room_data(spreadsheet_key, worksheet_index)
    return parse_room_data(room_data)

# Example usage
if __name__ == "__main__":
    '''
    try:
        room_data = get_room_data(spreadsheet_key, worksheet_index)
        
        # Get worksheet data and colors
        rooms_df = get_worksheet_values_and_colors(room_data)
        print(f"Worksheet dimensions: {rooms_df.shape}")
        
        # Count cells with background colors
        colored_cells = 0
        for i in range(len(rooms_df)):
            for j in range(len(rooms_df.columns)):
                cell_data = rooms_df.iloc[i, j]
                if isinstance(cell_data, tuple) and len(cell_data) == 2 and cell_data[1] is not None:
                    colored_cells += 1
        
        print(f"Found {colored_cells} cells with background colors")
        
        # Example: Print all cells with background colors
        for i in range(len(rooms_df)):
            for j in range(len(rooms_df.columns)):
                cell_data = rooms_df.iloc[i, j]
                if isinstance(cell_data, tuple) and len(cell_data) == 2 and cell_data[1] is not None:
                    cell_value, cell_color = cell_data
                    print(f"Cell ({i+1}, {j+1}): {cell_value} - Color: {cell_color}")
        
        print("Full rooms DataFrame:")
        print(rooms_df)
        
    except Exception as e:
        print(f"Error accessing room data: {str(e)}")
    '''
    room_data = get_room_data(spreadsheet_key, worksheet_index)
    parsed_room_data = parse_room_data(room_data)
    print(parsed_room_data)