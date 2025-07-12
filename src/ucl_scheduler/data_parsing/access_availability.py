#this file accesses the availability sheet and returns a pandas dataframe with the availability of each cast member

import numpy as np
import pandas as pd
import gspread
from gspread.auth import service_account
import os
from pathlib import Path

# Get the credentials file path relative to the package
package_dir = Path(__file__).parent.parent
credentials_path = package_dir / "credentials" / "ucl-scheduler-866343adad65.json"

# Create gspread client with service account credentials
gc = service_account(filename=credentials_path)

# Open the spreadsheet by key
spreadsheet = gc.open_by_key("1gQgJI0Ar2K9pkKZkmZxzxPTR95355VsVShk5_aMvxIo")

# Get worksheet data (assuming first worksheet)
worksheet = spreadsheet.get_worksheet(0)
data = worksheet.get_all_values()

# Convert to pandas DataFrame
availability_df = pd.DataFrame(data)