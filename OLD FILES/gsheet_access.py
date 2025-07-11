import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread

# Define the scope for Google Sheets API
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Load credentials from service account file
credentials = Credentials.from_service_account_file(
    'ucl-scheduler-866343adad65.json', 
    scopes=SCOPES
)

# Create gspread client with credentials
gc = gspread.authorize(credentials)

# Open the spreadsheet by key
booking_sheet = gc.open_by_key("11BIYel95aeO7tVaWOFu4AKDJwHQsXYeCN3kmFgGMwDA")

# Get worksheet data
worksheet = booking_sheet.worksheet('30.09.24').get_all_values()
print(worksheet)

# Process the data
bool_sheet = np.empty((26, 29))
for i in range(2, 28):
    for j in range(3, 31):
        if i == 2:
            bool_sheet[i][j] = worksheet[i][j]
        elif worksheet[i][j] != '':
            bool_sheet[i][j] = 1
        else:
            bool_sheet[i][j] = 0

# Create DataFrame
df = pd.DataFrame(bool_sheet, columns=worksheet[1])
print(df.head())

'''
Row ranges for different sections:
2-27 (2-28)
30-55
58-83
86-111
114-139
142-167
170-195
'''