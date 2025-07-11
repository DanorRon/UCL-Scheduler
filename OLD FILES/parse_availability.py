import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread
from ortools.sat.python import cp_model

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
full_doc = gc.open_by_key("15i-YrEjrj3Qqv-ohJv2WEfSi8NyfgoHmbSGwj6rZ1XI")

# Get worksheet data
worksheet = full_doc.worksheet('w/c 14/10').get_all_values()
print(worksheet)

worksheet_np = np.array(worksheet)

names = worksheet_np[3:37, 0]
week_availability = [worksheet_np[3:37, 2:14], worksheet_np[3:37, 15:27], worksheet_np[3:37, 28:40], worksheet_np[3:37, 41:53], worksheet_np[3:37, 54:66], worksheet_np[3:37, 67:79], worksheet_np[3:37, 80:92]]

num_cast = 27 #should be 28 right?
num_shifts = 12
num_days = 7
all_cast = range(num_cast)
all_shifts = range(num_shifts)
all_days = range(num_days)

model = cp_model.CpModel()

shifts = {}
for n in all_cast:
    for d in all_days:
        for s in all_shifts:
            shifts[(n, d, s)] = model.new_bool_var(f"shift_n{n}_d{d}_s{s}")


'''
#Constraint: All events must have a leader AND all members must be available
for n in all_cast:
    for d in all_days:
        for s in all_shifts:
            if shifts[(n, d, s)].get() == 1:
                #a leader must be available on day d, shift s
                #member n must be available on day d, shift s
                a=1 #TODO combine with gsheet reference

#Constraint: All members must be available for their rehearsals
for n in all_cast:
    for d in all_days:
        for s in all_shifts:
            if shifts[(n, d, s)].get() == 1:
                #a leader must be available
                a=1 #TODO combine with gsheet reference
'''

#Constraint: Limit hours per day
max_hours_per_day = 3 #max hours for each cast member per day
for n in all_cast:
    for d in all_days:
        daily_shifts_worked = []
        for s in all_shifts:
            daily_shifts_worked.append(shifts[(n, d, s)])
        model.add(sum(daily_shifts_worked) <= max_hours_per_day)

#Constraint: Don't schedule members at the same time unless they're in the same rehearsal (or leaders are different but implement this later)

#Constraint: Don't schedule more rehearsals than requested

event_requests = [[['Ollie', 'Sophia', 'Tumo'], 2],
                  [['Ollie'], 1],
                  [['Mary', 'Sade'], 2]]

def main_cost(): #TODO rename
    #Iterate through list of requested rehearsals; for each of them, check each time slot to see if it was fulfilled during that time slot
    #For now, assume that all rehearsal members requested are needed and that hours must be consecutive
    curr_free_shifts = np.full((num_days, num_shifts), True, dtype=bool) #initialize with True
    for request in event_requests:
        cost = 0 #number of mis-scheduled rehearsals
        member_labels = request[0]
        members = [] #TODO convert to list of member indices that are requested for the current request
        time = request[1]
        for d in all_days:
            scheduled_shifts = curr_free_shifts[d, :] #True except for the previously filled slots
            for s in all_shifts:
                for n in all_cast: #check that all (and only) members of the current request are scheduled for a rehearsal slot for the given time
                    #For now, just don't schedule any requests at the same time (which assumes a single leader)
                    if ((shifts[(n,d,s)] == 1) and (n not in members)) or ((shifts[(n,d,s)] != 1) and (n in members)):
                        scheduled_shifts[s] = False
            filled, shifts = consecutive_check(scheduled_shifts, time) #If the request was filled during the current day, true
            if filled:
                #update curr_free_shifts with shifts, false slots are the ones filled by previous requests
                cost = cost - 1 #cost is reduced if shift was filled somewhere
                break
        #check each day separately but only add 1 to the cost if the shift is filled anywhere
        #how to deal with the same group of people being scheduled multiple times in the same week? Deal with this later, minor issue

#maximize the number of filled requests
#minimize the total number of filled shifts; this should prevent excess

#check if there is a consecutive sequence of time shifts in scheduled_shifts
#Return a boolean for the check and a list of the indices of the consecutive block
def consecutive_check(scheduled_shifts, time): #TODO test more, I think this works though
    #xxxxxIterate through each d and s and see if there's a rehearsal then for some n; want to limit the number of 'isolated blocks'
    max_consecutive = 0
    for i in range(len(scheduled_shifts)):
        shift = scheduled_shifts[i]
        if shift:
            max_consecutive = max_consecutive + 1
        else:
            max_consecutive = 0
        
        if max_consecutive == time:
            print(max_consecutive)
            return True, [j for j in range(i - time + 1, i + 1)]
    
    return False, []


def objective_function():
    a=1

#model.minimize(objective_function)

#solver = cp_model.CpSolver()
#status = solver.solve(model)