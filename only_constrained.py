#this version does not optimize an objective function; it only finds all feasible solutions given the legal constraints
#assume one leader for all rehearsals; use Ronan's availability row

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# Import the database (availability DataFrame) from access_availability.py
#from access_availability import availability_df
from parse_availability import cast_members, cast_availability, leader_availability

num_cast = len(cast_members)
num_shifts = 12
num_days = 2
all_cast = range(num_cast)
all_shifts = range(num_shifts)
all_days = range(num_days)

# Convert a cast member name to an index
def member_to_index(member):
    return cast_members.index(member)

# Convert an index to a cast member name
def index_to_member(index):
    return cast_members[index]

model = cp_model.CpModel()

shifts = {}
for n in all_cast:
    for d in all_days:
        for s in all_shifts:
            shifts[(n, d, s)] = model.new_bool_var(f"shift_n{n}_d{d}_s{s}")

# For any shift where a cast member is scheduled, they must be:
for n in all_cast:
    for d in all_days:
        for s in all_shifts:
            if cast_availability[n][d][s] == 0:
                model.add(shifts[(n, d, s)] == 0)
            if leader_availability[0][d][s] == 0:
                model.add(shifts[(n, d, s)] == 0)


# Split requests into single and multi-hour requests
def split_requests(requests):
    single_requests = []
    multi_requests = {}
    for request in requests:
        if tuple(request[0]) in multi_requests:
            multi_requests[tuple(request[0])].append(request[1]) # If the request is already multi-hour, add the new request as well
        elif tuple(request[0]) in single_requests:
            single_requests.remove(tuple(request[0])) # If the request is already a single-hour request, make it multi-hour
            multi_requests[tuple(request[0])] = [request[1], 1] # Add the new request as a multi-hour request
        elif request[1] == 1:
            single_requests.append(tuple(request[0])) # If the request is a single-hour request, add it to the single-hour requests
        else:
            multi_requests[tuple(request[0])] = [request[1]] # If the request is a multi-hour request, add it to the multi-hour requests
    return single_requests, multi_requests


# If a group is requested for multiple slots, account for all combinations:
# For example, if a group is requested for 2 1hr slots and 1 2hr slot, the constraint is 4 1hr slots and 1 2hr slot overall


rehearsal_requests = [[['Ollie', 'Sophia', 'Tumo'], 1],
                      [['Ollie'], 1],
                      [['Ollie'], 1],
                      [['Mary', 'Sabine'], 2],
                      [['Mary', 'Sabine'], 1]]

single_requests, multi_requests = split_requests(rehearsal_requests)

# Create boolean variables for each request-shift combination
# request_fulfillment[(request, d, s)] will be True if request is fulfilled at day d, shift s
request_fulfillment = {}
for request in single_requests:
    for d in all_days:
        for s in all_shifts:
            request_fulfillment[(request, d, s)] = model.new_bool_var(f"fulfilled_{request}_d{d}_s{s}")

for request in single_requests:
    # Ensure each request is fulfilled exactly once
    # Sum of all fulfillment variables for this request must equal 1
    fulfillment_vars = [request_fulfillment[(request, d, s)] for d in all_days for s in all_shifts]
    model.add(sum(fulfillment_vars) == 1)
    
    # Link fulfillment variables to actual scheduling
    member_indices = [member_to_index(member) for member in request]
    for d in all_days:
        for s in all_shifts:
            # If request is fulfilled at this shift, all requested members must be scheduled
            for member_idx in member_indices:
                model.add(request_fulfillment[(request, d, s)] <= shifts[(member_idx, d, s)])
            
            # If request is fulfilled at this shift, no other members should be scheduled
            for n in all_cast:
                if n not in member_indices:
                    model.add(request_fulfillment[(request, d, s)] <= shifts[(n, d, s)].Not())

# Add support for multi-hour requests
multi_request_fulfillment = {}
for request, rehearsal_lengths in multi_requests.items():
    for rehearsal_length in rehearsal_lengths:
        for d in all_days:
            for s in range(num_shifts - rehearsal_length + 1):  # Can't start a 2-hour rehearsal at shift 11
                multi_request_fulfillment[(request, d, s, rehearsal_length)] = model.new_bool_var(f"multi_fulfilled_{request}_d{d}_s{s}_h{rehearsal_length}")

# Constraints for multi-hour requests
for request, rehearsal_lengths in multi_requests.items():
    # Each rehearsal length must be fulfilled exactly the number of times it appears in the list
    rehearsal_counts = {}
    for length in rehearsal_lengths:
        rehearsal_counts[length] = rehearsal_counts.get(length, 0) + 1
    
    for rehearsal_length, count in rehearsal_counts.items():
        fulfillment_vars = [multi_request_fulfillment[(request, d, s, rehearsal_length)] for d in all_days for s in range(num_shifts - rehearsal_length + 1)]
        model.add(sum(fulfillment_vars) == count)
    
    # Link multi-hour fulfillment to actual scheduling
    member_indices = [member_to_index(member) for member in request]
    for rehearsal_length in rehearsal_lengths:
        for d in all_days:
            for s in range(num_shifts - rehearsal_length + 1):
                # If multi-hour request is fulfilled starting at this shift, all members must be scheduled for all hours
                for hour in range(rehearsal_length):
                    for member_idx in member_indices:
                        model.add(multi_request_fulfillment[(request, d, s, rehearsal_length)] <= shifts[(member_idx, d, s + hour)])
                
                # If multi-hour request is fulfilled, no other members should be scheduled during these hours
                for hour in range(rehearsal_length):
                    for n in all_cast:
                        if n not in member_indices:
                            model.add(multi_request_fulfillment[(request, d, s, rehearsal_length)] <= shifts[(n, d, s + hour)].Not())

# Add constraints to prevent overlapping multi-hour rehearsals for the same group
for request, rehearsal_lengths in multi_requests.items():
    # Get all unique rehearsal lengths for this group
    unique_lengths = list(set(rehearsal_lengths))
    
    # For each pair of different rehearsal lengths, prevent overlap
    for i, length1 in enumerate(unique_lengths):
        for length2 in unique_lengths[i+1:]:
            for d in all_days:
                for s1 in range(num_shifts - length1 + 1):
                    for s2 in range(num_shifts - length2 + 1):
                        # Check if these two rehearsals would overlap
                        # Rehearsal 1: shifts s1 to s1+length1-1
                        # Rehearsal 2: shifts s2 to s2+length2-1
                        # They overlap if: s1 < s2+length2 AND s2 < s1+length1
                        # This covers: partial overlap, one contained in another, and exact overlap
                        if s1 < s2 + length2 and s2 < s1 + length1:
                            # Prevent both from being fulfilled at the same time
                            model.add(multi_request_fulfillment[(request, d, s1, length1)] + 
                                    multi_request_fulfillment[(request, d, s2, length2)] <= 1)

# Add constraint: people can only be scheduled if they're part of a fulfilled request
for n in all_cast:
    for d in all_days:
        for s in all_shifts:
            # A person can only be scheduled if they're part of a fulfilled request at this time
            member_scheduled = shifts[(n, d, s)]
            # Check if this person is part of any single-hour request fulfilled at this time
            member_in_fulfilled_request = []
            for request in single_requests:
                member_indices = [member_to_index(member) for member in request]
                if n in member_indices:
                    member_in_fulfilled_request.append(request_fulfillment[(request, d, s)])
            
            # Check if this person is part of any multi-hour request fulfilled at this time
            for request, rehearsal_lengths in multi_requests.items():
                member_indices = [member_to_index(member) for member in request]
                if n in member_indices:
                    # Check all possible start positions that could cover this shift
                    for rehearsal_length in rehearsal_lengths:
                        for start_s in range(max(0, s - rehearsal_length + 1), min(s + 1, num_shifts - rehearsal_length + 1)):
                            member_in_fulfilled_request.append(multi_request_fulfillment[(request, d, start_s, rehearsal_length)])
            
            # Person can only be scheduled if they're part of a fulfilled request
            if member_in_fulfilled_request:
                model.add(member_scheduled <= sum(member_in_fulfilled_request))
            else:
                # If person is not part of any request, they cannot be scheduled
                model.add(member_scheduled == 0)

class RehearsalSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate rehearsal scheduling solutions."""

    def __init__(self, shifts, num_cast, num_days, num_shifts, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._shifts = shifts
        self._num_cast = num_cast
        self._num_days = num_days
        self._num_shifts = num_shifts
        self._solution_count = 0
        self._solution_limit = limit

    def on_solution_callback(self):
        self._solution_count += 1
        
        # Create clean array format
        solution_array = []
        for d in range(self._num_days):
            day_schedule = []
            for s in range(self._num_shifts):
                scheduled_cast = []
                for n in range(self._num_cast):
                    if self.value(self._shifts[(n, d, s)]):
                        scheduled_cast.append(index_to_member(n))
                day_schedule.append(scheduled_cast)
            solution_array.append(day_schedule)
        
        # Store the solution array
        if not hasattr(self, '_solutions'):
            self._solutions = []
        self._solutions.append(solution_array)
        
        # Print minimal info
        print(f"Solution {self._solution_count}: {solution_array}")
        
        if self._solution_count >= self._solution_limit:
            print(f"Stop search after {self._solution_limit} solutions")
            self.stop_search()
    
    def get_solutions(self):
        """Return the clean solution arrays."""
        return getattr(self, '_solutions', [])

    def solutionCount(self):
        return self._solution_count

solution_limit = 1
solution_printer = RehearsalSolutionPrinter(
    shifts, num_cast, num_days, num_shifts, solution_limit
)

solver = cp_model.CpSolver()
solver.parameters.enumerate_all_solutions = True
status = solver.solve(model, solution_printer)

print(f"\nStatus: {solver.status_name(status)}")
print(f"Number of solutions found: {solution_printer.solutionCount()}")

# Convert solutions to HTML and display in terminal
from html_converter import convert_solutions_to_html
from terminal_viewer import view_schedule

solutions = solution_printer.get_solutions()
if solutions:
    # Display in terminal first
    print(f"\n{'-'*50}")
    print("📺 TERMINAL VIEW:")
    print(f"{'-'*50}")
    view_schedule(solutions)
    
    # Then create HTML
    html_filename = convert_solutions_to_html(solutions)
    print(f"\n📄 HTML schedule created: {html_filename}")
    print("Open this file in your browser to view the beautiful schedule!")
else:
    print("\n❌ No solutions found to convert to HTML.")