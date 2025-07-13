"""
Rehearsal Scheduler using Google OR-Tools CP-SAT solver.

This module provides a clean, maintainable implementation of the rehearsal
scheduling system with support for single and multi-hour rehearsals.
"""

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..solution_viewing.html_converter import convert_solutions_to_html
from ..solution_viewing.terminal_viewer import view_schedule

@dataclass
class RehearsalRequest:
    """Represents a rehearsal request with cast members, duration, and leader."""
    members: List[str]
    duration: int
    leader: str
    
    def __post_init__(self):
        # Convert to tuple for immutability in dict keys
        object.__setattr__(self, 'members', tuple(self.members))


class DataManager:
    """Manages cast member data and availability."""
    
    def __init__(self, cast_members: List[str], cast_availability: np.ndarray, leader_availability: np.ndarray):
        # Validate input data
        if not cast_members:
            raise ValueError("cast_members list is empty")
        if cast_availability.size == 0:
            raise ValueError("cast_availability array is empty")
        if leader_availability.size == 0:
            raise ValueError("leader_availability array is empty")
        
        if len(cast_availability.shape) != 3:
            raise ValueError(f"cast_availability should be 3D array, got shape {cast_availability.shape}")
        if len(leader_availability.shape) != 3:
            raise ValueError(f"leader_availability should be 3D array, got shape {leader_availability.shape}")
        
        if cast_availability.shape[0] != len(cast_members):
            raise ValueError(f"cast_availability first dimension ({cast_availability.shape[0]}) doesn't match number of cast members ({len(cast_members)})")
        
        self.cast_members = cast_members
        self.cast_availability = cast_availability
        self.leader_availability = leader_availability
        self.num_cast = len(cast_members)
        self.num_days = cast_availability.shape[1]
        self.num_shifts = cast_availability.shape[2]
        
    def member_to_index(self, member: str) -> int:
        """Convert cast member name to index."""
        return self.cast_members.index(member)
    
    def index_to_member(self, index: int) -> str:
        """Convert index to cast member name."""
        return self.cast_members[index]
    
    def get_all_cast(self):
        """Get range of all cast indices."""
        return range(self.num_cast)
    
    def get_all_shifts(self):
        """Get range of all shift indices."""
        return range(self.num_shifts)
    
    def get_all_days(self):
        """Get range of all day indices."""
        return range(self.num_days)


class RequestProcessor:
    """Processes and categorizes rehearsal requests."""
    
    @staticmethod
    def split_requests(requests: List[RehearsalRequest]) -> Tuple[List[Tuple], Dict[Tuple, List[int]]]:
        """
        Split requests into single and multi-hour categories.
        
        Returns:
            single_requests: List of single-hour request tuples (members, leader)
            multi_requests: Dict mapping (members, leader) tuples to list of rehearsal lengths
        """
        single_requests = []
        multi_requests = {}
        
        for request in requests:
            group = request.members
            duration = request.duration
            leader = request.leader
            request_key = (group, leader)
            
            if request_key in multi_requests:
                # Already a multi-hour group, add this duration
                multi_requests[request_key].append(duration)
            elif request_key in single_requests:
                # Convert from single to multi-hour
                single_requests.remove(request_key)
                multi_requests[request_key] = [duration, 1]
            elif duration == 1:
                # Single-hour request
                single_requests.append(request_key)
            else:
                # Multi-hour request
                multi_requests[request_key] = [duration]
        
        return single_requests, multi_requests


class ConstraintBuilder:
    """Builds OR-Tools constraints for the scheduling model."""
    
    def __init__(self, model: cp_model.CpModel, data_manager: DataManager):
        self.model = model
        self.data = data_manager
        self.shifts = {}
        self.request_fulfillment = {}
        self.multi_request_fulfillment = {}
        
    def create_shift_variables(self):
        """Create boolean variables for all cast member shifts."""
        for n in self.data.get_all_cast():
            for d in self.data.get_all_days():
                for s in self.data.get_all_shifts():
                    self.shifts[(n, d, s)] = self.model.new_bool_var(f"shift_n{n}_d{d}_s{s}")
    
    def add_availability_constraints(self):
        """Add constraints based on cast availability."""
        for n in self.data.get_all_cast():
            for d in self.data.get_all_days():
                for s in self.data.get_all_shifts():
                    # Cast member availability
                    if self.data.cast_availability[n][d][s] == 0:
                        self.model.add(self.shifts[(n, d, s)] == 0)
    
    def add_single_hour_constraints(self, single_requests: List[Tuple]):
        """Add constraints for single-hour rehearsal requests."""
        # Create fulfillment variables
        for request in single_requests:
            for d in self.data.get_all_days():
                for s in self.data.get_all_shifts():
                    self.request_fulfillment[(request, d, s)] = self.model.new_bool_var(
                        f"fulfilled_{request}_d{d}_s{s}")
        
        # Add constraints for each request
        for request in single_requests:
            # Each request must be fulfilled exactly once
            fulfillment_vars = [
                self.request_fulfillment[(request, d, s)] 
                for d in self.data.get_all_days() 
                for s in self.data.get_all_shifts()
            ]
            self.model.add(sum(fulfillment_vars) == 1)
            
            # Link fulfillment to actual scheduling
            member_indices = [self.data.member_to_index(member) for member in request[0]]
            for d in self.data.get_all_days():
                for s in self.data.get_all_shifts():
                    # All requested members must be scheduled if request is fulfilled
                    for member_idx in member_indices:
                        self.model.add(
                            self.request_fulfillment[(request, d, s)] <= 
                            self.shifts[(member_idx, d, s)]
                        )
                    
                    # No other members should be scheduled
                    for n in self.data.get_all_cast():
                        if n not in member_indices:
                            self.model.add(
                                self.request_fulfillment[(request, d, s)] <= 
                                self.shifts[(n, d, s)].Not()
                            )

            # Add leader availability constraint
            leader_index = self.data.member_to_index(request[1]) # Get leader index in cast_members
            for d in self.data.get_all_days():
                for s in self.data.get_all_shifts():
                    # Check that the leader is available (cast_availability includes leaders)
                    if self.data.cast_availability[leader_index][d][s] == 0:
                        self.model.add(self.request_fulfillment[(request, d, s)] == 0)
    
    def add_multi_hour_constraints(self, multi_requests: Dict[Tuple, List[int]]):
        """Add constraints for multi-hour rehearsal requests."""
        # Create fulfillment variables
        for request, rehearsal_lengths in multi_requests.items():
            for rehearsal_length in rehearsal_lengths:
                for d in self.data.get_all_days():
                    for s in range(self.data.num_shifts - rehearsal_length + 1):
                        self.multi_request_fulfillment[(request, d, s, rehearsal_length)] = \
                            self.model.new_bool_var(f"multi_fulfilled_{request}_d{d}_s{s}_h{rehearsal_length}")
        
        # Add constraints for each multi-hour request
        for request, rehearsal_lengths in multi_requests.items():
            # Count occurrences of each rehearsal length
            rehearsal_counts = {}
            for length in rehearsal_lengths:
                rehearsal_counts[length] = rehearsal_counts.get(length, 0) + 1
            
            # Each rehearsal length must be fulfilled the correct number of times
            for rehearsal_length, count in rehearsal_counts.items():
                fulfillment_vars = [
                    self.multi_request_fulfillment[(request, d, s, rehearsal_length)]
                    for d in self.data.get_all_days()
                    for s in range(self.data.num_shifts - rehearsal_length + 1)
                ]
                self.model.add(sum(fulfillment_vars) == count)
            
            # Link fulfillment to actual scheduling
            member_indices = [self.data.member_to_index(member) for member in request[0]]
            for rehearsal_length in rehearsal_lengths:
                for d in self.data.get_all_days():
                    for s in range(self.data.num_shifts - rehearsal_length + 1):
                        # All members must be scheduled for all hours
                        for hour in range(rehearsal_length):
                            for member_idx in member_indices:
                                self.model.add(
                                    self.multi_request_fulfillment[(request, d, s, rehearsal_length)] <=
                                    self.shifts[(member_idx, d, s + hour)]
                                )
                        
                        # No other members should be scheduled during these hours
                        for hour in range(rehearsal_length):
                            for n in self.data.get_all_cast():
                                if n not in member_indices:
                                    self.model.add(
                                        self.multi_request_fulfillment[(request, d, s, rehearsal_length)] <=
                                        self.shifts[(n, d, s + hour)].Not()
                                    )
            
            # Add leader availability constraints for multi-hour requests
            for request, rehearsal_lengths in multi_requests.items():
                leader_index = self.data.member_to_index(request[1])  # Get leader index in cast_members
                for rehearsal_length in rehearsal_lengths:
                    for d in self.data.get_all_days():
                        for s in range(self.data.num_shifts - rehearsal_length + 1):
                            # Check that the leader is available for all hours of the rehearsal
                            for hour in range(rehearsal_length):
                                if self.data.cast_availability[leader_index][d][s + hour] == 0:
                                    self.model.add(self.multi_request_fulfillment[(request, d, s, rehearsal_length)] == 0)
    
    def add_overlap_prevention_constraints(self, multi_requests: Dict[Tuple, List[int]]):
        """Add constraints to prevent overlapping multi-hour rehearsals for the same group."""
        for request, rehearsal_lengths in multi_requests.items():
            unique_lengths = list(set(rehearsal_lengths))
            
            # For each pair of different rehearsal lengths, prevent overlap
            for i, length1 in enumerate(unique_lengths):
                for length2 in unique_lengths[i+1:]:
                    for d in self.data.get_all_days():
                        for s1 in range(self.data.num_shifts - length1 + 1):
                            for s2 in range(self.data.num_shifts - length2 + 1):
                                # Check if rehearsals would overlap
                                if s1 < s2 + length2 and s2 < s1 + length1:
                                    # Prevent both from being fulfilled simultaneously
                                    self.model.add(
                                        self.multi_request_fulfillment[(request, d, s1, length1)] +
                                        self.multi_request_fulfillment[(request, d, s2, length2)] <= 1
                                    )
    
    def add_scheduling_constraints(self, single_requests: List[Tuple], multi_requests: Dict[Tuple, List[int]]):
        """Add constraints ensuring people are only scheduled for fulfilled requests."""
        for n in self.data.get_all_cast():
            for d in self.data.get_all_days():
                for s in self.data.get_all_shifts():
                    member_scheduled = self.shifts[(n, d, s)]
                    member_in_fulfilled_request = []
                    
                    # Check single-hour requests
                    for request in single_requests:
                        member_indices = [self.data.member_to_index(member) for member in request[0]]
                        if n in member_indices:
                            member_in_fulfilled_request.append(self.request_fulfillment[(request, d, s)])
                    
                    # Check multi-hour requests
                    for request, rehearsal_lengths in multi_requests.items():
                        member_indices = [self.data.member_to_index(member) for member in request[0]]
                        if n in member_indices:
                            # Check all possible start positions that could cover this shift
                            for rehearsal_length in rehearsal_lengths:
                                for start_s in range(
                                    max(0, s - rehearsal_length + 1),
                                    min(s + 1, self.data.num_shifts - rehearsal_length + 1)
                                ):
                                    member_in_fulfilled_request.append(
                                        self.multi_request_fulfillment[(request, d, start_s, rehearsal_length)]
                                    )
                    
                    # Person can only be scheduled if part of a fulfilled request
                    if member_in_fulfilled_request:
                        self.model.add(member_scheduled <= sum(member_in_fulfilled_request))
                    else:
                        self.model.add(member_scheduled == 0)
    
    def add_same_group_different_days_constraints(self, multi_requests: Dict[Tuple, List[int]]):
        """Add constraints to prevent multiple requests for the same group on the same day."""
        # For groups with multiple requests, ensure they're on different days
        for group, rehearsal_lengths in multi_requests.items():
            if len(rehearsal_lengths) > 1:
                # This group has multiple requests, so they must be on different days
                day_fulfillment_vars = []
                
                # Collect all fulfillment variables for this group across all days
                for d in self.data.get_all_days():
                    day_fulfilled = []
                    
                    # Check multi-hour requests for this group
                    for rehearsal_length in rehearsal_lengths:
                        for s in range(self.data.num_shifts - rehearsal_length + 1):
                            day_fulfilled.append(self.multi_request_fulfillment[(group, d, s, rehearsal_length)])
                    
                    # If any request for this group is fulfilled on this day, mark the day as used
                    if day_fulfilled:
                        day_fulfillment_vars.append(sum(day_fulfilled))
                
                # Ensure that the number of days used equals the number of requests
                if day_fulfillment_vars:
                    self.model.add(sum(day_fulfillment_vars) == len(rehearsal_lengths))
                    
                    # Ensure no more than one request per day for this group
                    for day_vars in day_fulfillment_vars:
                        self.model.add(day_vars <= 1)


class RehearsalSolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collects, stores, and formats rehearsal scheduling solutions.
    
    This class acts as a callback for the OR-Tools solver, collecting solutions
    as they are found and converting them into a clean, structured format
    suitable for display and further processing.
    """
    
    def __init__(self, shifts: Dict, request_fulfillment: Dict, multi_request_fulfillment: Dict, 
                 single_requests: List[Tuple], multi_requests: Dict[Tuple, List[int]], 
                 data_manager: DataManager, limit: int):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._shifts = shifts
        self._request_fulfillment = request_fulfillment
        self._multi_request_fulfillment = multi_request_fulfillment
        self._single_requests = single_requests
        self._multi_requests = multi_requests
        self._data = data_manager
        self._solution_count = 0
        self._solution_limit = limit
        self._solutions = []
    
    def on_solution_callback(self):
        self._solution_count += 1
        
        # Create clean array format with leader information
        solution_array = []
        for d in range(self._data.num_days):
            day_schedule = []
            for s in range(self._data.num_shifts):
                scheduled_cast = []
                leader = None
                
                # Check single-hour requests for this day/shift
                for request in self._single_requests:
                    if self.value(self._request_fulfillment[(request, d, s)]):
                        scheduled_cast = list(request[0])  # Get members from request
                        leader = request[1]  # Get leader from request
                        break
                
                # Check multi-hour requests for this day/shift
                if not scheduled_cast:
                    for request, rehearsal_lengths in self._multi_requests.items():
                        for rehearsal_length in rehearsal_lengths:
                            # Check if this shift is part of a multi-hour rehearsal
                            for start_s in range(max(0, s - rehearsal_length + 1), s + 1):
                                if self.value(self._multi_request_fulfillment[(request, d, start_s, rehearsal_length)]):
                                    scheduled_cast = list(request[0])  # Get members from request
                                    leader = request[1]  # Get leader from request
                                    break
                        if scheduled_cast:
                            break
                
                # Store both members and leader (empty if no rehearsal scheduled)
                day_schedule.append({
                    'members': scheduled_cast,
                    'leader': leader
                })
            solution_array.append(day_schedule)
        
        self._solutions.append(solution_array)
        print(f"Solution {self._solution_count}: {solution_array}")
        
        if self._solution_count >= self._solution_limit:
            print(f"Stop search after {self._solution_limit} solutions")
            self.stop_search()
    
    def get_solutions(self) -> List[List[List[Dict]]]:
        """Return the clean solution arrays with leader information."""
        return self._solutions
    
    def solutionCount(self) -> int:
        return self._solution_count


class RehearsalScheduler:
    """Main scheduler class that orchestrates the entire scheduling process."""
    
    def __init__(self, cast_members: List[str], cast_availability: np.ndarray, leader_availability: np.ndarray):
        self.data_manager = DataManager(cast_members, cast_availability, leader_availability)
        self.request_processor = RequestProcessor()
        self.model = cp_model.CpModel()
        self.constraint_builder = ConstraintBuilder(self.model, self.data_manager)
        
    def build_model(self, requests: List[RehearsalRequest]):
        """Build the complete scheduling model with all constraints."""
        # Store requests for later analysis
        self._current_requests = requests
        
        # Create variables
        self.constraint_builder.create_shift_variables()
        
        # Add constraints
        self.constraint_builder.add_availability_constraints()
        
        # Process requests
        self._single_requests, self._multi_requests = self.request_processor.split_requests(requests)
        
        # Add request-specific constraints
        self.constraint_builder.add_single_hour_constraints(self._single_requests)
        self.constraint_builder.add_multi_hour_constraints(self._multi_requests)
        self.constraint_builder.add_overlap_prevention_constraints(self._multi_requests)
        self.constraint_builder.add_scheduling_constraints(self._single_requests, self._multi_requests)
        self.constraint_builder.add_same_group_different_days_constraints(self._multi_requests)
        
        return self._single_requests, self._multi_requests
    
    def solve(self, solution_limit: int = 1) -> List[List[List[Dict]]]:
        """Solve the scheduling problem and return solutions."""
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        
        # Use the processed requests from build_model
        solution_collector = RehearsalSolutionCollector(
            self.constraint_builder.shifts,
            self.constraint_builder.request_fulfillment,
            self.constraint_builder.multi_request_fulfillment,
            self._single_requests,
            self._multi_requests,
            self.data_manager, 
            solution_limit
        )
        
        status = solver.solve(self.model, solution_collector)
        
        print(f"\nStatus: {solver.status_name(status)}")
        print(f"Number of solutions found: {solution_collector.solutionCount()}")
        
        # If infeasible, identify which requests couldn't be scheduled
        if status == cp_model.INFEASIBLE:
            self._identify_infeasible_requests()
        
        return solution_collector.get_solutions()
    
    def _identify_infeasible_requests(self):
        """Identify which requests couldn't be scheduled when the problem is infeasible."""
        print("\n🔍 ANALYZING INFEASIBLE REQUESTS:")
        print("=" * 50)
        
        # Test each request individually
        single_requests, multi_requests = self.request_processor.split_requests(self._current_requests)
        
        infeasible_requests = []
        
        # Test single-hour requests
        for request in single_requests:
            if not self._test_request_feasibility(request, 1):
                infeasible_requests.append(f"Single-hour: {list(request)}")
        
        # Test multi-hour requests
        for request, rehearsal_lengths in multi_requests.items():
            for length in rehearsal_lengths:
                if not self._test_request_feasibility(request, length):
                    infeasible_requests.append(f"Multi-hour ({length}h): {list(request)}")
        
        if infeasible_requests:
            print("❌ The following requests could not be scheduled:")
            for req in infeasible_requests:
                print(f"   • {req}")
            print("\n💡 Possible reasons:")
            print("   • Insufficient availability for all members")
            print("   • No leader available during required times")
            print("   • Requested duration exceeds available time slots")
        else:
            print("✅ All individual requests are feasible")
            print("💡 The infeasibility is due to conflicts between requests")
            #TODO: Add detailed conflict analysis to identify specific request conflicts
    
    def _test_request_feasibility(self, request: Tuple, duration: int) -> bool:
        """Test if a single request can be scheduled."""
        # Create a minimal model for this request
        test_model = cp_model.CpModel()
        
        # Create shift variables for this request only
        member_indices = [self.data_manager.member_to_index(member) for member in request[0]]
        leader_index = self.data_manager.member_to_index(request[1])  # Get leader index
        shifts = {}
        for n in member_indices:
            for d in self.data_manager.get_all_days():
                for s in self.data_manager.get_all_shifts():
                    shifts[(n, d, s)] = test_model.new_bool_var(f"test_shift_n{n}_d{d}_s{s}")
        
        # Add availability constraints
        for n in member_indices:
            for d in self.data_manager.get_all_days():
                for s in self.data_manager.get_all_shifts():
                    # Cast member availability
                    if self.data_manager.cast_availability[n][d][s] == 0:
                        test_model.add(shifts[(n, d, s)] == 0)
        
        # Add leader availability constraints
        for d in self.data_manager.get_all_days():
            for s in self.data_manager.get_all_shifts():
                if self.data_manager.cast_availability[leader_index][d][s] == 0:
                    test_model.add(shifts[(leader_index, d, s)] == 0)
        
        # Add request fulfillment constraint
        if duration == 1:
            # Single-hour request
            fulfillment_vars = []
            for d in self.data_manager.get_all_days():
                for s in self.data_manager.get_all_shifts():
                    fulfillment_vars.append(test_model.new_bool_var(f"fulfillment_d{d}_s{s}"))
            
            # Must be fulfilled exactly once
            test_model.add(sum(fulfillment_vars) == 1)
            
            # Link fulfillment to scheduling
            for d in self.data_manager.get_all_days():
                for s in self.data_manager.get_all_shifts():
                    for member_idx in member_indices:
                        test_model.add(fulfillment_vars[d * self.data_manager.num_shifts + s] <= shifts[(member_idx, d, s)])
        else:
            # Multi-hour request
            fulfillment_vars = []
            for d in self.data_manager.get_all_days():
                for s in range(self.data_manager.num_shifts - duration + 1):
                    fulfillment_vars.append(test_model.new_bool_var(f"fulfillment_d{d}_s{s}"))
            
            # Must be fulfilled exactly once
            test_model.add(sum(fulfillment_vars) == 1)
            
            # Link fulfillment to scheduling
            for d in self.data_manager.get_all_days():
                for s in range(self.data_manager.num_shifts - duration + 1):
                    for hour in range(duration):
                        for member_idx in member_indices:
                            test_model.add(fulfillment_vars[d * (self.data_manager.num_shifts - duration + 1) + s] <= shifts[(member_idx, d, s + hour)])
        
        # Solve the test model
        test_solver = cp_model.CpSolver()
        status = test_solver.solve(test_model)
        
        return status == cp_model.OPTIMAL or status == cp_model.FEASIBLE


def main():
    """Main function to run the rehearsal scheduler."""
    # Import availability data
    from ..data_parsing.availability_manager import get_parsed_availability, DEFAULT_SPREADSHEET_KEY
    cast_members, leaders, cast_availability, leader_availability = get_parsed_availability(DEFAULT_SPREADSHEET_KEY)
    
    # Define rehearsal requests
    rehearsal_requests = [
        RehearsalRequest(['Ollie', 'Sophia', 'Tumo'], 1, 'Ollie'),
        RehearsalRequest(['Ollie'], 1, 'Ollie'),
        RehearsalRequest(['Ollie'], 1, 'Ollie'),
        RehearsalRequest(['Mary', 'Sabine'], 2, 'Mary'),
        RehearsalRequest(['Mary', 'Sabine'], 1, 'Mary'),
    ]
    
    # Create and run scheduler with availability data
    scheduler = RehearsalScheduler(cast_members, cast_availability, leader_availability)
    scheduler.build_model(rehearsal_requests)
    solutions = scheduler.solve(solution_limit=1)
    
    # Display results
    if solutions:
        
        print(f"\n{'-'*50}")
        print("📺 TERMINAL VIEW:")
        print(f"{'-'*50}")
        view_schedule(solutions)
        
        html_filename = convert_solutions_to_html(solutions)
        print(f"\n📄 HTML schedule created: {html_filename}")
        print("Open this file in your browser to view the beautiful schedule!")
    else:
        print("\n❌ No solutions found to convert to HTML.")


if __name__ == "__main__":
    main()