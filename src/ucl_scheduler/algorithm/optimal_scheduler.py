"""
Skeleton for schedule optimization system.
Add features incrementally over time.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..data_parsing.room_manager import get_parsed_room_data
from .constrained_scheduler import RehearsalScheduler, RehearsalRequest, SolverStatus

from ..solution_viewing.terminal_viewer import view_schedule


@dataclass
class TimeOfDayPreferences:
    """Leader's preferences for time of day scheduling."""
    morning_weight: float = 0.0    # 9:00-12:00 (shifts 0-3)
    afternoon_weight: float = 0.7   # 12:00-17:00 (shifts 3-8) 
    evening_weight: float = 0.3     # 17:00-21:00 (shifts 8-12)
    
    def normalize(self):
        """Ensure weights sum to 1.0."""
        total = self.morning_weight + self.afternoon_weight + self.evening_weight
        self.morning_weight /= total
        self.afternoon_weight /= total
        self.evening_weight /= total


@dataclass
class RoomPreferences:
    """Leader's preferences for room usage."""
    available_rooms = ['Bloomsbury Theatre Conference Room', 'Bloomsbury Theatre Rehearsal Room', 'Bloomsbury Theatre 204', 'Lewis Building Dance Studio', 'Lewis Building 105', 'Lewis Building 206', 'Jeremy Bentham Room', 'Wilkins Building Garden Room', 'Wilkins Terrace', 'IOE 421 Nunn Hall', 'IOE 102 Punnet Hall (Drama Studio)', 'Christopher Ingold Building G21 Ramsay', 'Anatomy Building G04 Gavin De Beer', 'Anatomy Building B15', 'Darwin Building B05', 'Darwin Building B15', 'Foster Court 112/113', 'Foster Court 114', 'Foster Court 215', 'Foster Court 217', 'Foster Court 219', 'South Quad Teaching Block G01', 'South Quad Teaching Block 101', 'South Quad Teaching Block 102', 'South Quad Teaching Block 103', 'Chadwick Building G07', 'Chadwick Building G08', 'Chadwick Building 1.02A/B']
    room_weights = {
        'Bloomsbury Theatre Conference Room': 0.90,
        'Bloomsbury Theatre Rehearsal Room': 1.00,
        'Bloomsbury Theatre 204': 0.80,
        'Lewis Building Dance Studio': 0.70,
        'Lewis Building 105': 0.80,
        'Lewis Building 206': 0.90,
        'Jeremy Bentham Room': 0.90,
        'Wilkins Building Garden Room': 0.70,
        'Wilkins Terrace': 0.50,
        'IOE 421 Nunn Hall': 0.30,
        'IOE 102 Punnet Hall (Drama Studio)': 0.40,
        'Christopher Ingold Building G21 Ramsay': 0.20,
        'Anatomy Building G04 Gavin De Beer': 0.0,
        'Anatomy Building B15': 0.0,
        'Darwin Building B05': 0.0,
        'Darwin Building B15': 0.0,
        'Foster Court 112/113': 0.0,
        'Foster Court 114': 0.0,
        'Foster Court 215': 0.0,
        'Foster Court 217': 0.0,
        'Foster Court 219': 0.0,
        'South Quad Teaching Block G01': 0.70,
        'South Quad Teaching Block 101': 0.50,
        'South Quad Teaching Block 102': 0.50,
        'South Quad Teaching Block 103': 0.50,
        'Chadwick Building G07': 0.60,
        'Chadwick Building G08': 0.60,
        'Chadwick Building 1.02A/B': 0.40,
        'CMISGO': 0.0
    }  # Preference weights for each room
    
    # Not using as is--later we could use this to normalize the weights to have a max of 1.0 but not needed for now
    def normalize_weights(self):
        """Ensure room weights sum to 1.0."""
        if not self.room_weights:
            return
        
        total = sum(self.room_weights.values())
        if total > 0:
            for room in self.room_weights:
                self.room_weights[room] /= total


@dataclass
class ContinuityPreferences:
    """Preferences for rehearsal continuity (longer blocks, but with a limit)."""
    max_block_length: int = 5  # Maximum preferred consecutive rehearsal slots
    penalty_factor: float = 3.0  # How sharply to penalize blocks longer than max

@dataclass
class OptimizationWeights:
    """Weights for different optimization factors."""
    time_preference: float = 0.3
    room_preference: float = 0.4
    continuity_preference: float = 0.3
    
    def normalize(self):
        total = self.time_preference + self.room_preference + self.continuity_preference
        self.time_preference /= total
        self.room_preference /= total
        self.continuity_preference /= total


class FactorCalculator:
    """Calculate scores for different optimization factors."""
    
    def __init__(self, time_prefs: TimeOfDayPreferences, room_prefs: RoomPreferences, continuity_prefs: ContinuityPreferences, rooms_data):
        self.time_prefs = time_prefs
        self.room_prefs = room_prefs
        self.continuity_prefs = continuity_prefs
        self.parsed_room_data = rooms_data
    
    def calculate_time_preference_score(self, schedule) -> float:
        """Calculate time of day preference score (0-100)."""
        if not schedule:
            return 0
        
        total_rehearsals = 0
        morning_rehearsals = 0
        afternoon_rehearsals = 0
        evening_rehearsals = 0
        
        for day in schedule:
            for shift_idx, shift in enumerate(day):
                # Check if there's a rehearsal in this shift (has members)
                if shift and isinstance(shift, dict) and shift.get('members'):
                    total_rehearsals += 1
                    
                    # Categorize by time of day
                    if 0 <= shift_idx <= 3:  # 9:00-12:00
                        morning_rehearsals += 1
                    elif 4 <= shift_idx <= 8:  # 12:00-17:00
                        afternoon_rehearsals += 1
                    elif 9 <= shift_idx <= 12:  # 17:00-21:00
                        evening_rehearsals += 1
        
        if total_rehearsals == 0:
            return 0
        
        # Calculate how well the schedule matches preferences
        actual_morning_ratio = morning_rehearsals / total_rehearsals
        actual_afternoon_ratio = afternoon_rehearsals / total_rehearsals
        actual_evening_ratio = evening_rehearsals / total_rehearsals
        
        # Calculate score based on how close actual ratios are to preferred ratios
        morning_score = 100 - abs(actual_morning_ratio - self.time_prefs.morning_weight) * 100
        afternoon_score = 100 - abs(actual_afternoon_ratio - self.time_prefs.afternoon_weight) * 100
        evening_score = 100 - abs(actual_evening_ratio - self.time_prefs.evening_weight) * 100
        
        # Weighted average of the three time periods
        total_score = (
            morning_score * self.time_prefs.morning_weight +
            afternoon_score * self.time_prefs.afternoon_weight +
            evening_score * self.time_prefs.evening_weight
        )
        
        return max(0, total_score)
    
    def find_best_room(self, day: int, shift: int, rehearsal_length: int) -> Tuple[str, float]:
        """
        Find the best room for a given day, shift, and rehearsal length.
        Shift + rehearsal_length must be less than the length of the day
        """

        # assert parsed_room_data[day].keys() == self.room_prefs.available_rooms # To make sure the room names are consistent
        # TODO Make sure this passes
        
        #print(f"parsed_room_data[day].keys(): {parsed_room_data[day].keys()}")
        #print(f"self.room_prefs.available_rooms: {self.room_prefs.available_rooms}")

        # Find the best room for the given day, shift, and rehearsal length
        best_room = None
        best_score = 0.0 # Defaults to CMISGO essentially

        for room in self.room_prefs.available_rooms:
            # Check that the room is available for the entire rehearsal block
            if all(self.parsed_room_data[day][room][shift:shift + rehearsal_length]) and self.room_prefs.room_weights[room] > best_score:
                best_room = room
                best_score = self.room_prefs.room_weights[room]
        if best_room is None: # No rooms better than CMISGO were found
            best_room = 'CMISGO'

        return best_room, best_score
    
    def assign_rooms(self, schedule) -> List[List[str]]:
        bypass = False
        if bypass:
            return []
        """Assign rooms to the schedule."""
        # Assume that multi-hour rehearsals should be assigned to the same room for each hour. 
        # Multiple requests for the same group are already split to different days so we don't need to worry about that
        room_assignments = []
        for day_index, day in enumerate(schedule):
            day_room_assignments = []
            shift_index = 0
            while shift_index < len(day):
                shift = day[shift_index]
                if shift and isinstance(shift, dict) and shift.get('members'): # Requires shift.get('members') to be a non-empty list

                    rehearsal_length = 0
                    while shift_index + rehearsal_length < len(day) and shift == day[shift_index + rehearsal_length]: # If the current shift (i.e. scheduled rehearsal)
                        rehearsal_length += 1
                    room, score = self.find_best_room(day_index, shift_index, rehearsal_length)
                    day_room_assignments.extend([room] * rehearsal_length) # append room rehearsal_length times
                    shift_index += rehearsal_length
                else:
                    day_room_assignments.append('')
                    shift_index += 1
            room_assignments.append(day_room_assignments)
        return room_assignments


    def calculate_room_preference_score(self, schedule) -> float:
        """Calculate room preference score (0-100)."""

        room_assignments = self.assign_rooms(schedule)

        # Calculate the score for the room assignments
        total_score = 0
        num_rooms = 0
        for day_index, day in enumerate(room_assignments):
            for shift_index, room in enumerate(day):
                if room:
                    num_rooms += 1
                    total_score += self.room_prefs.room_weights[room]
        
        if num_rooms == 0:
            return 0.0 # No rooms assigned to the schedule
        score = total_score / num_rooms # The score is the average of the weights of the rooms assigned to the schedule
        
        if not schedule or not self.room_prefs.available_rooms:
            return 50.0  # Neutral score if no rooms configured
        
        return score * 100 # Normalize to 0-100

    def calculate_continuity_score(self, schedule) -> float:
        """Reward longer rehearsal blocks up to a limit, penalize longer blocks."""
        # Logic explained:
        # final_score = total_score / (total_blocks * max_length) ranges from 0 to 1 and represents the score (/ 100)
        # If all blocks have length == max_length, the final score is 100
        # If some blocks were shorter than max_length, total_score stays the same, but we need more blocks so the final score is lower by a factor of total_blocks / (total_blocks + k)
        # If some blocks were longer than max_length, total_score is penalized by the excess length times the penalty factor
        if not schedule:
            return 0
        max_length = self.continuity_prefs.max_block_length
        penalty = self.continuity_prefs.penalty_factor
        total_score = 0
        total_blocks = 0
        
        for day in schedule:
            # Find consecutive rehearsal blocks (any cast members)
            block_len = 0
            for shift in day:
                if shift and isinstance(shift, dict) and shift.get('members'):  # If there's any rehearsal in this shift
                    block_len += 1
                else:
                    if block_len > 0:
                        total_blocks += 1
                        if block_len <= max_length:
                            total_score += block_len
                        else:
                            total_score += max_length - (block_len - max_length) * penalty
                        block_len = 0
            
            # End of day
            if block_len > 0:
                total_blocks += 1
                if block_len <= max_length:
                    total_score += block_len
                else:
                    total_score += max_length - (block_len - max_length) * penalty
        
        if total_blocks == 0:
            return 0
        # Normalize to 0-100
        return max(0, min(100, (total_score / (total_blocks * max_length)) * 100))


class ScheduleOptimizer:
    """Optimize schedules based on weighted factors."""
    
    def __init__(self, weights: OptimizationWeights, time_prefs: TimeOfDayPreferences, room_prefs: RoomPreferences, continuity_prefs: ContinuityPreferences, rooms_data):
        self.weights = weights
        self.weights.normalize()
        self.calculator = FactorCalculator(time_prefs, room_prefs, continuity_prefs, rooms_data)
    
    def calculate_total_score(self, schedule) -> float:
        """Calculate weighted total score for a schedule."""
        time_score = self.calculator.calculate_time_preference_score(schedule)
        room_score = self.calculator.calculate_room_preference_score(schedule)
        continuity_score = self.calculator.calculate_continuity_score(schedule)
        total_score = (
            time_score * self.weights.time_preference +
            room_score * self.weights.room_preference +
            continuity_score * self.weights.continuity_preference
        )
        
        return total_score
    
    def find_best_schedule(self, solutions: List) -> Tuple[List, float]:
        """Find the best schedule from a list of solutions."""
        if not solutions:
            return [], 0.0

        best_schedule = None
        best_score = -1.0

        for solution in solutions:
            score = self.calculate_total_score(solution)
            if score > best_score:
                best_score = score
                best_schedule = solution

        if best_schedule is None:
            best_schedule = []
        return best_schedule, best_score


class OptimizedRehearsalScheduler(RehearsalScheduler): # inherits from the constrained scheduler
    """Enhanced scheduler with optimization capabilities."""
    
    def __init__(self, weights: OptimizationWeights, time_prefs: TimeOfDayPreferences, room_prefs: RoomPreferences, continuity_prefs: ContinuityPreferences, cast_members, cast_availability, leader_availability, rooms_data):
        # Validate that all required data is provided
        if cast_members is None:
            raise ValueError("cast_members is required but not provided")
        if cast_availability is None:
            raise ValueError("cast_availability is required but not provided")
        if leader_availability is None:
            raise ValueError("leader_availability is required but not provided")
        if rooms_data is None:
            raise ValueError("rooms_data is required but not provided")
        super().__init__(cast_members, cast_availability, leader_availability)
        self.optimizer = ScheduleOptimizer(weights, time_prefs, room_prefs, continuity_prefs, rooms_data)
    
    def solve_optimized(self, requests: List[RehearsalRequest], 
                       num_solutions: int = 200) -> dict:
        """Find and return the best schedule and its score as a dictionary."""
        # Build the model first
        print("Building scheduling model...")
        self.build_model(requests)
        
        # Get multiple feasible solutions
        print("Solving optimization problem...")
        status = self.solve_with_infeasible_requests(solution_limit=num_solutions)
        solutions = self.solutions
        infeasible_requests = self.infeasible_requests
        best_schedule, best_score = self.optimizer.find_best_schedule(solutions)
        room_assignments = self.optimizer.calculator.assign_rooms(best_schedule)

        result = {
            'schedule': best_schedule,
            'score': best_score,
            'infeasible_requests': infeasible_requests,
            'status': status,
            'room_assignments': room_assignments,
        }

        if status == SolverStatus.INFEASIBLE:
            print("No feasible solutions found")
            result['schedule'] = []
            result['score'] = 0.0
            result['infeasible_requests'] = []
            result['room_assignments'] = []
            result['alternate_schedules'] = []
            return result
        elif status == SolverStatus.PARTIALLY_FEASIBLE:
            print("Partially feasible solutions found")
            #print(f"Best schedule: {best_schedule}")
            print(f"Best score: {best_score}")
            print(f"Infeasible requests: {infeasible_requests}")
            return result
        elif status == SolverStatus.FEASIBLE:
            print("Feasible solutions found")
            #print(f"Best schedule: {best_schedule}")
            print(f"Best score: {best_score}")
            return result
        else:
            raise ValueError(f"Invalid solver status: {status}")
    
    def analyze_schedule(self, schedule) -> Dict:
        """Analyze a schedule and return factor scores."""
        return {
            'time_preference_score': self.optimizer.calculator.calculate_time_preference_score(schedule),
            'room_preference_score': self.optimizer.calculator.calculate_room_preference_score(schedule),
            'continuity_score': self.optimizer.calculator.calculate_continuity_score(schedule),
            'total_score': self.optimizer.calculate_total_score(schedule)
        }

class DragDropUtils:
    """Utility class for drag-drop operations."""

    def __init__(self, data_manager, factor_calculator):
        self.data_manager = data_manager
        self.factor_calculator = factor_calculator
        self.alternate_schedules = {} # maybe don't need this as an instance variable

    # Once this class is initialized, we can use it to get the alternate schedules for a given schedule
    # OR to calculate new room assignments for a given schedule using the factor_calculator

    # This method is part of this class to have access to the data_manager, but does not use the solutions architecture
    def get_alternate_schedules(self, schedule: List[List[Dict]]):
        """Get the alternate schedules for scheduled rehearsal."""
        # The output of this method should only be used for the first rehearsal in a merged block of rehearsals
        self.alternate_schedules = {} # key is a tuple (day, shift), value is a list of tuples (day, shift) that are the possible starting times for the rehearsal
        
        # Iterate through to find all the scheduled rehearsals
        for day_index, day in enumerate(schedule): # has size (num_days, num_shifts)
            for shift_index, shift in enumerate(day):
                # shift is a dictionary with the following keys: 'members', 'leader'; empty if no rehearsal is scheduled at this time
                if shift['members']: # If a rehearsal is scheduled at this time, get all the possible starting times for the rehearsal
                    members = shift['members']
                    leader = shift['leader']

                    # Find the post length of the rehearsal: The number of continuous hours scheduled for the group, starting (and including)from the current shift
                    post_length = 0
                    for time_index in range(shift_index, len(day)): # From the current index to the end of the day
                        if day[time_index]['members'] == members and day[time_index]['leader'] == leader:
                            post_length += 1
                        else:
                            break


                    # Find all the available times for the scheduled group (without considering length)
                    available_times = [] # list of all the available times for the scheduled group, each entry is a tuple (day, shift)
                    for d in range(self.data_manager.num_days):
                        for s in range(self.data_manager.num_shifts):
                            for member in shift['members'] + [shift['leader']]: # Leader is included in cast_availability so can be handled together
                                if self.data_manager.cast_availability[self.data_manager.member_to_index(member)][d][s] == 0:
                                    break # If any member is not available, break out of the loop
                            else:
                                available_times.append((d, s)) # If all members are available, add the time to the list
                    print(f"Available times: {available_times}")

                    # Find all the possible starting times for the rehearsal, taking into account the length of the rehearsal
                    possible_starting_times = [] # 2D array of all the possible starting times for the rehearsal
                    for time in available_times: # time is a tuple (day_index, shift_index)
                        for length_increment in range(1, post_length):
                            if (time[0], time[1] + length_increment) not in available_times: # If the index goes beyond the end of the day, it is not a valid starting time
                                break
                        else:
                            possible_starting_times.append((time[0], time[1])) # If all times are valid, add the time to the list


                    self.alternate_schedules[(day_index, shift_index)] = possible_starting_times # key is a tuple (day, shift), value is a 2D array of all the possible starting times for the rehearsal
        return self.alternate_schedules


def main():
    """Example usage of the optimization skeleton."""
    print("=== Schedule Optimization Demo ===\n")

    # Configure time preferences
    print("Configure time of day preferences:")
    print("1. Morning-focused (9:00-12:00)")
    print("2. Afternoon-focused (12:00-17:00)")
    print("3. Evening-focused (17:00-21:00)")
    print("4. Balanced")

    time_choice = input("Enter time preference (1-4): ").strip()

    if time_choice == "1":
        time_prefs = TimeOfDayPreferences(morning_weight=0.6, afternoon_weight=0.2, evening_weight=0.2)
        print("Selected: Morning-focused")
    elif time_choice == "2":
        time_prefs = TimeOfDayPreferences(morning_weight=0.2, afternoon_weight=0.6, evening_weight=0.2)
        print("Selected: Afternoon-focused")
    elif time_choice == "3":
        time_prefs = TimeOfDayPreferences(morning_weight=0.2, afternoon_weight=0.2, evening_weight=0.6)
        print("Selected: Evening-focused")
    else:
        time_prefs = TimeOfDayPreferences(morning_weight=0.33, afternoon_weight=0.34, evening_weight=0.33)
        print("Selected: Balanced")

    time_prefs.normalize()

    # Use default RoomPreferences (no add_room calls)
    print("\nUsing default room preferences.")
    room_prefs = RoomPreferences()

    print("Available rooms configured:")
    for room, weight in room_prefs.room_weights.items():
        print(f"  {room}: {weight:.2f}")

    # Configure continuity preferences
    print("\nConfigure continuity preferences:")
    max_length = int(input("Maximum preferred consecutive rehearsal slots (e.g. 2): ").strip() or 2)
    penalty = float(input("Penalty factor for exceeding block length (e.g. 2.0): ").strip() or 2.0)
    continuity_prefs = ContinuityPreferences(max_block_length=max_length, penalty_factor=penalty)

    # Configure optimization weights
    print("\nChoose optimization focus:")
    print("1. Time-focused")
    print("2. Room-focused")
    print("3. Continuity-focused")
    print("4. Balanced")
    choice = input("Enter choice (1-4): ").strip()
    if choice == "1":
        weights = OptimizationWeights(time_preference=0.7, room_preference=0.2, continuity_preference=0.1)
        print("Selected: Time-focused")
    elif choice == "2":
        weights = OptimizationWeights(time_preference=0.2, room_preference=0.7, continuity_preference=0.1)
        print("Selected: Room-focused")
    elif choice == "3":
        weights = OptimizationWeights(time_preference=0.2, room_preference=0.2, continuity_preference=0.6)
        print("Selected: Continuity-focused")
    else:
        weights = OptimizationWeights(time_preference=0.33, room_preference=0.33, continuity_preference=0.34)
        print("Selected: Balanced")

    # Get availability data for the scheduler
    from ucl_scheduler.data_parsing.availability_manager import get_parsed_availability, DEFAULT_SPREADSHEET_KEY
    cast_members, leaders, cast_availability, leader_availability = get_parsed_availability(DEFAULT_SPREADSHEET_KEY)

    # Create optimized scheduler
    scheduler = OptimizedRehearsalScheduler(weights, time_prefs, room_prefs, continuity_prefs, cast_members, cast_availability, leader_availability)

    # Define rehearsal requests
    requests = [
        RehearsalRequest(['Sophia', 'Tumo', 'Sabine'], 1, 'Sophia', 1),
        RehearsalRequest(['Ollie', 'Mary'], 2, 'Ollie', 2),
        RehearsalRequest(['Cal', 'Charlie'], 1, 'Cal', 3),
        RehearsalRequest(['Sophia', 'Tumo', 'Ollie'], 2, 'Sophia', 4),
    ]

    print(f"\nOptimizing schedule for {len(requests)} rehearsal requests...")

    # Build model and solve
    scheduler.build_model(requests)
    result = scheduler.solve_optimized(requests, num_solutions=200)
    best_schedule = result['schedule']
    best_score = result['score']
    infeasible_requests = result['infeasible_requests']
    status = result['status']
    room_assignments = result['room_assignments']

    if best_schedule:
        best_solution = best_schedule[0] if isinstance(best_schedule[0], list) else best_schedule
        analysis = scheduler.analyze_schedule(best_solution)

        print(f"\n=== OPTIMIZATION RESULTS ===")
        print(f"Time Preference Score: {analysis['time_preference_score']:.2f}")
        print(f"Room Preference Score: {analysis['room_preference_score']:.2f}")
        print(f"Continuity Score: {analysis['continuity_score']:.2f}")
        print(f"Total Score: {analysis['total_score']:.2f}")

        print(f"\n=== SCHEDULE ===")
        # Display the schedule
        view_schedule(best_schedule)

        print(f"\n=== ROOM ASSIGNMENTS ===")
        for day_idx, day in enumerate(room_assignments):
            print(f"Day {day_idx+1}: {day}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()