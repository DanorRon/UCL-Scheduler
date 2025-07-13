"""
Skeleton for schedule optimization system.
Add features incrementally over time.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from .constrained_scheduler import RehearsalScheduler, RehearsalRequest

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
    available_rooms: List[str]
    room_weights: Dict[str, float]  # Preference weights for each room
    
    def __init__(self):
        self.available_rooms = []
        self.room_weights = {}
    
    def add_room(self, room_name: str, weight: float = 1.0):
        """Add a room with its preference weight."""
        self.available_rooms.append(room_name)
        self.room_weights[room_name] = weight
    
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
    room_preference: float = 0.0
    continuity_preference: float = 0.7
    
    def normalize(self):
        total = self.time_preference + self.room_preference + self.continuity_preference
        self.time_preference /= total
        self.room_preference /= total
        self.continuity_preference /= total


class FactorCalculator:
    """Calculate scores for different optimization factors."""
    
    def __init__(self, time_prefs: TimeOfDayPreferences, room_prefs: RoomPreferences, continuity_prefs: ContinuityPreferences):
        self.time_prefs = time_prefs
        self.room_prefs = room_prefs
        self.continuity_prefs = continuity_prefs
    
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
    
    def calculate_room_preference_score(self, schedule) -> float:
        """Calculate room preference score (0-100)."""
        # Placeholder for room preference scoring
        # This will be implemented when room selection logic is added
        
        if not schedule or not self.room_prefs.available_rooms:
            return 50.0  # Neutral score if no rooms configured
        
        # For now, return a neutral score
        # TODO: Implement room assignment and scoring logic
        return 50.0

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
    
    def __init__(self, weights: OptimizationWeights, time_prefs: TimeOfDayPreferences, room_prefs: RoomPreferences, continuity_prefs: ContinuityPreferences):
        self.weights = weights
        self.weights.normalize()
        self.calculator = FactorCalculator(time_prefs, room_prefs, continuity_prefs)
    
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
    
    def __init__(self, weights: OptimizationWeights, time_prefs: TimeOfDayPreferences, room_prefs: RoomPreferences, continuity_prefs: ContinuityPreferences, cast_members, cast_availability, leader_availability):
        # Validate that all required data is provided
        if cast_members is None:
            raise ValueError("cast_members is required but not provided")
        if cast_availability is None:
            raise ValueError("cast_availability is required but not provided")
        if leader_availability is None:
            raise ValueError("leader_availability is required but not provided")
        
        super().__init__(cast_members, cast_availability, leader_availability)
        self.optimizer = ScheduleOptimizer(weights, time_prefs, room_prefs, continuity_prefs)
    
    def solve_optimized(self, requests: List[RehearsalRequest], 
                       num_solutions: int = 200) -> Tuple[List, float]:
        """Find and return the best schedule and its score."""
        # Build the model first
        print("Building scheduling model...")
        self.build_model(requests)
        
        # Get multiple feasible solutions
        print("Solving optimization problem...")
        solutions = self.solve(solution_limit=num_solutions)
        
        if not solutions:
            print("No feasible solutions found")
            return [], 0.0
        
        # Find the best one
        best_schedule, best_score = self.optimizer.find_best_schedule(solutions)
        
        print(f"Generated {len(solutions)} solutions")
        print(f"Best schedule score: {best_score:.2f}")
        return best_schedule, best_score
    
    def analyze_schedule(self, schedule) -> Dict:
        """Analyze a schedule and return factor scores."""
        return {
            'time_preference_score': self.optimizer.calculator.calculate_time_preference_score(schedule),
            'room_preference_score': self.optimizer.calculator.calculate_room_preference_score(schedule),
            'continuity_score': self.optimizer.calculator.calculate_continuity_score(schedule),
            'total_score': self.optimizer.calculate_total_score(schedule)
        }


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
    
    # Configure room preferences
    print("\nConfigure room preferences:")
    room_prefs = RoomPreferences()
    
    # Add some example rooms (can be customized)
    room_prefs.add_room("Studio A", 1.0)
    room_prefs.add_room("Studio B", 0.8)
    room_prefs.add_room("Rehearsal Room 1", 0.6)
    room_prefs.add_room("Rehearsal Room 2", 0.6)
    
    room_prefs.normalize_weights()
    
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
        RehearsalRequest(['Sophia', 'Tumo', 'Sabine'], 1, 'Sophia'),
        RehearsalRequest(['Ollie', 'Mary'], 2, 'Ollie'),
        RehearsalRequest(['Cal', 'Charlie'], 1, 'Cal'),
        RehearsalRequest(['Sophia', 'Tumo', 'Ollie'], 2, 'Sophia'),
    ]
    
    print(f"\nOptimizing schedule for {len(requests)} rehearsal requests...")
    
    # Build model and solve
    scheduler.build_model(requests)
    best_schedule, best_score = scheduler.solve_optimized(requests, num_solutions=200)
    
    if best_schedule:
        best_solution = best_schedule[0]
        analysis = scheduler.analyze_schedule(best_solution)
        
        print(f"\n=== OPTIMIZATION RESULTS ===")
        print(f"Time Preference Score: {analysis['time_preference_score']:.2f}")
        print(f"Room Preference Score: {analysis['room_preference_score']:.2f}")
        print(f"Continuity Score: {analysis['continuity_score']:.2f}")
        print(f"Total Score: {analysis['total_score']:.2f}")
        
        print(f"\n=== SCHEDULE ===")
        # Display the schedule
        view_schedule(best_schedule)
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()