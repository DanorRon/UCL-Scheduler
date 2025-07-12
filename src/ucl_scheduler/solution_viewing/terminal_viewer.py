def print_schedule_terminal(solutions, num_days=2, num_shifts=12):
    """Display schedule in terminal format."""
    
    def print_header():
        print("="*60)
        print("REHEARSAL SCHEDULE")
        print("="*60)
    
    def print_day_header(day):
        print(f"\n--- DAY {day} ---")
    
    def print_time_slot(shift, members, is_rehearsal=False):
        hour = 9 + shift
        time_str = f"{hour:02d}:00"
        
        if is_rehearsal and members:
            print(f"{time_str}: {', '.join(members)}")
        else:
            print(f"{time_str}: No rehearsals")
    
    def print_solution_header(solution_num):
        print(f"\n--- SOLUTION {solution_num} ---")
    
    # Print main header
    print_header()
    
    # Print each solution
    for i, solution in enumerate(solutions, 1):
        print_solution_header(i)
        
        for day in range(num_days):
            print_day_header(day)
            
            for shift in range(num_shifts):
                if solution[day][shift]:
                    print_time_slot(shift, solution[day][shift], is_rehearsal=True)
                else:
                    print_time_slot(shift, [], is_rehearsal=False)

def print_simple_schedule(solutions):
    """Display a simple version."""
    print("\n" + "="*50)
    print("REHEARSAL SCHEDULE")
    print("="*50)
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n--- SOLUTION {i} ---")
        
        for day in range(len(solution)):
            print(f"\nDay {day}:")
            for shift in range(len(solution[day])):
                hour = 9 + shift
                time_str = f"{hour:02d}:00"
                
                if solution[day][shift]:
                    members = ", ".join(solution[day][shift])
                    print(f"  {time_str}: {members}")
                else:
                    print(f"  {time_str}: No rehearsals")

def view_schedule(solutions, use_colors=True):
    """Main function to view schedule in terminal."""
    if not solutions:
        print("No solutions to display!")
        return
    
    try:
        if use_colors:
            print_schedule_terminal(solutions)
        else:
            print_simple_schedule(solutions)
    except:
        print("Using simple view...")
        print_simple_schedule(solutions)

# Example usage:
if __name__ == "__main__":
    # Example solution array
    example_solutions = [
        [
            [['Ollie', 'Sophia', 'Tumo'], [], ['Ollie'], [], ['Mary', 'Sade'], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], [], [], []]
        ]
    ]
    
    view_schedule(example_solutions) 