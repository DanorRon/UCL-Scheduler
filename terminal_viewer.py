from colorama import init, Fore, Back, Style
import os

# Initialize colorama for cross-platform colored output
init()

def print_schedule_terminal(solutions, num_days=2, num_shifts=12):
    """Display schedule in a beautiful terminal format."""
    
    def print_header():
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}🎭  REHEARSAL SCHEDULE  🎭")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def print_day_header(day):
        print(f"\n{Fore.GREEN}{'─'*30}")
        print(f"{Fore.WHITE}📅 DAY {day}")
        print(f"{Fore.GREEN}{'─'*30}{Style.RESET_ALL}")
    
    def print_time_slot(shift, members, is_rehearsal=False):
        hour = 9 + shift
        time_str = f"{hour:02d}:00"
        
        if is_rehearsal:
            # Check if this is part of a multi-hour rehearsal
            if len(members) > 0:
                print(f"{Fore.WHITE}🕐 {time_str} │ {Fore.GREEN}👥 {', '.join(members)}{Style.RESET_ALL}")
            else:
                print(f"{Fore.WHITE}🕐 {time_str} │ {Fore.RED}❌ No rehearsals{Style.RESET_ALL}")
        else:
            print(f"{Fore.WHITE}🕐 {time_str} │ {Fore.RED}❌ No rehearsals{Style.RESET_ALL}")
    
    def print_solution_header(solution_num):
        print(f"\n{Fore.MAGENTA}{'═'*50}")
        print(f"{Fore.YELLOW}📋 SOLUTION {solution_num}")
        print(f"{Fore.MAGENTA}{'═'*50}{Style.RESET_ALL}")
    
    def print_stats(solutions):
        total_rehearsals = sum(len([s for s in day if s]) for solution in solutions for day in solution)
        total_solutions = len(solutions)
        
        print(f"\n{Fore.CYAN}📊 STATISTICS:")
        print(f"{Fore.WHITE}   • Solutions found: {Fore.YELLOW}{total_solutions}")
        print(f"{Fore.WHITE}   • Total rehearsals: {Fore.YELLOW}{total_rehearsals}")
        print(f"{Style.RESET_ALL}")
    
    # Print main header
    print_header()
    print_stats(solutions)
    
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
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}✨ Schedule display complete!")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

def print_simple_schedule(solutions):
    """Display a simpler version without colors (for compatibility)."""
    
    print("\n" + "="*50)
    print("🎭 REHEARSAL SCHEDULE")
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
    
    print("\n" + "="*50)

def view_schedule(solutions, use_colors=True):
    """Main function to view schedule in terminal."""
    if not solutions:
        print("❌ No solutions to display!")
        return
    
    try:
        if use_colors:
            print_schedule_terminal(solutions)
        else:
            print_simple_schedule(solutions)
    except ImportError:
        print("⚠️  Colorama not available, using simple view...")
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