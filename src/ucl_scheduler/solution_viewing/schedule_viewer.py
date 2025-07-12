import webbrowser
import os
from datetime import datetime

def create_html_schedule(solutions, cast_members, num_days=2, num_shifts=12):
    """Create an HTML file to display the rehearsal schedule."""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rehearsal Schedule</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .solution { margin-bottom: 30px; border: 2px solid #ddd; padding: 15px; border-radius: 8px; }
            .solution h2 { color: #333; margin-top: 0; }
            .day { margin-bottom: 20px; }
            .day h3 { color: #666; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 10px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; font-weight: bold; }
            .shift-time { background-color: #e8f4f8; font-weight: bold; }
            .rehearsal { background-color: #d4edda; }
            .empty { background-color: #f8f9fa; color: #6c757d; }
            .timestamp { color: #666; font-size: 12px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>🎭 Rehearsal Schedule</h1>
    """
    
    for i, solution in enumerate(solutions, 1):
        html_content += f"""
        <div class="solution">
            <h2>Solution {i}</h2>
        """
        
        for day in range(num_days):
            html_content += f"""
            <div class="day">
                <h3>Day {day}</h3>
                <table>
                    <tr>
                        <th>Shift</th>
                        <th>Time</th>
                        <th>Cast Members</th>
                    </tr>
            """
            
            for shift in range(num_shifts):
                # Convert shift number to time (assuming 1-hour shifts starting at 9 AM)
                hour = 9 + shift
                time_str = f"{hour:02d}:00"
                
                if solution[day][shift]:
                    members = ", ".join(solution[day][shift])
                    html_content += f"""
                    <tr class="rehearsal">
                        <td class="shift-time">{shift}</td>
                        <td class="shift-time">{time_str}</td>
                        <td>{members}</td>
                    </tr>
                    """
                else:
                    html_content += f"""
                    <tr>
                        <td class="shift-time">{shift}</td>
                        <td class="shift-time">{time_str}</td>
                        <td class="empty">No rehearsals</td>
                    </tr>
                    """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += "</div>"
    
    html_content += f"""
        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    # Write to file
    filename = "rehearsal_schedule.html"
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename

def parse_solution_output(output_text):
    """Parse the solution output from the solver to extract schedule data."""
    solutions = []
    current_solution = None
    current_day = None
    
    lines = output_text.split('\n')
    
    for line in lines:
        if "=== Rehearsal Schedule Solution" in line:
            if current_solution is not None:
                solutions.append(current_solution)
            current_solution = [[[] for _ in range(12)] for _ in range(2)]  # 2 days, 12 shifts
            current_day = None
        elif "Day" in line and ":" in line:
            current_day = int(line.split("Day")[1].split(":")[0].strip())
        elif "Shift" in line and ":" in line and current_day is not None and current_solution is not None:
            parts = line.split(":")
            shift_num = int(parts[0].split("Shift")[1].strip())
            if len(parts) > 1 and parts[1].strip() != "No rehearsals":
                members = [m.strip() for m in parts[1].split(",")]
                current_solution[current_day][shift_num] = members
    
    if current_solution is not None:
        solutions.append(current_solution)
    
    return solutions

# Example usage:
if __name__ == "__main__":
    # This would be called after running your solver
    # You would capture the output and pass it to parse_solution_output
    print("Schedule viewer created! Run your solver and then use this to create HTML view.") 