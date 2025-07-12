import webbrowser
from datetime import datetime

def create_html_schedule(solutions, cast_members, num_days=2, num_shifts=12):
    """Convert solution arrays to HTML schedule view."""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rehearsal Schedule</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #2c3e50; 
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .solution { 
                margin-bottom: 40px; 
                border: 2px solid #3498db; 
                padding: 20px; 
                border-radius: 10px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }
            .solution h2 { 
                color: #2c3e50; 
                margin-top: 0; 
                text-align: center;
                font-size: 1.8em;
            }
            .days-container {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }
            .day { 
                flex: 1;
                min-width: 300px;
                margin-bottom: 20px; 
                border: 1px solid #dee2e6; 
                padding: 15px; 
                border-radius: 8px;
                background: white;
            }
            .day h3 { 
                color: #495057; 
                margin-top: 0;
                text-align: center;
                font-size: 1.3em;
                padding-bottom: 10px;
                border-bottom: 2px solid #e9ecef;
            }
            table { 
                border-collapse: collapse; 
                width: 100%; 
                margin-bottom: 10px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            th, td { 
                border: 1px solid #dee2e6; 
                padding: 10px; 
                text-align: center; 
                font-size: 14px;
            }
            th { 
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            .shift-time { 
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                font-weight: bold; 
                color: #1976d2;
                font-size: 12px;
            }
            .rehearsal { 
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-left: 4px solid #28a745;
                font-weight: 500;
            }
            .multi-hour {
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%) !important;
                border-left: 4px solid #ffc107 !important;
                font-weight: 600;
            }
            .empty { 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                color: #6c757d; 
                font-style: italic;
            }
            .timestamp { 
                color: #6c757d; 
                font-size: 12px; 
                margin-top: 30px;
                text-align: center;
                border-top: 1px solid #dee2e6;
                padding-top: 20px;
            }
            .stats {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                text-align: center;
            }
            .stats h3 {
                margin: 0 0 10px 0;
                color: #1976d2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Rehearsal Schedule</h1>
    """
    
    # Add statistics
    total_rehearsals = sum(len([s for s in day if s]) for solution in solutions for day in solution)
    total_solutions = len(solutions)
    
    html_content += f"""
            <div class="stats">
                <h3>📊 Schedule Statistics</h3>
                <p><strong>{total_solutions}</strong> solution(s) found | <strong>{total_rehearsals}</strong> total rehearsals scheduled</p>
            </div>
    """
    
    for i, solution in enumerate(solutions, 1):
        html_content += f"""
            <div class="solution">
                <h2>Solution {i}</h2>
                <div class="days-container">
        """
        
        for day in range(num_days):
            html_content += f"""
                    <div class="day">
                        <h3>Day {day}</h3>
                        <table>
                            <tr>
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
                    # Check if this might be part of a multi-hour rehearsal
                    is_multi_hour = False
                    if shift > 0 and solution[day][shift-1] == solution[day][shift]:
                        is_multi_hour = True
                    
                    rehearsal_class = "rehearsal multi-hour" if is_multi_hour else "rehearsal"
                    html_content += f"""
                    <tr class="{rehearsal_class}">
                        <td class="shift-time">{time_str}</td>
                        <td>{members}</td>
                    </tr>
                    """
                else:
                    html_content += f"""
                    <tr>
                        <td class="shift-time">{time_str}</td>
                        <td class="empty">No rehearsals</td>
                    </tr>
                    """
            
            html_content += """
                        </table>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
    
    html_content += f"""
            <div class="timestamp">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    filename = "rehearsal_schedule.html"
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename

def convert_solutions_to_html(solutions):
    """Simple wrapper to convert solutions to HTML."""
    return create_html_schedule(solutions, cast_members=None)

# Example usage:
if __name__ == "__main__":
    # Example solution array format
    example_solutions = [
        [
            [['Ollie', 'Sophia', 'Tumo'], [], ['Ollie'], [], ['Mary', 'Sade'], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], [], [], []]
        ]
    ]
    
    filename = convert_solutions_to_html(example_solutions)
    print(f"HTML schedule created: {filename}")
    print("Open this file in your browser to view the schedule!") 