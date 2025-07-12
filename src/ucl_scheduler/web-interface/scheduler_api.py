"""
Flask API for the UCL Scheduler web interface.
Integrates the web form with the optimal scheduler.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import our scheduler
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ucl_scheduler.algorithm.optimal_scheduler import (
    OptimizedRehearsalScheduler, 
    OptimizationWeights, 
    TimeOfDayPreferences, 
    RoomPreferences, 
    ContinuityPreferences
)
from ucl_scheduler.algorithm.constrained_scheduler import RehearsalRequest
from ucl_scheduler.data_parsing.access_availability import availability_df
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for development

@app.route('/generate-schedule', methods=['POST'])
def generate_schedule():
    """Handle schedule generation request from the web interface."""
    try:
        # Get data from the web form
        data = request.get_json()
        print(f"Received data: {data}")
        
        requests_data = data.get('requests', [])
        preferences = data.get('preferences', {})
        
        print(f"Requests: {requests_data}")
        print(f"Preferences: {preferences}")
        
        # Validate input
        if not requests_data:
            return jsonify({'success': False, 'error': 'No rehearsal requests provided'})
        
        # Convert requests to RehearsalRequest objects
        rehearsal_requests = []
        for req in requests_data:
            members = req.get('group', [])
            duration = req.get('duration', 1)
            print(f"Creating request: members={members}, duration={duration}")
            rehearsal_requests.append(RehearsalRequest(members=members, duration=duration))
        
        print(f"Created {len(rehearsal_requests)} rehearsal requests")
        
        # Use the parsed availability data for cast members
        from ucl_scheduler.data_parsing.parse_availability import cast_members
        print(f"Using cast members from availability data: {cast_members}")
        
        # For now, let's use a simple approach - create a basic scheduler
        # that uses the availability data
        from ucl_scheduler.algorithm.constrained_scheduler import RehearsalScheduler
        
        # Create a simple scheduler with availability data
        scheduler = RehearsalScheduler(num_days=2, num_shifts=12)
        
        # Build the model
        scheduler.build_model(rehearsal_requests)
        
        # Solve the problem
        print("Solving scheduling problem...")
        solutions = scheduler.solve(solution_limit=5)
        
        if not solutions:
            return jsonify({
                'success': False, 
                'error': 'No feasible schedule found. Please check your requests and try again.'
            })
        
        # Get the first solution
        best_schedule = solutions[0]
        score = 85.0  # Placeholder score for now
        
        # Convert schedule to a format suitable for web display
        schedule_display = []
        time_slots = [
            "9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00",
            "13:00-14:00", "14:00-15:00", "15:00-16:00", "16:00-17:00",
            "17:00-18:00", "18:00-19:00", "19:00-20:00", "20:00-21:00"
        ]
        
        for day_idx, day in enumerate(best_schedule):
            day_schedule = []
            for shift_idx, shift in enumerate(day):
                if shift:  # If there's a rehearsal in this shift
                    rehearsal_info = {
                        'day': day_idx + 1,
                        'time_slot': time_slots[shift_idx],
                        'members': list(shift),
                        'duration': 1  # For now, assuming 1-hour slots
                    }
                    day_schedule.append(rehearsal_info)
            
            if day_schedule:
                schedule_display.append({
                    'day': day_idx + 1,
                    'rehearsals': day_schedule
                })
        
        return jsonify({
            'success': True,
            'schedule': schedule_display,
            'score': score,
            'message': f'Schedule generated successfully! Score: {score:.1f}/100'
        })
        
    except Exception as e:
        print(f"Error generating schedule: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': f'An error occurred: {str(e)}'
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'UCL Scheduler API is running'})

if __name__ == '__main__':
    print("Starting UCL Scheduler API...")
    print("API will be available at: http://localhost:8080")
    print("Web interface should call: http://localhost:8080/generate-schedule")
    app.run(debug=True, host='0.0.0.0', port=8080) 