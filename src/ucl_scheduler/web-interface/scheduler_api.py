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
from ucl_scheduler.data_parsing.availability_manager import parse_spreadsheet_url, get_parsed_availability
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
        
        sheets_url = data.get('sheets_url', '')
        requests_data = data.get('requests', [])
        preferences = data.get('preferences', {})
        
        print(f"Google Sheets URL: {sheets_url}")
        print(f"Requests: {requests_data}")
        print(f"Preferences: {preferences}")
        
        # Validate input
        if not sheets_url:
            return jsonify({'success': False, 'error': 'Google Sheets URL is required'})
        if not requests_data:
            return jsonify({'success': False, 'error': 'No rehearsal requests provided'})
        
        # Extract spreadsheet key from URL
        try:
            spreadsheet_key = parse_spreadsheet_url(sheets_url)
            print(f"Extracted spreadsheet key: {spreadsheet_key}")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid Google Sheets URL: {str(e)}'})
        
        # Get availability data from Google Sheets
        try:
            cast_members, leaders, cast_availability, leader_availability = get_parsed_availability(spreadsheet_key)
            print(f"Retrieved availability data for {len(cast_members)} members")
            print(f"Found {len(leaders)} leaders")
            print(f"Cast availability shape: {cast_availability.shape}")
            print(f"Leader availability shape: {leader_availability.shape}")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid data format: {str(e)}'})
        except FileNotFoundError as e:
            return jsonify({'success': False, 'error': f'Credentials not found: {str(e)}'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to retrieve availability data: {str(e)}'})
        
        # Convert requests to RehearsalRequest objects
        rehearsal_requests = []
        for req in requests_data:
            members = req.get('group', [])
            duration = req.get('duration', 1)
            print(f"Creating request: members={members}, duration={duration}")
            if not members:
                return jsonify({'success': False, 'error': 'Rehearsal request must have at least one member'})
            rehearsal_requests.append(RehearsalRequest(members=members, duration=duration))
        
        print(f"Created {len(rehearsal_requests)} rehearsal requests")
        
        # Validate that all requested members exist in cast_members
        all_requested_members = set()
        for req in rehearsal_requests:
            all_requested_members.update(req.members)
        
        missing_members = all_requested_members - set(cast_members)
        if missing_members:
            return jsonify({'success': False, 'error': f'Requested members not found in availability data: {list(missing_members)}'})
        
        print(f"All requested members found in cast data: {list(all_requested_members)}")
        
        # Create optimization preferences
        time_prefs = TimeOfDayPreferences(
            morning_weight=preferences.get('morning', 30) / 100,
            afternoon_weight=preferences.get('afternoon', 40) / 100,
            evening_weight=preferences.get('evening', 30) / 100
        )
        time_prefs.normalize()
        
        room_prefs = RoomPreferences()
        continuity_prefs = ContinuityPreferences()
        
        weights = OptimizationWeights(
            time_preference=0.6,
            room_preference=0.2,
            continuity_preference=0.2
        )
        weights.normalize()
        
        # Create optimized scheduler with availability data
        try:
            print(f"Creating scheduler with {len(cast_members)} members, {len(rehearsal_requests)} requests")
            scheduler = OptimizedRehearsalScheduler(weights, time_prefs, room_prefs, continuity_prefs, cast_members, cast_availability, leader_availability)
            print("Scheduler created successfully")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid scheduler configuration: {str(e)}'})
        except Exception as e:
            print(f"Error creating scheduler: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Failed to create scheduler: {str(e)}'})
        
        # Build and solve the model
        print("Building and solving scheduling problem...")
        try:
            best_schedule, score = scheduler.solve_optimized(rehearsal_requests, num_solutions=5)
            print("Best schedule and score obtained from solve_optimized.")
        except Exception as e:
            print(f"Error solving optimization: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Failed to solve optimization: {str(e)}'})
        
        if not best_schedule:
            return jsonify({
                'success': False, 
                'error': 'No feasible schedule found. Please check your requests and try again.'
            })
        
        # Return the best schedule directly (already in 2D array format)
        return jsonify({
            'success': True,
            'schedule': best_schedule,
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

@app.route('/fetch-cast-members', methods=['POST'])
def fetch_cast_members():
    """Fetch cast members from a Google Sheets URL."""
    try:
        data = request.get_json()
        sheets_url = data.get('sheets_url', '')
        
        if not sheets_url:
            return jsonify({'success': False, 'error': 'Google Sheets URL is required'})
        
        # Extract spreadsheet key from URL
        try:
            spreadsheet_key = parse_spreadsheet_url(sheets_url)
            print(f"Extracted spreadsheet key for cast members: {spreadsheet_key}")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid Google Sheets URL: {str(e)}'})
        
        # Get availability data from Google Sheets
        try:
            cast_members, leaders, cast_availability, leader_availability = get_parsed_availability(spreadsheet_key)
            print(f"Retrieved cast members: {cast_members}")
            print(f"Retrieved leaders: {leaders}")
            return jsonify({
                'success': True,
                'cast_members': cast_members,
                'leaders': leaders
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to retrieve cast members: {str(e)}'})
            
    except Exception as e:
        print(f"Error fetching cast members: {e}")
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