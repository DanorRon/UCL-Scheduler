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
from ucl_scheduler.algorithm.constrained_scheduler import RehearsalRequest, SolverStatus
from ucl_scheduler.data_parsing.availability_manager import parse_spreadsheet_url, get_parsed_availability, get_worksheet_names
from ucl_scheduler.data_parsing.room_manager import get_parsed_room_data, rooms_spreadsheet_url
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
        availability_worksheet_name = data.get('availability_worksheet_name', None)
        room_worksheet_name = data.get('room_worksheet_name', None)
        requests_data = data.get('requests', [])
        
        print(f"Google Sheets URL: {sheets_url}")
        print(f"Availability worksheet name: {availability_worksheet_name}")
        print(f"Room worksheet name: {room_worksheet_name}")
        print(f"Requests: {requests_data}")
        
        # Validate input
        if not sheets_url:
            return jsonify({'success': False, 'error': 'Google Sheets URL is required'})
        if not requests_data:
            return jsonify({'success': False, 'error': 'No rehearsal requests provided'})
        if not availability_worksheet_name:
            return jsonify({'success': False, 'error': 'Availability worksheet name is required'})
        if not room_worksheet_name:
            return jsonify({'success': False, 'error': 'Room worksheet name is required'})
        
        # Extract availability spreadsheet key from URL
        try:
            availability_spreadsheet_key = parse_spreadsheet_url(sheets_url)
            print(f"Extracted spreadsheet key: {availability_spreadsheet_key}")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid Google Sheets URL: {str(e)}'})
        
        # Get availability data from Google Sheets
        try:
            cast_members, leaders, cast_availability, leader_availability = get_parsed_availability(availability_spreadsheet_key, availability_worksheet_name)
            print(f"Retrieved availability data for {len(cast_members)} members")
            print(f"Found {len(leaders)} leaders")
            print(f"Cast availability shape: {cast_availability.shape}")
            print(f"Leader availability shape: {leader_availability.shape}")
        except ValueError as e:
            error_msg = str(e)
            if 'environment variable' in error_msg or 'Invalid JSON' in error_msg:
                return jsonify({'success': False, 'error': f'Credentials configuration error: {error_msg}'})
            else:
                return jsonify({'success': False, 'error': f'Invalid data format: {error_msg}'})
        except FileNotFoundError as e:
            error_msg = str(e)
            if 'environment variable' in error_msg:
                return jsonify({'success': False, 'error': f'Credentials not configured: {error_msg}'})
            else:
                return jsonify({'success': False, 'error': f'Credentials file not found: {error_msg}'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to retrieve availability data: {str(e)}'})
        
        # Extract rooms spreadsheet key from URL
        try:
            rooms_spreadsheet_key = parse_spreadsheet_url(rooms_spreadsheet_url)
            print(f"Extracted rooms spreadsheet key: {rooms_spreadsheet_key}")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid Google Sheets URL: {str(e)}'})

        # Get parsed rooms data from Google Sheets
        try:
            rooms_data = get_parsed_room_data(rooms_spreadsheet_key, room_worksheet_name)
            print(f"Room worksheet name: {room_worksheet_name}")
            print(f"Retrieved rooms data: {rooms_data}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to retrieve rooms data: {str(e)}'})
        
        # Convert requests to RehearsalRequest objects
        rehearsal_requests = []
        for i, req in enumerate(requests_data):
            members = req.get('group', [])
            duration = req.get('duration', 1)
            leader = req.get('leader', '')
            if not members:
                return jsonify({'success': False, 'error': 'Rehearsal request must have at least one member'})
            index = i + 1  # Index is simply the position in the list (plus 1 to correspond to the website numbering)
            print(f"Creating request: members={members}, duration={duration}, leader={leader}, index={index}")
            if not leader:
                return jsonify({'success': False, 'error': 'Rehearsal request must have a leader'})
            if leader not in leaders:
                return jsonify({'success': False, 'error': f'Leader "{leader}" not found in leaders list: {leaders}'})
            rehearsal_requests.append(RehearsalRequest(members=members, duration=duration, leader=leader, index=index))
        
        print(f"Created {len(rehearsal_requests)} rehearsal requests")
        
        # Validate that all requested members exist in cast_members
        all_requested_members = set()
        for req in rehearsal_requests:
            all_requested_members.update(req.members)
        
        missing_members = all_requested_members - set(cast_members)
        if missing_members:
            return jsonify({'success': False, 'error': f'Requested members not found in availability data: {list(missing_members)}'})
        
        print(f"All requested members found in cast data: {list(all_requested_members)}")
        
        # Create optimization preferences (assume these are already normalized and set in optimal_scheduler)
        time_prefs = TimeOfDayPreferences()
        
        room_prefs = RoomPreferences()
        continuity_prefs = ContinuityPreferences()
        
        weights = OptimizationWeights()
        
        # Create optimized scheduler with availability data
        try:
            print(f"Creating scheduler with {len(cast_members)} members, {len(rehearsal_requests)} requests")
            scheduler = OptimizedRehearsalScheduler(weights, time_prefs, room_prefs, continuity_prefs, cast_members, cast_availability, leader_availability, rooms_data)
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
            best_schedule, score, infeasible_requests, status, room_assignments = scheduler.solve_optimized(rehearsal_requests, num_solutions=200)
            print("Best schedule and score obtained from solve_optimized.")
        except Exception as e:
            print(f"Error in solve_optimized: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Failed to execute solve_optimized: {str(e)}'})
        
        ''' # This does not work because the default value of best_schedule is []
        if not best_schedule:
            return jsonify({
                'success': False, 
                'error': 'No schedule returned from solve_optimized.'
            })
        '''
        
        if status == SolverStatus.INFEASIBLE:
            return jsonify({
                'success': False,
                'status': status.name.lower(),
                'error': 'No feasible schedule found. Please check your requests and try again.'
            })
        elif status == SolverStatus.PARTIALLY_FEASIBLE:
            return jsonify({
                'success': True,
                'schedule': best_schedule,
                'score': score,
                'infeasible_requests': infeasible_requests,
                'status': status.name.lower(),
                'room_assignments': room_assignments,
                'message': f'Schedule generated successfully! Score: {score:.1f}/100'
            })
        elif status == SolverStatus.FEASIBLE:
            print(f"Room assignments: {room_assignments}")
            return jsonify({
                'success': True,
                'schedule': best_schedule,
                'score': score,
                'status': status.name.lower(),
                'room_assignments': room_assignments,
                'message': f'Schedule generated successfully! Score: {score:.1f}/100'
            })
        else:
            raise ValueError(f"Invalid solver status: {status}")
        
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
        availability_worksheet_name = data.get('availability_worksheet_name', None)
        
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
            cast_members, leaders, cast_availability, leader_availability = get_parsed_availability(spreadsheet_key, availability_worksheet_name)
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

# Retrieve the worksheet names from the availability spreadsheet URL.
@app.route('/fetch-availability-worksheets', methods=['POST'])
def fetch_availability_worksheet_names():
    """Fetch worksheet names from the availability spreadsheet URL."""
    try:
        data = request.get_json()
        sheets_url = data.get('sheets_url', '') # This is the URL of the availability spreadsheet.
        
        if not sheets_url:
            return jsonify({'success': False, 'error': 'Google Sheets URL is required'})
        
        # Extract spreadsheet key from URL
        try:
            availability_spreadsheet_key = parse_spreadsheet_url(sheets_url)
            print(f"Extracted spreadsheet key for availability worksheets: {availability_spreadsheet_key}")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid Google Sheets URL: {str(e)}'})
        
        # Get worksheet names from Google Sheets
        try:
            worksheet_names = get_worksheet_names(availability_spreadsheet_key)
            print(f"Retrieved worksheet names: {worksheet_names}")
            return jsonify({
                'success': True,
                'worksheets': worksheet_names
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to retrieve worksheet names: {str(e)}'})
            
    except Exception as e:
        print(f"Error fetching worksheets: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': f'An error occurred: {str(e)}'
        })

# Retrieve the worksheet names from the rooms spreadsheet URL.
@app.route('/fetch-rooms-worksheets', methods=['POST'])
def fetch_rooms_worksheet_names():
    """Fetch worksheet names from the rooms spreadsheet URL."""
    try:
        # rooms_spreadsheet_url is a global variable in room_manager.py. The URL is hardcoded in the file.
        if not rooms_spreadsheet_url:
            return jsonify({'success': False, 'error': 'Google Sheets URL is required'})
        
        # Extract spreadsheet key from URL
        try:
            rooms_spreadsheet_key = parse_spreadsheet_url(rooms_spreadsheet_url)
            print(f"Extracted spreadsheet key for rooms worksheets: {rooms_spreadsheet_key}")
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid Google Sheets URL: {str(e)}'})

        # Get worksheet names from Google Sheets
        try:
            worksheet_names = get_worksheet_names(rooms_spreadsheet_key)
            print(f"Retrieved worksheet names: {worksheet_names}")
            return jsonify({
                'success': True,
                'worksheets': worksheet_names
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to retrieve worksheet names: {str(e)}'})
            
    except Exception as e:
        print(f"Error fetching worksheets: {e}")
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