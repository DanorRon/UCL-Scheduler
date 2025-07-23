"""
Main Flask application for the UCL Scheduler web interface.
This file serves the HTML interface and provides the API endpoints.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import our scheduler
# This works both locally and on Railway
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent  # Go up from web_interface to src
sys.path.insert(0, str(src_dir))

# Create the main Flask app
# Set the static folder to the current folder so index.html is found wherever the start point for the program is
app = Flask(__name__, 
           template_folder=str(current_dir / 'templates'),
           static_folder=str(current_dir / 'static'))


# Load secret key from credentials file
# Get the credentials file path relative to the package
package_dir = Path(__file__).parent.parent
secret_key_path = package_dir / "credentials" / "flask_secret_key.txt"

if secret_key_path.exists():
    print("Using secret key from credentials file")
    with open(secret_key_path, 'r') as f:
        app.secret_key = f.read().strip()
else:
    # Fall back to environment variable (for deployment)
    print("Secret key file not found, checking environment variable")
    secret_key = os.environ.get('FLASK_SECRET_KEY')

    if not secret_key:
        raise ValueError("FLASK_SECRET_KEY environment variable not set")
    app.secret_key = secret_key


CORS(app, origins=['http://localhost:8080', 'https://web-production-38d52.up.railway.app/'])  # Allow cross-origin requests for development


# Import and register the API routes
from ucl_scheduler.web_interface.scheduler_api import (
    generate_schedule,
    fetch_cast_members,
    fetch_availability_worksheet_names,
    fetch_rooms_worksheet_names,
    health_check,
    test_session,
    calculate_alternates,
    recalculate_rooms
)

# Register the routes
app.route('/generate-schedule', methods=['POST'])(generate_schedule)
app.route('/fetch-cast-members', methods=['POST'])(fetch_cast_members)
app.route('/fetch-availability-worksheets', methods=['POST'])(fetch_availability_worksheet_names)
app.route('/fetch-rooms-worksheets', methods=['POST'])(fetch_rooms_worksheet_names)
app.route('/health', methods=['GET'])(health_check)
app.route('/test-session', methods=['GET'])(test_session)
app.route('/calculate-alternates', methods=['POST'])(calculate_alternates)
app.route('/recalculate-rooms', methods=['POST'])(recalculate_rooms)

@app.route('/')
def index():
    """Serve the main HTML interface."""
    static_dir = Path(__file__).parent
    return send_from_directory(static_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    static_dir = Path(__file__).parent
    return send_from_directory(static_dir, filename)

# This is only run when this file is run directly
if __name__ == '__main__':
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    # Non production server
    #app.run(host='0.0.0.0', port=port, debug=False)

    from waitress import serve
    serve(app, host="0.0.0.0", port=port)