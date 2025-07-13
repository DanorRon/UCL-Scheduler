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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scheduler_api import app as api_app

# Create the main Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Allow cross-origin requests

# Import and register the API routes
from scheduler_api import generate_schedule, fetch_cast_members, health_check

# Register the routes
app.route('/generate-schedule', methods=['POST'])(generate_schedule)
app.route('/fetch-cast-members', methods=['POST'])(fetch_cast_members)
app.route('/health', methods=['GET'])(health_check)

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False) 