#!/usr/bin/env python3
"""
Simple entry point for Railway deployment.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Debug: Print path information
print(f"DEBUG: Current working directory: {os.getcwd()}")
print(f"DEBUG: Added src_path to sys.path: {src_path}")
print(f"DEBUG: src_path exists: {src_path.exists()}")

# Import the Flask app
import ucl_scheduler.web_interface.app as web_app
app = web_app.app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 