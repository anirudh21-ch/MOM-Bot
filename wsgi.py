"""
WSGI Entry Point for MOM-Bot Production Deployment with Gunicorn
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Flask app
from src.main.main import app

# Application callable for Gunicorn
application = app

if __name__ == "__main__":
    # For development/testing only
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
