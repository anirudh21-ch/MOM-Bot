#!/usr/bin/env python3
"""
MOM-Bot Streamlit Application Launcher

Run this file with: streamlit run app.py
Or: python -m streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    
    # Run Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
