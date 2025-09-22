#!/usr/bin/env python3
"""
Run the AI-Powered Delivery Failure Analysis Streamlit App

This script launches the Streamlit web interface for the delivery failure analysis POC.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("🚀 Starting AI-Powered Delivery Failure Analysis App...")
    print("=" * 60)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'src', 'app.py')
    
    if not os.path.exists(app_path):
        print(f"❌ Error: App file not found at {app_path}")
        sys.exit(1)
    
    print(f"📁 App location: {app_path}")
    print("🌐 Starting Streamlit server...")
    print("📝 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
