#!/usr/bin/env python3
"""
Video Simulation App Launcher
Simple script to run the video simulation web application
"""

import os
import sys
import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")    

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import cv2
        import numpy
        import torch
        import transformers
        import timm
        import sklearn
        import openai
        print("✅ All required dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Please install requirements: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment variables"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("   GPT-based features will not work")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        return False
    else:
        print("✅ OpenAI API key found")
        return True

def check_directories():
    """Check required directories and files"""
    current_dir = Path(__file__).parent
    
    required_dirs = [
        'videos',
        'data_json',
        'templates',
        'scene_change_detection',
        'gpt_scene_description',
        'live_descriptor'
    ]
    
    required_files = [
        'video_simulation_app.py',
        'templates/video_simulation.html',
        'scene_change_detection/change_detector.py',
        'gpt_scene_description/gpt_descriptor.py',
        'live_descriptor/live_descriptor.py',
        'locomotion_infer.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not (current_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    for file_path in required_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
    
    if not missing_dirs and not missing_files:
        print("✅ All required directories and files found")
        return True
    
    return False

def list_available_data():
    """List available videos and data files"""
    current_dir = Path(__file__).parent
    
    videos_dir = current_dir / 'videos'
    data_dir = current_dir / 'data_json'
    
    print("\n📁 Available files:")
    
    if videos_dir.exists():
        videos = list(videos_dir.glob('*.mp4'))
        print(f"   Videos ({len(videos)}): {', '.join(v.name for v in videos)}")
    else:
        print("   Videos: directory not found")
    
    if data_dir.exists():
        data_files = list(data_dir.glob('*.json'))
        print(f"   Data files ({len(data_files)}): {', '.join(d.name for d in data_files)}")
    else:
        print("   Data files: directory not found")

def main():
    """Main launcher function"""
    print("🚀 Video Simulation App Launcher")
    print("=" * 50)
    
    # Check system requirements
    print("\n1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n2. Checking environment...")
    env_ok = check_environment()
    
    print("\n3. Checking directories and files...")
    files_ok = check_directories()
    
    if not files_ok:
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        return 1
    
    # List available data
    list_available_data()
    
    # Show usage instructions
    print("\n🌐 Starting Video Simulation Web App...")
    print("   📱 Open your browser and go to: http://localhost:5000")
    print("   ⏹️  Press Ctrl+C to stop the server")
    
    if not env_ok:
        print("\n⚠️  Note: Some features may not work without OpenAI API key")
    
    print("\n" + "=" * 50)
    
    # Run the Flask app
    try:
        # Add current directory to Python path
        current_dir = str(Path(__file__).parent)
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import and run the app
        from video_simulation_app import app
        
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
