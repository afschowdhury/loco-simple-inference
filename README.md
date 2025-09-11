# Video Frame Analyzer with AI Descriptions

A Flask web application that extracts frames from videos and generates AI-powered descriptions using the Moondream vision model via Ollama.

## Features

🎬 **Video Frame Extraction**: Extract frames at selected FPS (1, 5, or 10)  
🤖 **AI Descriptions**: Generate natural language descriptions using Moondream model  
🖼️ **Side-by-Side Viewer**: Display images and descriptions in an elegant web interface  
📱 **Responsive Design**: Modern, mobile-friendly user interface  

## Prerequisites

1. **Ollama** with Moondream model installed:
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Moondream model
ollama pull moondream
```

2. **Python environment** (recommended to use the existing loco-env):
```bash
source loco-env/bin/activate
```

## Setup

1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

2. Ensure Ollama is running:
```bash
ollama serve
```

## Usage

### 1. Start the Application
```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

### 2. Process Video Frames
- Upload a video file (MP4, AVI, MOV, etc.)
- Select target FPS (1, 5, or 10)
- Frames will be extracted to `outputs/` directory

### 3. Generate AI Descriptions
- After frame extraction, click "View Frames with AI Descriptions"
- Click "Generate AI Descriptions" to analyze all frames with Moondream
- View the side-by-side display of images and descriptions

### 4. Standalone Frame Processing
You can also process frames directly:
```bash
python frame_processor.py
```

## File Structure

```
loco-simple/
├── app.py                    # Main Flask application
├── frame_processor.py        # AI description generator
├── processed_frames.json     # Generated descriptions data
├── templates/
│   ├── frames.html          # Frame viewer template
│   └── ...
├── outputs/                 # Extracted frames
└── requirements.txt         # Dependencies
```

## API Endpoints

- `GET /` - Main page for video upload
- `GET /frames/<dirname>` - View frames with descriptions
- `GET /generate_descriptions/<dirname>` - Generate AI descriptions
- `GET /frame/<path>` - Serve frame images

## Technical Details

- **Frame Extraction**: Uses OpenCV for efficient video processing
- **AI Model**: Moondream vision model via Ollama API
- **Web Framework**: Flask with modern HTML/CSS templates
- **Image Processing**: Base64 encoding for API communication
- **Data Storage**: JSON format for descriptions

## Example Output

The system generates descriptions like:
- "iced glass windows reflect the overcast sky, while a tree and building with white doors frame the scene on a gray sidewalk."
- "iced concrete sidewalk, tree with green leaves, white building, and black car parked on street."

## Troubleshooting

1. **Ollama Connection Issues**: Ensure Ollama is running on `localhost:11434`
2. **Model Not Found**: Run `ollama pull moondream` to download the model
3. **Memory Issues**: Moondream requires ~2GB RAM for optimal performance
