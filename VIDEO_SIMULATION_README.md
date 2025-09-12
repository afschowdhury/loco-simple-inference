# Video Simulation App - Locomotion Mode Prediction

A comprehensive web application for simulating real-time locomotion mode prediction using computer vision, GPT-4o vision descriptions, live scene descriptions, and machine learning inference.

## 🚀 Features

- **Video Processing**: Extract frames at 1FPS from MP4 videos
- **Scene Change Detection**: Real-time detection using MobileCLIP or FasterViT models
- **GPT-4o Vision**: Advanced scene description using OpenAI's GPT-4o model
- **Live Description**: Scene analysis using Moondream via Ollama
- **Locomotion Inference**: ML-powered prediction of locomotion modes
- **Real-time Simulation**: Synchronized timing based on timestamp data
- **Web Interface**: Modern, responsive web UI with real-time updates
- **Background Processing**: Multi-threaded processing for optimal performance

## 📋 Supported Locomotion Modes

- `ground walk`: Normal walking locomotion
- `crouch walk`: Crouched walking movement
- `end crouch walk`: Transition from crouched to upright
- `stationary`: No movement/standing still
- `unknown`: Uncertain or unrecognized mode

## 🏗️ Architecture

### Core Components

1. **Scene Change Detector** (`scene_change_detection/change_detector.py`)
   - Uses MobileCLIP-S1 or FasterViT-0 for feature extraction
   - Cosine similarity for change detection
   - Optimized for RTX 4070 SUPER with TensorRT support

2. **GPT Scene Descriptor** (`gpt_scene_description/gpt_descriptor.py`)
   - GPT-4o Vision API integration
   - Context-aware descriptions with narrative flow
   - Memory management for scene continuity

3. **Live Descriptor** (`live_descriptor/live_descriptor.py`)
   - Moondream model via Ollama
   - Real-time frame description
   - Construction site specific prompts

4. **Locomotion Inference** (`locomotion_infer.py`)
   - GPT-4o-mini powered classification
   - Multi-modal input processing (voice, scene, context)
   - Confidence scoring

### Web Application

- **Flask Backend**: RESTful API with real-time status updates
- **Bootstrap Frontend**: Modern, responsive UI design
- **Background Workers**: Multi-process architecture for performance
- **Real-time Updates**: WebSocket-like polling for live data

## 📁 File Structure

```
loco-simple/
├── video_simulation_app.py          # Main Flask application
├── run_video_simulation.py          # Application launcher
├── templates/
│   └── video_simulation.html        # Web interface template
├── videos/                          # Video files (.mp4)
│   ├── demo_data.mp4
│   ├── loco_data_2.mp4
│   ├── loco_data_3.mp4
│   └── loco_data_4.mp4
├── data_json/                       # Timestamp and command data
│   ├── demo_data.json
│   ├── loco_data_2.json
│   ├── loco_data_3.json
│   └── loco_data_4.json
├── scene_change_detection/
│   └── change_detector.py           # Scene change detection
├── gpt_scene_description/
│   └── gpt_descriptor.py           # GPT-4o vision descriptions
├── live_descriptor/
│   └── live_descriptor.py          # Live scene descriptions
├── locomotion_infer.py              # Locomotion mode inference
└── requirements.txt                 # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- OpenAI API key
- Ollama (for live descriptions)

### Setup Steps

1. **Clone and Navigate**
   ```bash
   cd /home/cmuser/ASIF/loco-simple
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

4. **Install Ollama (for live descriptions)**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Moondream model
   ollama pull moondream
   
   # Start Ollama service
   ollama serve
   ```

## 🚀 Usage

### Quick Start

1. **Launch the Application**
   ```bash
   python run_video_simulation.py
   ```

2. **Open Web Interface**
   - Navigate to `http://localhost:5000` in your browser

3. **Load Data**
   - Select a video file from the dropdown (e.g., `loco_data_4.mp4`)
   - Select corresponding data file (e.g., `loco_data_4.json`)
   - Click "Load Data"

4. **Initialize Models**
   - Click "Initialize AI Models" (may take a few minutes)

5. **Start Simulation**
   - Click "Start Simulation"
   - Watch real-time processing and predictions

### Data Format

JSON data files should follow this structure:

```json
[
  {
    "timestamp": "00:00:00",          // HH:MM:SS format
    "demo_voice_command": "start walking",
    "level": "ground walk"            // Expected locomotion mode
  },
  {
    "timestamp": "00:05:00",
    "demo_voice_command": "I am crouch walking",
    "level": "crouch walk"
  }
]
```

## 🎯 How It Works

### Processing Pipeline

1. **Frame Extraction**: Video is processed at 1FPS to extract individual frames
2. **Scene Change Detection**: Every 2nd frame is analyzed for scene changes using computer vision
3. **Description Generation**: 
   - When scene changes are detected, GPT-4o generates detailed descriptions
   - Live descriptions are generated for every 2nd frame using Moondream
4. **Locomotion Prediction**: Combined voice commands, scene descriptions, and context are used to predict locomotion mode
5. **Timing Simulation**: Processing is synchronized with timestamp data to simulate real-time operation

### Background Processing

- **Change Detection Worker**: Dedicated process for scene change detection
- **Description Worker**: Separate process for live descriptions
- **Main Thread**: Coordinates processing and handles web interface
- **Real-time Updates**: Status polling provides live feedback to the web interface

## 🔧 Configuration

### Scene Change Detection

Modify `change_detector.py` parameters:
- `similarity_threshold`: Lower values = more sensitive (default: 0.85)
- `model_name`: "mobileclip" or "fastervit"
- `use_fp16`: Enable FP16 for performance (default: True)

### GPT Descriptions

Configure `gpt_descriptor.py`:
- `max_tokens`: Response length limit (default: 100)
- `temperature`: Creativity level (default: 0.7)
- `max_memory_size`: Context window size (default: 10)

### Locomotion Inference

Adjust `locomotion_infer.py`:
- `model`: GPT model selection (default: "gpt-4o-mini")
- `temperature`: Prediction consistency (default: 0.1)

## 📊 Performance Monitoring

The web interface provides real-time statistics:

- **Frames Processed**: Total frames analyzed
- **Scene Changes**: Number of detected scene changes
- **GPT Descriptions**: Count of GPT-generated descriptions
- **Live Descriptions**: Count of live descriptions
- **Processing Log**: Detailed activity log
- **Confidence Scores**: Prediction confidence visualization

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Check GPU memory: Use smaller models if needed

2. **API Errors**
   - Verify OpenAI API key is set correctly
   - Check API quota and rate limits

3. **Ollama Connection Issues**
   - Ensure Ollama service is running: `ollama serve`
   - Verify Moondream model is installed: `ollama list`

4. **Video Loading Problems**
   - Ensure videos are in MP4 format
   - Check file paths and permissions

### Debug Mode

Run with debug logging:
```bash
FLASK_DEBUG=1 python video_simulation_app.py
```

## 🎮 Example Session

1. Load `loco_data_4.mp4` with `loco_data_4.json`
2. Initialize models (30-60 seconds)
3. Start simulation
4. Observe real-time processing:
   - Frame extraction at 1FPS
   - Scene change detection every 2nd frame
   - GPT descriptions for scene changes
   - Live descriptions for all processed frames
   - Locomotion predictions with confidence scores
   - Synchronized timing based on voice command timestamps

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is part of the ASIF locomotion analysis system.

## 🔗 Related Components

- Scene Change Detection: MobileCLIP, FasterViT models
- GPT Integration: OpenAI GPT-4o Vision API
- Live Descriptions: Ollama + Moondream
- Web Framework: Flask + Bootstrap
- Computer Vision: OpenCV, PyTorch

---

**Note**: This application requires significant computational resources for optimal performance. GPU acceleration is highly recommended for real-time processing.
