# GPT Scene Descriptor

This module provides GPT-4o Vision-based scene description with memory functionality for the scene change detection system.

## Features

- **GPT-4o Vision Integration**: Uses OpenAI's GPT-4o model to analyze and describe scene changes
- **Memory System**: Maintains a rolling memory of scene descriptions with configurable size
- **Context Awareness**: Provides context from previous scenes to generate more coherent descriptions
- **Performance Optimized**: Image compression and optimization for fast API responses
- **Error Handling**: Robust error handling and fallback mechanisms

## Installation

```bash
pip install openai pillow numpy opencv-python
```

## Usage

### Basic Usage

```python
from gpt_scene_description.gpt_descriptor import update_gpt_description_memory
import cv2

# Initialize memory
gpt_description_memory = []

# Get scene frame from change detector
scene_frame = cv2.imread("scene.jpg")  # This would come from change_detector.py

# Update memory with new scene description
gpt_description_memory = update_gpt_description_memory(gpt_description_memory, scene_frame)

# Print descriptions
for desc in gpt_description_memory:
    print(f"Scene {desc['scene_id']}: {desc['description']}")
```

### Advanced Usage with Custom Configuration

```python
from gpt_scene_description.gpt_descriptor import GPTSceneDescriptor

# Initialize with custom settings
descriptor = GPTSceneDescriptor(
    api_key="your-openai-api-key",  # or set OPENAI_API_KEY env var
    model="gpt-4o",
    max_tokens=200,
    temperature=0.7,
    max_memory_size=15,
    enable_context=True
)

# Initialize memory
gpt_description_memory = []

# Process scene changes
gpt_description_memory = descriptor.update_description_memory(gpt_description_memory, scene_frame)

# Get performance stats
stats = descriptor.get_memory_summary(gpt_description_memory)
print(stats)

# Save memory to file
descriptor.save_memory_to_file(gpt_description_memory, "scene_descriptions.json")
```

### Integration with Scene Change Detector

```python
from scene_change_detection.change_detector import SceneChangeDetector
from gpt_scene_description.gpt_descriptor import update_gpt_description_memory
import cv2

# Initialize systems
detector = SceneChangeDetector()
gpt_description_memory = []

# Process video
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Check for scene change
    is_changed, changed_frame = detector.detect_scene_change(frame)
    
    if is_changed and changed_frame is not None:
        # Generate description for the new scene
        gpt_description_memory = update_gpt_description_memory(
            gpt_description_memory, changed_frame
        )
        
        print(f"New scene detected: {gpt_description_memory[-1]['description']}")

cap.release()
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Parameters

- `model`: GPT model to use (default: "gpt-4o")
- `max_tokens`: Maximum tokens per description (default: 150)
- `temperature`: Response creativity (0-2, default: 0.7)
- `max_memory_size`: Maximum descriptions to keep in memory (default: 10)
- `enable_context`: Whether to provide context from previous scenes (default: True)

## Performance Optimization

The module includes several optimizations for fast response:

1. **Image Compression**: Automatically resizes and compresses images before sending to API
2. **Memory Management**: Limits memory size to prevent excessive context
3. **Async Ready**: Can be easily adapted for async operations
4. **Caching**: Previous embeddings and descriptions are cached

## Memory Format

Each memory entry contains:

```json
{
    "timestamp": "2024-01-15T10:30:45.123456",
    "description": "A bustling city street with pedestrians crossing...",
    "scene_id": 1
}
```

## Error Handling

The module handles various error scenarios:

- API key missing or invalid
- Network connectivity issues
- Image encoding failures
- API rate limiting
- Invalid input formats

## Testing

Run the built-in test:

```python
python gpt_descriptor.py
```

This will test the system with sample frames if a test video is available.
