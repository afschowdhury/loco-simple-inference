#!/usr/bin/env python3
"""
Frame processor that uses Moondream model via Ollama to generate descriptions for video frames.
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Optional


class FrameProcessor:
    """Handles processing of video frames with Moondream model."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "moondream"
    
    def describe_image(self, image_path: str, prompt: str = "Describe this image in a short sentence.") -> str:
        """Generate description for a single image using Moondream."""
        
        try:
            # Read and encode the image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare the request
            url = f"{self.ollama_url}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get('response', 'No description available').strip()
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"
        except FileNotFoundError:
            return "Image file not found"
        except Exception as e:
            return f"Error processing image: {e}"
    
    def process_frames_directory(self, frames_dir: str, output_file: str = None) -> List[Dict]:
        """Process all frames in a directory and generate descriptions."""
        
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            raise ValueError(f"Frames directory not found: {frames_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file_path in frames_path.iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        # Sort files by name for consistent ordering
        image_files.sort(key=lambda x: x.name)
        
        results = []
        total_files = len(image_files)
        
        print(f"Processing {total_files} frames...")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing frame {i}/{total_files}: {image_file.name}")
            
            description = self.describe_image(str(image_file))
            
            frame_data = {
                "filename": image_file.name,
                "path": str(image_file),
                "description": description,
                "frame_number": i - 1
            }
            
            results.append(frame_data)
            print(f"  Description: {description}")
        
        # Save results to JSON file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {output_file}")
        
        return results
    
    def load_processed_data(self, json_file: str) -> List[Dict]:
        """Load previously processed frame data from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            return []


def main():
    """Main function to process frames in the outputs directory."""
    
    # Default paths
    frames_dir = "/home/cmuser/ASIF/loco-simple/outputs/demo_data_at_1fps_0c379378"
    output_file = "/home/cmuser/ASIF/loco-simple/processed_frames.json"
    
    processor = FrameProcessor()
    
    try:
        results = processor.process_frames_directory(frames_dir, output_file)
        print(f"\nProcessing completed! Generated descriptions for {len(results)} frames.")
        print(f"Data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
