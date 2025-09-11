#!/usr/bin/env python3
"""
Live descriptor module for real-time frame description using Moondream via Ollama.
"""

import base64
import requests
import cv2
import numpy as np
from typing import List, Dict, Union
from datetime import datetime
from pathlib import Path


def process_live_frame(live_desc_memory: List[Dict], frame: Union[np.ndarray, str], 
                      ollama_url: str = "http://localhost:11434",
                      prompt: str = "Describe this image in a short sentence.") -> List[Dict]:
    """
    Process a live frame with Moondream via Ollama and add description to memory.
    
    Args:
        live_desc_memory: List of dictionaries containing previous frame descriptions
        frame: Either a numpy array (cv2 frame) or path to image file
        ollama_url: URL for Ollama API endpoint
        prompt: Prompt to use for image description
        
    Returns:
        Updated live_desc_memory list with new frame description added
    """
    
    try:
        # Handle different frame input types
        if isinstance(frame, str):
            # Frame is a file path
            with open(frame, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            frame_identifier = Path(frame).name
        elif isinstance(frame, np.ndarray):
            # Frame is a numpy array (cv2 frame)
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                raise ValueError("Failed to encode frame as JPEG")
            image_data = base64.b64encode(buffer).decode('utf-8')
            frame_identifier = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        else:
            raise ValueError("Frame must be either a file path (str) or numpy array")
        
        # Prepare the request to Ollama
        url = f"{ollama_url}/api/generate"
        data = {
            "model": "moondream",
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }
        
        # Make the request
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        description = result.get('response', 'No description available').strip()
        
        # Create frame description entry
        frame_description = {
            "timestamp": datetime.now().isoformat(),
            "frame_identifier": frame_identifier,
            "description": description,
            "frame_index": len(live_desc_memory),
            "prompt_used": prompt
        }
        
        # Add to memory and return updated memory
        updated_memory = live_desc_memory.copy()
        updated_memory.append(frame_description)
        
        return updated_memory
        
    except requests.exceptions.RequestException as e:
        # Add error entry to memory
        error_description = {
            "timestamp": datetime.now().isoformat(),
            "frame_identifier": frame_identifier if 'frame_identifier' in locals() else "unknown",
            "description": f"Error connecting to Ollama: {e}",
            "frame_index": len(live_desc_memory),
            "prompt_used": prompt,
            "error": True
        }
        updated_memory = live_desc_memory.copy()
        updated_memory.append(error_description)
        return updated_memory
        
    except FileNotFoundError:
        # Add error entry to memory
        error_description = {
            "timestamp": datetime.now().isoformat(),
            "frame_identifier": frame if isinstance(frame, str) else "unknown",
            "description": "Image file not found",
            "frame_index": len(live_desc_memory),
            "prompt_used": prompt,
            "error": True
        }
        updated_memory = live_desc_memory.copy()
        updated_memory.append(error_description)
        return updated_memory
        
    except Exception as e:
        # Add error entry to memory
        error_description = {
            "timestamp": datetime.now().isoformat(),
            "frame_identifier": frame_identifier if 'frame_identifier' in locals() else "unknown",
            "description": f"Error processing frame: {e}",
            "frame_index": len(live_desc_memory),
            "prompt_used": prompt,
            "error": True
        }
        updated_memory = live_desc_memory.copy()
        updated_memory.append(error_description)
        return updated_memory


def get_recent_descriptions(live_desc_memory: List[Dict], count: int = 5) -> List[Dict]:
    """
    Get the most recent descriptions from memory.
    
    Args:
        live_desc_memory: List of frame descriptions
        count: Number of recent descriptions to return
        
    Returns:
        List of most recent descriptions
    """
    return live_desc_memory[-count:] if len(live_desc_memory) >= count else live_desc_memory


def clear_memory(live_desc_memory: List[Dict], keep_recent: int = 0) -> List[Dict]:
    """
    Clear memory, optionally keeping recent entries.
    
    Args:
        live_desc_memory: List of frame descriptions
        keep_recent: Number of recent entries to keep (0 to clear all)
        
    Returns:
        Cleared memory list
    """
    if keep_recent <= 0:
        return []
    return live_desc_memory[-keep_recent:] if len(live_desc_memory) > keep_recent else live_desc_memory


def get_memory_summary(live_desc_memory: List[Dict]) -> Dict:
    """
    Get summary statistics about the memory.
    
    Args:
        live_desc_memory: List of frame descriptions
        
    Returns:
        Dictionary with memory statistics
    """
    if not live_desc_memory:
        return {
            "total_frames": 0,
            "error_count": 0,
            "success_count": 0,
            "first_timestamp": None,
            "last_timestamp": None
        }
    
    error_count = sum(entry.get('error', False) for entry in live_desc_memory)
    success_count = len(live_desc_memory) - error_count
    
    return {
        "total_frames": len(live_desc_memory),
        "error_count": error_count,
        "success_count": success_count,
        "first_timestamp": live_desc_memory[0].get('timestamp'),
        "last_timestamp": live_desc_memory[-1].get('timestamp')
    }
