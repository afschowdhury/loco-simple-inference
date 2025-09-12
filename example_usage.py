#!/usr/bin/env python3
"""
Example usage of the Locomotion Inference System
Demonstrates how to integrate with live descriptor and GPT descriptor
"""

import os
import cv2
import numpy as np
from locomotion_infer import LocomotionInferenceEngine, LocomotionState, update_locomotion_stm
from live_descriptor.live_descriptor import process_live_frame
from gpt_scene_description.gpt_descriptor import update_gpt_description_memory

def example_real_time_integration():
    """
    Example of how to integrate locomotion inference in a real-time system
    """
    print("🚀 Real-time Locomotion Detection Example")
    
    # Set your OpenAI API key
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    # Initialize the locomotion inference engine
    engine = LocomotionInferenceEngine(
        voice_data_path="data_json/loco_data_4.json",
        model="gpt-4o-mini",  # Fast model for real-time
        max_memory_size=10,
        confidence_threshold=0.7
    )
    
    # Initialize memory stores
    gpt_description_memory = []
    live_description_memory = []
    locomotion_stm = []
    
    # Example: Process a single frame
    print("\n📸 Processing example frame...")
    
    # Create a dummy frame (in real usage, this would come from your camera/video)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        # 1. Process frame with live descriptor
        live_description_memory = process_live_frame(
            live_description_memory,
            dummy_frame,
            prompt="Describe the person's posture and movement in this image."
        )
        
        # 2. Process frame with GPT descriptor (if needed for scene context)
        gpt_description_memory = update_gpt_description_memory(
            gpt_description_memory,
            dummy_frame
        )
        
        # 3. Update locomotion inference
        locomotion_stm = engine.update_locomotion_stm(
            gpt_description_memory,
            live_description_memory,
            locomotion_stm
        )
        
        # 4. Get current locomotion mode
        current_mode = engine.get_current_mode(locomotion_stm)
        print(f"🚶 Current locomotion mode: {current_mode}")
        
        # 5. Get performance stats
        stats = engine.get_performance_stats()
        print(f"⚡ Inference time: {stats.get('last_inference_time_ms', 0):.1f}ms")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        print("💡 Make sure OPENAI_API_KEY is set and Ollama is running for live descriptor")

def example_standalone_function():
    """
    Example using the standalone update_locomotion_stm function
    """
    print("\n🔧 Standalone Function Example")
    
    # Initialize memory with some dummy data
    gpt_description_memory = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "description": "Person standing upright in a room",
            "scene_id": 1
        }
    ]
    
    live_description_memory = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "description": "Person in normal standing position",
            "frame_index": 0
        }
    ]
    
    locomotion_stm = []
    
    # Update locomotion using standalone function
    updated_stm = update_locomotion_stm(
        gpt_description_memory=gpt_description_memory,
        live_description_memory=live_description_memory,
        locomotion_stm=locomotion_stm,
        voice_data_path="data_json/loco_data_4.json"
    )
    
    print(f"📊 Updated locomotion STM: {len(updated_stm)} states")
    if updated_stm:
        latest = updated_stm[-1]
        print(f"Latest state: {latest.mode} (confidence: {latest.confidence:.2f})")

def example_with_video_simulation():
    """
    Example simulating video processing with locomotion detection
    """
    print("\n🎬 Video Simulation Example")
    
    # Initialize engine
    engine = LocomotionInferenceEngine(
        voice_data_path="data_json/loco_data_4.json"
    )
    
    # Initialize memories
    gpt_description_memory = []
    live_description_memory = []
    locomotion_stm = []
    
    # Simulate video frames with different scenarios
    scenarios = [
        ("00:00:00", "Person starts walking normally", "ground walk"),
        ("00:05:00", "Person begins to crouch down", "crouch walk"),
        ("00:13:00", "Person stands up from crouch", "end crouch walk"),
        ("00:16:00", "Person walking forward normally", "ground walk"),
        ("00:19:00", "Person crouching and moving forward", "crouch walk"),
    ]
    
    for timestamp, description, expected_mode in scenarios:
        print(f"\n⏰ Processing timestamp: {timestamp}")
        print(f"📝 Scene: {description}")
        
        # Add descriptions to memory
        live_description_memory.append({
            "timestamp": timestamp,
            "description": description,
            "frame_index": len(live_description_memory)
        })
        
        gpt_description_memory.append({
            "timestamp": timestamp,
            "description": description,
            "scene_id": len(gpt_description_memory) + 1
        })
        
        # Update locomotion
        locomotion_stm = engine.update_locomotion_stm(
            gpt_description_memory,
            live_description_memory,
            locomotion_stm,
            timestamp
        )
        
        # Check result
        if locomotion_stm:
            predicted = locomotion_stm[-1]
            print(f"🎯 Predicted: {predicted.mode} (expected: {expected_mode})")
            print(f"📊 Confidence: {predicted.confidence:.2f}")
            print(f"🔍 Source: {predicted.source}")
        
        print("-" * 50)
    
    # Final summary
    print(f"\n📈 Final Summary:")
    print(f"Total predictions: {len(locomotion_stm)}")
    mode_sequence = [state.mode for state in locomotion_stm]
    print(f"Mode sequence: {' → '.join(mode_sequence)}")
    
    # Save results
    engine.save_locomotion_memory(locomotion_stm, "example_results.json")
    print("💾 Results saved to example_results.json")

if __name__ == "__main__":
    print("🤖 Locomotion Inference System - Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_standalone_function()
        example_with_video_simulation()
        
        # Uncomment to test with real API calls
        # example_real_time_integration()
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("💡 Make sure to set OPENAI_API_KEY environment variable")
    
    print("\n✅ Examples completed!")