#!/usr/bin/env python3
"""
Integration test script for Scene Change Detection + GPT Description
This script demonstrates how to use both systems together
"""

import os
import cv2
import numpy as np
from scene_change_detection.change_detector import SceneChangeDetector
from gpt_scene_description.gpt_descriptor import update_gpt_description_memory

def test_integration_without_api():
    """Test the integration without making actual API calls"""
    
    print("🚀 Testing Scene Change Detection + GPT Description Integration")
    print("📝 Note: This test runs without API key to demonstrate the integration")
    
    # Initialize scene change detector
    print("\n1️⃣ Initializing Scene Change Detector...")
    detector = SceneChangeDetector(
        model_name="mobileclip",
        similarity_threshold=0.85,
        use_fp16=True
    )
    
    # Initialize GPT description memory
    print("2️⃣ Initializing GPT Description Memory...")
    gpt_description_memory = []
    
    # Test with sample video if available
    video_path = "/home/cmuser/ASIF/loco-simple/videos/demo_data.mp4"
    
    if os.path.exists(video_path):
        print(f"3️⃣ Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        scene_changes = 0
        max_frames = 100  # Process first 100 frames
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                # Detect scene change
                is_changed, changed_frame = detector.detect_scene_change(frame)
                
                if is_changed and changed_frame is not None:
                    scene_changes += 1
                    print(f"🔄 Scene change detected at frame {frame_count}")
                    
                    # Here's where you would call the GPT descriptor
                    # (commented out to avoid API key requirement)
                    """
                    # With API key set, you would call:
                    gpt_description_memory = update_gpt_description_memory(
                        gpt_description_memory, changed_frame
                    )
                    print(f"📝 Scene described: {gpt_description_memory[-1]['description']}")
                    """
                    
                    # Simulate adding a description
                    mock_description = {
                        'timestamp': f"2024-01-15T10:{30+scene_changes}:45.123456",
                        'description': f"Mock description for scene change {scene_changes}",
                        'scene_id': scene_changes
                    }
                    gpt_description_memory.append(mock_description)
                    print(f"📝 Mock description added: Scene {scene_changes}")
                
                frame_count += 1
                
                # Progress report
                if frame_count % 25 == 0:
                    print(f"🎬 Processed {frame_count} frames, detected {scene_changes} scene changes")
            
        except KeyboardInterrupt:
            print("\n⏹️  Test stopped by user")
        
        finally:
            cap.release()
            
            # Print results
            print(f"\n📊 Integration Test Results:")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Scene changes detected: {scene_changes}")
            print(f"   Descriptions in memory: {len(gpt_description_memory)}")
            
            # Print memory contents
            if gpt_description_memory:
                print(f"\n📝 Description Memory Contents:")
                for desc in gpt_description_memory:
                    print(f"   Scene {desc['scene_id']}: {desc['description']}")
    
    else:
        print(f"❌ Test video not found: {video_path}")
        print("💡 To test with real video, place a video file at the specified path")
    
    print(f"\n✅ Integration test completed!")
    print(f"🔑 To use with GPT-4o, set your OpenAI API key:")
    print(f"   export OPENAI_API_KEY='your-api-key-here'")

def test_with_api_key():
    """Test with actual API calls (requires OPENAI_API_KEY)"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set. Please set it to test with actual API calls.")
        return
    
    print("🚀 Testing with actual OpenAI API...")
    
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test Scene", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Initialize memory
    gpt_description_memory = []
    
    try:
        # Test the actual GPT descriptor
        gpt_description_memory = update_gpt_description_memory(gpt_description_memory, test_image)
        
        if gpt_description_memory:
            print(f"✅ API test successful!")
            print(f"📝 Description: {gpt_description_memory[0]['description']}")
        else:
            print("❌ API test failed - no description generated")
    
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    # Test integration without API
    test_integration_without_api()
    
    print("\n" + "="*60)
    
    # Test with API if key is available
    test_with_api_key()
