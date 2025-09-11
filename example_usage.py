#!/usr/bin/env python3
"""
Example usage of Scene Change Detection + GPT Description system
This demonstrates the complete workflow from scene detection to description
"""

import os
import cv2
from scene_change_detection.change_detector import SceneChangeDetector
from gpt_scene_description.gpt_descriptor import update_gpt_description_memory

def main():
    """Main example demonstrating the complete workflow"""
    
    print("🎬 Scene Change Detection + GPT Description Example")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not found!")
        print("💡 Set your API key: export OPENAI_API_KEY='your-key-here'")
        print("📝 This example will run but skip GPT descriptions")
        use_gpt = False
    else:
        print("✅ OpenAI API key found")
        use_gpt = True
    
    # Initialize systems
    print("\n🔧 Initializing systems...")
    
    # 1. Initialize Scene Change Detector
    detector = SceneChangeDetector(
        model_name="mobileclip",  # or "fastervit"
        similarity_threshold=0.85,  # Lower = more sensitive
        use_fp16=True,
        enable_tensorrt=True
    )
    
    # 2. Initialize GPT description memory
    gpt_description_memory = []
    
    # 3. Process video or webcam
    video_source = "/home/cmuser/ASIF/loco-simple/videos/demo_data.mp4"  # or 0 for webcam
    
    if os.path.exists(video_source):
        print(f"📹 Processing video: {video_source}")
    else:
        print("📹 Video file not found, using webcam (press 'q' to quit)")
        video_source = 0
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("❌ Could not open video source")
        return
    
    frame_count = 0
    scene_count = 0
    
    print("\n🎬 Starting processing...")
    print("Press 'q' to quit, 's' to save descriptions to file")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video or failed to read frame")
                break
            
            # Detect scene change
            is_changed, changed_frame = detector.detect_scene_change(frame)
            
            if is_changed and changed_frame is not None:
                scene_count += 1
                print(f"\n🔄 Scene change #{scene_count} detected at frame {frame_count}")
                
                if use_gpt:
                    # Generate GPT description
                    print("🤖 Generating description with GPT-4o...")
                    try:
                        gpt_description_memory = update_gpt_description_memory(
                            gpt_description_memory, changed_frame
                        )
                        
                        if gpt_description_memory:
                            latest_desc = gpt_description_memory[-1]
                            print(f"📝 Description: {latest_desc['description']}")
                        
                    except Exception as e:
                        print(f"❌ GPT description failed: {e}")
                else:
                    # Add mock description
                    mock_desc = {
                        'timestamp': '2024-01-15T10:30:45',
                        'description': f'Scene change {scene_count} detected (GPT disabled)',
                        'scene_id': scene_count
                    }
                    gpt_description_memory.append(mock_desc)
                    print(f"📝 Mock description added (set OPENAI_API_KEY for real descriptions)")
                
                # Optionally save the frame
                frame_filename = f"scene_change_{scene_count}.jpg"
                cv2.imwrite(frame_filename, changed_frame)
                print(f"💾 Saved frame: {frame_filename}")
            
            frame_count += 1
            
            # Progress indicator (every 50 frames)
            if frame_count % 50 == 0:
                print(f"📊 Progress: {frame_count} frames, {scene_count} scene changes")
            
            # For webcam, show the frame
            if video_source == 0:
                cv2.imshow('Scene Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_descriptions(gpt_description_memory)
            
            # Limit processing for demo video
            if video_source != 0 and frame_count >= 300:  # Process first 300 frames
                print("📊 Demo limit reached (300 frames)")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final results
        print_results(frame_count, scene_count, gpt_description_memory, detector)
        
        # Save descriptions
        if gpt_description_memory:
            save_descriptions(gpt_description_memory)

def print_results(frame_count, scene_count, memory, detector):
    """Print processing results"""
    print("\n" + "=" * 60)
    print("📊 PROCESSING RESULTS")
    print("=" * 60)
    print(f"📹 Total frames processed: {frame_count}")
    print(f"🔄 Scene changes detected: {scene_count}")
    print(f"📝 Descriptions generated: {len(memory)}")
    
    # Performance stats
    stats = detector.get_performance_stats()
    print(f"\n⚡ Performance:")
    print(f"   Average inference time: {stats.get('average_inference_time_ms', 0):.1f}ms")
    print(f"   FPS: {stats.get('fps', 0):.1f}")
    print(f"   Device: {stats.get('device', 'unknown')}")
    
    # Show descriptions
    if memory:
        print(f"\n📝 Generated Descriptions:")
        for desc in memory[-5:]:  # Show last 5
            print(f"   Scene {desc['scene_id']}: {desc['description'][:100]}...")

def save_descriptions(memory):
    """Save descriptions to file"""
    if not memory:
        print("📝 No descriptions to save")
        return
    
    import json
    from datetime import datetime
    
    filename = f"scene_descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(memory, f, indent=2)
        print(f"💾 Descriptions saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save descriptions: {e}")

if __name__ == "__main__":
    main()
