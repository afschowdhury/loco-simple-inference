#!/usr/bin/env python3
"""
Demo comparing List vs String-based memory styles for GPT scene descriptions
Shows the difference in coherence and output quality
"""

import cv2
import numpy as np
from gpt_scene_description.gpt_descriptor import GPTSceneDescriptor

def create_demo_scenes():
    """Create demo scenes for testing"""
    scenes = []
    
    # Scene 1: City street
    scene1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(scene1, (0, 300), (640, 480), (100, 100, 100), -1)  # Road
    cv2.rectangle(scene1, (200, 200), (300, 300), (150, 150, 150), -1)  # Building
    cv2.putText(scene1, "CITY STREET", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    scenes.append(("City Street Scene", scene1))
    
    # Scene 2: Park
    scene2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(scene2, (0, 0), (640, 480), (34, 139, 34), -1)  # Green background
    cv2.circle(scene2, (320, 240), 80, (139, 69, 19), -1)  # Tree trunk
    cv2.circle(scene2, (320, 160), 60, (0, 128, 0), -1)  # Tree top
    cv2.putText(scene2, "PARK", (280, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    scenes.append(("Park Scene", scene2))
    
    # Scene 3: Office
    scene3 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(scene3, (0, 0), (640, 480), (240, 240, 240), -1)  # Office background
    cv2.rectangle(scene3, (100, 200), (540, 350), (200, 200, 200), -1)  # Desk
    cv2.rectangle(scene3, (250, 150), (390, 200), (0, 0, 0), -1)  # Monitor
    cv2.putText(scene3, "OFFICE", (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    scenes.append(("Office Scene", scene3))
    
    return scenes

def demo_memory_styles():
    """Demonstrate different memory styles"""
    
    print("🎬 GPT Scene Description Memory Style Comparison")
    print("=" * 70)
    
    # Check for API key
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set!")
        print("💡 This demo requires an OpenAI API key")
        print("📝 Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\n🔄 Running with mock descriptions instead...")
        run_mock_demo()
        return
    
    # Create demo scenes
    scenes = create_demo_scenes()
    
    # Test both memory styles
    for style_name, use_narrative in [("STRUCTURED LIST", False), ("NARRATIVE FLOW", True)]:
        print(f"\n🧠 Testing {style_name} Memory Style")
        print("-" * 50)
        
        # Initialize descriptor
        descriptor = GPTSceneDescriptor(
            max_memory_size=5,
            enable_context=True,
            use_narrative_style=use_narrative,
            temperature=0.8  # More creative for better narrative
        )
        
        # Initialize memory
        gpt_description_memory = []
        
        # Process each scene
        for i, (scene_name, scene_frame) in enumerate(scenes, 1):
            print(f"\n📸 Processing {scene_name}...")
            
            try:
                # Generate description
                gpt_description_memory = descriptor.update_description_memory(
                    gpt_description_memory, scene_frame
                )
                
                if gpt_description_memory:
                    latest_desc = gpt_description_memory[-1]
                    print(f"✅ Scene {i}: {latest_desc['description']}")
                else:
                    print(f"❌ Failed to generate description for {scene_name}")
                    
            except Exception as e:
                print(f"❌ Error processing {scene_name}: {e}")
        
        # Show final memory state
        print(f"\n📋 Final {style_name} Memory:")
        if gpt_description_memory:
            for j, desc in enumerate(gpt_description_memory, 1):
                print(f"   {j}. {desc['description']}")
        
        # Show context that would be sent to next GPT call
        if descriptor.enable_context and gpt_description_memory:
            context = descriptor._get_context_prompt(use_narrative=use_narrative)
            print(f"\n🔗 Context for next scene:")
            print(f"   {context}")
    
    print(f"\n💡 ANALYSIS:")
    print(f"📋 STRUCTURED LIST: Better for data management, precise tracking")
    print(f"📖 NARRATIVE FLOW: Better for coherent storytelling, natural progression")

def run_mock_demo():
    """Run demo with mock descriptions when API key is not available"""
    
    mock_descriptions = {
        "structured": [
            "A busy urban street scene with tall buildings, moving cars, and pedestrians walking on sidewalks during daytime.",
            "A peaceful park setting with green grass, mature trees, and open spaces for recreation under clear skies.",
            "A modern office environment with desks, computer equipment, and professional workspace lighting."
        ],
        "narrative": [
            "The scene opens on a bustling urban thoroughfare where life moves at a rapid pace, with towering structures framing the street as people navigate the concrete pathways.",
            "Transitioning from the urban energy, we now find ourselves in a tranquil park sanctuary where nature provides respite, with verdant landscapes and towering trees creating a peaceful contrast to the previous metropolitan setting.",
            "The narrative then shifts to the structured world of professional life, revealing a contemporary office space where technology and human productivity converge in a clean, organized environment that speaks to modern work culture."
        ]
    }
    
    print("\n📝 MOCK DEMONSTRATION (without API calls)")
    print("=" * 60)
    
    for style_name, descriptions in [("STRUCTURED LIST", "structured"), ("NARRATIVE FLOW", "narrative")]:
        print(f"\n🧠 {style_name} Style:")
        print("-" * 40)
        
        for i, desc in enumerate(mock_descriptions[descriptions], 1):
            print(f"Scene {i}: {desc}")
        
        # Show how context would be built
        print(f"\n🔗 Context for next scene:")
        if descriptions == "structured":
            context = "Previous scenes:\n"
            for i, desc in enumerate(mock_descriptions[descriptions], 1):
                context += f"Scene {i}: {desc}\n"
            context += "Continuing this sequence, "
        else:
            context = "Following the visual narrative where "
            narrative_parts = []
            for i, desc in enumerate(mock_descriptions[descriptions]):
                if i == 0:
                    narrative_parts.append(f"the sequence began with {desc.lower()}")
                elif i == len(mock_descriptions[descriptions]) - 1:
                    narrative_parts.append(f"most recently showing {desc.lower()}")
                else:
                    narrative_parts.append(f"then transitioned to {desc.lower()}")
            context += ", ".join(narrative_parts) + ", "
        
        print(f"   {context}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"🎯 For better coherence and storytelling: Use NARRATIVE FLOW")
    print(f"🔧 For structured data and precise control: Use STRUCTURED LIST")
    print(f"⚖️  For best of both: Use the hybrid approach in the main descriptor")

def main():
    """Run the memory style demonstration"""
    
    print("🚀 Starting Memory Style Demonstration...")
    demo_memory_styles()
    
    print(f"\n✅ Demonstration completed!")
    print(f"💡 To use narrative style in your code:")
    print(f"   update_gpt_description_memory(memory, frame, use_narrative_style=True)")

if __name__ == "__main__":
    main()
