import os
import base64
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime
import json
from openai import OpenAI
import io
from PIL import Image

class GPTSceneDescriptor:
    """
    GPT-4o Vision-based scene descriptor with memory functionality
    Optimized for fast response times and efficient API usage
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 max_tokens: int = 100,
                 temperature: float = 0.7,
                 max_memory_size: int = 10,
                 enable_context: bool = True,
                 use_narrative_style: bool = True):
        """
        Initialize the GPT Scene Descriptor
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: GPT model to use (gpt-4o recommended for vision)
            max_tokens: Maximum tokens for each description
            temperature: Temperature for GPT responses (0-2)
            max_memory_size: Maximum number of descriptions to keep in memory
            enable_context: Whether to provide context from previous descriptions
            use_narrative_style: If True, use narrative flow for better coherence
        """
        # Set up OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_memory_size = max_memory_size
        self.enable_context = enable_context
        self.use_narrative_style = use_narrative_style
        
        # Initialize memory
        self.gpt_description_memory: List[Dict] = []
        
        # Performance tracking
        self.api_call_times = []
        self.total_descriptions = 0
        
        print(f"✅ GPT Scene Descriptor initialized")
        print(f"🤖 Model: {self.model}")
        print(f"🧠 Memory size: {self.max_memory_size}")
        print(f"🔗 Context enabled: {self.enable_context}")
    
    def _encode_image_to_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """
        Encode OpenCV frame to base64 string for GPT-4o Vision API
        
        Args:
            frame: OpenCV frame (BGR format)
            quality: JPEG quality (1-100, higher = better quality but larger size)
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Optimize image size for faster API calls
            # Resize if too large (GPT-4o can handle up to 2048x2048)
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            raise
    
    def _get_context_prompt(self, use_narrative: bool = True) -> str:
        """
        Generate context prompt from previous descriptions
        Enhanced for better coherence with narrative flow
        
        Args:
            use_narrative: If True, create narrative-style context for better coherence
        
        Returns:
            Context string optimized for GPT coherence
        """
        if not self.enable_context or not self.gpt_description_memory:
            return ""
        
        # Get last few descriptions for context
        recent_descriptions = self.gpt_description_memory[-3:]  # Last 3 for context
        
        if use_narrative:
            # Create narrative-style context for better coherence
            if len(recent_descriptions) == 1:
                desc = recent_descriptions[0]['description']
                return f"Continuing from the previous scene where {desc.lower()}, "
            else:
                narrative_parts = []
                for i, desc_data in enumerate(recent_descriptions):
                    if i == 0:
                        narrative_parts.append(f"the sequence began with {desc_data['description'].lower()}")
                    elif i == len(recent_descriptions) - 1:
                        narrative_parts.append(f"most recently showing {desc_data['description'].lower()}")
                    else:
                        narrative_parts.append(f"then transitioned to {desc_data['description'].lower()}")
                
                narrative = ", ".join(narrative_parts)
                return f"Following the visual narrative where {narrative}, "
        else:
            # Original structured context
            context_parts = []
            for i, desc_data in enumerate(recent_descriptions, 1):
                context_parts.append(f"Previous scene {i}: {desc_data['description']}")
            
            context = "\n".join(context_parts)
            return f"Context from recent scenes:\n{context}\n\n"
    
    def _describe_scene_with_gpt(self, frame: np.ndarray) -> Optional[str]:
        """
        Generate scene description using GPT-4o Vision API
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Scene description or None if failed
        """
        try:
            start_time = time.time()
            
            # Encode image
            image_base64 = self._encode_image_to_base64(frame)
            
            # Prepare context with enhanced narrative flow
            context = self._get_context_prompt(use_narrative=self.use_narrative_style)
            
            # Enhanced prompts for better coherence
            system_prompt = (
                "You are an expert to analyse the scenes from a head cam of a person walking in a construction "
                "site. Your scene description will be used to an expert with the person's voice command for locomotion mode prediction. "
                "You will be given previous scene descriptions as well. Please describe the scene that may help the locomotion mode prediction."
            )
            
            if context:
                user_prompt = (
                    f"{context}describe this new scene. Focus on how it continues or "
                    "contrasts with the visual narrative, maintaining story flow and coherence."
                )
            else:
                user_prompt = (
                    "Describe this scene."
                )
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "auto"  # auto, low, or high
                                }
                            }
                        ]
                    }
                ],
                # max_tokens=self.max_tokens,
                # temperature=self.temperature
            )
            
            # Extract description
            description = response.choices[0].message.content.strip()
            
            # Track performance
            api_time = time.time() - start_time
            self.api_call_times.append(api_time)
            
            return description
            
        except Exception as e:
            print(f"❌ Error calling GPT-4o API: {e}")
            return None
    
    def _manage_memory_size(self):
        """Manage memory size by removing oldest entries if needed"""
        if len(self.gpt_description_memory) > self.max_memory_size:
            # Remove oldest entries
            excess = len(self.gpt_description_memory) - self.max_memory_size
            self.gpt_description_memory = self.gpt_description_memory[excess:]
    
    def describe_scene(self, scene_frame: np.ndarray) -> Optional[str]:
        """
        Generate description for a single scene frame
        
        Args:
            scene_frame: OpenCV frame (BGR format) from change detector
            
        Returns:
            Scene description or None if failed
        """
        return self._describe_scene_with_gpt(scene_frame)
    
    def update_description_memory(self, gpt_description_memory: List[Dict], scene_frame: np.ndarray) -> List[Dict]:
        """
        Main function: Update GPT description memory with new scene description
        
        Args:
            gpt_description_memory: Current memory list (will be updated in-place)
            scene_frame: OpenCV frame (BGR format) from change detector
            
        Returns:
            Updated memory list with new description added
        """
        try:
            # Update internal memory reference
            self.gpt_description_memory = gpt_description_memory
            
            # Generate description
            description = self._describe_scene_with_gpt(scene_frame)
            
            if description:
                # Create memory entry
                memory_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'description': description,
                    'scene_id': len(gpt_description_memory) + 1
                }
                
                # Add to memory
                gpt_description_memory.append(memory_entry)
                self.total_descriptions += 1
                
                # Manage memory size
                if len(gpt_description_memory) > self.max_memory_size:
                    excess = len(gpt_description_memory) - self.max_memory_size
                    gpt_description_memory = gpt_description_memory[excess:]
                
                print(f"🎬 Scene {memory_entry['scene_id']} described: {description[:100]}...")
                
            else:
                print("❌ Failed to generate scene description")
            
            return gpt_description_memory
            
        except Exception as e:
            print(f"❌ Error updating description memory: {e}")
            return gpt_description_memory
    
    def get_memory_summary(self, memory: List[Dict]) -> Dict:
        """
        Get summary of the description memory
        
        Args:
            memory: Description memory list
            
        Returns:
            Summary statistics
        """
        if not memory:
            return {"message": "No descriptions in memory"}
        
        avg_api_time = np.mean(self.api_call_times[-10:]) if self.api_call_times else 0
        
        return {
            "total_descriptions": len(memory),
            "memory_size_limit": self.max_memory_size,
            "latest_description": memory[-1]['description'] if memory else None,
            "average_api_time_ms": avg_api_time * 1000,
            "total_api_calls": len(self.api_call_times),
            "context_enabled": self.enable_context
        }
    
    def save_memory_to_file(self, memory: List[Dict], filename: str = "scene_descriptions.json"):
        """
        Save memory to JSON file
        
        Args:
            memory: Description memory list
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(memory, f, indent=2)
            print(f"💾 Memory saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
    
    def load_memory_from_file(self, filename: str = "scene_descriptions.json") -> List[Dict]:
        """
        Load memory from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded memory list
        """
        try:
            with open(filename, 'r') as f:
                memory = json.load(f)
            print(f"📂 Memory loaded from {filename}")
            return memory
        except FileNotFoundError:
            print(f"📂 No existing memory file found: {filename}")
            return []
        except Exception as e:
            print(f"❌ Error loading memory: {e}")
            return []


def update_gpt_description_memory(gpt_description_memory: List[Dict], scene_frame: np.ndarray, 
                                api_key: Optional[str] = None, use_narrative_style: bool = True) -> List[Dict]:
    """
    Standalone function to update GPT description memory with new scene
    
    Args:
        gpt_description_memory: Current memory list
        scene_frame: OpenCV frame (BGR format) from change detector
        api_key: OpenAI API key (optional, uses env var if not provided)
        use_narrative_style: If True, use narrative flow for better coherence
        
    Returns:
        Updated memory list
    """
    # Create or reuse descriptor instance
    if not hasattr(update_gpt_description_memory, 'descriptor'):
        update_gpt_description_memory.descriptor = GPTSceneDescriptor(
            api_key=api_key, 
            use_narrative_style=use_narrative_style
        )
    
    return update_gpt_description_memory.descriptor.update_description_memory(
        gpt_description_memory, scene_frame
    )


# Example usage and testing
def test_gpt_descriptor():
    """Test the GPT Scene Descriptor with sample images"""
    
    # Initialize descriptor
    descriptor = GPTSceneDescriptor(
        max_memory_size=5,
        enable_context=True
    )
    
    # Initialize memory
    gpt_description_memory = []
    
    # Test with sample video
    video_path = "/home/cmuser/ASIF/loco-simple/videos/demo_data.mp4"
    
    if os.path.exists(video_path):
        print(f"🎬 Testing with video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        max_test_frames = 5  # Test with first 5 scene changes
        
        try:
            while frame_count < max_test_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to simulate scene changes
                if frame_count % 30 == 0:  # Every 30th frame
                    print(f"\n📸 Processing frame {frame_count}")
                    
                    # Update memory
                    gpt_description_memory = descriptor.update_description_memory(
                        gpt_description_memory, frame
                    )
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹️  Test stopped by user")
        
        finally:
            cap.release()
            
            # Print results
            print(f"\n📊 Test Results:")
            summary = descriptor.get_memory_summary(gpt_description_memory)
            for key, value in summary.items():
                print(f"   {key}: {value}")
            
            # Save memory
            descriptor.save_memory_to_file(gpt_description_memory, "test_descriptions.json")
    
    else:
        print(f"❌ Test video not found: {video_path}")
        print("💡 Create some test images to verify the implementation")


if __name__ == "__main__":
    # Set your OpenAI API key as environment variable:
    # export OPENAI_API_KEY="your-api-key-here"
    
    print("🚀 GPT Scene Descriptor Test")
    test_gpt_descriptor()
