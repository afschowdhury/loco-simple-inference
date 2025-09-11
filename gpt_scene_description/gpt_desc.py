import os
import base64
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import openai

@dataclass
class SceneDescription:
    """Individual scene description with metadata"""
    timestamp: str
    frame_number: int
    description: str
    key_objects: List[str]
    scene_type: str
    confidence_score: float = 0.0
    importance_score: float = 0.5  # For smart memory management

@dataclass
class NarrativeMemory:
    """Coherent narrative memory for GPT context"""
    current_narrative: str = ""
    scene_count: int = 0
    max_narrative_length: int = 1000  # characters
    
    def add_scene_to_narrative(self, scene: SceneDescription, transition_context: str = ""):
        """Add scene to narrative in a coherent way"""
        if self.scene_count == 0:
            # First scene
            self.current_narrative = f"Scene 1: {scene.description}"
        else:
            # Add with smooth transition
            if transition_context:
                addition = f" {transition_context} Scene {self.scene_count + 1}: {scene.description}"
            else:
                addition = f" Then, {scene.description.lower()}"
            
            self.current_narrative += addition
        
        self.scene_count += 1
        
        # Trim if too long
        if len(self.current_narrative) > self.max_narrative_length:
            self._trim_narrative()
    
    def _trim_narrative(self):
        """Intelligently trim narrative while maintaining coherence"""
        # Find natural break points (scene boundaries)
        sentences = self.current_narrative.split('. ')
        
        # Keep last portion that fits within limit
        trimmed = ""
        for sentence in reversed(sentences):
            test_length = len(sentence + ". " + trimmed)
            if test_length <= self.max_narrative_length:
                trimmed = sentence + ". " + trimmed
            else:
                break
        
        self.current_narrative = trimmed.strip()
        
        # Add transition indicator
        if not self.current_narrative.startswith("..."):
            self.current_narrative = "..." + self.current_narrative

class HybridSceneMemory:
    """Hybrid memory system combining structured data and narrative coherence"""
    
    def __init__(self, 
                 max_scenes: int = 15,
                 narrative_length: int = 1000,
                 importance_threshold: float = 0.3):
        self.scenes: List[SceneDescription] = []
        self.narrative = NarrativeMemory(max_narrative_length=narrative_length)
        self.max_scenes = max_scenes
        self.importance_threshold = importance_threshold
        self.total_scenes_processed = 0
        
        # Analytics
        self.scene_types_count = {}
        self.common_objects = {}
    
    def add_scene(self, scene: SceneDescription, transition_hint: str = ""):
        """Add scene with intelligent memory management"""
        self.total_scenes_processed += 1
        
        # Update analytics
        self._update_analytics(scene)
        
        # Calculate importance score
        scene.importance_score = self._calculate_importance(scene)
        
        # Add to structured memory
        self.scenes.append(scene)
        
        # Add to narrative memory with context
        transition = self._generate_transition(scene, transition_hint)
        self.narrative.add_scene_to_narrative(scene, transition)
        
        # Manage memory size
        self._manage_memory()
    
    def _calculate_importance(self, scene: SceneDescription) -> float:
        """Calculate importance score for smart memory management"""
        importance = 0.5  # base score
        
        # Boost for unique scene types
        if scene.scene_type not in self.scene_types_count:
            importance += 0.2
        
        # Boost for many objects (complex scenes)
        importance += min(len(scene.key_objects) * 0.1, 0.3)
        
        # Boost for high confidence
        importance += scene.confidence_score * 0.2
        
        # Boost for action scenes
        action_keywords = ['moving', 'running', 'driving', 'flying', 'fighting', 'dancing']
        if any(keyword in scene.description.lower() for keyword in action_keywords):
            importance += 0.2
        
        return min(importance, 1.0)
    
    def _generate_transition(self, scene: SceneDescription, hint: str = "") -> str:
        """Generate smooth transitions between scenes"""
        if hint:
            return hint
        
        if self.scenes and len(self.scenes) > 1:
            prev_scene = self.scenes[-2]
            
            # Scene type transitions
            if prev_scene.scene_type != scene.scene_type:
                transitions = {
                    'indoor_to_outdoor': "Moving outside,",
                    'outdoor_to_indoor': "Going inside,", 
                    'action_to_dialogue': "The action pauses as",
                    'dialogue_to_action': "Suddenly,",
                    'day_to_night': "As time passes,",
                    'static_to_motion': "The scene becomes dynamic as"
                }
                
                transition_key = f"{prev_scene.scene_type}_to_{scene.scene_type}"
                if transition_key in transitions:
                    return transitions[transition_key]
            
            # Object-based transitions
            common_objects = set(prev_scene.key_objects) & set(scene.key_objects)
            if common_objects:
                return f"Continuing with {', '.join(list(common_objects)[:2])},"
            
            # Default transitions
            default_transitions = [
                "Next,", "Then,", "Subsequently,", "Meanwhile,", 
                "Following this,", "In the next moment,"
            ]
            return default_transitions[len(self.scenes) % len(default_transitions)]
        
        return ""
    
    def _manage_memory(self):
        """Intelligent memory management keeping important scenes"""
        if len(self.scenes) <= self.max_scenes:
            return
        
        # Sort by importance, keep most important + recent scenes
        sorted_scenes = sorted(self.scenes[:-3], key=lambda x: x.importance_score, reverse=True)
        
        # Keep top important scenes + last 3 recent scenes
        keep_important = sorted_scenes[:self.max_scenes-5]  
        keep_recent = self.scenes[-5:]  # Always keep last 5
        
        self.scenes = keep_important + keep_recent
        
        # Rebuild narrative from kept scenes
        self._rebuild_narrative()
    
    def _rebuild_narrative(self):
        """Rebuild narrative from current scenes"""
        self.narrative = NarrativeMemory(self.narrative.max_narrative_length)
        
        for i, scene in enumerate(self.scenes):
            if i == 0:
                self.narrative.add_scene_to_narrative(scene)
            else:
                transition = self._generate_transition(scene)
                self.narrative.add_scene_to_narrative(scene, transition)
    
    def _update_analytics(self, scene: SceneDescription):
        """Update analytics for better memory management"""
        # Scene types
        self.scene_types_count[scene.scene_type] = \
            self.scene_types_count.get(scene.scene_type, 0) + 1
        
        # Common objects
        for obj in scene.key_objects:
            self.common_objects[obj] = self.common_objects.get(obj, 0) + 1
    
    def get_context_for_gpt(self, include_objects: bool = True) -> str:
        """Get optimized context for GPT (prioritizes narrative)"""
        context_parts = []
        
        # Main narrative (most important)
        if self.narrative.current_narrative:
            context_parts.append(f"Story so far: {self.narrative.current_narrative}")
        
        # Recent objects context (optional)
        if include_objects and self.scenes:
            recent_objects = []
            for scene in self.scenes[-3:]:  # Last 3 scenes
                recent_objects.extend(scene.key_objects)
            
            if recent_objects:
                unique_objects = list(set(recent_objects))[:8]  # Limit objects
                context_parts.append(f"Recent objects: {', '.join(unique_objects)}")
        
        # Scene count context
        context_parts.append(f"Total scenes analyzed: {self.total_scenes_processed}")
        
        return "\n".join(context_parts)
    
    def get_structured_data(self) -> Dict:
        """Get structured data for analysis/debugging"""
        return {
            "scenes": [asdict(scene) for scene in self.scenes],
            "narrative": asdict(self.narrative),
            "total_processed": self.total_scenes_processed,
            "scene_types": self.scene_types_count,
            "common_objects": self.common_objects
        }
    
    def search_scenes(self, 
                     query: str = "", 
                     scene_type: str = "", 
                     min_confidence: float = 0.0) -> List[SceneDescription]:
        """Search scenes with filters"""
        results = []
        
        for scene in self.scenes:
            matches = True
            
            if query and query.lower() not in scene.description.lower():
                matches = False
            
            if scene_type and scene.scene_type != scene_type:
                matches = False
                
            if scene.confidence_score < min_confidence:
                matches = False
            
            if matches:
                results.append(scene)
        
        return results
    
    def get_summary_stats(self) -> Dict:
        """Get memory statistics"""
        if not self.scenes:
            return {"message": "No scenes processed yet"}
        
        return {
            "total_scenes": len(self.scenes),
            "total_processed": self.total_scenes_processed,
            "avg_confidence": sum(s.confidence_score for s in self.scenes) / len(self.scenes),
            "scene_types": dict(self.scene_types_count),
            "narrative_length": len(self.narrative.current_narrative),
            "most_common_objects": sorted(self.common_objects.items(), 
                                        key=lambda x: x[1], reverse=True)[:5]
        }

class OptimizedSceneDescriptor:
    """Enhanced scene descriptor using hybrid memory"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4o-mini",
                 max_tokens: int = 120,  # Reduced for faster responses
                 temperature: float = 0.2):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    async def describe_scene_async(self, 
                                 frame: np.ndarray, 
                                 frame_number: int,
                                 memory: HybridSceneMemory) -> SceneDescription:
        """Describe scene using hybrid memory context"""
        
        # Compress frame
        base64_image = self._compress_frame(frame)
        
        # Get optimized context (narrative-focused)
        context = memory.get_context_for_gpt()
        
        # Optimized prompt for coherent descriptions
        system_prompt = """You are a video scene analyzer. Create brief, coherent descriptions that flow naturally with the ongoing story.

Focus on:
1. Main subjects and actions
2. Key changes from previous scenes  
3. Setting/environment
4. Story continuity

Respond in JSON:
{
    "description": "Brief scene description that flows with story",
    "key_objects": ["obj1", "obj2"],
    "scene_type": "action/dialogue/indoor/outdoor/transition",
    "confidence_score": 0.95
}"""

        user_prompt = f"""Analyze frame #{frame_number}. 

Context: {context}

Describe this scene maintaining story flow and noting significant changes."""

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            result = self._parse_json_response(response_text)
            
            return SceneDescription(
                timestamp=datetime.now().isoformat(),
                frame_number=frame_number,
                description=result.get("description", "Scene analysis"),
                key_objects=result.get("key_objects", []),
                scene_type=result.get("scene_type", "unknown"),
                confidence_score=result.get("confidence_score", 0.5)
            )
            
        except Exception as e:
            print(f"Error describing scene: {e}")
            return SceneDescription(
                timestamp=datetime.now().isoformat(),
                frame_number=frame_number,
                description=f"Error processing frame: {str(e)}",
                key_objects=[],
                scene_type="error",
                confidence_score=0.0
            )
    
    def _compress_frame(self, frame: np.ndarray) -> str:
        """Compress frame for API"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(frame)
        pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from response with fallback"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback
        return {
            "description": response_text[:100],
            "key_objects": [],
            "scene_type": "unknown",
            "confidence_score": 0.5
        }

# Main enhanced function
async def update_hybrid_scene_memory(
    memory: HybridSceneMemory,
    scene_frame: np.ndarray,
    frame_number: int = 0,
    transition_hint: str = "",
    api_key: Optional[str] = None
) -> HybridSceneMemory:
    """
    Update hybrid scene memory with new frame
    
    Args:
        memory: Current hybrid memory
        scene_frame: OpenCV frame
        frame_number: Frame number
        transition_hint: Optional transition context
        api_key: OpenAI API key
    
    Returns:
        Updated HybridSceneMemory
    """
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
    
    descriptor = OptimizedSceneDescriptor(api_key=api_key)
    
    scene_description = await descriptor.describe_scene_async(
        frame=scene_frame,
        frame_number=frame_number,
        memory=memory
    )
    
    memory.add_scene(scene_description, transition_hint)
    return memory

# Sync wrapper
def update_hybrid_scene_memory_sync(
    memory: HybridSceneMemory,
    scene_frame: np.ndarray,
    frame_number: int = 0,
    transition_hint: str = "",
    api_key: Optional[str] = None
) -> HybridSceneMemory:
    """Synchronous wrapper"""
    return asyncio.run(update_hybrid_scene_memory(
        memory, scene_frame, frame_number, transition_hint, api_key
    ))

# Example usage
if __name__ == "__main__":
    # Initialize hybrid memory
    memory = HybridSceneMemory(max_scenes=10, narrative_length=800)
    
    # Example usage
    print("Hybrid Memory System Example")
    print("Narrative-focused context with structured data backend")
    
    # Simulate processing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        updated_memory = update_hybrid_scene_memory_sync(
            memory=memory,
            scene_frame=dummy_frame,
            frame_number=1,
            transition_hint="Starting the video"
        )
        
        print(f"Context for GPT:\n{updated_memory.get_context_for_gpt()}")
        print(f"\nStats: {updated_memory.get_summary_stats()}")
        
    except Exception as e:
        print(f"Error: {e}")