#!/usr/bin/env python3
"""
Simple Locomotion Mode Inference System
Takes string inputs and returns JSON with detected mode and confidence.
"""

import os
import json
from typing import Optional, Dict, Any
from openai import OpenAI


class LocomotionInferenceEngine:
    """
    Simple locomotion mode inference using GPT-4o-mini
    Takes string inputs and returns JSON with mode and confidence
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the simple locomotion inference
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: GPT model for inference (gpt-4o-mini for speed)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Define locomotion modes
        self.locomotion_modes = [
            "ground walk", "crouch walk", "end crouch walk", "stationary",  "unknown"
        ]
    
    def detect_locomotion_mode(self, 
                              gpt_description: str = "",
                              live_description: str = "",
                              previous_mode: str = "",
                              voice_command: str = "") -> Dict[str, Any]:
        """
        Detect locomotion mode from string inputs
        
        Args:
            gpt_description: GPT scene description (string)
            live_description: Live scene description (string)
            previous_mode: Previous locomotion mode (string)
            voice_command: Voice command (string)
            
        Returns:
            Dictionary with mode_detected and confidence_score
            Format: {"mode_detected": "detected_mode", "confidence_score": 0.85}
        """
        try:
            # Build context from inputs
            context_parts = []
            
            if voice_command:
                context_parts.append(f"Voice command: \"{voice_command}\"")
            
            if previous_mode:
                context_parts.append(f"Previous mode: {previous_mode}")
            
            if gpt_description:
                context_parts.append(f"Scene analysis: {gpt_description}")
            
            if live_description:
                context_parts.append(f"Live description: {live_description}")
            
            # If no input provided, return unknown
            if not context_parts:
                return {
                    "mode_detected": "unknown",
                    "confidence_score": 0.1
                }
            
            context = "\n".join(context_parts)
            
            # Create prompt for locomotion detection
            system_prompt = (
                f"You are a locomotion mode classifier. Based on the provided information, "
                f"classify the current locomotion into one of these modes: "
                f"{', '.join(self.locomotion_modes)}. "
                f"Respond ONLY with valid JSON format: "
                f'{{\"mode_detected\": \"<mode>\", \"confidence_score\": <0.0-1.0>}}'
            )
            
            user_prompt = f"""
Analyze the following information to determine the current locomotion mode:

{context}

Instructions:
- Prioritize voice commands when available
- Use scene descriptions to validate and provide context
- Consider the previous mode for transitions
- Provide a confidence score between 0.0 and 1.0
- Return only the JSON response, no additional text

Available modes: {', '.join(self.locomotion_modes)}
"""
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,  # Very limited for fast response
                temperature=0.1,  # Low temperature for consistent results
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Validate response format
            if "mode_detected" not in result or "confidence_score" not in result:
                raise ValueError("Invalid response format from GPT")
            
            # Validate mode
            mode = result["mode_detected"]
            if mode not in self.locomotion_modes:
                mode = "unknown"
            
            # Validate confidence score
            confidence = float(result["confidence_score"])
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
            return {
                "mode_detected": mode,
                "confidence_score": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"❌ Error in locomotion detection: {e}")
            # Fallback logic
            if voice_command:
                # Simple voice command mapping
                voice_lower = voice_command.lower()
                if "crouch" in voice_lower:
                    return {"mode_detected": "crouch walk", "confidence_score": 0.8}
                elif "stand" in voice_lower or "end" in voice_lower:
                    return {"mode_detected": "end crouch walk", "confidence_score": 0.8}
                elif "walk" in voice_lower or "move" in voice_lower:
                    return {"mode_detected": "ground walk", "confidence_score": 0.8}
                elif "stop" in voice_lower or "stay" in voice_lower:
                    return {"mode_detected": "stationary", "confidence_score": 0.8}
            
            # Ultimate fallback
            return {
                "mode_detected": previous_mode if previous_mode else "unknown",
                "confidence_score": 0.2
            }


# Convenience function for standalone usage
def detect_locomotion_mode(gpt_description: str = "",
                          live_description: str = "",
                          previous_mode: str = "",
                          voice_command: str = "",
                          api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to detect locomotion mode
    
    Args:
        gpt_description: GPT scene description (string)
        live_description: Live scene description (string)
        previous_mode: Previous locomotion mode (string)
        voice_command: Voice command (string)
        api_key: OpenAI API key (optional)
        
    Returns:
        Dictionary: {"mode_detected": "mode", "confidence_score": 0.85}
    """
    # Create or reuse detector instance
    if not hasattr(detect_locomotion_mode, 'detector'):
        detect_locomotion_mode.detector = LocomotionInferenceEngine(api_key=api_key)
    
    return detect_locomotion_mode.detector.detect_locomotion_mode(
        gpt_description=gpt_description,
        live_description=live_description,
        previous_mode=previous_mode,
        voice_command=voice_command
    )


# Example usage and testing
def test_simple_detection():
    """Test the simple locomotion detection with examples"""
    
    print("🚀 Testing Simple Locomotion Detection")
    
    # Initialize detector
    detector = LocomotionInferenceEngine()
    
    # Test cases
    test_cases = [
        {
            "name": "Voice command only",
            "voice_command": "I am crouch walking",
            "expected": "crouch walk"
        },
        {
            "name": "Scene description only", 
            "gpt_description": "Person crouching down and moving forward slowly",
            "expected": "crouch walk"
        },
        {
            "name": "Combined inputs",
            "voice_command": "standing up",
            "live_description": "Person transitioning from crouch to upright position",
            "previous_mode": "crouch walk",
            "expected": "end crouch walk"
        },
        {
            "name": "Normal walking",
            "voice_command": "start walking",
            "gpt_description": "Person walking normally across the room",
            "expected": "ground walk"
        },
        {
            "name": "Previous mode continuation",
            "live_description": "Person continuing similar movement",
            "previous_mode": "ground walk",
            "expected": "ground walk"
        }
    ]
    
    print(f"\n📊 Running {len(test_cases)} test cases:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        
        # Extract test inputs
        inputs = {k: v for k, v in test_case.items() 
                 if k not in ['name', 'expected']}
        
        # Print inputs
        for key, value in inputs.items():
            if value:
                print(f"  {key}: \"{value}\"")
        
        try:
            # Run detection
            result = detector.detect_locomotion_mode(**inputs)
            
            # Print results
            mode = result["mode_detected"]
            confidence = result["confidence_score"]
            expected = test_case["expected"]
            
            status = "✅ PASS" if mode == expected else "❌ FAIL"
            print(f"  Result: {mode} (confidence: {confidence:.3f}) {status}")
            
            if mode != expected:
                print(f"  Expected: {expected}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print("-" * 40)
    
    print("\n✅ Testing completed!")


def example_usage():
    """Show example usage of the simple API"""
    
    print("\n💡 Example Usage:")
    print("=" * 40)
    
    # Method 1: Using the class
    print("\n1. Using the class:")
    detector = LocomotionInferenceEngine()
    
    result = detector.detect_locomotion_mode(
        voice_command="I am crouch walking",
        gpt_description="Person in crouching position moving forward",
        previous_mode="ground walk"
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Method 2: Using the standalone function  
    print("\n2. Using the standalone function:")
    result = detect_locomotion_mode(
        voice_command="standing up",
        live_description="Person transitioning to upright position"
    )
    print(f"Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    print("🤖 Simple Locomotion Detection System")
    print("Set OPENAI_API_KEY environment variable before running")
    
    try:
        # Show example usage
        example_usage()
        
        # Run tests (uncomment to test with real API)
        # test_simple_detection()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure OPENAI_API_KEY is set")