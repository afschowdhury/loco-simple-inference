# Add this method to VideoProcessor class in video_simulation_app.py

def get_immediate_live_description(self, frame):
    """Get live description immediately for current frame (synchronous)"""
    try:
        from live_descriptor.live_descriptor import process_live_frame
        
        # Process frame immediately
        updated_memory = process_live_frame(
            [],  # Empty memory for immediate processing
            frame,
            prompt="Describe this scene from a person's perspective walking in a construction site. Focus on elements that might affect locomotion."
        )
        
        if updated_memory:
            return updated_memory[-1].get("description", "")
        return ""
        
    except Exception as e:
        print(f"❌ Error getting immediate description: {e}")
        return ""

# Then modify the prediction section around line 412:

# Generate locomotion prediction (only when there's a voice command)
if voice_command:
    try:
        # Get immediate live description for this specific frame
        immediate_description = self.get_immediate_live_description(frame)
        
        # Prepare context from recent descriptions + immediate
        live_context = []
        if recent_descriptions:
            for desc in recent_descriptions:
                live_context.append(f"Frame {desc['frame_idx']}: {desc['description']}")
        
        # Add immediate description as most recent
        if immediate_description:
            live_context.append(f"Frame {frame_idx} (current): {immediate_description}")
        
        # Prepare previous modes context
        modes_context = ", ".join(recent_modes) if recent_modes else ""
        
        # Get most recent single descriptions for compatibility
        recent_gpt = (
            self.gpt_description_memory[-1]["description"]
            if self.gpt_description_memory
            else ""
        )
        recent_live = "\n".join(live_context) if live_context else ""
        
        print(f"🎯 Making prediction with {len(live_context)} descriptions including immediate")
        
        prediction = self.locomotion_engine.detect_locomotion_mode(
            gpt_description=recent_gpt,
            live_description=recent_live,
            previous_mode=modes_context,
            voice_command=voice_command,
        )

        # ... rest of prediction processing
