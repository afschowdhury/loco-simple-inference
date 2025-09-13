# Alternative: Pre-process descriptions for voice command frames


def preprocess_voice_command_frames(self, frames, timestamp_data_map):
    """Pre-process live descriptions for all voice command frames"""
    print("🔄 Pre-processing voice command frames...")

    for frame_idx in timestamp_data_map.keys():
        if frame_idx < len(frames):
            try:
                frame = frames[frame_idx]
                description = self.get_immediate_live_description(frame)

                if description:
                    desc_entry = {
                        "frame_idx": frame_idx,
                        "description": description,
                        "timestamp": datetime.now().isoformat(),
                        "pre_processed": True,
                    }
                    self.collected_live_descriptions.append(desc_entry)
                    print(f"📝 Pre-processed frame {frame_idx}: {description[:50]}...")

            except Exception as e:
                print(f"❌ Error pre-processing frame {frame_idx}: {e}")


# Then call this in run_simulation() before the main loop:
# video_processor.preprocess_voice_command_frames(frames, timestamp_data_map)
