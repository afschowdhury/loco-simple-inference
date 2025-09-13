# Option 3: Wait briefly for background results


def collect_background_results_with_timeout(self, timeout_seconds=0.5):
    """Collect results with a brief timeout for voice command frames"""
    import time

    start_time = time.time()
    initial_count = len(self.collected_live_descriptions)

    while (time.time() - start_time) < timeout_seconds:
        # Try to collect results
        scene_result = self.collect_background_results()

        # Check if we got new descriptions
        if len(self.collected_live_descriptions) > initial_count:
            print(
                f"✅ Got {len(self.collected_live_descriptions) - initial_count} new descriptions after {time.time() - start_time:.2f}s"
            )
            return scene_result

        time.sleep(0.1)  # Small delay

    print(f"⏰ Timeout waiting for descriptions after {timeout_seconds}s")
    return None


# Then modify the prediction section:
if voice_command:
    # Wait a bit for recent descriptions
    scene_change_result = self.collect_background_results_with_timeout(0.5)
    recent_descriptions, recent_modes = self.get_recent_context()
    # ... rest of prediction
