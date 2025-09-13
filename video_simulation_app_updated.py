#!/usr/bin/env python3
"""
Video Simulation Web App
Real-time locomotion mode prediction simulation with scene change detection,
GPT descriptions, live descriptions, and timing synchronization.
"""

import base64
import json
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.serving import make_server

from gpt_scene_description.gpt_descriptor import (
    GPTSceneDescriptor,
    update_gpt_description_memory,
)
from live_descriptor.live_descriptor import process_live_frame
from locomotion_infer import LocomotionInferenceEngine

# Import our modules
from scene_change_detection.change_detector import SceneChangeDetector

app = Flask(__name__)
app.secret_key = "video_simulation_secret_key"

# Global variables for shared state
video_processor = None
simulation_state = {
    "is_running": False,
    "current_video": None,
    "current_data": None,
    "current_frame": None,
    "frame_count": 0,
    "scene_changes": [],
    "gpt_descriptions": [],
    "live_descriptions": [],
    "locomotion_predictions": [],
    "current_voice_command": "",
    "current_prediction": None,
    "video_duration": 0,
    "fps": 1,  # 1 FPS for extraction
    "processing_fps": 0.5,  # Every 2nd frame
    "prediction_accuracy": {"total": 0, "correct": 0, "percentage": 0},
    "latency_stats": {
        "avg_latency": 0,
        "max_latency": 0,
        "min_latency": 0,
        "realtime_factor": 0,
    },
}


class VideoProcessor:
    """
    Main video processor that handles all the simulation logic
    """

    def __init__(self):
        self.change_detector = None
        self.gpt_descriptor = None
        self.locomotion_engine = None
        self.manager = Manager()

        # Shared data structures
        self.shared_state = self.manager.dict()
        self.frame_queue = Queue()
        self.result_queue = Queue()

        # Background processes
        self.change_detection_process = None
        self.description_process = None

        # Memory stores
        self.gpt_description_memory = []
        self.live_desc_memory = []

        # Context windows for richer predictions
        self.collected_live_descriptions = (
            []
        )  # All live descriptions from every 2nd frame
        self.previous_modes_history = []  # History of previous locomotion modes
        self.max_context_size = 3  # Maximum context items to keep

        self.reset_state()

    def reset_state(self):
        """Reset all processing state"""
        self.shared_state.update(
            {
                "is_processing": False,
                "frame_count": 0,
                "scene_changes_detected": 0,
                "current_frame_data": None,
            }
        )

        self.gpt_description_memory = []
        self.live_desc_memory = []
        self.collected_live_descriptions = []
        self.previous_modes_history = []

    def collect_background_results(self):
        """Continuously collect results from background workers"""
        # Collect all available live description results
        collected_count = 0
        while True:
            try:
                live_desc_result = self.desc_result_queue.get_nowait()
                if (
                    live_desc_result
                    and live_desc_result.get("type") == "live_description"
                ):
                    # Add to our collected descriptions
                    desc_entry = {
                        "frame_idx": live_desc_result["frame_idx"],
                        "description": live_desc_result["description"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.collected_live_descriptions.append(desc_entry)

                    # Also add to simulation state for UI display
                    simulation_state["live_descriptions"].append(desc_entry)

                    collected_count += 1
                    print(
                        f"📝 Collected live description for frame {live_desc_result['frame_idx']}: {live_desc_result['description'][:50]}..."
                    )

            except queue.Empty:
                break

        # Collect scene change results too
        scene_change_result = None
        while True:
            try:
                scene_result = self.change_result_queue.get_nowait()
                if scene_result and scene_result.get("is_changed"):
                    scene_change_result = scene_result
                    break
            except queue.Empty:
                break

        return scene_change_result

    def get_recent_context(self):
        """Get the 3 most recent live descriptions and previous modes"""
        # Get up to 3 most recent live descriptions
        recent_descriptions = (
            self.collected_live_descriptions[-self.max_context_size :]
            if self.collected_live_descriptions
            else []
        )

        # Get up to 3 most recent previous modes
        recent_modes = (
            self.previous_modes_history[-self.max_context_size :]
            if self.previous_modes_history
            else []
        )

        return recent_descriptions, recent_modes

    def initialize_models(self):
        """Initialize all AI models"""
        try:
            print("🚀 Initializing AI models...")

            # Initialize scene change detector
            self.change_detector = SceneChangeDetector(
                model_name="mobileclip", similarity_threshold=0.85, use_fp16=True
            )

            # Initialize GPT descriptor
            self.gpt_descriptor = GPTSceneDescriptor(
                max_memory_size=10, enable_context=True, use_narrative_style=True
            )

            # Initialize locomotion inference engine
            self.locomotion_engine = LocomotionInferenceEngine()

            print("✅ All models initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            return False

    def extract_video_frames(
        self, video_path: str, fps: float = 1.0
    ) -> List[np.ndarray]:
        """Extract frames from video at specified FPS"""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / fps)

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append(frame.copy())
                extracted_count += 1

            frame_count += 1

        cap.release()
        print(f"📹 Extracted {extracted_count} frames from {frame_count} total frames")
        return frames

    def change_detection_worker(self, frame_queue: Queue, result_queue: Queue):
        """Background worker for scene change detection"""
        print("🔍 Starting change detection worker...")

        try:
            detector = SceneChangeDetector(
                model_name="mobileclip", similarity_threshold=0.85, use_fp16=True
            )

            while True:
                try:
                    # Get frame from queue
                    frame_data = frame_queue.get(timeout=1)
                    if frame_data is None:  # Shutdown signal
                        break

                    frame_idx, frame = frame_data

                    # Detect scene change
                    is_changed, changed_frame = detector.detect_scene_change(frame)

                    # Send result back
                    result_queue.put(
                        {
                            "type": "scene_change",
                            "frame_idx": frame_idx,
                            "is_changed": is_changed,
                            "changed_frame": changed_frame,
                        }
                    )

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Error in change detection: {e}")

        except Exception as e:
            print(f"❌ Error initializing change detection worker: {e}")

    def description_worker(self, frame_queue: Queue, result_queue: Queue):
        """Background worker for live description"""
        print("📝 Starting description worker...")

        live_desc_memory = []

        while True:
            try:
                # Get frame from queue
                frame_data = frame_queue.get(timeout=1)
                if frame_data is None:  # Shutdown signal
                    break

                frame_idx, frame = frame_data

                # Process live frame description
                updated_memory = process_live_frame(
                    live_desc_memory,
                    frame,
                    prompt="Describe this image shortly so that it is useful for locomotion mode prediction.",
                )

                live_desc_memory = updated_memory

                # Send result back
                if live_desc_memory:
                    latest_desc = live_desc_memory[-1]
                    result_queue.put(
                        {
                            "type": "live_description",
                            "frame_idx": frame_idx,
                            "description": latest_desc.get("description", ""),
                            "memory_size": len(live_desc_memory),
                        }
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in description worker: {e}")

    def start_background_workers(self):
        """Start background processing workers"""
        try:
            # Create queues for communication
            self.change_queue = Queue()
            self.desc_queue = Queue()
            self.change_result_queue = Queue()
            self.desc_result_queue = Queue()

            # Start change detection worker
            self.change_detection_process = Process(
                target=self.change_detection_worker,
                args=(self.change_queue, self.change_result_queue),
            )
            self.change_detection_process.start()

            # Start description worker
            self.description_process = Process(
                target=self.description_worker,
                args=(self.desc_queue, self.desc_result_queue),
            )
            self.description_process.start()

            print("✅ Background workers started")
            return True

        except Exception as e:
            print(f"❌ Error starting background workers: {e}")
            return False

    def stop_background_workers(self):
        """Stop background processing workers"""
        try:
            # Send shutdown signals
            if hasattr(self, "change_queue"):
                self.change_queue.put(None)
            if hasattr(self, "desc_queue"):
                self.desc_queue.put(None)

            # Wait for processes to finish
            if (
                self.change_detection_process
                and self.change_detection_process.is_alive()
            ):
                self.change_detection_process.join(timeout=5)
                if self.change_detection_process.is_alive():
                    self.change_detection_process.terminate()

            if self.description_process and self.description_process.is_alive():
                self.description_process.join(timeout=5)
                if self.description_process.is_alive():
                    self.description_process.terminate()

            print("✅ Background workers stopped")

        except Exception as e:
            print(f"❌ Error stopping background workers: {e}")

    def process_simulation_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        voice_command: str = "",
        previous_mode: str = "",
        video_start_time: float = None,
    ) -> Dict:
        """Process a single frame through the simulation pipeline"""
        result = {
            "frame_idx": frame_idx,
            "timestamp": datetime.now().isoformat(),
            "scene_changed": False,
            "gpt_description": "",
            "live_description": "",
            "voice_command": voice_command,
            "locomotion_prediction": None,
            "processing_time": 0,
            "video_timestamp": frame_idx,  # Video time in seconds
            "latency": 0,  # Time difference between video time and processing time
        }

        start_time = time.time()
        video_start_time = video_start_time or start_time

        try:
            # Send frame to background workers (every 2nd frame)
            if frame_idx % 2 == 0:
                self.change_queue.put((frame_idx, frame))
                self.desc_queue.put((frame_idx, frame))

            # Collect ALL available results from background workers
            scene_change_result = self.collect_background_results()

            # Get recent context for predictions
            recent_descriptions, recent_modes = self.get_recent_context()

            # Process scene change if detected
            if scene_change_result and scene_change_result["is_changed"]:
                result["scene_changed"] = True

                # Generate GPT description for scene change
                try:
                    gpt_desc = self.gpt_descriptor.describe_scene(
                        scene_change_result["changed_frame"]
                    )
                    if gpt_desc:
                        self.gpt_description_memory.append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "description": gpt_desc,
                                "frame_idx": frame_idx,
                            }
                        )
                        result["gpt_description"] = gpt_desc
                except Exception as e:
                    print(f"❌ Error generating GPT description: {e}")

            # Generate locomotion prediction (only when there's a voice command)
            if voice_command:
                try:
                    # Prepare context from recent descriptions
                    live_context = []
                    if recent_descriptions:
                        for desc in recent_descriptions:
                            live_context.append(
                                f"Frame {desc['frame_idx']}: {desc['description']}"
                            )

                    # Prepare previous modes context
                    modes_context = ", ".join(recent_modes) if recent_modes else ""

                    # Get most recent single descriptions for compatibility
                    recent_gpt = (
                        self.gpt_description_memory[-1]["description"]
                        if self.gpt_description_memory
                        else ""
                    )
                    recent_live = "\n".join(live_context) if live_context else ""

                    prediction = self.locomotion_engine.detect_locomotion_mode(
                        gpt_description=recent_gpt,
                        live_description=recent_live,
                        previous_mode=modes_context,  # Now contains multiple modes
                        voice_command=voice_command,
                    )

                    result["locomotion_prediction"] = prediction

                    # Extract API latency if available
                    if "api_latency" in prediction:
                        result["api_latency"] = prediction["api_latency"]
                        print(
                            f"⚡ OpenAI API latency: {prediction['api_latency']:.3f}s"
                        )

                    # Update previous modes history
                    detected_mode = prediction.get("mode_detected", "unknown")
                    self.previous_modes_history.append(detected_mode)
                    # Keep only recent modes
                    if len(self.previous_modes_history) > self.max_context_size:
                        self.previous_modes_history = self.previous_modes_history[
                            -self.max_context_size :
                        ]

                    print(
                        f"🤖 Predicted: {detected_mode} using {len(live_context)} descriptions and {len(recent_modes)} previous modes"
                    )

                except Exception as e:
                    print(f"❌ Error generating locomotion prediction: {e}")
                    result["locomotion_prediction"] = {
                        "mode_detected": "unknown",
                        "confidence_score": 0.1,
                        "api_latency": 0.0,
                    }

            # Calculate latency metrics
            end_time = time.time()
            result["processing_time"] = end_time - start_time

            # Calculate latency: how much time has passed in real processing vs video time
            elapsed_real_time = end_time - video_start_time
            video_time_expected = frame_idx  # Since we're processing at 1 FPS
            result["latency"] = elapsed_real_time - video_time_expected

            return result

        except Exception as e:
            print(f"❌ Error processing frame {frame_idx}: {e}")
            end_time = time.time()
            result["processing_time"] = end_time - start_time
            elapsed_real_time = end_time - video_start_time
            video_time_expected = frame_idx
            result["latency"] = elapsed_real_time - video_time_expected
            return result


# Initialize global video processor
video_processor = VideoProcessor()


@app.route("/")
def index():
    """Main page with video selection and simulation controls"""
    # Get available videos and data files
    videos_dir = Path("/home/cmuser/ASIF/loco-simple/videos")
    data_dir = Path("/home/cmuser/ASIF/loco-simple/data_json")

    videos = [f.name for f in videos_dir.glob("*.mp4")] if videos_dir.exists() else []
    data_files = [f.name for f in data_dir.glob("*.json")] if data_dir.exists() else []

    return render_template(
        "video_simulation.html",
        videos=videos,
        data_files=data_files,
        current_state=simulation_state,
    )


@app.route("/api/load_data", methods=["POST"])
def load_data():
    """Load video and corresponding JSON data"""
    try:
        data = request.get_json()
        video_file = data.get("video_file")
        data_file = data.get("data_file")

        if not video_file or not data_file:
            return jsonify({"error": "Video and data files are required"}), 400

        # Load video path
        video_path = f"/home/cmuser/ASIF/loco-simple/videos/{video_file}"
        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_file}"}), 404

        # Load JSON data
        data_path = f"/home/cmuser/ASIF/loco-simple/data_json/{data_file}"
        if not os.path.exists(data_path):
            return jsonify({"error": f"Data file not found: {data_file}"}), 404

        with open(data_path, "r") as f:
            json_data = json.load(f)

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        # Update simulation state
        simulation_state.update(
            {
                "current_video": video_path,
                "current_data": json_data,
                "video_duration": duration,
                "frame_count": 0,
                "is_running": False,
            }
        )

        return jsonify(
            {
                "success": True,
                "video_info": {
                    "file": video_file,
                    "duration": duration,
                    "fps": fps,
                    "frame_count": frame_count,
                },
                "data_info": {"file": data_file, "entries": len(json_data)},
                "data_entries": json_data,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/initialize_models", methods=["POST"])
def initialize_models():
    """Initialize AI models"""
    try:
        success = video_processor.initialize_models()
        if success:
            return jsonify(
                {"success": True, "message": "Models initialized successfully"}
            )
        else:
            return jsonify({"error": "Failed to initialize models"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/start_simulation", methods=["POST"])
def start_simulation():
    """Start the video simulation"""
    try:
        if (
            not simulation_state["current_video"]
            or not simulation_state["current_data"]
        ):
            return jsonify({"error": "Video and data must be loaded first"}), 400

        # Reset state
        video_processor.reset_state()
        simulation_state.update(
            {
                "is_running": True,
                "frame_count": 0,
                "scene_changes": [],
                "gpt_descriptions": [],
                "live_descriptions": [],
                "locomotion_predictions": [],
                "prediction_accuracy": {"total": 0, "correct": 0, "percentage": 0},
                "latency_stats": {
                    "avg_latency": 0,
                    "max_latency": 0,
                    "min_latency": 0,
                    "realtime_factor": 0,
                },
            }
        )

        # Start background workers
        success = video_processor.start_background_workers()
        if not success:
            return jsonify({"error": "Failed to start background workers"}), 500

        # Start simulation thread
        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.daemon = True
        simulation_thread.start()

        return jsonify({"success": True, "message": "Simulation started"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop_simulation", methods=["POST"])
def stop_simulation():
    """Stop the video simulation"""
    try:
        simulation_state["is_running"] = False
        video_processor.stop_background_workers()
        return jsonify({"success": True, "message": "Simulation stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulation_status")
def simulation_status():
    """Get current simulation status"""
    try:
        return jsonify(
            {
                "is_running": simulation_state["is_running"],
                "frame_count": simulation_state["frame_count"],
                "current_voice_command": simulation_state["current_voice_command"],
                "current_prediction": simulation_state["current_prediction"],
                "scene_changes_count": len(simulation_state["scene_changes"]),
                "gpt_descriptions_count": len(simulation_state["gpt_descriptions"]),
                "live_descriptions_count": len(simulation_state["live_descriptions"]),
                "prediction_accuracy": simulation_state["prediction_accuracy"],
                "latency_stats": simulation_state["latency_stats"],
                "current_frame_base64": simulation_state.get(
                    "current_frame_base64", ""
                ),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def calculate_prediction_accuracy(predictions, data_entries):
    """Calculate prediction accuracy statistics"""
    if not predictions or not data_entries:
        return {"total": 0, "correct": 0, "percentage": 0}

    # Create a lookup map for expected levels by timestamp with tolerance
    expected_levels = {}
    for entry in data_entries:
        timestamp_seconds = parse_timestamp(entry["timestamp"])
        frame_idx = int(timestamp_seconds * simulation_state["fps"])
        expected_levels[frame_idx] = entry["level"]
        # Also add nearby frames (±2 seconds tolerance)
        for offset in range(-2, 3):
            if frame_idx + offset >= 0:
                expected_levels[frame_idx + offset] = entry["level"]

    total_predictions = 0
    correct_predictions = 0

    print(f"🔍 Calculating accuracy for {len(predictions)} predictions")
    print(f"📋 Expected levels map: {expected_levels}")

    for prediction_result in predictions:
        if prediction_result.get("locomotion_prediction"):
            frame_idx = prediction_result["frame_idx"]
            predicted_mode = prediction_result["locomotion_prediction"]["mode_detected"]

            if frame_idx in expected_levels:
                total_predictions += 1
                expected_level = expected_levels[frame_idx]
                is_correct = predicted_mode.lower() == expected_level.lower()

                print(
                    f"🎯 Frame {frame_idx}: Predicted '{predicted_mode}' vs Expected '{expected_level}' = {'✅' if is_correct else '❌'}"
                )

                if is_correct:
                    correct_predictions += 1
            else:
                print(f"⚠️ No expected level found for frame {frame_idx}")

    percentage = (
        (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    )

    print(
        f"📊 Final accuracy: {correct_predictions}/{total_predictions} = {percentage:.1f}%"
    )

    return {
        "total": total_predictions,
        "correct": correct_predictions,
        "percentage": round(percentage, 1),
    }


def calculate_latency_stats(predictions):
    """Calculate OpenAI API latency statistics from prediction results"""
    if not predictions:
        return {
            "avg_latency": 0,
            "max_latency": 0,
            "min_latency": 0,
            "realtime_factor": 0,
        }

    api_latencies = []
    processing_times = []

    for prediction_result in predictions:
        # Use API latency if available, otherwise fall back to processing time
        if "api_latency" in prediction_result:
            api_latencies.append(prediction_result["api_latency"])
        elif "processing_time" in prediction_result:
            processing_times.append(prediction_result["processing_time"])

    # Prefer API latencies over processing times
    latencies = api_latencies if api_latencies else processing_times

    if not latencies:
        return {
            "avg_latency": 0,
            "max_latency": 0,
            "min_latency": 0,
            "realtime_factor": 0,
        }

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    # Calculate real-time factor based on API response time:
    # - < 0.5s = Excellent for real-time
    # - 0.5-1.0s = Good for real-time
    # - > 1.0s = Too slow for real-time
    # Real-time factor: how much slower than ideal (0.5s target)
    realtime_factor = avg_latency / 0.5  # 0.5 second target for real-time

    latency_type = "API" if api_latencies else "Processing"
    print(
        f"📊 {latency_type} Latency Stats: Avg={avg_latency:.3f}s, Max={max_latency:.3f}s, Min={min_latency:.3f}s, RT Factor={realtime_factor:.2f}x"
    )

    return {
        "avg_latency": round(avg_latency, 3),
        "max_latency": round(max_latency, 3),
        "min_latency": round(min_latency, 3),
        "realtime_factor": round(realtime_factor, 2),
    }


def parse_timestamp(timestamp_str: str) -> float:
    """Parse timestamp string (HH:MM:SS) to seconds"""
    try:
        parts = timestamp_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except:
        return 0.0


def run_simulation():
    """Main simulation loop running in background thread"""
    try:
        print("🎬 Starting video simulation...")

        # Extract frames from video at 1 FPS
        frames = video_processor.extract_video_frames(
            simulation_state["current_video"], fps=simulation_state["fps"]
        )

        if not frames:
            print("❌ No frames extracted from video")
            return

        # Get data entries for reference
        data_entries = simulation_state["current_data"]
        previous_mode = ""

        # Track video start time for latency calculation
        video_start_time = time.time()

        # Create a lookup map for timestamp-based data
        timestamp_data_map = {}
        for entry in data_entries:
            timestamp_seconds = parse_timestamp(entry["timestamp"])
            frame_idx = int(timestamp_seconds * simulation_state["fps"])
            timestamp_data_map[frame_idx] = entry

        # Process ALL extracted frames (1 FPS)
        for frame_idx, current_frame in enumerate(frames):
            if not simulation_state["is_running"]:
                break

            # Update simulation state
            simulation_state.update({"frame_count": frame_idx})

            # Check if we have specific data for this frame
            current_voice_command = ""
            expected_level = ""

            if frame_idx in timestamp_data_map:
                entry = timestamp_data_map[frame_idx]
                current_voice_command = entry["demo_voice_command"]
                expected_level = entry["level"]
                simulation_state["current_voice_command"] = current_voice_command
                print(f"🎯 Frame {frame_idx} - Voice command: {current_voice_command}")
                print(f"📊 Expected mode: {expected_level}")
            else:
                # No specific data for this frame, use empty command
                simulation_state["current_voice_command"] = ""

            # Encode frame for display
            _, buffer = cv2.imencode(".jpg", current_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            simulation_state["current_frame_base64"] = frame_base64

            print(f"🎬 Showing frame {frame_idx}/{len(frames)-1} (time: {frame_idx}s)")

            # Process frame through simulation pipeline
            result = video_processor.process_simulation_frame(
                current_frame,
                frame_idx,
                voice_command=current_voice_command,
                previous_mode=previous_mode,
                video_start_time=video_start_time,
            )

            # Update simulation state with results
            if result["scene_changed"]:
                simulation_state["scene_changes"].append(result)

            if result["gpt_description"]:
                simulation_state["gpt_descriptions"].append(result)

            # Note: live_descriptions are now added directly in collect_background_results()

            if result["locomotion_prediction"]:
                simulation_state["locomotion_predictions"].append(result)
                simulation_state["current_prediction"] = result["locomotion_prediction"]
                previous_mode = result["locomotion_prediction"]["mode_detected"]

                print(
                    f"🤖 Predicted: {result['locomotion_prediction']['mode_detected']} "
                    f"(confidence: {result['locomotion_prediction']['confidence_score']:.3f})"
                )

                # Update prediction accuracy statistics
                simulation_state["prediction_accuracy"] = calculate_prediction_accuracy(
                    simulation_state["locomotion_predictions"], data_entries
                )

                # Update latency statistics
                simulation_state["latency_stats"] = calculate_latency_stats(
                    simulation_state["locomotion_predictions"]
                )

            # Sleep for 1 second to maintain 1 FPS display rate
            time.sleep(1.0)

        print("✅ Simulation completed")

        # Final accuracy calculation
        print("🔄 Calculating final accuracy...")
        simulation_state["prediction_accuracy"] = calculate_prediction_accuracy(
            simulation_state["locomotion_predictions"], data_entries
        )

        # Final latency calculation
        print("🔄 Calculating final latency stats...")
        simulation_state["latency_stats"] = calculate_latency_stats(
            simulation_state["locomotion_predictions"]
        )

        simulation_state["is_running"] = False

    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        simulation_state["is_running"] = False

    finally:
        video_processor.stop_background_workers()


if __name__ == "__main__":
    print("🚀 Starting Video Simulation App")
    print("📁 Videos folder: /home/cmuser/ASIF/loco-simple/videos")
    print("📁 Data folder: /home/cmuser/ASIF/loco-simple/data_json")

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
