#!/usr/bin/env python3
"""
Locomotion pipeline orchestrator for loco_data_4.mp4 using:
- 1 FPS frame sampling
- Scene change detection (scene_change_detection/change_detector.py)
- GPT scene description on every frame (gpt_scene_description/gpt_descriptor.py)
- Live descriptor every 2nd sampled frame (live_descriptor/live_descriptor.py)
- Locomotion mode prediction only at JSON timestamps (locomotion_infer.py)

Notes:
- GPT and OpenAI locomotion inference require OPENAI_API_KEY; gracefully handled if missing
- Live descriptor uses Ollama (moondream) at http://localhost:11434; gracefully handled if unavailable
- Results saved to outputs/loco_data_4_results.json
"""

import os
import json
import cv2
import argparse
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")    

# Local imports
from scene_change_detection.change_detector import SceneChangeDetector
from gpt_scene_description.gpt_descriptor import GPTSceneDescriptor
from live_descriptor.live_descriptor import process_live_frame
from locomotion_infer import LocomotionInferenceEngine


def parse_hh_mm_ss_to_seconds(time_str: str) -> int:
    parts = time_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format (expected HH:MM:SS): {time_str}")
    hours, minutes, seconds = [int(p) for p in parts]
    return hours * 3600 + minutes * 60 + seconds


def load_voice_script(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    # Normalize with seconds
    enriched = []
    for item in data:
        try:
            sec = parse_hh_mm_ss_to_seconds(item.get("timestamp", "00:00:00"))
        except Exception:
            sec = 0
        enriched.append({
            "timestamp": item.get("timestamp", "00:00:00"),
            "seconds": sec,
            "voice_command": item.get("demo_voice_command", ""),
            "level": item.get("level", "")
        })
    enriched.sort(key=lambda x: x["seconds"])  # ascending
    return enriched


def voice_for_second(voice_timeline: List[Dict[str, Any]], second_index: int) -> Tuple[str, str]:
    """Return the most recent voice_command and level at or before second_index."""
    latest_voice = ""
    latest_level = ""
    for item in voice_timeline:
        if item["seconds"] <= second_index:
            latest_voice = item["voice_command"]
            latest_level = item["level"]
        else:
            break
    return latest_voice, latest_level


def ensure_outputs_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def seconds_to_hhmmss(sec: int) -> str:
    return str(timedelta(seconds=int(sec)))


def sample_video_1fps(video_path: str) -> Tuple[int, int]:
    """
    Return (total_seconds, source_fps) for planning iteration.
    Use random access via CAP_PROP_POS_MSEC when fetching frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_seconds = int(math.floor(frame_count / fps)) if fps > 0 else 0
    cap.release()
    return duration_seconds, int(round(fps))


def get_frame_at_second(cap: cv2.VideoCapture, second_index: int) -> Optional[Any]:
    target_msec = second_index * 1000.0
    cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def simple_rule_locomotion_infer(voice_command: str, gpt_description: str, live_description: str,
                                 previous_mode: str) -> Dict[str, Any]:
    """Fallback mapping when OpenAI locomotion engine is not available."""
    text_blobs = " ".join(filter(None, [voice_command, gpt_description, live_description])).lower()
    if any(k in text_blobs for k in ["crouch", "crouching", "crouched"]):
        mode = "crouch walk"
        conf = 0.8
    elif any(k in text_blobs for k in ["stand", "standing", "end crouch", "upright"]):
        mode = "end crouch walk"
        conf = 0.8
    elif any(k in text_blobs for k in ["walk", "walking", "move", "moving", "forward"]):
        mode = "ground walk"
        conf = 0.75
    elif any(k in text_blobs for k in ["stop", "stay", "stationary", "idle"]):
        mode = "stationary"
        conf = 0.7
    else:
        mode = previous_mode or "unknown"
        conf = 0.2
    return {"mode_detected": mode, "confidence_score": round(conf, 3)}


def run_pipeline(
    video_path: str,
    json_path: str,
    model_name: str = "mobileclip",
    similarity_threshold: float = 0.85,
    use_fp16: bool = True,
    enable_tensorrt: bool = False,
    enable_gpt: bool = True,
    ollama_url: str = "http://localhost:11434",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    base_dir = str(Path(video_path).resolve().parent.parent)
    outputs_dir = ensure_outputs_dir(base_dir)

    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Prepare timeline
    voice_timeline = load_voice_script(json_path)
    sec_to_voice = {item["seconds"]: (item["voice_command"], item["level"]) for item in voice_timeline}
    request_seconds = set(sec_to_voice.keys())

    # Prepare detectors/descriptors
    detector = SceneChangeDetector(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        use_fp16=use_fp16,
        enable_tensorrt=enable_tensorrt,
    )

    gpt_descriptor: Optional[GPTSceneDescriptor] = None
    if enable_gpt:
        try:
            gpt_descriptor = GPTSceneDescriptor(
                api_key=openai_api_key,
                model= "gpt-4.1-nano",
                max_tokens=100,
            )
        except Exception as e:
            print(f"⚠️  GPT descriptor disabled (reason: {e})")
            gpt_descriptor = None

    # Locomotion inference engine
    locomotion_engine: Optional[LocomotionInferenceEngine] = None
    try:
        locomotion_engine = LocomotionInferenceEngine(
            api_key=openai_api_key,
            model= "gpt-4.1-nano",
            max_memory_size=10,
            confidence_threshold=0.7,
        )
    except Exception as e:
        print(f"⚠️  OpenAI locomotion engine unavailable, using rule-based fallback (reason: {e})")
        locomotion_engine = None

    # Memories
    gpt_description_memory: List[Dict[str, Any]] = []
    live_desc_memory: List[Dict[str, Any]] = []

    # Video iteration planning
    total_seconds, src_fps = sample_video_1fps(video_path)
    print(f"🎥 Video: {video_path} | src_fps={src_fps} | duration≈{total_seconds}s | sampling=1 FPS ({total_seconds+1} frames)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    results: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    previous_mode: str = ""

    try:
        for sec in range(total_seconds + 1):
            frame = get_frame_at_second(cap, sec)
            if frame is None:
                print(f"⚠️  Could not read frame at t={sec}s; skipping")
                continue

            # Voice context (most recent for logging)
            voice_command_context, expected_level_context = voice_for_second(voice_timeline, sec)

            # Scene change detection (always)
            scene_changed, changed_frame = detector.detect_scene_change(frame)

            # GPT description (always if enabled)
            gpt_desc_text: str = ""
            if gpt_descriptor is not None:
                try:
                    # Prefer the changed frame if available, otherwise current frame
                    if scene_changed:
                        
                        gpt_description_memory = gpt_descriptor.update_description_memory(
                        gpt_description_memory, changed_frame
                    )
                    gpt_desc_text = gpt_description_memory[-1]["description"] if gpt_description_memory else ""
                except Exception as e:
                    print(f"⚠️  GPT description failed at t={sec}s: {e}")

            # Live descriptor (every 2nd frame)
            live_desc_text: str = ""
            try:
                if sec % 2 == 0:
                    live_desc_memory = process_live_frame(live_desc_memory, frame, ollama_url=ollama_url)
                last_live = live_desc_memory[-1] if live_desc_memory else None
                if last_live and not last_live.get("error", False):
                    live_desc_text = last_live.get("description", "")
            except Exception as e:
                print(f"⚠️  Live descriptor error at t={sec}s: {e}")

            # Locomotion inference only at exact request seconds
            inference_triggered = sec in request_seconds
            infer_result: Optional[Dict[str, Any]] = None
            if inference_triggered:
                exact_voice_command, exact_expected_level = sec_to_voice.get(sec, (voice_command_context, expected_level_context))
                if locomotion_engine is not None:
                    infer_result = locomotion_engine.detect_locomotion_mode(
                        gpt_description=gpt_desc_text,
                        live_description=live_desc_text,
                        previous_mode=previous_mode,
                        voice_command=exact_voice_command,
                    )
                else:
                    infer_result = simple_rule_locomotion_infer(
                        voice_command=exact_voice_command,
                        gpt_description=gpt_desc_text,
                        live_description=live_desc_text,
                        previous_mode=previous_mode,
                    )
                previous_mode = infer_result.get("mode_detected", previous_mode)

                predictions.append({
                    "t_seconds": sec,
                    "timestamp": seconds_to_hhmmss(sec),
                    "voice_command": exact_voice_command,
                    "expected_level": exact_expected_level,
                    "locomotion_result": infer_result,
                })

            # Aggregate per-frame log
            record = {
                "t_seconds": sec,
                "timestamp": seconds_to_hhmmss(sec),
                "voice_command_context": voice_command_context,
                "expected_level_context": expected_level_context,
                "scene_changed": bool(scene_changed),
                "gpt_description": gpt_desc_text,
                "live_description": live_desc_text,
                "inference_triggered": inference_triggered,
                "locomotion_result": infer_result,
                "current_mode_state": previous_mode,
            }
            results.append(record)

            if sec % 10 == 0:
                print(f"⏱️  t={sec:>4}s | change={record['scene_changed']} | infer={inference_triggered} | mode={previous_mode}")

    finally:
        cap.release()

    output = {
        "video": str(Path(video_path).resolve()),
        "json": str(Path(json_path).resolve()),
        "model_name": model_name,
        "similarity_threshold": similarity_threshold,
        "sampling_fps": 1,
        "results": results,
        "predictions": predictions,
        "prediction_count": len(predictions),
        "gpt_descriptions_total": len(gpt_description_memory),
        "live_descriptions_total": len(live_desc_memory),
    }

    # Save
    if save_path is None:
        save_path = os.path.join(outputs_dir, "loco_data_4_results.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"💾 Saved results to {save_path}")

    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locomotion pipeline for loco_data_4.mp4")
    default_root = Path(__file__).resolve().parent
    default_video = default_root / "videos" / "loco_data_4.mp4"
    default_json = default_root / "data_json" / "loco_data_4.json"

    parser.add_argument("--video", type=str, default=str(default_video), help="Path to input video")
    parser.add_argument("--json", type=str, default=str(default_json), help="Path to voice command JSON")
    parser.add_argument("--model", type=str, default="mobileclip", choices=["mobileclip", "fastervit"], help="Scene change model")
    parser.add_argument("--threshold", type=float, default=0.85, help="Scene change similarity threshold")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--tensorrt", action="store_true", help="Enable TensorRT optimizations (if available)")
    parser.add_argument("--no-gpt", action="store_true", help="Disable GPT scene description")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--save", type=str, default=None, help="Output JSON path")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    run_pipeline(
        video_path=args.video,
        json_path=args.json,
        model_name=args.model,
        similarity_threshold=args.threshold,
        use_fp16=(not args.no_fp16),
        enable_tensorrt=args.tensorrt,
        enable_gpt=(not args.no_gpt),
        ollama_url=args.ollama_url,
        save_path=args.save,
    )
