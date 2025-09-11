import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import cv2
from frame_processor import FrameProcessor

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
for directory in (VIDEOS_DIR, OUTPUT_DIR, TEMPLATES_DIR):
	directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


def get_available_videos() -> List[str]:
	"""Get list of video files in the videos directory."""
	video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
	videos = []
	for file_path in VIDEOS_DIR.iterdir():
		if file_path.is_file() and file_path.suffix.lower() in video_extensions:
			videos.append(file_path.name)
	return sorted(videos)


def extract_frames(video_path: Path, target_fps: float, dest_dir: Path) -> Tuple[int, float]:
	"""Extract frames from video at approximately target_fps.

	Returns (num_saved, source_fps).
	"""
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise RuntimeError("Failed to open video.")

	source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
	if source_fps <= 0:
		# Fallback if FPS not available. Assume 30 for step calculation.
		source_fps = 30.0

	if target_fps <= 0:
		target_fps = 1.0

	# Determine frame step; at least 1
	step = max(1, int(round(source_fps / target_fps)))

	dest_dir.mkdir(parents=True, exist_ok=True)

	frame_index = 0
	saved_count = 0

	while True:
		success, frame = capture.read()
		if not success:
			break
		if frame_index % step == 0:
			filename = f"frame_{frame_index:06d}.jpg"
			save_path = dest_dir / filename
			cv2.imwrite(str(save_path), frame)
			saved_count += 1
		frame_index += 1

	capture.release()
	return saved_count, source_fps


@app.route("/", methods=["GET", "POST"])
def index():
	available_videos = get_available_videos()
	
	if request.method == "POST":
		video_name = request.form.get("video")
		fps_value = request.form.get("fps", type=int)
		
		if not video_name:
			flash("Please select a video file.")
			return redirect(url_for("index"))
		if not fps_value or fps_value not in [1, 5, 10]:
			flash("Please select a valid FPS (1, 5, or 10).")
			return redirect(url_for("index"))

		video_path = VIDEOS_DIR / video_name
		if not video_path.exists():
			flash("Selected video file not found.")
			return redirect(url_for("index"))

		# Output dir: base name + target fps + unique id
		unique_id = uuid.uuid4().hex[:8]
		base_stem = video_path.stem
		out_dir = OUTPUT_DIR / f"{base_stem}_at_{fps_value}fps_{unique_id}"

		try:
			saved_count, source_fps = extract_frames(video_path, float(fps_value), out_dir)
		except Exception as exc:
			flash(f"Error processing video: {exc}")
			return redirect(url_for("index"))

		return redirect(url_for(
			"result",
			dirname=out_dir.name,
			saved=saved_count,
			target_fps=fps_value,
			source_fps=f"{source_fps:.2f}",
			video_name=video_name,
		))

	return render_template("index.html", videos=available_videos)


@app.route("/video/<filename>")
def serve_video(filename):
	"""Serve video files from the videos directory."""
	video_path = VIDEOS_DIR / filename
	if not video_path.exists():
		return "Video not found", 404
	return send_file(video_path)


@app.route("/result")
def result():
	dirname = request.args.get("dirname")
	saved = request.args.get("saved")
	target_fps = request.args.get("target_fps")
	source_fps = request.args.get("source_fps")
	video_name = request.args.get("video_name")

	output_path = OUTPUT_DIR / (dirname or "")
	frames_dir_exists = output_path.exists()

	return render_template(
		"result.html",
		dirname=dirname,
		output_path=str(output_path),
		saved=saved,
		target_fps=target_fps,
		source_fps=source_fps,
		video_name=video_name,
		frames_dir_exists=frames_dir_exists,
	)


@app.route("/frames/<dirname>")
def view_frames(dirname):
	"""View frames with AI-generated descriptions."""
	frames_dir = OUTPUT_DIR / dirname
	if not frames_dir.exists():
		flash("Frames directory not found.")
		return redirect(url_for("index"))
	
	# Check if descriptions are already generated
	json_file = BASE_DIR / "processed_frames.json"
	processor = FrameProcessor()
	
	frame_data = processor.load_processed_data(str(json_file))
	
	# Filter data for this specific directory
	filtered_data = [frame for frame in frame_data if dirname in frame.get('path', '')]
	
	return render_template(
		"frames.html",
		frames=filtered_data,
		dirname=dirname,
		has_descriptions=len(filtered_data) > 0
	)


@app.route("/generate_descriptions/<dirname>")
def generate_descriptions(dirname):
	"""Generate descriptions for frames using Moondream."""
	frames_dir = OUTPUT_DIR / dirname
	if not frames_dir.exists():
		return jsonify({"error": "Frames directory not found"}), 404
	
	try:
		processor = FrameProcessor()
		json_file = BASE_DIR / "processed_frames.json"
		
		# Load existing data
		existing_data = processor.load_processed_data(str(json_file))
		
		# Process new frames
		new_data = processor.process_frames_directory(str(frames_dir))
		
		# Merge with existing data (remove old entries for this directory)
		filtered_existing = [frame for frame in existing_data if dirname not in frame.get('path', '')]
		all_data = filtered_existing + new_data
		
		# Save updated data
		with open(json_file, 'w', encoding='utf-8') as f:
			json.dump(all_data, f, indent=2, ensure_ascii=False)
		
		return jsonify({
			"success": True,
			"message": f"Generated descriptions for {len(new_data)} frames",
			"frames": new_data
		})
		
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.route("/frame/<path:frame_path>")
def serve_frame(frame_path):
	"""Serve frame images."""
	frame_file = BASE_DIR / frame_path
	if not frame_file.exists():
		return "Frame not found", 404
	return send_file(frame_file)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
