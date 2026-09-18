"""
app.py
Flask backend for the CRC WSI analysis frontend.

Run with:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import uuid
import threading

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference_runner import run_job

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 * 1024  # 5 GB max upload

JOBS = {}  # job_id -> state dict, in-memory (fine for a single-user local demo)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "slide" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["slide"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Upload a .svs, .tif, .tiff, or .ndpi file."}), 400

    job_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
    file.save(save_path)

    JOBS[job_id] = {
        "phase": "queued",
        "total_patches": 0,
        "processed": 0,
        "normal_count": 0,
        "tumor_count": 0,
        "stage_counts": {},
        "result": None,
        "error": None,
        "slide_name": filename,
    }

    thread = threading.Thread(target=run_job, args=(job_id, save_path, JOBS), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    state = JOBS.get(job_id)
    if state is None:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(state)


if __name__ == "__main__":
    print("\nCRC WSI Analyzer running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
