"""
app.py - GyanGuru AI-Powered ML Learning Assistant
Main Flask Application
"""

import os
import sys
import logging
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, abort
)
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── App Initialization ────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "gyanguru-dev-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

GENERATED_FOLDER = os.getenv("GENERATED_FOLDER", "generated")
AUDIO_DIR = os.path.join(GENERATED_FOLDER, "audio")
IMAGE_DIR = os.path.join(GENERATED_FOLDER, "images")
CODE_DIR  = os.path.join(GENERATED_FOLDER, "code")

for d in [AUDIO_DIR, IMAGE_DIR, CODE_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── Import utility modules ────────────────────────────────────────────────────
from utils.genai_utils import (
    generate_text_explanation,
    generate_code_example,
    generate_audio_script,
    generate_image_prompts
)
from utils.audio_utils import generate_audio, cleanup_old_audio
from utils.image_utils import generate_educational_images
from utils.code_executor import save_code_file, get_colab_instructions, get_local_instructions


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/text")
def text_page():
    return render_template("text.html")

@app.route("/code")
def code_page():
    return render_template("code.html")

@app.route("/audio")
def audio_page():
    return render_template("audio.html")

@app.route("/image")
def image_page():
    return render_template("image.html")


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES — CONTENT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/generate/text", methods=["POST"])
def api_generate_text():
    """Generate structured ML concept explanation."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    depth = data.get("depth", "Comprehensive")

    if not topic:
        return jsonify({"status": "error", "message": "Topic is required"}), 400

    logger.info(f"Generating text explanation: '{topic}' [{depth}]")
    result = generate_text_explanation(topic, depth)
    return jsonify(result)


@app.route("/api/generate/code", methods=["POST"])
def api_generate_code():
    """Generate complete Python ML algorithm implementation."""
    data = request.get_json(silent=True) or {}
    algorithm = data.get("algorithm", "").strip()
    complexity = data.get("complexity", "Detailed")

    if not algorithm:
        return jsonify({"status": "error", "message": "Algorithm name is required"}), 400

    logger.info(f"Generating code: '{algorithm}' [{complexity}]")
    result = generate_code_example(algorithm, complexity)

    if result.get("status") == "success":
        # Save code file
        code = result.get("code", "")
        save_result = save_code_file(code, algorithm, CODE_DIR)
        result["filename"] = save_result.get("filename", "")
        result["syntax_valid"] = save_result.get("syntax_valid", False)
        result["line_count"] = save_result.get("line_count", 0)

        # Add run instructions
        deps = result.get("dependencies", [])
        result["colab_instructions"] = get_colab_instructions(result["filename"], deps)
        result["local_instructions"] = get_local_instructions(result["filename"], deps)

    return jsonify(result)


@app.route("/api/generate/audio", methods=["POST"])
def api_generate_audio():
    """Generate educational audio lesson (script + MP3)."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    length = data.get("length", "Medium")

    if not topic:
        return jsonify({"status": "error", "message": "Topic is required"}), 400

    logger.info(f"Generating audio: '{topic}' [{length}]")

    # Step 1: Generate script
    script_result = generate_audio_script(topic, length)
    if script_result.get("status") != "success":
        return jsonify(script_result), 500

    script = script_result.get("script", "")

    # Step 2: Convert to audio
    audio_result = generate_audio(script, topic, AUDIO_DIR)
    audio_result["script"] = script
    audio_result["topic"] = topic

    # Cleanup old files
    cleanup_old_audio(AUDIO_DIR)

    return jsonify(audio_result)


@app.route("/api/generate/image", methods=["POST"])
def api_generate_image():
    """Generate educational ML concept diagrams."""
    data = request.get_json(silent=True) or {}
    concept = data.get("concept", "").strip()
    backend = data.get("backend", "gemini")

    if not concept:
        return jsonify({"status": "error", "message": "Concept is required"}), 400

    logger.info(f"Generating images: '{concept}' [{backend}]")

    # Step 1: Get image prompts
    prompt_result = generate_image_prompts(concept)
    if prompt_result.get("status") != "success":
        return jsonify(prompt_result), 500

    prompts = prompt_result.get("prompts", [])

    # Step 2: Generate images
    image_result = generate_educational_images(prompts, concept, backend, IMAGE_DIR)
    image_result["prompts"] = prompts
    image_result["concept"] = concept

    return jsonify(image_result)


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/download/audio/<filename>")
def download_audio(filename):
    """Serve MP3 audio file for download."""
    safe = _safe_filename(filename)
    if not safe:
        abort(400)
    return send_from_directory(AUDIO_DIR, safe, as_attachment=True)


@app.route("/download/code/<filename>")
def download_code(filename):
    """Serve Python code file for download."""
    safe = _safe_filename(filename)
    if not safe:
        abort(400)
    return send_from_directory(CODE_DIR, safe, as_attachment=True)


@app.route("/download/image/<filename>")
def download_image(filename):
    """Serve generated image for download."""
    safe = _safe_filename(filename)
    if not safe:
        abort(400)
    return send_from_directory(IMAGE_DIR, safe, as_attachment=True)


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_filename(filename: str) -> str:
    """Validate filename to prevent path traversal."""
    from werkzeug.utils import secure_filename
    safe = secure_filename(filename)
    # Ensure only alphanumeric, dash, underscore, dot
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not all(c in allowed_chars for c in safe):
        return ""
    return safe


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"🚀 GyanGuru starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
