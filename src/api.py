import os
import re
import tempfile
import traceback

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from src.asr import run_asr
from src.config_loader import load_config
from src.diarization import get_transcript, init_diarization, run_diarization, save_transcript
from src.manifest import create_manifest
from src.preprocessing import prepare_audio

# Initialize Flask app
app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "ogg"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
DATA_DIR = "data"


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_transcript(file_path):
    """Parse transcript file into structured format."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"\[(\d+:\d+\.\d+) - (\d+:\d+\.\d+)\]\s+(\w+):\s+(.*)", line)
            if match:
                start, end, speaker, text = match.groups()
                results.append(
                    {
                        "speaker": speaker,
                        "start_time": start,
                        "end_time": end,
                        "transcription": text,
                    }
                )
    return results


def run_pipeline(audio_file_path, output_dir=None):
    """
    Run the complete MOMbot pipeline on an audio file.

    Args:
        audio_file_path (str): Path to the input audio file
        output_dir (str): Directory to save outputs (default: temp directory)

    Returns:
        dict: Pipeline results with transcript JSON
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]

    try:
        print(f"Processing audio file: {audio_file_path}")

        # 1. Preprocess: Convert to mono
        mono_audio_path, signal, sample_rate = prepare_audio(audio_file_path, output_dir=output_dir)

        # 2. Load config
        cfg = load_config(data_dir=output_dir, domain_type="meeting")

        # 3. Create manifest
        manifest_path = create_manifest(mono_audio_path, output_dir=output_dir)
        cfg.diarizer.manifest_filepath = manifest_path

        # 4. Run ASR
        sentence_hyp, sentence_ts_hyp = run_asr(cfg, level="sentence")

        # 5. Initialize diarization
        asr_diar_offline = init_diarization(cfg)

        # 6. Run diarization
        diar_hyp, diar_score, asr_diar_offline = run_diarization(
            cfg, sentence_ts_hyp, asr_diar_offline
        )

        # 7. Extract transcript
        transcript_df = get_transcript(
            asr_diar_offline, diar_hyp, sentence_hyp, sentence_ts_hyp, return_df=True
        )

        # 8. Save transcript
        extract_path = save_transcript(base_name=base_name, data_dir=output_dir)

        # 9. Parse transcript for JSON response
        parsed_transcript = parse_transcript(extract_path)

        return {
            "status": "success",
            "transcript": parsed_transcript,
            "metadata": {
                "num_segments": len(parsed_transcript),
                "speakers": list(set([seg["speaker"] for seg in parsed_transcript])),
                "audio_file": audio_file_path,
            },
        }

    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "MOMbot Audio Transcription API",
            "version": "1.0.0",
            "description": "Convert audio files to JSON transcript with speaker diarization",
        }
    )


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Process an uploaded audio file and return JSON transcript.

    Form data:
    - file: Audio file upload (required)
    """
    try:
        # Check if file is in request
        if "file" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        file = request.files["file"]

        # Check if file is selected
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        if not allowed_file(file.filename):
            return (
                jsonify(
                    {"error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}
                ),
                400,
            )

        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()

        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_audio_path = os.path.join(temp_dir, filename)
        file.save(temp_audio_path)

        # Run pipeline
        result = run_pipeline(temp_audio_path, temp_dir)

        # Clean up uploaded file
        try:
            os.remove(temp_audio_path)
        except:
            pass

        if result["status"] == "error":
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return (
            jsonify({"error": f"Processing error: {str(e)}", "traceback": traceback.format_exc()}),
            500,
        )


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return (
        jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"}),
        413,
    )


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "available_endpoints": ["GET /", "POST /transcribe", "GET /help"],
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return (
        jsonify(
            {
                "error": "Internal server error",
                "message": "Please check the server logs for details",
            }
        ),
        500,
    )


# Configure Flask app
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


@app.route("/help", methods=["GET"])
def api_help():
    """API documentation endpoint."""
    return jsonify(
        {
            "MOMbot API Documentation": {
                "description": "AI-powered audio transcription service",
                "endpoints": {
                    "GET /": {"description": "Health check and service info", "parameters": "None"},
                    "POST /transcribe": {
                        "description": "Upload audio file and get JSON transcript",
                        "content_type": "multipart/form-data",
                        "parameters": {"file": "file (required) - Audio file upload"},
                        "example": "curl -X POST -F 'file=@meeting.wav' http://localhost:5000/transcribe",
                    },
                    "GET /help": {"description": "This documentation", "parameters": "None"},
                },
                "supported_formats": list(ALLOWED_EXTENSIONS),
                "max_file_size": f"{MAX_FILE_SIZE // (1024*1024)}MB",
                "response_format": {
                    "status": "success|error",
                    "transcript": "array of transcript segments with speaker identification",
                    "metadata": "object with processing information",
                },
            }
        }
    )


if __name__ == "__main__":
    print("=" * 50)
    print("MOMbot Transcription API")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET  /          - Health check")
    print("  POST /transcribe - Upload audio & get JSON transcript")
    print("  GET  /help      - API documentation")
    print("=" * 50)

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    app.run(host="0.0.0.0", port=5000, debug=True)
