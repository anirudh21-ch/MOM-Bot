import os
import re
import tempfile
import traceback
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# Load environment variables from .env file
load_dotenv()

from src.audio_processing import run_asr, load_config
from src.diarization import get_transcript, init_diarization, run_diarization, save_transcript
from src.diarization.diarize import diarize_audio
from src.openai_integration import summarize_transcript
from src.manifest import create_manifest
from src.preprocessing import prepare_audio


def create_fallback_transcript(sentence_hyp, sentence_ts_hyp):
    """
    Create a simple transcript without speaker diarization when diarization fails.
    """
    transcript_data = []
    
    # Handle different timestamp formats
    try:
        if isinstance(sentence_ts_hyp, list) and len(sentence_ts_hyp) > 0:
            for i, text in enumerate(sentence_hyp):
                if i < len(sentence_ts_hyp):
                    # Handle various timestamp formats
                    ts = sentence_ts_hyp[i]
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        start_time, end_time = ts[0], ts[1]
                    elif hasattr(ts, 'start') and hasattr(ts, 'end'):
                        start_time, end_time = ts.start, ts.end
                    else:
                        # Fallback: estimate timestamps
                        start_time = i * 3.0  # Assume 3 seconds per sentence
                        end_time = (i + 1) * 3.0
                else:
                    start_time = i * 3.0
                    end_time = (i + 1) * 3.0
                    
                transcript_data.append({
                    "speaker": "SPEAKER_00",
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "transcription": str(text).strip()
                })
        else:
            # Simple fallback when no timestamps available
            for i, text in enumerate(sentence_hyp):
                transcript_data.append({
                    "speaker": "SPEAKER_00",
                    "start_time": float(i * 3.0),
                    "end_time": float((i + 1) * 3.0),
                    "transcription": str(text).strip()
                })
    
    except Exception as e:
        print(f"Timestamp processing error: {e}")
        # Ultra-simple fallback
        full_text = " ".join(str(text) for text in sentence_hyp)
        transcript_data.append({
            "speaker": "SPEAKER_00",
            "start_time": 0.0,
            "end_time": 30.0,  # Default duration
            "transcription": full_text
        })
    
    return pd.DataFrame(transcript_data)


# Initialize Flask app
app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "ogg"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
DATA_DIR = "data"

# Global variable to store the last processed transcript
last_transcript = []
# Global variable to store the last output directory
last_output_directory = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_transcript(file_path):
    """Parse transcript file into structured format."""
    results = []
    current_time = 0.0
    
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Try format with timestamps: [MM:SS.mmm - MM:SS.mmm] SPEAKER: text
            timestamp_match = re.match(r"\[(\d+:\d+\.\d+) - (\d+:\d+\.\d+)\]\s+(\w+):\s+(.*)", line)
            if timestamp_match:
                start, end, speaker, text = timestamp_match.groups()
                start_seconds = convert_time_to_seconds(start)
                end_seconds = convert_time_to_seconds(end)
                results.append({
                    "speaker": speaker,
                    "start_time": start_seconds,
                    "end_time": end_seconds,
                    "text": text,
                    "transcription": text,
                })
                current_time = end_seconds
                continue
            
            # Try simple format: speaker_X: text (current format)
            speaker_match = re.match(r"(speaker_\d+):\s*(.*)", line)
            if speaker_match:
                speaker, text = speaker_match.groups()
                # Estimate duration based on text length (roughly 3 words per second)
                words = len(text.split())
                duration = max(2.0, words / 3.0)  # Minimum 2 seconds
                
                results.append({
                    "speaker": speaker.upper(),
                    "start_time": current_time,
                    "end_time": current_time + duration,
                    "text": text,
                    "transcription": text,
                })
                current_time += duration
                continue
            
            # Try simple format: SPEAKER: text
            simple_match = re.match(r"([A-Z_]+\d*):\s*(.*)", line)
            if simple_match:
                speaker, text = simple_match.groups()
                words = len(text.split())
                duration = max(2.0, words / 3.0)
                
                results.append({
                    "speaker": speaker,
                    "start_time": current_time,
                    "end_time": current_time + duration,
                    "text": text,
                    "transcription": text,
                })
                current_time += duration
    
    return results


def convert_time_to_seconds(time_str):
    """Convert MM:SS.mmm format to seconds."""
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return 0.0


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

                # Load configuration
        cfg = load_config(data_dir=output_dir, domain_type="meeting")
        
        # Ensure device parameters are available for NeMo models
        try:
            from omegaconf import OmegaConf
            # Temporarily disable struct mode to allow device parameter addition
            OmegaConf.set_struct(cfg, False)
            
            # Set device parameters using direct assignment since struct mode is disabled
            cfg.diarizer.device = "auto"
            if hasattr(cfg.diarizer, 'vad'):
                cfg.diarizer.vad.device = "auto"
            if hasattr(cfg.diarizer, 'clustering'):  
                cfg.diarizer.clustering.device = "auto"
            if hasattr(cfg.diarizer, 'speaker_embeddings'):
                cfg.diarizer.speaker_embeddings.device = "auto"
            
            # Re-enable struct mode
            OmegaConf.set_struct(cfg, True)
            print("Successfully configured device parameters for NeMo models")
            
        except Exception as e:
            print(f"Warning: Could not set device parameters: {e}")
            # Try fallback approach without struct mode manipulation
            try:
                cfg.diarizer.device = "auto"
                print("Applied fallback device configuration")
            except:
                print("Could not apply any device configuration")

        # 2. Create manifest
        manifest_path = create_manifest(mono_audio_path, output_dir=output_dir)
        cfg.diarizer.manifest_filepath = manifest_path

        # 4. Run ASR with debugging
        print("🎤 Running ASR transcription...")
        sentence_hyp, sentence_ts_hyp = run_asr(cfg, level="sentence")
        
        print(f"📝 ASR Results:")
        print(f"   - Sentences detected: {len(sentence_hyp) if sentence_hyp else 0}")
        print(f"   - Timestamps detected: {len(sentence_ts_hyp) if sentence_ts_hyp else 0}")
        
        if sentence_hyp:
            # Handle different data types safely
            try:
                if isinstance(sentence_hyp, (list, tuple)):
                    print(f"   - First few transcriptions: {sentence_hyp[:3] if len(sentence_hyp) > 0 else 'empty'}")
                elif isinstance(sentence_hyp, dict):
                    items = list(sentence_hyp.items())[:3]
                    print(f"   - First few transcriptions: {items}")
                else:
                    print(f"   - Transcription type: {type(sentence_hyp)}")
                    print(f"   - Transcription content: {str(sentence_hyp)[:200]}...")
            except Exception as print_error:
                print(f"   - Transcription data available (type: {type(sentence_hyp)})")
                print(f"   - Print error: {print_error}")
        else:
            print("   ⚠️ NO AUDIO TRANSCRIPTION DETECTED!")
            print("   📁 Checking audio file...")
            import librosa
            import numpy as np
            try:
                audio_check, sr = librosa.load(mono_audio_path, sr=None)
                duration = len(audio_check) / sr
                max_amplitude = np.max(np.abs(audio_check)) if len(audio_check) > 0 else 0
                print(f"   📊 Audio file stats:")
                print(f"      - Duration: {duration:.2f} seconds")
                print(f"      - Sample rate: {sr} Hz")
                print(f"      - Max amplitude: {max_amplitude:.4f}")
                print(f"      - Samples: {len(audio_check)}")
                
                if max_amplitude < 0.001:
                    print("   ⚠️ Audio amplitude very low - might be silent!")
                elif duration < 1.0:
                    print("   ⚠️ Audio too short for transcription!")
                else:
                    print("   ✅ Audio seems valid - ASR model issue?")
            except Exception as audio_error:
                print(f"   ❌ Could not analyze audio: {audio_error}")

        # 5. Initialize diarization
        asr_diar_offline = init_diarization(cfg)

        # Check if we have valid ASR results before diarization
        if not sentence_hyp or len(sentence_hyp) == 0:
            return {
                "status": "error",
                "error": "No speech detected in audio file",
                "details": "The ASR model could not detect any speech in the uploaded audio. Please check that:",
                "suggestions": [
                    "The audio file contains clear speech",
                    "The audio is not silent or too quiet",
                    "The file format is supported (wav, mp3, m4a, etc.)",
                    "The audio quality is sufficient for transcription"
                ]
            }
        
        # 6. Run REAL diarization using notebook approach - NO FALLBACK
        print("🎯 RUNNING NOTEBOOK-STYLE SPEAKER DIARIZATION - No fallback allowed!")
        diarization_results = diarize_audio(
            audio_file=audio_file_path,  # Use original audio file path
            num_speakers=None  # Let it auto-detect
        )
        
        print(f"Diarization completed successfully: {diarization_results['success']}")
        print(f"Number of speakers detected: {len(diarization_results['speakers'])}")
        
        # 7. Process results from notebook-style diarization
        if diarization_results['success']:
            # Get transcript text lines
            transcript_lines = diarization_results.get('transcript_text', [])
            
            # Save transcript 
            extract_path = os.path.join(output_dir, "extract.txt")
            with open(extract_path, "w") as f:
                for line in transcript_lines:
                    f.write(line + "\n")
            print(f"Transcript saved to {extract_path}")
            
        else:
            # Fallback - should not happen with notebook approach
            extract_path = os.path.join(output_dir, "extract.txt")
            with open(extract_path, "w") as f:
                f.write("Error: Diarization failed\n")

        # 8. Parse transcript for JSON response
        parsed_transcript = parse_transcript(extract_path)
        
        # 9. Save JSON transcript file
        json_path = os.path.join(output_dir, "transcript.json")
        import json
        transcript_data = {
            "status": "success",
            "transcript": parsed_transcript,
            "metadata": {
                "num_segments": len(parsed_transcript),
                "speakers": list(set([seg["speaker"] for seg in parsed_transcript])),
                "audio_file": audio_file_path,
                "processing_date": str(pd.Timestamp.now()),
            },
        }
        
        with open(json_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        print(f"JSON transcript saved to {json_path}")
        
        # 10. Also ensure we have the .txt file (extract.txt is already created above)
        print(f"TXT transcript saved to {extract_path}")

        # Return both file paths and transcript data
        result = transcript_data.copy()
        result["files"] = {
            "txt_file": extract_path,
            "json_file": json_path
        }
        result["output_directory"] = output_dir  # Store output directory for summary saving
        
        return result

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

        # Store transcript globally for /transcript endpoint
        global last_transcript, last_output_directory
        if result["status"] == "success":
            last_transcript = result["transcript"]
            last_output_directory = result.get("output_directory")

        # Clean up uploaded file (but keep generated files for download)
        try:
            os.remove(temp_audio_path)
        except:
            pass

        if result["status"] == "error":
            return jsonify(result), 500

        # Prepare response with download links
        response_data = {
            "status": result["status"],
            "transcript": result["transcript"],
            "metadata": result["metadata"],
            "downloads": {
                "json_file": "/transcript/download/json",
                "txt_file": "/transcript/download/txt",
                "description": "Use these endpoints to download the transcript files"
            },
            "message": "Transcription completed successfully. Use the download links to get TXT and JSON files."
        }

        return jsonify(response_data)

    except Exception as e:
        return (
            jsonify({"error": f"Processing error: {str(e)}", "traceback": traceback.format_exc()}),
            500,
        )


@app.route("/transcript", methods=["GET"])
def get_transcript():
    """
    Get the last processed transcript.
    
    Returns:
        JSON: The transcript from the last successful transcription, or empty array if none.
    """
    global last_transcript
    return jsonify(last_transcript)


@app.route("/transcript/download/json", methods=["GET"])
def download_transcript_json():
    """
    Download the last processed transcript as a JSON file.
    
    Returns:
        File: JSON file download of the transcript.
    """
    global last_transcript
    if not last_transcript:
        return jsonify({"error": "No transcript available to download"}), 400
    
    from flask import send_file
    import tempfile
    import json
    
    # Create a temporary JSON file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    transcript_data = {
        "status": "success",
        "transcript": last_transcript,
        "metadata": {
            "num_segments": len(last_transcript),
            "speakers": list(set([seg.get("speaker", "Unknown") for seg in last_transcript])),
            "download_date": str(pd.Timestamp.now()),
        }
    }
    
    json.dump(transcript_data, temp_file, indent=2)
    temp_file.close()
    
    return send_file(temp_file.name, as_attachment=True, download_name="transcript.json", mimetype="application/json")


@app.route("/transcript/download/txt", methods=["GET"])
def download_transcript_txt():
    """
    Download the last processed transcript as a TXT file.
    
    Returns:
        File: TXT file download of the transcript.
    """
    global last_transcript
    if not last_transcript:
        return jsonify({"error": "No transcript available to download"}), 400
    
    from flask import send_file
    import tempfile
    
    # Create a temporary TXT file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    
    # Write transcript in readable format
    for segment in last_transcript:
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', segment.get('transcription', ''))
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)
        
        # Format time as MM:SS
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)
        
        temp_file.write(f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] {speaker}: {text}\n")
    
    temp_file.close()
    
    return send_file(temp_file.name, as_attachment=True, download_name="transcript.txt", mimetype="text/plain")


@app.route("/summary/download", methods=["GET"])
def download_summary():
    """
    Download the last generated summary as a TXT file.
    
    Returns:
        File: TXT file download of the summary.
    """
    global last_summary
    if not last_summary:
        return "Error: No summary available to download. Generate a summary first using /summary endpoint.", 400
    
    from flask import send_file
    import tempfile
    
    # Create a temporary TXT file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    
    # Write comprehensive summary
    temp_file.write(f"MEETING SUMMARY\n")
    temp_file.write(f"Generated: {last_summary['generated_at']}\n")
    temp_file.write(f"Transcript segments: {last_summary['transcript_segments']}\n")
    temp_file.write(f"{'='*50}\n\n")
    temp_file.write(last_summary['text'])
    
    temp_file.close()
    
    return send_file(temp_file.name, as_attachment=True, download_name="meeting_summary.txt", mimetype="text/plain")


# Global variable to store the last summary
last_summary = None

@app.route("/summary", methods=["GET", "POST"])
def get_summary():
    """
    Generate a summary of the transcript using OpenAI.
    
    GET: Summarize the last processed JSON transcript
    POST: Summarize a provided JSON transcript data
    
    Returns:
        TEXT: Generated summary in plain text format.
    """
    global last_transcript, last_summary, last_output_directory
    
    transcript_to_summarize = None
    
    if request.method == "POST":
        # Handle POST request with JSON transcript data
        if request.is_json:
            data = request.get_json()
            transcript_to_summarize = data.get('transcript', [])
        elif 'json_file' in request.files:
            # Handle JSON file upload
            file = request.files['json_file']
            if file and file.filename.endswith('.json'):
                try:
                    import json
                    content = file.read().decode('utf-8')
                    json_data = json.loads(content)
                    transcript_to_summarize = json_data.get('transcript', [])
                except Exception as e:
                    return f"Error: Failed to parse JSON file: {str(e)}", 400
        else:
            # Use last transcript if no data provided in POST
            transcript_to_summarize = last_transcript
    else:
        # GET request - use last transcript JSON data
        transcript_to_summarize = last_transcript
    
    if not transcript_to_summarize:
        return "Error: No transcript available. Please upload and transcribe an audio file first, or provide JSON transcript data in the request.", 400
    
    try:
        from src.openai_integration import summarize_transcript
        summary_data = summarize_transcript(transcript_to_summarize)
        
        # Format summary as readable text
        if isinstance(summary_data, dict):
            summary_text = format_summary_as_text(summary_data)
        else:
            summary_text = str(summary_data)
        
        # Store summary globally for download
        last_summary = {
            "text": summary_text,
            "structured_data": summary_data,
            "transcript_segments": len(transcript_to_summarize),
            "generated_at": str(pd.Timestamp.now())
        }
        
        # Save summary to the same directory as transcript files
        output_dir = last_output_directory
        if not output_dir or not os.path.exists(output_dir):
            # Fallback: Create a data directory for outputs if it doesn't exist
            output_dir = os.path.join(os.getcwd(), "data", "outputs")
            os.makedirs(output_dir, exist_ok=True)
        
        summary_file_path = os.path.join(output_dir, "summary.txt")
        
        # Write comprehensive summary with metadata
        try:
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                f.write(f"MEETING SUMMARY\n")
                f.write(f"Generated: {last_summary['generated_at']}\n")
                f.write(f"Transcript segments: {last_summary['transcript_segments']}\n")
                f.write(f"{'='*50}\n\n")
                f.write(summary_text)
                f.write(f"\n\n{'='*50}\n")
                f.write(f"File saved at: {summary_file_path}\n")
            
            print(f"✅ Summary successfully saved to: {summary_file_path}")
        except Exception as e:
            print(f"❌ Failed to save summary: {e}")
            summary_file_path = "Failed to save file"
        
        # Return only the clean summary text
        return summary_text, 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        error_text = f"Error: Failed to generate summary: {str(e)}\n\nPlease check that:\n- OpenAI API key is configured in .env file\n- Internet connection is available\n- Transcript data is valid JSON format"
        return error_text, 500, {'Content-Type': 'text/plain'}


def format_summary_as_text(summary_data: Dict[str, Any]) -> str:
    """
    Format structured summary data as readable text.
    
    Args:
        summary_data: Dictionary containing summary information
        
    Returns:
        Formatted text summary
    """
    text_parts = []
    
    # Executive Summary
    if 'executive_summary' in summary_data:
        text_parts.append("=== EXECUTIVE SUMMARY ===")
        text_parts.append(summary_data['executive_summary'])
        text_parts.append("")
    
    # Key Points
    if 'key_points' in summary_data and summary_data['key_points']:
        text_parts.append("=== KEY DISCUSSION POINTS ===")
        for i, point in enumerate(summary_data['key_points'], 1):
            text_parts.append(f"{i}. {point}")
        text_parts.append("")
    
    # Decisions Made
    if 'decisions' in summary_data and summary_data['decisions']:
        text_parts.append("=== DECISIONS MADE ===")
        for i, decision in enumerate(summary_data['decisions'], 1):
            text_parts.append(f"{i}. {decision}")
        text_parts.append("")
    
    # Action Items
    if 'action_items' in summary_data and summary_data['action_items']:
        text_parts.append("=== ACTION ITEMS ===")
        for i, action in enumerate(summary_data['action_items'], 1):
            text_parts.append(f"{i}. {action}")
        text_parts.append("")
    
    # Participants
    if 'participants' in summary_data and summary_data['participants']:
        text_parts.append("=== PARTICIPANTS ===")
        text_parts.append(", ".join(summary_data['participants']))
        text_parts.append("")
    
    # Next Steps
    if 'next_steps' in summary_data and summary_data['next_steps']:
        text_parts.append("=== NEXT STEPS ===")
        for i, step in enumerate(summary_data['next_steps'], 1):
            text_parts.append(f"{i}. {step}")
        text_parts.append("")
    
    # Fallback for simple summary text
    if not text_parts and 'summary_text' in summary_data:
        text_parts.append("=== MEETING SUMMARY ===")
        text_parts.append(summary_data['summary_text'])
    
    return "\n".join(text_parts)


def parse_extract_file_content(content: str) -> List[Dict[str, Any]]:
    """
    Parse extract.txt file content to create transcript structure.
    
    Args:
        content: Raw text content from extract.txt
        
    Returns:
        List of transcript segments
    """
    transcript = []
    lines = content.strip().split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('='):
            # Try to parse speaker and text
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker_part = parts[0].strip()
                    text_part = parts[1].strip()
                    
                    # Extract timestamp if present [MM:SS]
                    start_time = 0
                    if '[' in speaker_part and ']' in speaker_part:
                        try:
                            time_str = speaker_part[speaker_part.find('[')+1:speaker_part.find(']')]
                            if ':' in time_str:
                                minutes, seconds = map(int, time_str.split(':'))
                                start_time = minutes * 60 + seconds
                            speaker_part = speaker_part.split(']')[-1].strip()
                        except:
                            pass
                    
                    transcript.append({
                        'speaker': speaker_part,
                        'text': text_part,
                        'start_time': start_time,
                        'end_time': start_time + 5,  # Approximate
                        'segment_id': i
                    })
            else:
                # Line without speaker, treat as continuation
                if transcript:
                    transcript[-1]['text'] += ' ' + line
                else:
                    transcript.append({
                        'speaker': 'Unknown',
                        'text': line,
                        'start_time': 0,
                        'end_time': 5,
                        'segment_id': i
                    })
    
    return transcript


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
                "available_endpoints": ["GET /", "POST /transcribe", "GET /transcript", "GET /summary", "GET /help"],
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
                    "GET /transcript": {
                        "description": "Get the last processed transcript",
                        "parameters": "None",
                        "example": "curl http://localhost:5000/transcript",
                    },
                    "GET /transcript/download/json": {
                        "description": "Download the last processed transcript as JSON file",
                        "parameters": "None",
                        "example": "curl -O http://localhost:5000/transcript/download/json",
                    },
                    "GET /transcript/download/txt": {
                        "description": "Download the last processed transcript as TXT file",
                        "parameters": "None",
                        "example": "curl -O http://localhost:5000/transcript/download/txt",
                    },
                    "GET /summary": {
                        "description": "Generate AI summary from JSON transcript data (returns plain text)",
                        "parameters": "None - uses last processed transcript",
                        "example": "curl http://localhost:5000/summary",
                        "note": "Requires OpenAI API key to be configured",
                    },
                    "POST /summary": {
                        "description": "Generate AI summary from provided JSON transcript data",
                        "content_type": "application/json or multipart/form-data",
                        "parameters": {"transcript": "array - JSON transcript data", "json_file": "file - JSON transcript file"},
                        "example": "curl -X POST -H 'Content-Type: application/json' -d '{\"transcript\": [...]}' http://localhost:5000/summary",
                    },
                    "GET /summary/download": {
                        "description": "Download the last generated summary as TXT file",
                        "parameters": "None",
                        "example": "curl -O http://localhost:5000/summary/download",
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
    print("  GET  /transcript - Get last processed transcript")
    print("  GET  /transcript/download/json - Download transcript JSON file")
    print("  GET  /transcript/download/txt  - Download transcript TXT file")
    print("  GET  /summary    - Generate AI summary from JSON transcript (TXT output)")
    print("  GET  /summary/download - Download summary TXT file")
    print("  GET  /help      - API documentation")
    print("=" * 50)

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Use PORT environment variable or default to 5000
    port = int(os.getenv('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)