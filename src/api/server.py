"""MOM-Bot REST API Endpoint

Simple Flask API for the speech processing pipeline.
Handles audio file uploads and returns transcription with diarization.

Run: python src/api/server.py
     or gunicorn -w 4 -b 0.0.0.0:5000 src.api.server:app
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
import logging
from datetime import datetime
import traceback

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import MOMBotPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent.parent.parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent.parent.parent / "output"
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Create folders
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = str(OUTPUT_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'MOM-Bot Speech Processing Pipeline',
        'version': '1.0',
        'endpoints': {
            'POST /process': 'Process audio file',
            'GET /models': 'List available Whisper models',
            'GET /health': 'Health check'
        }
    }), 200


@app.route('/models', methods=['GET'])
def get_models():
    """Get available Whisper models."""
    return jsonify({
        'models': [
            {
                'name': 'tiny',
                'size': '39M',
                'ram': '~1GB',
                'speed': 'Very Fast',
                'quality': 'Good',
                'recommended_for': '15-20 min audio'
            },
            {
                'name': 'base',
                'size': '74M',
                'ram': '~2GB',
                'speed': 'Fast',
                'quality': 'Very Good',
                'recommended_for': '10-15 min audio'
            },
            {
                'name': 'small',
                'size': '244M',
                'ram': '~4GB',
                'speed': 'Moderate',
                'quality': 'Excellent',
                'recommended_for': '5-10 min audio'
            },
            {
                'name': 'medium',
                'size': '769M',
                'ram': '~8GB',
                'speed': 'Slow',
                'quality': 'Best',
                'recommended_for': 'High accuracy, GPU recommended'
            },
            {
                'name': 'large',
                'size': '1.5B',
                'ram': '~10GB',
                'speed': 'Very Slow',
                'quality': 'Best+',
                'recommended_for': 'Maximum accuracy, GPU required'
            }
        ]
    }), 200


@app.route('/process', methods=['POST'])
def process_audio():
    """
    Process audio file and return transcription with diarization.
    
    Request:
        - audio_file: Audio file (multipart/form-data)
        - model: Whisper model (optional, default: tiny)
        - num_speakers: Number of speakers (optional, auto-detect if omitted)
    
    Response:
        - status: 'success' or 'error'
        - file: Original filename
        - duration: Audio duration in seconds
        - speakers: Number of speakers detected
        - segments: List of transcribed segments with speaker attribution
        - language: Detected language code
        - output_file: Path to saved JSON results
        - processing_time: Time taken in seconds
    """
    try:
        # Check if file is in request
        if 'audio_file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No audio file provided. Use "audio_file" field.'
            }), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get parameters
        model = request.form.get('model', 'tiny')
        num_speakers = request.form.get('num_speakers', type=int)
        
        # Validate model
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if model not in valid_models:
            return jsonify({
                'status': 'error',
                'message': f'Invalid model. Choose from: {", ".join(valid_models)}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(filepath))
        
        logger.info(f"Processing file: {filename}")
        
        # Process audio
        start_time = datetime.now()
        pipeline = MOMBotPipeline(
            whisper_model_size=model,
            num_speakers=num_speakers
        )
        results = pipeline.process(str(filepath))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if 'error' in results:
            return jsonify({
                'status': 'error',
                'message': results['error'],
                'file': filename
            }), 400
        
        # Prepare response
        response = {
            'status': 'success',
            'file': filename,
            'original_filename': file.filename,
            'duration': results.get('duration', 0),
            'speakers': results.get('num_speakers', 0),
            'segments': results.get('segments', []),
            'language': results.get('language', 'unknown'),
            'processing_time': round(processing_time, 2),
            'model': model,
            'summary': {
                'total_words': sum(s.get('word_count', 0) for s in results.get('segments', [])),
                'speakers_list': list(set(s.get('speaker', '') for s in results.get('segments', []))),
                'average_confidence': round(
                    sum(s.get('confidence', 0) for s in results.get('segments', [])) / 
                    max(len(results.get('segments', [])), 1), 3
                )
            }
        }
        
        # Save response to output folder
        output_file = Path(app.config['OUTPUT_FOLDER']) / f"{timestamp}_results.json"
        with open(output_file, 'w') as f:
            json.dump(response, f, indent=2)
        
        response['output_file'] = str(output_file)
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Processing failed: {str(e)}'
        }), 500


@app.route('/results/<filename>', methods=['GET'])
def get_results(filename):
    """Retrieve saved results file."""
    try:
        filepath = Path(app.config['OUTPUT_FOLDER']) / filename
        if not filepath.exists():
            return jsonify({
                'status': 'error',
                'message': 'Results file not found'
            }), 404
        
        return send_file(str(filepath), as_attachment=True)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get service status and statistics."""
    try:
        upload_count = len(list(UPLOAD_FOLDER.glob('*')))
        results_count = len(list(OUTPUT_FOLDER.glob('*')))
        
        return jsonify({
            'status': 'operational',
            'uploads': upload_count,
            'results': results_count,
            'uptime': 'running',
            'version': '1.0'
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'status': 'error',
        'message': f'File too large. Maximum size: 500MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'GET /models',
            'GET /status',
            'POST /process',
            'GET /results/<filename>'
        ]
    }), 404


if __name__ == '__main__':
    # Development server
    logger.info("Starting MOM-Bot API Server...")
    logger.info("Available endpoints:")
    logger.info("  GET  / - Home & endpoints")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /models - Available models")
    logger.info("  GET  /status - Service status")
    logger.info("  POST /process - Process audio")
    logger.info("  GET  /results/<filename> - Get results")
    logger.info("\nServer running at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
