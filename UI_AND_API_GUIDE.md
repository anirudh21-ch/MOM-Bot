# MOM-Bot User Interface & REST API Guide

Complete guide for using the Streamlit web interface and REST API for the MOM-Bot speech processing pipeline.

## Overview

The MOM-Bot system now provides **three ways** to process audio:

1. **Python CLI** - For automation and scripting
2. **REST API** - For programmatic integration  
3. **Streamlit Web UI** - For interactive use

---

## Streamlit Web Application

### Features

✨ **Interactive Interface**
- Drag-and-drop audio file upload
- Real-time processing status
- Beautiful results visualization
- Export to JSON or TXT

👥 **Speaker Management**
- Automatic speaker detection
- Specify number of speakers
- View speaker breakdown
- Timeline visualization

📊 **Results Viewing**
- Full transcription with timestamps
- Per-speaker duration tracking
- Confidence scores
- Search and filter capabilities

⚙️ **Model Selection**
- Choose Whisper model (tiny → large)
- View model specifications
- See expected processing time

### Getting Started

#### Launch the App
```bash
cd "/Users/anirudhbabuch/Downloads/MOM-Bot-master copy 2"
streamlit run app.py
```

The app opens at: **http://localhost:8501**

#### Upload Audio
1. Navigate to **"Upload & Process"** tab
2. Click "Choose an audio file"
3. Select your audio file (WAV, MP3, M4A, FLAC, OGG, WebM)
4. Select desired Whisper model from sidebar
5. Set number of speakers (or leave auto)
6. Click **"▶️ Process Audio"**

#### View Results
1. Navigate to **"Results"** tab
2. See speaker summary with duration breakdown
3. Read full transcription with speaker labels
4. Search for specific words or filter by speaker
5. Download as JSON or TXT

#### Analyze Statistics
1. Navigate to **"Statistics"** tab
2. View metrics:
   - Total duration
   - Number of segments
   - Total words
   - Average confidence
3. See speaker breakdown chart
4. View speaker distribution pie chart

### Models Guide

| Model | Size | RAM | Speed | Best For |
|-------|------|-----|-------|----------|
| **tiny** | 39M | ~1GB | ⚡⚡⚡⚡⚡ | Quick processing, long audio |
| **base** | 74M | ~2GB | ⚡⚡⚡⚡ | Balanced, recommended |
| **small** | 244M | ~4GB | ⚡⚡⚡ | Better accuracy |
| **medium** | 769M | ~8GB | ⚡⚡ | High accuracy |
| **large** | 1.5B | ~10GB | ⚡ | Best accuracy (GPU recommended) |

### Example Workflow

```
1. Open http://localhost:8501
2. Upload: meeting_recording.wav (5 minutes)
3. Model: tiny
4. Speakers: 2
5. Click "Process Audio"
6. Wait 30-45 seconds
7. View results with speaker breakdown
8. Download JSON for further analysis
```

### Keyboard Shortcuts

- `R` - Rerun app
- `C` - Clear cache
- `S` - Open settings

### File Limits

- **Max file size**: 500MB
- **Max duration**: No strict limit (40+ minute files work)
- **Supported formats**: WAV, MP3, M4A, FLAC, OGG, WebM

---

## REST API

### Overview

The REST API allows programmatic access to the pipeline. Perfect for:
- Integrating with other applications
- Batch processing multiple files
- Building custom interfaces
- Automation scripts

### Starting the API

```bash
python src/api/server.py
```

Server runs at: **http://localhost:5000**

### Endpoints

#### 1. Health Check

**GET** `/`

Returns service information and available endpoints.

```bash
curl http://localhost:5000/
```

Response:
```json
{
  "status": "ok",
  "message": "MOM-Bot Pipeline API v1.0",
  "endpoints": [
    "/health",
    "/models",
    "/status",
    "/process",
    "/results/<filename>"
  ]
}
```

#### 2. Service Health

**GET** `/health`

Get current service status.

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "ok",
  "uptime": 125.34,
  "timestamp": "2025-03-06 10:30:45"
}
```

#### 3. Available Models

**GET** `/models`

List all available Whisper models.

```bash
curl http://localhost:5000/models
```

Response:
```json
{
  "models": [
    {
      "name": "tiny",
      "size": "39M",
      "ram": "~1GB",
      "speed": "Very Fast"
    },
    {
      "name": "base",
      "size": "74M",
      "ram": "~2GB",
      "speed": "Fast"
    },
    ...
  ]
}
```

#### 4. Service Status

**GET** `/status`

Get service statistics.

```bash
curl http://localhost:5000/status
```

Response:
```json
{
  "status": "ok",
  "files_processed": 15,
  "results_available": 12,
  "uptime_seconds": 3600
}
```

#### 5. Process Audio (Main Endpoint)

**POST** `/process`

Process an audio file and return transcription with speaker diarization.

**Parameters:**
- `audio_file` (required): Audio file (multipart/form-data)
- `model` (optional): Whisper model - `tiny`, `base`, `small`, `medium`, `large` (default: `tiny`)
- `num_speakers` (optional): Number of speakers - 1-10 (default: auto-detect)

**Example:**
```bash
curl -X POST \
  -F "audio_file=@meeting.wav" \
  -F "model=tiny" \
  -F "num_speakers=2" \
  http://localhost:5000/process
```

**Response (Success):**
```json
{
  "status": "success",
  "file": "meeting.wav",
  "duration": 237.77,
  "speakers": 2,
  "segments": [
    {
      "segment_id": 1,
      "speaker": "Speaker 1",
      "start_time": "00:00:10",
      "end_time": "00:00:25",
      "duration": 15.5,
      "text": "Good morning everyone, thanks for joining this meeting.",
      "confidence": 0.95
    },
    {
      "segment_id": 2,
      "speaker": "Speaker 2",
      "start_time": "00:00:26",
      "end_time": "00:00:35",
      "duration": 9.2,
      "text": "Thanks for having us. Let's get started.",
      "confidence": 0.92
    }
  ],
  "language": "en",
  "processing_time": 23.45,
  "model": "tiny",
  "summary": {
    "total_words": 712,
    "speakers_list": ["Speaker 1", "Speaker 2"],
    "average_confidence": 0.88,
    "speaker_durations": {
      "Speaker 1": 118.0,
      "Speaker 2": 100.0
    }
  }
}
```

**Response (Error):**
```json
{
  "status": "error",
  "error": "File too large. Maximum size: 500MB",
  "code": 413
}
```

#### 6. Retrieve Results

**GET** `/results/<filename>`

Get previously processed results.

```bash
curl http://localhost:5000/results/meeting.wav
```

Returns the same JSON response as `/process` endpoint.

### Error Codes

| Code | Error | Solution |
|------|-------|----------|
| 400 | Bad Request | Missing required parameter |
| 413 | File Too Large | File exceeds 500MB |
| 415 | Unsupported File Type | Use WAV, MP3, M4A, FLAC, OGG, WebM |
| 422 | Processing Failed | Invalid audio or insufficient resources |
| 500 | Server Error | Check API logs |

### Rate Limiting

Currently no rate limiting. For production deployment, implement:
- 10 concurrent requests max
- 100 MB/minute per IP

### Response Times

Processing time depends on:
- **Model size**: tiny (5-10s) → large (30-60s) for 1 minute audio
- **Audio duration**: Linear scaling
- **Hardware**: M2 Mac vs Intel vs Cloud instance

**Example timings** (on Mac M2):
- 1 min audio, tiny model: 5-10 seconds
- 5 min audio, tiny model: 20-40 seconds
- 5 min audio, base model: 40-80 seconds

### Python Integration Example

```python
import requests
import json

# Setup
API_URL = "http://localhost:5000"

# Process audio file
with open("meeting.wav", "rb") as f:
    files = {"audio_file": f}
    data = {"model": "tiny", "num_speakers": "2"}
    
    response = requests.post(
        f"{API_URL}/process",
        files=files,
        data=data
    )

# Handle response
if response.status_code == 200:
    results = response.json()
    print(f"✅ Processed: {results['file']}")
    print(f"👥 Speakers: {results['speakers']}")
    print(f"⏱️ Duration: {results['duration']:.1f}s")
    print(f"⏱️ Processing time: {results['processing_time']:.2f}s")
    
    # Iterate through segments
    for segment in results["segments"]:
        print(f"\n[{segment['start_time']}] {segment['speaker']}")
        print(f"{segment['text']}")

else:
    error = response.json()
    print(f"❌ Error: {error['error']}")
```

### JavaScript/Node.js Integration

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

// Setup
const API_URL = "http://localhost:5000";

// Prepare file
const form = new FormData();
form.append('audio_file', fs.createReadStream('meeting.wav'));
form.append('model', 'tiny');
form.append('num_speakers', '2');

// Send request
axios.post(`${API_URL}/process`, form, {
    headers: form.getHeaders()
})
.then(response => {
    const results = response.data;
    console.log(`✅ Processed: ${results.file}`);
    console.log(`👥 Speakers: ${results.speakers}`);
    console.log(`⏱️ Duration: ${results.duration.toFixed(1)}s`);
})
.catch(error => {
    console.error(`❌ Error: ${error.response.data.error}`);
});
```

### cURL Examples

**Process with model selection:**
```bash
curl -X POST \
  -F "audio_file=@audio.wav" \
  -F "model=base" \
  http://localhost:5000/process | jq .
```

**Process with speaker count:**
```bash
curl -X POST \
  -F "audio_file=@audio.wav" \
  -F "num_speakers=3" \
  http://localhost:5000/process | jq .
```

**Get formatted output:**
```bash
curl -X POST \
  -F "audio_file=@audio.wav" \
  http://localhost:5000/process | jq '.segments[0]'
```

**Save results to file:**
```bash
curl -X POST \
  -F "audio_file=@audio.wav" \
  http://localhost:5000/process > results.json
```

---

## Combining API and Web UI

### Scenario 1: API Backend with Custom Frontend

```bash
# Terminal 1: Start API
python src/api/server.py

# Terminal 2: Start your custom app (connects to API)
python my_custom_app.py
```

### Scenario 2: Both Services Running

```bash
# Terminal 1: Start API
python src/api/server.py

# Terminal 2: Start Streamlit
streamlit run app.py

# Now use either:
# - Web UI: http://localhost:8501
# - REST API: http://localhost:5000
```

### Scenario 3: Automated Batch Processing

```bash
# Use API for batch processing
python batch_processor.py  # Calls http://localhost:5000/process

# While web UI is available for interactive use
streamlit run app.py      # http://localhost:8501
```

---

## Configuration

### API Server Settings

Edit `src/api/server.py` to customize:

```python
# Port and host
port = 5000
host = "0.0.0.0"  # or "127.0.0.1" for local-only

# File size limit (500MB default)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
```

### Streamlit Settings

Edit `.streamlit/config.toml` (create if needed):

```toml
[server]
port = 8501
headless = true

[client]
showErrorDetails = true

[logger]
level = "info"
```

### Pipeline Settings

Edit `src/pipeline/pipeline.py`:

```python
# Default settings
DEFAULT_MODEL = "tiny"
DEFAULT_NUM_SPEAKERS = 2
DIARIZATION_THRESHOLD = 0.55
```

---

## Advanced Usage

### Batch Processing Script

```python
import requests
import os
from pathlib import Path

API_URL = "http://localhost:5000"
AUDIO_DIR = "audio_files"
OUTPUT_DIR = "results"

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Process all audio files
for audio_file in os.listdir(AUDIO_DIR):
    if not audio_file.endswith(('.wav', '.mp3', '.m4a')):
        continue
    
    filepath = os.path.join(AUDIO_DIR, audio_file)
    
    print(f"Processing {audio_file}...")
    
    with open(filepath, 'rb') as f:
        response = requests.post(
            f"{API_URL}/process",
            files={"audio_file": f},
            data={"model": "tiny"}
        )
    
    if response.status_code == 200:
        results = response.json()
        
        # Save results
        output_file = os.path.join(OUTPUT_DIR, f"{audio_file}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✅ {results['speakers']} speakers, {results['duration']:.1f}s")
    else:
        print(f"  ❌ Failed")
```

### Health Check Loop

```python
import requests
import time

API_URL = "http://localhost:5000"

while True:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print(f"✅ API Healthy - {response.json()['status']}")
    except:
        print(f"❌ API Unreachable")
    
    time.sleep(30)  # Check every 30 seconds
```

---

## Troubleshooting

### Issue: "Connection refused" on localhost:5000
**Solution:** Make sure API server is running
```bash
python src/api/server.py
```

### Issue: "Connection refused" on localhost:8501
**Solution:** Make sure Streamlit is running
```bash
streamlit run app.py
```

### Issue: File upload fails silently
**Solution:** Check file size and format
- Max size: 500MB
- Formats: WAV, MP3, M4A, FLAC, OGG, WebM

### Issue: "Out of memory" error
**Solution:** Use smaller model or process shorter audio
- Try: `tiny` model instead of `base`
- Split long audio into segments

### Issue: Poor transcription quality
**Solution:** Try larger model or ensure audio quality
- Upgrade from `tiny` to `base` or `small`
- Check audio has good signal-to-noise ratio
- Minimize background noise

### Issue: Wrong speaker count detected
**Solution:** Specify number explicitly
- Set `num_speakers` parameter instead of auto-detect
- API: POST `/process` with `num_speakers=2`
- Streamlit: Set in settings sidebar

---

## Performance Optimization

### For Real-Time Inference
- Use `tiny` model
- Split audio into 10-minute segments
- Process in parallel using multiple API instances

### For Batch Processing
- Use `base` or `small` model for better quality
- Process files sequentially to manage memory
- Enable results caching to avoid reprocessing

### For High Accuracy
- Use `medium` or `large` model
- Ensure server has 8-10GB RAM
- Consider GPU for faster processing

---

## Next Steps

1. **Start the Web UI:** `streamlit run app.py`
2. **Test with sample audio:** Use audio in `data/` directory
3. **Try the REST API:** Use one of the cURL examples
4. **Integrate with your apps:** Use API endpoints
5. **Deploy to production:** See DEPLOYMENT_GUIDE.md

---

## Support

For issues or questions:
1. Check DEPLOYMENT_GUIDE.md for common issues
2. Review README.md for project overview
3. Check Diarization_IMPROVEMENT_PLAN.md for technical details

---

**Last Updated**: 2025-03-06
**Version**: 1.0.0
