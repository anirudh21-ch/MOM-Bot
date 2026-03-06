# MOM-Bot Meeting Transcription Pipeline

Unified speech processing pipeline with Whisper ASR for meeting transcription.

**Features:**
- ✅ Automatic Speech Recognition (Whisper)
- ✅ Speaker Diarization
- ✅ Voice Activity Detection
- ✅ Supports audio files up to 20+ minutes
- ✅ Multilingual support (99+ languages)
- ✅ JSON output with speaker metadata

---

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline

**Basic usage** (default model: base):
```bash
python -m src.main.pipeline meeting.wav
```

**For 15-20 minute audio** (use tiny model for speed):
```bash
python -m src.main.pipeline long_meeting.wav --model tiny
```

**Specify output file**:
```bash
python -m src.main.pipeline meeting.wav --output results.json
```

**Print results to console**:
```bash
python -m src.main.pipeline meeting.wav --print
```

---

## Command-line Options

```
python -m src.main.pipeline <audio_file> [options]

Positional arguments:
  audio_file            Path to audio file (WAV, MP3, M4A, FLAC, etc.)

Optional arguments:
  --model, -m           Whisper model size (default: base)
                        tiny   - ⚡ Fastest (~1GB RAM)
                        base   - 🚀 Recommended (~2GB RAM)
                        small  - 📊 Better accuracy (~4GB RAM)
                        medium - High accuracy (~8GB RAM)
                        large  - Best accuracy (~10GB RAM)
  
  --output, -o          Output JSON file (default: output/results.json)
  
  --num-speakers        Number of speakers (auto-detect if not specified)
  
  --sample-rate         Target sample rate (default: 16000)
  
  --print               Print formatted results to console
  
  --help                Show help message
```

---

## Examples

### Example 1: Process short meeting with default settings
```bash
python -m src.main.pipeline audio/meeting.wav
```

### Example 2: Process long meeting (15+ minutes) with fast model
```bash
python -m src.main.pipeline audio/long_meeting.wav --model tiny -o results/long_meeting.json
```

### Example 3: High accuracy transcription with debug output
```bash
python -m src.main.pipeline audio/meeting.wav --model small --print
```

### Example 4: Specify speaker count
```bash
python -m src.main.pipeline audio/meeting.wav --num-speakers 2 --output meeting_output.json
```

---

## Whisper Model Sizes

| Model  | Speed        | Accuracy | RAM   | Use Case |
|--------|-------------|----------|-------|----------|
| tiny   | ⚡ ~2-3min/20min | Good  | ~1GB  | 15-20min audio, speed critical |
| base   | 🚀 ~10-15s/4min | Very Good | ~2GB | **RECOMMENDED** - best balance |
| small  | 📊 ~30-45s/4min | Excellent | ~4GB | Accuracy important, time available |
| medium | 🐢 ~90-120s/4min | Excellent+ | ~8GB | High accuracy, GPU recommended |
| large  | 🐌 ~2-3min/4min | Best | ~10GB | Highest accuracy, GPU required |

**Recommendation for 15-20 minute audio:**
- Use `tiny` for fastest processing
- Use `base` for balanced speed/accuracy
- Use `small` if accuracy is critical

---

## Output Format

Pipeline saves results as JSON with the following structure:

```json
{
  "audio_file": "meeting.wav",
  "total_duration": 237.77,
  "num_segments": 11,
  "num_speakers": 2,
  "asr_model": "Whisper-base",
  "language": "en",
  "segments": [
    {
      "segment_id": 1,
      "start_time": "0.00s",
      "end_time": "34.00s",
      "duration": 34.0,
      "speaker": "Speaker 1",
      "text": "Meeting transcript text...",
      "confidence": 0.95,
      "language": "en"
    }
  ],
  "speaker_summary": {
    "Speaker 1": {
      "duration": 122.0,
      "segments": 6,
      "words": 415
    },
    "Speaker 2": {
      "duration": 96.0,
      "segments": 5,
      "words": 308
    }
  }
}
```

---

## Supported Audio Formats

- WAV  (.wav)
- MP3  (.mp3)
- FLAC (.flac)
- M4A  (.m4a)
- OGG  (.ogg)
- And more...

---

## Troubleshooting

### Whisper Model Too Large

If you get memory errors:
1. Use a smaller model (`tiny` or `base`)
2. Ensure you have at least 2GB RAM free
3. On GPU, ensure CUDA is properly installed

### Slow Processing

If the pipeline is too slow:
1. Use `tiny` model instead of `base`
2. For GPU acceleration, ensure CUDA/PyTorch is configured
3. Consider splitting long audio files

### Audio Not Recognized

If no speech is detected:
1. Ensure audio is not corrupted
2. Check audio format is supported
3. Verify audio has speech content (not music/silence)

---

## Architecture

Pipeline stages:

```
Audio File
    ↓
[Load Audio] - Load and resample to 16kHz
    ↓
[VAD] - Detect speech segments (marblenet)
    ↓
[Whisper ASR] - Transcribe to text (Whisper model)
    ↓
[Diarization] - Identify speakers (titanet_large)
    ↓
[Combine] - Merge all results with metadata
    ↓
JSON Output + Console Display
```

---

## Configuration

Default configuration in `src/pipeline/pipeline.py`:
- Sample rate: 16000 Hz
- VAD model: marblenet
- ASR model: Whisper (configurable size)
- Diarization: titanet_large
- Language: Auto-detected

---

## License

MIT License - see LICENSE file for details
