# MOM-Bot: Meeting Transcription Pipeline

Professional speech processing pipeline with **Whisper ASR** for meeting transcription, speaker diarization, and meeting minutes.

## ✨ Features

- ✅ **Whisper ASR**: OpenAI Whisper for high-accuracy transcription (99+ languages)
- ✅ **Speaker Diarization**: Automatically identify who spoke when
- ✅ **Long Audio Support**: Handles 15-20+ minute meetings efficiently
- ✅ **Multiple Models**: Choose speed vs accuracy (tiny → large)
- ✅ **Audio Format Support**: WAV, MP3, M4A, FLAC, OGG, and more
- ✅ **JSON Output**: Structured results with timestamps and metadata
- ✅ **Language Detection**: Automatic language detection
- ✅ **CLI & Python API**: Both command-line and programmatic access

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Command Line Usage

**Process audio with default settings:**
```bash
python -m src.main.pipeline meeting.wav
```

**For long meetings (15-20 minutes) - use tiny model:**
```bash
python -m src.main.pipeline long_meeting.wav --model tiny
```

**Save output to specific file:**
```bash
python -m src.main.pipeline meeting.wav -o results.json
## 🎯 Whisper Model Selection

For different audio durations:

| Audio Length | Recommended Model | Speed | Accuracy |
|--------------|------------------|-------|----------|
| < 5 min      | base             | fast  | excellent |
| 5-15 min     | base             | medium | excellent |
| 15-20 min    | **tiny**         | ⚡ fast | good |
| 20+ min      | tiny             | ⚡ fastest | good |
| Accuracy critical | small | slower | excellent+ |

## 📁 Project Structure

```
MOM-Bot-master copy 2/
├── src/
│   ├── main/
│   │   ├── __init__.py
│   │   └── pipeline.py          # CLI entry point
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── whisper_asr.py       # Whisper engine (ONLY ASR)
│   │   └── pipeline.py          # Main pipeline orchestration
│   ├── audio_processing/
│   │   ├── asr.py
│   │   ├── audio_loader.py
│   │   └── config_loader.py
│   ├── diarization/             # Speaker identification
│   ├── preprocessing/
│   ├── utils/
│   └── openai_integration.py
├── config/                      # Model configurations
├── data/                        # Audio files (add yours here)
├── output/                      # Results (generated)
├── requirements.txt             # Dependencies
├── QUICKSTART.md               # Detailed usage guide
├── test_pipeline.py            # Example usage script
└── README.md                   # This file
```

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify Whisper installation
python -c "import whisper; print('✓ Whisper installed')"
```

## 📝 Examples

### Example 1: Process meeting with fast model
```bash
python -m src.main.pipeline meeting.wav --model tiny --output results.json
```

### Example 2: Process with accuracy focus
```bash
python -m src.main.pipeline meeting.wav --model small --print
```

### Example 3: Process multiple audio formats
```bash
# WAV
python -m src.main.pipeline audio/meeting.wav

# MP3
python -m src.main.pipeline audio/recording.mp3

# M4A
python -m src.main.pipeline audio/podcast.m4a
```

## 🎙️ Output Example

Results are saved as JSON with speaker information:

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
      "speaker": "Speaker 1",
      "text": "Good morning everyone...",
      "confidence": 0.95,
      "language": "en"
    }
  ],
  "speaker_summary": {
    "Speaker 1": {"duration": 122.0, "segments": 6, "words": 415},
    "Speaker 2": {"duration": 96.0, "segments": 5, "words": 308}
  }
}
```

## 💾 Supported Audio Formats

WAV • MP3 • M4A • FLAC • OGG • And more

## 🔧 Technologies

- **Speech Recognition**: OpenAI Whisper (99+ languages)
- **Voice Activity Detection**: NeMo MarbleNet
- **Speaker Diarization**: NeMo TitanET
- **Deep Learning**: PyTorch with CPU/CUDA/MPS support
- **Audio Processing**: librosa, soundfile

## 📚 Documentation

- [QUICKSTART.md](QUICKSTART.md) - Detailed usage guide
- [requirements.txt](requirements.txt) - All dependencies
- **Flask**: REST API framework
- **librosa**: Audio processing and analysis

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review `SETUP_COMPLETE.md` for detailed setup info