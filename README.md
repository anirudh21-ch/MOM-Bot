# MOM-Bot: Meeting Minutes Automation Bot

AI-powered audio transcription and meeting summarization with speaker diarization, optimized for Apple Silicon Macs with MPS acceleration.

## Features

- 🎯 **Speaker Diarization**: Automatically identify and separate different speakers
- 🚀 **MPS Acceleration**: Optimized for Apple Silicon GPUs (M1/M2/M3)
- 🤖 **AI Summarization**: GPT-powered meeting summaries with action items
- 📝 **Multiple Formats**: Support for WAV, MP3, MP4, FLAC audio files
- 🌐 **REST API**: Simple HTTP endpoints for integration
- ⚡ **Real-time Processing**: Fast transcription with hardware acceleration

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MOM-Bot.git
cd MOM-Bot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Set your OpenAI API key for summarization (optional)
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run the Server

```bash
python -m src.main.main
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Transcribe Audio
```bash
curl -X POST -F "audio=@your-audio-file.wav" http://localhost:8000/transcribe
```

### Generate Summary
```bash
# Summarize last transcription
curl http://localhost:8000/summary

# Upload extract.txt file
curl -X POST -F "extract_file=@extract.txt" http://localhost:8000/summary
```

## Example Response

### Transcription
```json
{
  "status": "success",
  "transcript": [
    {
      "speaker": "Speaker_0",
      "text": "Good morning everyone, let's start the meeting.",
      "start_time": 0.0,
      "end_time": 3.2
    }
  ],
  "metadata": {
    "num_segments": 15,
    "speakers": ["Speaker_0", "Speaker_1"],
    "audio_file": "/path/to/audio.wav"
  }
}
```

### Summary
```json
{
  "status": "success",
  "summary": {
    "executive_summary": "Team meeting to discuss Q4 planning...",
    "key_points": ["Budget approval needed", "Timeline finalized"],
    "decisions": ["Approved Q4 budget increase"],
    "action_items": ["John to prepare proposal by Friday"],
    "participants": ["Speaker_0", "Speaker_1"],
    "next_steps": ["Follow-up meeting scheduled for next week"]
  }
}
```

## System Requirements

- **macOS**: Apple Silicon Mac (M1/M2/M3) recommended for MPS acceleration
- **Python**: 3.9+ 
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB for models and dependencies

## Architecture

```
MOM-Bot/
├── src/
│   ├── main/               # Flask API server
│   ├── audio_processing/   # ASR and preprocessing
│   ├── diarization/        # Speaker diarization with MPS
│   └── openai_integration/ # AI summarization
├── config/                 # NeMo model configurations
└── data/                   # Audio files and outputs
```

## Technologies

- **NeMo Toolkit**: NVIDIA's toolkit for speech AI
- **PyTorch**: With Metal Performance Shaders (MPS) support
- **OpenAI GPT**: For intelligent summarization
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