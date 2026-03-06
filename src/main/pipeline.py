"""MOM-Bot Pipeline - Command-line entry point

Run the complete pipeline from command line:

    python -m src.main.pipeline <audio_file> [options]

Examples:

    # Use default (base model) for 4-10 minute audio
    python -m src.main.pipeline meeting.wav

    # Use tiny model for 15-20 minute audio (faster, smaller model)
    python -m src.main.pipeline long_meeting.wav --model tiny

    # Use small model for better accuracy
    python -m src.main.pipeline meeting.wav --model small

    # Specify output file
    python -m src.main.pipeline meeting.wav -o results.json

    # Show help
    python -m src.main.pipeline --help

WHISPER MODEL SIZES FOR DIFFERENT AUDIO LENGTHS:

    < 5 minutes:  tiny or base ⚡ (seconds)
    5-10 minutes: base (good balance)
    10-15 minutes: base or small (minutes)
    15-20+ minutes: tiny ⚡ (fastest) or base (balanced)
    High accuracy critical: small or medium
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import MOMBotPipeline


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="MOM-Bot Speech Processing Pipeline - VAD → Whisper ASR → Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Process with default settings (base model)
    python -m src.main.pipeline meeting.wav
    
    # Process 15+ minute audio with tiny model (faster)
    python -m src.main.pipeline long_meeting.wav --model tiny
    
    # Process with small model (better accuracy, slower)
    python -m src.main.pipeline meeting.wav --model small
    
    # Save output to specific file
    python -m src.main.pipeline meeting.wav -o my_results.json

WHISPER MODELS:
    tiny    - ⚡ Fastest (~1GB RAM) - USE FOR 15-20min audio
    base    - 🚀 Fast & accurate (~2GB RAM) - RECOMMENDED
    small   - 📊 Better accuracy (~4GB RAM) - Use if time available
    medium  - High accuracy (~8GB RAM) - Slow on CPU
    large   - Best accuracy (~10GB RAM) - Very slow, use GPU if possible
        """
    )
    
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to audio file (WAV, MP3, M4A, FLAC, etc.)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='tiny',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base). Use "tiny" for 15-20min audio'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file path (default: output/results.json)'
    )
    
    parser.add_argument(
        '--num-speakers',
        type=int,
        default=None,
        help='Expected number of speakers (optional, auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Target sample rate in Hz (default: 16000)'
    )
    
    parser.add_argument(
        '--print',
        action='store_true',
        help='Print formatted results to console'
    )
    
    args = parser.parse_args()
    
    # Validate audio file
    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        print(f"✗ Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('output') / 'results.json'
    
    # Run pipeline
    print(f"\n🎙️  MOM-BOT PIPELINE")
    print(f"📁 Input: {audio_file}")
    print(f"🤖 ASR Model: Whisper-{args.model}")
    print(f"💾 Output: {output_path}\n")
    
    pipeline = MOMBotPipeline(
        whisper_model_size=args.model,
        num_speakers=args.num_speakers,
        sample_rate=args.sample_rate
    )
    
    # Process
    results = pipeline.process(str(audio_file))
    
    if 'error' in results:
        print(f"\n✗ Pipeline failed: {results['error']}")
        sys.exit(1)
    
    # Save results
    pipeline.save_results(str(output_path))
    
    # Print if requested
    if args.print:
        pipeline.print_results()
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
