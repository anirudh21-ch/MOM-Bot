#!/usr/bin/env python3
"""
Simple test script for MOM-Bot pipeline
Shows how to programmatically use the pipeline
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import MOMBotPipeline


def test_pipeline():
    """Test the pipeline with an audio file."""
    
    # Audio file to process
    audio_file = "your_audio_file.wav"  # Change to your audio file
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("\nUsage:")
        print("  1. Place your audio file in the current directory")
        print("  2. Update 'audio_file' variable in this script")
        print("  3. Run: python test_pipeline.py")
        return
    
    # Create pipeline with tiny model (fast for 15-20min audio)
    print("🚀 Initializing pipeline...")
    pipeline = MOMBotPipeline(
        whisper_model_size="tiny",  # tiny for fast, base for balanced
        num_speakers=None,           # Auto-detect speakers
        sample_rate=16000
    )
    
    # Process audio
    print(f"\n📁 Processing: {audio_file}")
    results = pipeline.process(audio_file)
    
    # Save results
    output_file = "results.json"
    pipeline.save_results(output_file)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print formatted results
    pipeline.print_results()


if __name__ == '__main__':
    test_pipeline()
