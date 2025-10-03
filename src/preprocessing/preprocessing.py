import glob
import os

import librosa
from pydub import AudioSegment


def prepare_audio(original_audio_path: str, output_dir: str = "data"):
    """
    Convert any audio format to mono WAV and save in output_dir.
    Also returns the loaded signal and sample rate.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"🎵 Processing audio file: {original_audio_path}")
    
    try:
        # Try to detect file format and load accordingly
        file_ext = os.path.splitext(original_audio_path)[1].lower()
        
        if file_ext == '.wav':
            audio = AudioSegment.from_wav(original_audio_path)
        elif file_ext == '.mp3':
            audio = AudioSegment.from_mp3(original_audio_path)
        elif file_ext in ['.m4a', '.mp4', '.aac']:
            audio = AudioSegment.from_file(original_audio_path, format="mp4")
        elif file_ext in ['.flac']:
            audio = AudioSegment.from_file(original_audio_path, format="flac")
        elif file_ext in ['.ogg']:
            audio = AudioSegment.from_ogg(original_audio_path)
        else:
            # Try generic loader as fallback
            print(f"⚠️ Unknown format {file_ext}, trying generic loader...")
            audio = AudioSegment.from_file(original_audio_path)
        
        print(f"✅ Audio loaded successfully:")
        print(f"   - Duration: {len(audio) / 1000:.2f} seconds")
        print(f"   - Channels: {audio.channels}")
        print(f"   - Sample rate: {audio.frame_rate} Hz")
        print(f"   - Format: {file_ext}")
        
        # Normalize audio levels (important for speech detection)
        audio = audio.normalize()
        
        # Convert to mono (16kHz is preferred for NeMo)
        mono_audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Save mono file
        mono_audio_path = os.path.join(output_dir, "mono_audio.wav")
        mono_audio.export(mono_audio_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        
        print(f"✅ Converted to mono WAV: {mono_audio_path}")

        # Load mono file with librosa for verification
        signal, sample_rate = librosa.load(mono_audio_path, sr=16000)
        
        print(f"📊 Final audio stats:")
        print(f"   - Sample rate: {sample_rate} Hz")  
        print(f"   - Duration: {len(signal) / sample_rate:.2f} seconds")
        print(f"   - Max amplitude: {max(abs(signal)) if len(signal) > 0 else 0:.4f}")
        
        if len(signal) == 0:
            raise ValueError("Processed audio file is empty!")
            
        if max(abs(signal)) < 0.001:
            print("⚠️ Warning: Audio amplitude is very low - might not be detected by ASR")

        return mono_audio_path, signal, sample_rate
        
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        raise Exception(f"Could not process audio file: {e}")


if __name__ == "__main__":
    # Example usage
    original_audio = "data/example_stereo.wav"
    mono_path, signal, sr = prepare_audio(original_audio)
    print(f"Mono audio saved at: {mono_path}")