import glob
import os

import librosa
from pydub import AudioSegment


def prepare_audio(original_audio_path: str, output_dir: str = "data"):
    """
    Convert stereo audio to mono and save in output_dir.
    Also returns the loaded signal and sample rate.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to mono
    audio = AudioSegment.from_wav(original_audio_path)
    mono_audio = audio.set_channels(1)

    # Save mono file
    mono_audio_path = os.path.join(output_dir, "mono_audio.wav")
    mono_audio.export(mono_audio_path, format="wav")

    # Load mono file
    signal, sample_rate = librosa.load(mono_audio_path, sr=None)

    return mono_audio_path, signal, sample_rate


if __name__ == "__main__":
    # Example usage
    original_audio = "data/example_stereo.wav"
    mono_path, signal, sr = prepare_audio(original_audio)
    print(f"Mono audio saved at: {mono_path}")
