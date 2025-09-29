import json
import os
from typing import List, Optional, Union


def create_manifest(
    audio_filepath: str,
    output_dir: str = "data",
    num_speakers: Optional[int] = None,
    rttm_filepath: Optional[str] = None,
    ctm_filepath: Optional[str] = None,
    uem_filepath: Optional[str] = None,
    label: str = "infer",
    text: str = "-",
) -> str:
    """
    Create a NeMo-compatible manifest file for diarization/ASR.

    Args:
        audio_filepath (str): Path to the audio file (.wav).
        output_dir (str): Directory to save the manifest file.
        num_speakers (int, optional): Known number of speakers. Default: None.
        rttm_filepath (str, optional): Path to RTTM file (if available).
        ctm_filepath (str, optional): Path to CTM file (if available).
        uem_filepath (str, optional): Path to UEM file (if available).
        label (str): Label field for manifest (default "infer").
        text (str): Text field for manifest (default "-").

    Returns:
        str: Path to the created manifest file.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "input_manifest.json")

    meta = {
        "audio_filepath": os.path.abspath(audio_filepath),
        "offset": 0,
        "duration": None,
        "label": label,
        "text": text,
        "num_speakers": num_speakers,
        "rttm_filepath": rttm_filepath,
        "ctm_filepath": ctm_filepath,
        "uem_filepath": uem_filepath,
    }

    with open(manifest_path, "w") as f:
        json.dump(meta, f)
        f.write("\n")

    return manifest_path


def create_manifest_for_multiple(
    audio_filepaths: List[str],
    output_dir: str = "data",
    num_speakers: Optional[Union[int, List[int]]] = None,
    rttm_filepaths: Optional[List[str]] = None,
    ctm_filepaths: Optional[List[str]] = None,
    uem_filepaths: Optional[List[str]] = None,
    label: str = "infer",
    text: str = "-",
) -> str:
    """
    Create a manifest file for multiple audio files.

    Args:
        audio_filepaths (list): List of audio file paths.
        output_dir (str): Directory to save manifest.
        num_speakers (int | list, optional): Known speaker counts. Can be single int or list.
        rttm_filepaths (list, optional): List of RTTM file paths.
        ctm_filepaths (list, optional): List of CTM file paths.
        uem_filepaths (list, optional): List of UEM file paths.
        label (str): Label field for manifest.
        text (str): Text field for manifest.

    Returns:
        str: Path to the created manifest file.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "input_manifest.json")

    entries = []
    for i, audio in enumerate(audio_filepaths):
        entry = {
            "audio_filepath": os.path.abspath(audio),
            "offset": 0,
            "duration": None,
            "label": label,
            "text": text,
            "num_speakers": (num_speakers[i] if isinstance(num_speakers, list) else num_speakers),
            "rttm_filepath": None if not rttm_filepaths else rttm_filepaths[i],
            "ctm_filepath": None if not ctm_filepaths else ctm_filepaths[i],
            "uem_filepath": None if not uem_filepaths else uem_filepaths[i],
        }
        entries.append(entry)

    with open(manifest_path, "w") as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")

    return manifest_path


if __name__ == "__main__":
    # Example usage (single file)
    mono_audio = "data/mono_audio.wav"
    manifest_path = create_manifest(mono_audio, output_dir="data")
    print(f"Manifest created: {manifest_path}")

    # Example usage (multiple files)
    audio_files = ["data/mono_audio.wav", "data/another_audio.wav"]
    manifest_path_multi = create_manifest_for_multiple(audio_files, output_dir="data")
    print(f"Multi-file manifest created: {manifest_path_multi}")