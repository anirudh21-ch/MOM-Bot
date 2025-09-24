from os.path import join

import pandas as pd
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

from src.utils import read_file


def init_diarization(cfg, asr_decoder_ts=None):
    """
    Initialize diarization pipeline.

    Args:
        cfg: OmegaConf config with diarizer parameters.
        asr_decoder_ts: Optional ASRDecoderTimeStamps instance (for word-level diarization).

    Returns:
        OfflineDiarWithASR instance.
    """
    asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)

    # If word-level ASR was run, connect the anchor offset
    if asr_decoder_ts and hasattr(asr_decoder_ts, "word_ts_anchor_offset"):
        asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset

    return asr_diar_offline


def run_diarization(cfg, ts_hyp, asr_diar_offline=None):
    """
    Run diarization with timestamps.

    Args:
        cfg: OmegaConf config with diarizer parameters.
        ts_hyp: Timestamps hypothesis (sentence-level or word-level).
        asr_diar_offline: Optional initialized diarization pipeline.

    Returns:
        tuple: (diar_hyp, diar_score, asr_diar_offline)
    """
    if asr_diar_offline is None:
        asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)

    diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, ts_hyp)
    return diar_hyp, diar_score, asr_diar_offline


def get_transcript(asr_diar_offline, diar_hyp, hyp, ts_hyp, return_df=False):
    """
    Get transcript with speaker labels.

    Args:
        asr_diar_offline: OfflineDiarWithASR instance.
        diar_hyp: Diarization hypothesis.
        hyp: ASR hypothesis (sentence_hyp or word_hyp).
        ts_hyp: ASR timestamps (sentence_ts_hyp or word_ts_hyp).
        return_df (bool): If True, returns a Pandas DataFrame.

    Returns:
        dict or pd.DataFrame: Transcript structured by speaker and timestamp.
    """
    trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, hyp, ts_hyp)

    if return_df:
        rows = []
        for speaker, segments in trans_info_dict.items():
            for seg in segments:
                rows.append(
                    {
                        "speaker": speaker,
                        "start_time": float(seg["start_time"]),
                        "end_time": float(seg["end_time"]),
                        "transcription": seg["text"],
                    }
                )
        return pd.DataFrame(rows)

    return trans_info_dict


def load_rttm(base_name: str, data_dir: str = "data"):
    """
    Load RTTM file and convert to labels.

    Args:
        base_name (str): Base filename of the audio (without extension).
        data_dir (str): Directory where pred_rttms is stored.

    Returns:
        tuple: (list of RTTM lines, list of parsed labels)
    """
    predicted_rttm_path = join(data_dir, f"pred_rttms/{base_name}.rttm")

    # Read RTTM lines
    with open(predicted_rttm_path) as f:
        rttm_lines = f.read().splitlines()

    # Convert RTTM to structured labels
    labels = rttm_to_labels(predicted_rttm_path)

    return rttm_lines, labels


def save_transcript(base_name: str, data_dir: str = "data") -> str:
    """
    Load the generated transcript file and save it as extract.txt.

    Args:
        base_name (str): Base filename of the audio (without extension).
        data_dir (str): Directory containing pred_rttms and where to save extract.txt.

    Returns:
        str: Path to the saved extract.txt file.
    """
    transcription_path = join(data_dir, f"pred_rttms/{base_name}.txt")
    transcript_lines = read_file(transcription_path)

    extract_path = join(data_dir, "extract.txt")
    with open(extract_path, "w") as f:
        for line in transcript_lines:
            f.write(line + "\n")

    print(f"Transcript saved to {extract_path}")
    return extract_path
