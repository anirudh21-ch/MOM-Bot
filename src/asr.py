from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps

def run_asr(cfg, level: str = "sentence"):
    """
    Run ASR with NeMo ASRDecoderTimeStamps.

    Args:
        cfg: OmegaConf config with diarizer parameters.
        level (str): 'sentence' or 'word' for output granularity.

    Returns:
        tuple: (hypotheses, timestamps)
    """
    asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
    asr_model = asr_decoder_ts.set_asr_model()

    if level == "word":
        hyp, ts_hyp = asr_decoder_ts.run_ASR(asr_model)
    else:  # sentence
        hyp, ts_hyp = asr_decoder_ts.run_ASR(asr_model)

    return hyp, ts_hyp
