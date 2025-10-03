from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
import torch
import logging
import numpy as np

# Patch for BFloat16 compatibility with Apple Silicon MPS
def patch_tensor_numpy():
    """Patch tensor.numpy() to handle BFloat16 on MPS"""
    original_numpy = torch.Tensor.numpy
    
    def patched_numpy(self, force=False):
        if self.dtype == torch.bfloat16:
            # Convert BFloat16 to Float32 first, then to numpy
            return self.float().cpu().numpy(force=force)
        else:
            return original_numpy(self, force=force)
    
    torch.Tensor.numpy = patched_numpy

# Apply the patch
patch_tensor_numpy()


def run_asr(cfg, level: str = "sentence"):
    """
    Run ASR with NeMo ASRDecoderTimeStamps with MPS support.

    Args:
        cfg: OmegaConf config with diarizer parameters.
        level (str): 'sentence' or 'word' for output granularity.

    Returns:
        tuple: (hypotheses, timestamps)
    """
    logger = logging.getLogger(__name__)
    
    # Check device configuration
    device = getattr(cfg.diarizer, 'device', 'auto')
    if device == 'mps' and torch.backends.mps.is_available():
        logger.info("🚀 ASR using MPS acceleration")
    elif device == 'cuda' and torch.cuda.is_available():
        logger.info("🚀 ASR using CUDA acceleration") 
    else:
        logger.info("🔄 ASR using CPU")
        
    asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
    asr_model = asr_decoder_ts.set_asr_model()

    if level == "word":
        hyp, ts_hyp = asr_decoder_ts.run_ASR(asr_model)
    else:  # sentence
        hyp, ts_hyp = asr_decoder_ts.run_ASR(asr_model)

    return hyp, ts_hyp