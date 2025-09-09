import os
import wget
from omegaconf import OmegaConf

def load_config(data_dir="data", domain_type="meeting"):
    """
    Load and customize diarization config for given domain.
    Downloads from NeMo GitHub if not already present.
    """
    os.makedirs(data_dir, exist_ok=True)

    config_file_name = f"diar_infer_{domain_type}.yaml"
    config_url = (
        f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/"
        f"speaker_tasks/diarization/conf/inference/{config_file_name}"
    )
    config_path = os.path.join(data_dir, config_file_name)

    # Download if not available
    if not os.path.exists(config_path):
        print(f"Downloading config: {config_file_name}")
        wget.download(config_url, data_dir)

    cfg = OmegaConf.load(config_path)

    # -------------------------
    # Custom overrides
    # -------------------------
    pretrained_speaker_model = "titanet_large"
    cfg.diarizer.out_dir = data_dir
    cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    cfg.diarizer.clustering.parameters.oracle_num_speakers = False

    # VAD + ASR setup
    cfg.diarizer.vad.model_path = "vad_multilingual_marblenet"
    cfg.diarizer.asr.model_path = "QuartzNet15x5Base-En"
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.asr.parameters.word_ts = False
    cfg.diarizer.asr.parameters.sentence_ts = True
    cfg.diarizer.asr.parameters.asr_based_vad = False
    cfg.diarizer.asr.parameters.get_full_text = True
    cfg.diarizer.asr.parameters.use_rnnt_decoder_timestamps = False
    cfg.diarizer.asr.parameters.use_cer = False

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
