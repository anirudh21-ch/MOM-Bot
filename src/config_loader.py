import os
import wget
from omegaconf import OmegaConf
import requests
from pathlib import Path

def load_config(data_dir="data", domain_type="meeting"):
    """
    Load and customize diarization config for given domain.
    Uses local config first, falls back to download only if needed.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    config_file_name = f"diar_infer_{domain_type}.yaml"
    
    # Priority 1: Check local config directory first
    local_config_path = os.path.join("config", config_file_name)
    if os.path.exists(local_config_path):
        print(f"Loading local config: {local_config_path}")
        cfg = OmegaConf.load(local_config_path)
    else:
        # Priority 2: Check data directory cache
        cached_config_path = os.path.join(data_dir, config_file_name)
        if os.path.exists(cached_config_path):
            print(f"Loading cached config: {cached_config_path}")
            cfg = OmegaConf.load(cached_config_path)
        else:
            # Priority 3: Download from GitHub (fallback)
            print(f"Local config not found, downloading: {config_file_name}")
            config_url = (
                f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/"
                f"speaker_tasks/diarization/conf/inference/{config_file_name}"
            )
            try:
                # Use requests instead of wget for better error handling
                response = requests.get(config_url, timeout=10)
                response.raise_for_status()
                
                with open(cached_config_path, 'w') as f:
                    f.write(response.text)
                
                cfg = OmegaConf.load(cached_config_path)
                print(f"Config downloaded and cached to: {cached_config_path}")
                
            except Exception as e:
                # Priority 4: Use embedded default config as final fallback
                print(f"Download failed ({e}), using embedded default config")
                cfg = get_default_config()
    
    # Apply custom overrides
    cfg = apply_custom_overrides(cfg, data_dir)
    return cfg

def get_default_config():
    """
    Embedded default configuration as fallback when network is unavailable.
    """
    default_config = {
        'diarizer': {
            'manifest_filepath': None,
            'out_dir': './data',
            'oracle_vad': False,
            'collar': 0.25,
            'ignore_overlap': True,
            'speaker_embeddings': {
                'model_path': 'titanet_large',
                'parameters': {
                    'window_length_in_sec': 0.96,
                    'shift_length_in_sec': 0.48,
                    'multiscale_weights': None
                }
            },
            'clustering': {
                'parameters': {
                    'oracle_num_speakers': False,
                    'max_num_speakers': 20,
                    'enhanced_count_thres': 80,
                    'max_rp_threshold': 0.25,
                    'sparse_search_volume': 30
                }
            },
            'vad': {
                'model_path': 'vad_multilingual_marblenet',
                'parameters': {
                    'onset': 0.8,
                    'offset': 0.6,
                    'pad_onset': 0.05,
                    'pad_offset': -0.05,
                    'min_duration_on': 0.2,
                    'min_duration_off': 0.2,
                    'filter_speech_first': True
                }
            },
            'asr': {
                'model_path': 'QuartzNet15x5Base-En',
                'parameters': {
                    'asr_based_vad': False,
                    'asr_based_vad_threshold': 1.0,
                    'asr_batch_size': None,
                    'decoder_delay_in_sec': 0.2,
                    'word_ts': False,
                    'sentence_ts': True,
                    'get_full_text': True,
                    'use_rnnt_decoder_timestamps': False,
                    'use_cer': False
                }
            }
        }
    }
    return OmegaConf.create(default_config)

def apply_custom_overrides(cfg, data_dir):
    """
    Apply custom configuration overrides.
    """
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
