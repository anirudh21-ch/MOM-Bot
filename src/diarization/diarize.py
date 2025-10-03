import os
import json
import tempfile
import logging
from typing import Optional, Dict, Any, List, Tuple
from omegaconf import OmegaConf
import numpy as np
import librosa
import torch
from pydub import AudioSegment

# Try to import wget for fallback, but don't require it
try:
    import wget
    WGET_AVAILABLE = True
except ImportError:
    WGET_AVAILABLE = False

# NeMo imports
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

def diarize_audio(audio_file: str, num_speakers: Optional[int] = None) -> Dict[str, Any]:
    """
    Run speaker diarization using the exact approach from the working notebook.
    
    Args:
        audio_file: Path to audio file
        num_speakers: Optional number of speakers (if known)
        
    Returns:
        Dictionary containing diarization results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting notebook-style diarization for {audio_file}")
    
    # Setup MPS acceleration for Apple Silicon Macs
    import torch
    
    # Determine best available device
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("✅ Using MPS (Metal Performance Shaders) acceleration")
        # Don't force CPU-only mode for MPS
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("✅ Using CUDA GPU acceleration")
    else:
        device = "cpu"
        logger.info("⚠️ Using CPU (no GPU acceleration available)")
        # Only force single-threaded for CPU to avoid pickle errors
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(1)
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare mono audio file
            mono_audio_path = _prepare_mono_audio(audio_file, temp_dir)
            
            # Create manifest file
            manifest_path = _create_notebook_manifest(mono_audio_path, temp_dir)
            
            # Get diarization config using existing config_loader (no GitHub download)
            diar_config = _get_notebook_config(temp_dir)
            
            # Disable struct mode to allow parameter modifications
            OmegaConf.set_struct(diar_config, False)
            
            # Set up configuration exactly like notebook
            diar_config.diarizer.manifest_filepath = manifest_path
            diar_config.diarizer.out_dir = temp_dir
            diar_config.diarizer.speaker_embeddings.model_path = 'titanet_large'
            diar_config.diarizer.clustering.parameters.oracle_num_speakers = False
            
            # VAD and ASR settings
            diar_config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
            diar_config.diarizer.asr.model_path = "QuartzNet15x5Base-En"
            diar_config.diarizer.oracle_vad = False
            
            # Ensure ASR parameters exist before setting them
            if not hasattr(diar_config.diarizer.asr, 'parameters'):
                diar_config.diarizer.asr.parameters = OmegaConf.create({})
            
            diar_config.diarizer.asr.parameters.word_ts = False
            diar_config.diarizer.asr.parameters.sentence_ts = True
            diar_config.diarizer.asr.parameters.asr_based_vad = False
            diar_config.diarizer.asr.parameters.get_full_text = True
            diar_config.diarizer.asr.parameters.use_rnnt_decoder_timestamps = False
            diar_config.diarizer.asr.parameters.use_cer = False
            
            # Fix multiprocessing pickle error by ensuring all parameters exist
            # Ensure VAD parameters exist
            if not hasattr(diar_config.diarizer.vad, 'parameters'):
                diar_config.diarizer.vad.parameters = OmegaConf.create({})
            if not hasattr(diar_config.diarizer.vad.parameters, 'vad_inference'):
                diar_config.diarizer.vad.parameters.vad_inference = OmegaConf.create({})
            
            # Set batch sizes and disable multiprocessing
            diar_config.diarizer.vad.parameters.vad_inference.batch_size = 1
            diar_config.diarizer.vad.parameters.vad_inference.num_workers = 0
            
            # Ensure speaker embeddings parameters exist
            if not hasattr(diar_config.diarizer.speaker_embeddings, 'parameters'):
                diar_config.diarizer.speaker_embeddings.parameters = OmegaConf.create({})
                
            diar_config.diarizer.speaker_embeddings.parameters.batch_size = 1
            diar_config.diarizer.speaker_embeddings.parameters.num_workers = 0
            
            # Add global dataloader parameters
            diar_config.num_workers = 0
            if not hasattr(diar_config, 'dataloader_params'):
                diar_config.dataloader_params = OmegaConf.create({})
            diar_config.dataloader_params.num_workers = 0
            diar_config.dataloader_params.pin_memory = False
            
            # Configure device settings for MPS/CUDA/CPU
            device_config = "mps" if device == "mps" else ("cuda" if device == "cuda" else "cpu")
            
            # Set device for all NeMo components
            diar_config.device = device_config
            diar_config.diarizer.device = device_config
            if hasattr(diar_config.diarizer, 'vad'):
                diar_config.diarizer.vad.device = device_config
            if hasattr(diar_config.diarizer, 'asr'):
                diar_config.diarizer.asr.device = device_config  
            if hasattr(diar_config.diarizer, 'speaker_embeddings'):
                diar_config.diarizer.speaker_embeddings.device = device_config
            if hasattr(diar_config.diarizer, 'clustering'):
                diar_config.diarizer.clustering.device = device_config
                
            logger.info(f"📱 Configured all NeMo components for device: {device_config}")
            
            # Override num_speakers if provided
            if num_speakers:
                diar_config.diarizer.clustering.parameters.oracle_num_speakers = True
                diar_config.diarizer.clustering.parameters.max_num_speakers = num_speakers
            
            # Add missing sample_rate parameter (required by NeMo)
            if not hasattr(diar_config, 'sample_rate'):
                diar_config.sample_rate = 16000
                
            # Add verbose parameter for progress display
            if not hasattr(diar_config.diarizer, 'verbose'):
                diar_config.diarizer.verbose = True
            
            logger.info("Running ASR with timestamps...")
            
            # We'll patch the verbose attribute after diarizer instantiation
            
            # Monkey patch DataLoader to force num_workers=0 everywhere
            from torch.utils.data import DataLoader
            original_init = DataLoader.__init__
            
            def patched_init(self, *args, **kwargs):
                kwargs['num_workers'] = 0
                kwargs['pin_memory'] = False
                return original_init(self, *args, **kwargs)
            
            DataLoader.__init__ = patched_init
            
            # Initialize ASR decoder with timestamps (exactly like notebook)
            asr_decoder_ts = ASRDecoderTimeStamps(diar_config.diarizer)
            asr_model = asr_decoder_ts.set_asr_model()
            sentence_hyp, sentence_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
            
            # Restore original DataLoader
            DataLoader.__init__ = original_init
            
            logger.info("Running diarization...")
            
            # Apply DataLoader patch for diarization too
            from torch.utils.data import DataLoader
            original_init = DataLoader.__init__
            
            def patched_init(self, *args, **kwargs):
                kwargs['num_workers'] = 0
                kwargs['pin_memory'] = False
                kwargs['batch_size'] = 1  # Force batch size to 1 to avoid tensor issues
                return original_init(self, *args, **kwargs)
            
            DataLoader.__init__ = patched_init
            
            # Patch the collate function to handle 0-d tensors
            from nemo.collections.asr.data.audio_to_label import _fixed_seq_collate_fn
            import torch
            
            original_collate_fn = _fixed_seq_collate_fn
            
            def patched_collate_fn(dataset, batch):
                """Handle 0-d tensor iteration issues in collate function."""
                try:
                    return original_collate_fn(dataset, batch)
                except TypeError as e:
                    if "iteration over a 0-d tensor" in str(e):
                        # Handle the case where batch items are 0-d tensors
                        processed_batch = []
                        for item in batch:
                            if isinstance(item, tuple):
                                # Convert 0-d tensors to 1-d if needed
                                new_item = []
                                for element in item:
                                    if isinstance(element, torch.Tensor) and element.dim() == 0:
                                        new_item.append(element.unsqueeze(0))
                                    else:
                                        new_item.append(element)
                                processed_batch.append(tuple(new_item))
                            else:
                                processed_batch.append(item)
                        return original_collate_fn(dataset, processed_batch)
                    else:
                        raise e
            
            # Apply the collate function patch
            import nemo.collections.asr.data.audio_to_label
            nemo.collections.asr.data.audio_to_label._fixed_seq_collate_fn = patched_collate_fn
            
            # Patch ClusteringDiarizer to add verbose attribute globally before creating any instances
            from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer
            
            # Add verbose property to the entire ClusteringDiarizer class
            class VerboseProperty:
                def __get__(self, obj, objtype=None):
                    return False
                def __set__(self, obj, value):
                    pass  # Ignore attempts to set verbose
            
            # Store original verbose if it exists and apply patch
            original_verbose_attr = getattr(ClusteringDiarizer, 'verbose', None)
            ClusteringDiarizer.verbose = VerboseProperty()
            
            # Initialize offline diarization (exactly like notebook)
            asr_diar_offline = OfflineDiarWithASR(diar_config.diarizer)
            original_run_vad = ClusteringDiarizer._run_vad
            
            def patched_run_vad(self, manifest_vad_input):
                """Run VAD without verbose attribute dependency and fix JSON parsing issues."""
                import json
                import os
                import re
                
                # Use monkey patching to add verbose property that always returns False
                class VerboseProperty:
                    def __get__(self, obj, objtype=None):
                        return False
                    def __set__(self, obj, value):
                        pass  # Ignore attempts to set verbose
                
                # Store original if it exists
                verbose_was_attr = hasattr(type(self), 'verbose')
                original_verbose_attr = getattr(type(self), 'verbose', None) if verbose_was_attr else None
                
                # Set the property on the class
                type(self).verbose = VerboseProperty()
                
                # Create fixed manifest path
                fixed_manifest_path = manifest_vad_input.replace('.json', '_fixed.json')
                
                try:
                    # Read the original manifest content
                    with open(manifest_vad_input, 'r') as f_in:
                        content = f_in.read().strip()
                    
                    # Handle different JSON formatting issues
                    fixed_lines = []
                    
                    if content:
                        # Check if it's multiple JSON objects concatenated without newlines
                        if content.count('{') > 1 and '\n' not in content:
                            # Split by '}{' pattern and fix each part
                            parts = re.split(r'}\s*{', content)
                            
                            for i, part in enumerate(parts):
                                # Add missing braces
                                if i == 0:
                                    if not part.endswith('}'):
                                        part += '}'
                                elif i == len(parts) - 1:
                                    if not part.startswith('{'):
                                        part = '{' + part
                                else:
                                    if not part.startswith('{'):
                                        part = '{' + part
                                    if not part.endswith('}'):
                                        part += '}'
                                
                                # Try to parse and fix this JSON object
                                try:
                                    data = json.loads(part)
                                    # Replace 'infer' with 'speech'
                                    if data.get('label') == 'infer':
                                        data['label'] = 'speech'
                                    fixed_lines.append(json.dumps(data))
                                except json.JSONDecodeError:
                                    continue
                        else:
                            # Handle line-by-line processing
                            for line in content.split('\n'):
                                line = line.strip()
                                if line:
                                    try:
                                        data = json.loads(line)
                                        # Replace 'infer' with 'speech'
                                        if data.get('label') == 'infer':
                                            data['label'] = 'speech'
                                        fixed_lines.append(json.dumps(data))
                                    except json.JSONDecodeError:
                                        continue
                    
                    # Write fixed manifest
                    with open(fixed_manifest_path, 'w') as f_out:
                        for line in fixed_lines:
                            f_out.write(line + '\n')
                    
                    # Call original method with fixed manifest (verbose already set)
                    result = original_run_vad(self, fixed_manifest_path)
                    return result
                    
                finally:
                    # Restore original verbose attribute on the class
                    if verbose_was_attr:
                        type(self).verbose = original_verbose_attr
                    else:
                        if hasattr(type(self), 'verbose'):
                            delattr(type(self), 'verbose')
                    
                    # Clean up temporary file
                    if os.path.exists(fixed_manifest_path):
                        try:
                            os.remove(fixed_manifest_path)
                        except:
                            pass
            
            ClusteringDiarizer._run_vad = patched_run_vad
            
            try:
                # Run diarization with sentence timestamps
                diar_hyp, diar_score = asr_diar_offline.run_diarization(diar_config, sentence_ts_hyp)
            finally:
                # Restore original method
                ClusteringDiarizer._run_vad = original_run_vad
                # Restore original verbose attribute
                if original_verbose_attr is not None:
                    ClusteringDiarizer.verbose = original_verbose_attr
                else:
                    if hasattr(ClusteringDiarizer, 'verbose'):
                        delattr(ClusteringDiarizer, 'verbose')
            
            # Restore original DataLoader and collate function
            DataLoader.__init__ = original_init
            nemo.collections.asr.data.audio_to_label._fixed_seq_collate_fn = original_collate_fn
            
            logger.info("Processing results...")
            
            # Get transcript with speaker labels
            trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(
                diar_hyp, sentence_hyp, sentence_ts_hyp
            )
            
            # Process results like in notebook
            speakers = []
            rttm_output = []
            transcript = []
            
            # Get audio basename
            audio_basename = os.path.splitext(os.path.basename(mono_audio_path))[0]
            
            # Read RTTM file
            rttm_path = os.path.join(temp_dir, "pred_rttms", f"{audio_basename}.rttm")
            if os.path.exists(rttm_path):
                pred_labels = rttm_to_labels(rttm_path)
                
                # Convert to required format with error handling
                speaker_segments = {}
                for label in pred_labels:
                    try:
                        # Handle potential parsing issues
                        start_time = float(label[0]) if label[0] != '.' else 0.0
                        end_time = float(label[1]) if label[1] != '.' else 0.0
                        speaker_id = label[2]
                        
                        # Skip invalid entries
                        if start_time == 0.0 and end_time == 0.0:
                            continue
                            
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid label {label}: {e}")
                        continue
                    
                    if speaker_id not in speaker_segments:
                        speaker_segments[speaker_id] = []
                    speaker_segments[speaker_id].append((start_time, end_time))
                    
                    rttm_output.append({
                        'start': start_time,
                        'end': end_time,
                        'speaker': speaker_id
                    })
                
                # Convert to speakers format
                for speaker_id, segments in speaker_segments.items():
                    speakers.append({
                        'speaker': speaker_id,
                        'segments': segments
                    })
            
            # Get transcript
            if audio_basename in trans_info_dict:
                transcript = trans_info_dict[audio_basename].get('words', [])
            
            # Also read transcript file if available
            transcript_path = os.path.join(temp_dir, "pred_rttms", f"{audio_basename}.txt")
            transcript_text = []
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r') as f:
                    transcript_text = f.read().splitlines()
            
            return {
                'speakers': speakers,
                'rttm_output': rttm_output,
                'transcript': transcript,
                'transcript_text': transcript_text,
                'success': True,
                'sentence_hyp': sentence_hyp,
                'sentence_ts_hyp': sentence_ts_hyp,
                'trans_info_dict': trans_info_dict
            }
            
    except Exception as e:
        logger.error(f"Notebook-style diarization failed: {e}")
        raise e

def _prepare_mono_audio(audio_file: str, temp_dir: str) -> str:
    """Prepare mono audio file like in the notebook."""
    # Load audio and convert to mono
    audio = AudioSegment.from_file(audio_file)
    mono_audio = audio.set_channels(1)
    
    # Save mono file
    mono_audio_path = os.path.join(temp_dir, "mono_audio.wav")
    mono_audio.export(mono_audio_path, format="wav")
    
    return mono_audio_path

def _create_notebook_manifest(audio_file: str, temp_dir: str) -> str:
    """Create manifest file exactly like the notebook."""
    manifest_path = os.path.join(temp_dir, 'input_manifest.json')
    
    meta = {
        'audio_filepath': audio_file,
        'offset': 0,
        'duration': None,
        'label': 'speech',  # Use 'speech' instead of 'infer' for VAD compatibility
        'text': '-',
        'num_speakers': None,
        'rttm_filepath': None,
        'uem_filepath': None
    }
    
    with open(manifest_path, 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')
    
    return manifest_path

def _get_notebook_config(temp_dir: str) -> OmegaConf:
    """Get diarization config using local config files instead of GitHub."""
    # Use existing config_loader instead of downloading from GitHub
    try:
        # Import the existing config loader
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(project_root)
        
        from src.audio_processing.config_loader import load_config
        
        # Use the existing load_config function with meeting domain
        cfg = load_config(data_dir=temp_dir, domain_type="meeting")
        logger = logging.getLogger(__name__)
        logger.info("Using local config from config_loader")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not use config_loader, falling back to local config file: {e}")
        
        # Fallback: Use local config files directly
        DOMAIN_TYPE = "meeting"  # Can be meeting or telephonic
        CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
        
        # Look for config in the project's config directory first
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_config_path = os.path.join(project_root, "config", CONFIG_FILE_NAME)
        
        if os.path.exists(local_config_path):
            logger.info(f"Using local config file: {local_config_path}")
            cfg = OmegaConf.load(local_config_path)
        else:
            # Last resort: copy to temp dir if needed
            config_path = os.path.join(temp_dir, CONFIG_FILE_NAME)
            if not os.path.exists(config_path):
                # Only download if absolutely no local config exists and wget is available
                if WGET_AVAILABLE:
                    logger.warning("No local config found, downloading as last resort")
                    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
                    wget.download(CONFIG_URL, temp_dir)
                    cfg = OmegaConf.load(config_path)
                else:
                    logger.error("No local config found and wget not available")
                    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE_NAME}")
            else:
                cfg = OmegaConf.load(config_path)
    
    # Disable struct mode temporarily to add multiprocessing fixes
    OmegaConf.set_struct(cfg, False)
    
    # Add dataloader parameters to prevent pickle errors
    if not hasattr(cfg, 'dataloader_params'):
        cfg.dataloader_params = OmegaConf.create({})
    cfg.dataloader_params.num_workers = 0
    cfg.dataloader_params.pin_memory = False
    
    # Ensure all nested configs have num_workers = 0
    def set_num_workers_recursive(config_obj):
        if isinstance(config_obj, OmegaConf):
            for key, value in config_obj.items():
                if key == 'num_workers':
                    config_obj[key] = 0
                elif isinstance(value, OmegaConf):
                    set_num_workers_recursive(value)
    
    set_num_workers_recursive(cfg)
    
    # Re-enable struct mode
    OmegaConf.set_struct(cfg, True)
    
    return cfg