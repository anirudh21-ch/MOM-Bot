"""Automatic Speech Recognition Engine using OpenAI Whisper - ONLY ASR ENGINE"""

import numpy as np
from typing import List, Dict, Optional
import whisper
import logging
import os

logger = logging.getLogger(__name__)


class WhisperASR:
    """
    Automatic Speech Recognition using OpenAI Whisper.
    ONLY ASR engine - Supports all audio files up to 20+ minutes.
    
    MODEL SIZES (for 15-20 minute audio):
    ├─ tiny    (39M)   - ✓ Fastest (~5-10s/4min), ~1GB RAM ⚡ RECOMMENDED for 15-20min
    ├─ base    (74M)   - ✓ Good balance (~10-15s/4min), ~2GB RAM [DEFAULT]
    ├─ small   (244M)  - ✓ Better accuracy (~30-45s/4min), ~4GB RAM
    ├─ medium  (769M)  - ✓ High accuracy (~90-120s/4min), ~8GB RAM
    └─ large   (1.5B)  - ✓ Highest accuracy (~2-3min/4min), ~10GB RAM
    
    For 15-20 minute audio:
    - Use 'tiny' for fastest processing (~2-3 minutes for 20min audio)
    - Use 'base' for balanced speed/accuracy
    - Use 'small' if accuracy is critical
    """
    
    MODEL_SIZES = {
        "tiny": {"params": "39M", "speed": "⚡ Fastest", "ram": "~1GB"},
        "base": {"params": "74M", "speed": "🚀 Fast", "ram": "~2GB"},
        "small": {"params": "244M", "speed": "📊 Medium", "ram": "~4GB"},
        "medium": {"params": "769M", "speed": "🐢 Slow", "ram": "~8GB"},
        "large": {"params": "1.5B", "speed": "🐌 Slowest", "ram": "~10GB"}
    }
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize Whisper ASR.
        
        Args:
            model_size (str): Model size - 'tiny', 'base', 'small', 'medium', 'large' (default: 'base')
            device (str): Device to use - 'cpu', 'cuda', 'mps' (default: 'cpu')
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model_size. Choose from: {list(self.MODEL_SIZES.keys())}")
        
        self.model_size = model_size
        self.device = device
        self.model = None
        self.detected_language = None
        self.transcripts = []
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper-{self.model_size} model...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"✓ Whisper-{self.model_size} loaded")
            print(f"✓ Whisper-{self.model_size} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Try CPU fallback
            if self.device != "cpu":
                logger.info("Falling back to CPU...")
                self.device = "cpu"
                self.model = whisper.load_model(self.model_size, device="cpu")
                logger.info(f"✓ Whisper-{self.model_size} loaded on CPU")
            else:
                raise
    
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Transcribe full audio.
        
        Args:
            audio (np.ndarray): Audio waveform
            sr (int): Sample rate
        
        Returns:
            Dict: Transcription with language metadata
        """
        try:
            import tempfile
            import soundfile as sf
            
            # Save to temp file (Whisper needs file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            sf.write(temp_path, audio.astype(np.float32), sr)
            
            # Transcribe
            result = self.model.transcribe(temp_path, verbose=False, fp16=False)
            
            self.detected_language = result.get('language', 'en')
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
            
            return {
                "text": result.get('text', ''),
                "language": self.detected_language,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"text": "", "language": None, "success": False}
    
    def transcribe_segments(self, audio: np.ndarray, segments: List[Dict], sr: int = 16000) -> List[Dict]:
        """
        Transcribe VAD segments.
        
        Args:
            audio (np.ndarray): Full audio
            segments (List[Dict]): VAD segments with start/end times
            sr (int): Sample rate
        
        Returns:
            List[Dict]: Transcribed segments
        """
        import tempfile
        import soundfile as sf
        
        results = []
        
        for idx, seg in enumerate(segments):
            try:
                # Extract segment
                start_idx = int(seg['start'] * sr)
                end_idx = int(seg['end'] * sr)
                seg_audio = audio[start_idx:end_idx]
                
                if len(seg_audio) == 0:
                    continue
                
                # Save to temp
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                
                sf.write(temp_path, seg_audio.astype(np.float32), sr)
                
                # Transcribe
                result = self.model.transcribe(temp_path, verbose=False, fp16=False)
                text = result.get('text', '').strip()
                
                results.append({
                    'segment_id': idx + 1,
                    'start': seg['start'],
                    'end': seg['end'],
                    'duration': seg['end'] - seg['start'],
                    'text': text,
                    'confidence': self._confidence(text),
                    'language': result.get('language', self.detected_language)
                })
                
                # Cleanup
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                logger.info(f"Segment {idx + 1}/{len(segments)}: {text[:50]}...")
            
            except Exception as e:
                logger.warning(f"Error on segment {idx + 1}: {e}")
                results.append({
                    'segment_id': idx + 1,
                    'start': seg['start'],
                    'end': seg['end'],
                    'duration': seg['end'] - seg['start'],
                    'text': '',
                    'confidence': 0.0,
                    'language': None
                })
        
        self.transcripts = results
        return results
    
    def _confidence(self, text: str) -> float:
        """Estimate confidence based on text length."""
        if not text.strip():
            return 0.0
        word_count = len(text.split())
        confidence = min(1.0, 0.5 + (word_count / 50.0))
        return round(confidence, 2)
    
    def get_language(self) -> Optional[str]:
        """Get detected language code."""
        return self.detected_language
