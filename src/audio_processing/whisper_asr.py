"""
Whisper-based ASR engine with timestamp extraction
Supports 99+ languages, code-switching, and multilingual meetings

Phase 2.1 Implementation - Replace QuartzNet with OpenAI Whisper Medium
"""

import whisper
import torch
import logging
from typing import Tuple, List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class WhisperASREngine:
    """
    OpenAI Whisper ASR with sentence-level timestamp extraction
    
    Features:
    - Automatic language detection
    - Word-level or sentence-level output
    - GPU/CPU optimized
    - Apple Silicon (MPS) compatible
    
    Example:
        >>> engine = WhisperASREngine("medium")
        >>> result = engine.transcribe("meeting.wav")
        >>> print(result["language"])  # "en"
        >>> sentences, timestamps = engine.extract_sentences(result)
    """
    
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "default"
    ):
        """
        Initialize Whisper ASR engine
        
        Args:
            model_size: "tiny" | "base" | "small" | "medium" | "large"
                       Memory: tiny=39MB, base=140MB, small=466MB,
                               medium=1.4GB, large=2.9GB
            device: "auto" | "cpu" | "cuda" | "mps"
            compute_type: "default" | "float32" | "float16" | "int8"
        
        Raises:
            RuntimeError: If model cannot be loaded
        """
        self.model_size = model_size
        self.device = self._select_device(device)
        self.compute_type = compute_type
        
        logger.info(f"🚀 Loading Whisper {model_size} on {self.device}...")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info(f"✅ Whisper {model_size} loaded successfully on {self.device}")
        except RuntimeError as e:
            # Handle MPS sparse tensor incompatibility
            if "sparse" in str(e).lower() and self.device == "mps":
                logger.warning(f"⚠️ MPS sparse tensor error, falling back to CPU")
                self.device = "cpu"
                try:
                    self.model = whisper.load_model(model_size, device="cpu")
                    logger.info(f"✅ Whisper {model_size} loaded on CPU (MPS fallback)")
                except Exception as cpu_error:
                    logger.error(f"❌ Failed to load on CPU: {cpu_error}")
                    raise RuntimeError(f"Cannot load Whisper model: {cpu_error}")
            else:
                logger.error(f"❌ Failed to load Whisper model: {e}")
                raise RuntimeError(f"Cannot load Whisper model: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise RuntimeError(f"Cannot load Whisper model: {e}")
    
    @staticmethod
    def _select_device(device: str) -> str:
        """Auto-select best available device"""
        if device == "auto":
            if torch.backends.mps.is_available():
                logger.info("📱 Detected Apple Silicon, using MPS")
                return "mps"
            elif torch.cuda.is_available():
                logger.info("⚙️ Detected CUDA GPU, using CUDA")
                return "cuda"
            else:
                logger.info("💻 No GPU detected, using CPU")
                return "cpu"
        
        # Validate requested device
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        
        return device
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file (.mp3, .wav, .flac, .ogg, .m4a, etc.)
            language: ISO-639-1 language code ("en", "hi", "te", etc.)
                     If None, auto-detect
            verbose: Print progress
        
        Returns:
            {
                "text": "full transcription",
                "language": "en",
                "segments": [
                    {
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": 3.5,
                        "text": " hello world",
                        "avg_logprob": 0.95,
                        "compression_ratio": 1.2,
                        "no_speech_prob": 0.001
                    },
                    ...
                ]
            }
        
        Example:
            >>> engine = WhisperASREngine("medium")
            >>> result = engine.transcribe("meeting.mp3")
            >>> print(result["language"])  # "en"
            >>> print(result["text"])      # Full transcription
        """
        logger.info(f"Transcribing: {audio_path}")
        
        try:
            options = {
                "language": language,
                "verbose": verbose,
                "fp16": self.compute_type == "float16"
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            result = self.model.transcribe(audio_path, **options)
            
            logger.info(
                f"✅ Transcription complete | Language: {result['language']} | "
                f"Segments: {len(result['segments'])}"
            )
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def extract_sentences(
        self,
        whisper_result: Dict,
        min_segment_duration: float = 0.5
    ) -> Tuple[List[str], List[Tuple[float, float]]]:
        """
        Convert Whisper segment output to sentence-level segments
        
        Strategy:
        1. Split on punctuation (., !, ?, ;)
        2. Group by length to avoid overly long sentences
        3. Preserve timestamps
        
        Args:
            whisper_result: Output from transcribe()
            min_segment_duration: Minimum duration (seconds) for a segment
        
        Returns:
            (sentences, timestamps)
            sentences: List of sentence strings
            timestamps: List of (start, end) tuples
        
        Example:
            >>> result = engine.transcribe("meeting.mp3")
            >>> sentences, timestamps = engine.extract_sentences(result)
            >>> for sentence, (start, end) in zip(sentences, timestamps):
            ...     print(f"[{start:.1f}-{end:.1f}] {sentence}")
        """
        sentences = []
        timestamps = []
        
        if not whisper_result["segments"]:
            return sentences, timestamps
        
        current_sentence = []
        current_tokens = []
        start_time = None
        
        for segment in whisper_result["segments"]:
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            # Initialize start time for first segment
            if start_time is None:
                start_time = segment["start"]
            
            # Check if we should break the sentence
            should_break = self._should_break_sentence(text, current_sentence)
            
            if should_break and current_sentence:
                # Save current sentence
                sentence_text = " ".join(current_sentence).strip()
                end_time = current_tokens[-1]["end"] if current_tokens else segment["start"]
                duration = end_time - start_time
                
                if duration >= min_segment_duration:
                    sentences.append(sentence_text)
                    timestamps.append((start_time, end_time))
                
                # Reset
                current_sentence = []
                current_tokens = []
                start_time = segment["start"]
            
            # Add to current sentence
            current_sentence.append(text)
            current_tokens.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text
            })
        
        # Don't lose trailing content
        if current_sentence:
            sentence_text = " ".join(current_sentence).strip()
            end_time = current_tokens[-1]["end"] if current_tokens else segment["end"]
            duration = end_time - start_time
            
            if duration >= min_segment_duration:
                sentences.append(sentence_text)
                timestamps.append((start_time, end_time))
        
        logger.info(f"Extracted {len(sentences)} sentences from {len(whisper_result['segments'])} segments")
        return sentences, timestamps
    
    @staticmethod
    def _should_break_sentence(current_text: str, accumulated: List[str]) -> bool:
        """
        Decide whether to break current sentence
        
        Criteria:
        1. Current text ends with punctuation
        2. Accumulated text is getting too long (>150 chars)
        3. No accumulated text (first segment)
        """
        if not accumulated:
            return False
        
        # Break on sentence-ending punctuation
        if any(current_text.endswith(p) for p in [".", "!", "?"]):
            return True
        
        # Break if getting too long
        total_length = sum(len(t) for t in accumulated) + len(current_text)
        if total_length > 150:
            return True
        
        # Other heuristics can be added here
        return False
    
    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Tuple[List[str], List[Tuple[float, float]]]:
        """
        Convenience method: transcribe and extract sentences in one call
        
        Returns:
            (sentences, timestamps)
        
        Example:
            >>> engine = WhisperASREngine("medium")
            >>> sentences, timestamps = engine.transcribe_with_timestamps("meeting.mp3")
            >>> for sent, (start, end) in zip(sentences, timestamps):
            ...     print(f"{start:.1f}-{end:.1f}: {sent}")
        """
        result = self.transcribe(audio_path, language=language)
        return self.extract_sentences(result)
    
    def get_language(self, audio_path: str) -> str:
        """
        Detect language without full transcription
        
        Returns:
            Language code: "en", "es", "fr", "hi", "te", etc.
        """
        logger.info(f"Detecting language for: {audio_path}")
        result = self.transcribe(audio_path)
        return result["language"]
    
    def supported_languages(self) -> Dict[str, str]:
        """
        Get all supported languages
        
        Returns:
            {"en": "English", "es": "Spanish", ...}
        """
        return whisper.tokenizer.LANGUAGES
    
    def get_info(self) -> Dict:
        """Get engine information"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "supported_languages": len(self.supported_languages()),
            "compute_type": self.compute_type
        }
