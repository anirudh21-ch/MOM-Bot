"""Voice Activity Detection Engine using NVIDIA NeMo"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path


class VADEngine:
    """
    Voice Activity Detection Engine using NeMo pretrained models.
    Detects speech segments and non-speech (silence) segments.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize VAD Engine.
        
        Args:
            sample_rate (int): Audio sampling rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        self.vad_model = None
        self.speech_segments = []
        self.silence_segments = []
        self.audio_length = 0
        self.is_initialized = False
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize NeMo VAD model.
        This will download the pretrained model on first use.
        """
        try:
            # Import here to avoid dependency issues if nemo not installed
            from nemo.collections.asr.models import EncDecClassificationModel
            
            # Load pretrained VAD model from NeMo
            # marblenet is a lightweight VAD model optimized for speech detection
            self.vad_model = EncDecClassificationModel.from_pretrained(
                model_name="vad_marblenet",
                map_location="cpu"  # Change to "cuda" if GPU available
            )
            
            # Set model to evaluation mode (disables augmentation, dropout, etc)
            self.vad_model.eval()
            
            self.is_initialized = True
            print("✓ VAD Model initialized successfully (marblenet)")
        
        except ImportError:
            print("⚠ NeMo not installed. Install with: pip install nemo_toolkit[asr]")
            self.is_initialized = False
        except Exception as e:
            print(f"⚠ Failed to initialize VAD model: {str(e)}")
            self.is_initialized = False
    
    def detect_speech_segments(self, audio: np.ndarray) -> Dict:
        """
        Detect speech segments in audio.
        
        Args:
            audio (np.ndarray): Audio waveform (mono)
        
        Returns:
            Dict: Contains speech and silence segments with timestamps
        """
        if not self.is_initialized:
            raise RuntimeError("VAD model not initialized. Check NeMo installation.")
        
        self.audio_length = len(audio) / self.sample_rate
        
        try:
            import torch
            
            # Normalize audio
            audio = audio.astype(np.float32)
            audio_max = np.abs(audio).max()
            if audio_max > 0:
                audio = audio / audio_max
            
            # Process audio in larger chunks (16000 samples = 1 second at 16kHz)
            frame_size = 16000
            predictions_list = []
            
            for i in range(0, len(audio), frame_size):
                frame = audio[i:i+frame_size]
                
                # Pad frame if necessary
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
                
                # Convert to tensor (batch, time) - 2D tensor
                frame_tensor = torch.from_numpy(frame).float().unsqueeze(0)
                signal_length = torch.tensor([frame_tensor.shape[-1]], dtype=torch.long)
                
                # Run VAD model in eval mode
                with torch.no_grad():
                    self.vad_model.eval()
                    logits = self.vad_model(input_signal=frame_tensor, input_signal_length=signal_length)
                
                # Get prediction (speech=1, non-speech=0)
                # logits shape is [batch_size, num_labels, time_steps]
                prediction = torch.argmax(logits, dim=1).cpu().numpy()  # [batch_size, time_steps]
                # Flatten to 1D and append
                predictions_list.append(prediction.flatten())
            
            # Combine predictions
            predictions = np.concatenate(predictions_list)
            
            # Smooth predictions to reduce noise
            predictions = self._smooth_predictions(predictions)
            
            # Extract segments
            self._extract_segments(predictions)
            
            return self._format_output()
        
        except Exception as e:
            print(f"Error during speech detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _smooth_predictions(self, predictions: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Apply smoothing to reduce noise in predictions.
        
        Args:
            predictions (np.ndarray): Raw predictions
            window_size (int): Window size for smoothing
        
        Returns:
            np.ndarray: Smoothed predictions
        """
        from scipy.ndimage import uniform_filter1d
        
        # Apply uniform filter for smoothing
        smoothed = uniform_filter1d(predictions.astype(float), size=window_size, mode='nearest')
        return (smoothed > 0.5).astype(int)
    
    def _extract_segments(self, predictions: np.ndarray) -> None:
        """
        Extract continuous speech and silence segments from predictions.
        
        Args:
            predictions (np.ndarray): Binary predictions (1=speech, 0=silence)
        """
        self.speech_segments = []
        self.silence_segments = []
        
        # Frame parameters (frame_size = 16000 samples at 16kHz = 1 second per frame)
        frame_size = 16000
        frame_shift = frame_size / self.sample_rate  # 1 second per frame
        
        current_label = None
        segment_start = 0
        
        for frame_idx in range(len(predictions)):
            label = predictions[frame_idx]
            
            if label != current_label:
                if current_label is not None:
                    # Save previous segment
                    start_time = segment_start * frame_shift
                    end_time = frame_idx * frame_shift
                    
                    if current_label == 1:  # Speech segment
                        self.speech_segments.append({
                            'start': start_time,
                            'end': end_time,
                            'duration': end_time - start_time
                        })
                    else:  # Silence segment
                        self.silence_segments.append({
                            'start': start_time,
                            'end': end_time,
                            'duration': end_time - start_time
                        })
                
                current_label = label
                segment_start = frame_idx
        
        # Handle last segment
        if current_label is not None:
            start_time = segment_start * frame_shift
            end_time = len(predictions) * frame_shift
            
            if current_label == 1:
                self.speech_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
            else:
                self.silence_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
    
    def _format_output(self) -> Dict:
        """Format VAD results as dictionary."""
        return {
            'audio_duration': self.audio_length,
            'total_speech_duration': sum(seg['duration'] for seg in self.speech_segments),
            'total_silence_duration': sum(seg['duration'] for seg in self.silence_segments),
            'speech_segments': self.speech_segments,
            'silence_segments': self.silence_segments,
            'num_speech_regions': len(self.speech_segments),
            'num_silence_regions': len(self.silence_segments)
        }
    
    def get_speech_segments(self) -> List[Dict]:
        """Get detected speech segments."""
        return self.speech_segments
    
    def get_silence_segments(self) -> List[Dict]:
        """Get detected silence segments."""
        return self.silence_segments
    
    def extract_speech_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract only speech portions from audio.
        
        Args:
            audio (np.ndarray): Full audio waveform
        
        Returns:
            Tuple[np.ndarray, List[Dict]]: Speech-only audio and segment info
        """
        if not self.speech_segments:
            return audio, self.speech_segments
        
        speech_audio_list = []
        
        for segment in self.speech_segments:
            start_sample = int(segment['start'] * self.sample_rate)
            end_sample = int(segment['end'] * self.sample_rate)
            speech_audio_list.append(audio[start_sample:end_sample])
        
        # Concatenate all speech segments
        speech_audio = np.concatenate(speech_audio_list) if speech_audio_list else np.array([])
        
        return speech_audio, self.speech_segments
    
    def save_results(self, output_path: str) -> None:
        """
        Save VAD results to JSON file.
        
        Args:
            output_path (str): Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'audio_duration': self.audio_length,
            'speech_duration': sum(seg['duration'] for seg in self.speech_segments),
            'silence_duration': sum(seg['duration'] for seg in self.silence_segments),
            'speech_segments': self.speech_segments,
            'silence_segments': self.silence_segments
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ VAD results saved to: {output_path}")
    
    def print_summary(self) -> None:
        """Print a summary of VAD results."""
        print("\n" + "="*70)
        print("VOICE ACTIVITY DETECTION (VAD) SUMMARY")
        print("="*70)
        print(f"Total Audio Duration:     {self.audio_length:.2f}s")
        print(f"Speech Duration:          {sum(seg['duration'] for seg in self.speech_segments):.2f}s")
        print(f"Silence Duration:         {sum(seg['duration'] for seg in self.silence_segments):.2f}s")
        print(f"Number of Speech Regions: {len(self.speech_segments)}")
        print(f"Number of Silence Regions:{len(self.silence_segments)}")
        print("-"*70)
        print("Speech Segments:")
        for i, seg in enumerate(self.speech_segments, 1):
            print(f"  [{i}] {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {seg['duration']:.2f}s)")
        print("="*70 + "\n")
