"""Unified Speech Processing Pipeline - VAD → Whisper ASR → Diarization"""

import numpy as np
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path
import librosa

# Import from src modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.whisper_asr import WhisperASR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MOMBotPipeline:
    """
    Complete meeting transcription pipeline:
    
    1. Audio Loading - Load audio file
    2. Voice Activity Detection (VAD) - Detect speech segments
    3. Whisper ASR - Transcribe to text (ONLY ASR engine)
    4. Speaker Diarization - Identify who spoke when
    5. Output - Generate JSON with all metadata
    
    Features:
    - Supports all audio formats (WAV, MP3, M4A, FLAC, etc.)
    - Handles 15-20+ minute files
    - Automatic language detection
    - Multiple Whisper model sizes for speed/accuracy trade-off
    - Speaker identification with timing
    - Confidence scores and metadata
    """
    
    def __init__(self, whisper_model_size: str = "base", num_speakers: Optional[int] = None, sample_rate: int = 16000):
        """
        Initialize pipeline.
        
        Args:
            whisper_model_size (str): Model size - 'tiny', 'base', 'small', 'medium', 'large'
                For 15-20 min audio, use 'tiny' (fast) or 'base' (balanced)
            num_speakers (int, optional): Number of speakers (auto-detect if None)
            sample_rate (int): Target sample rate (default: 16000)
        """
        self.whisper_model_size = whisper_model_size
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.file_path = None
        
        # Initialize Whisper ASR
        self.whisper_asr = WhisperASR(model_size=whisper_model_size, device="cpu")
        
        # Results storage
        self.audio = None
        self.sr = None
        self.vad_segments = []
        self.transcripts = []
        self.speaker_assignments = []
        self.combined_results = None
    
    def process(self, audio_path: str) -> Dict:
        """
        Process audio through complete pipeline.
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            Dict: Complete results with segments, speakers, timestamps, transcripts
        """
        print("\n" + "="*80)
        print(f"🎙️  MOM-BOT SPEECH PROCESSING PIPELINE")
        print(f"📊 Whisper Model: {self.whisper_model_size}")
        print("="*80)
        
        # Step 1: Load Audio
        print("\n[1/4] Loading audio...")
        try:
            self.audio, self.sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            self.file_path = audio_path
            duration = len(self.audio) / self.sr
            print(f"✓ Audio loaded: {len(self.audio):,} samples, {duration:.2f}s @ {self.sr}Hz")
        except Exception as e:
            print(f"✗ Failed to load audio: {e}")
            return {"error": f"Audio loading failed: {e}"}
        
        # Step 2: VAD (Voice Activity Detection)
        print("\n[2/4] Detecting speech segments (VAD)...")
        try:
            self.vad_segments = self._detect_speech_segments()
            if not self.vad_segments:
                print("✗ No speech detected")
                return {"error": "No speech segments detected"}
            
            total_speech = sum(s['end'] - s['start'] for s in self.vad_segments)
            print(f"✓ Detected {len(self.vad_segments)} speech segments ({total_speech:.2f}s of speech)")
        except Exception as e:
            print(f"✗ VAD failed: {e}")
            return {"error": f"VAD failed: {e}"}
        
        # Step 3: Whisper ASR Transcription
        print("\n[3/4] Transcribing with Whisper ASR...")
        try:
            self.transcripts = self.whisper_asr.transcribe_segments(self.audio, self.vad_segments, self.sr)
            if not self.transcripts:
                print("✗ Transcription failed")
                return {"error": "Transcription failed"}
            
            avg_confidence = sum(t['confidence'] for t in self.transcripts) / len(self.transcripts)
            language = self.whisper_asr.get_language()
            print(f"✓ Transcribed {len(self.transcripts)} segments")
            print(f"  Language: {language}")
            print(f"  Avg confidence: {avg_confidence:.2f}")
        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            return {"error": f"Transcription failed: {e}"}
        
        # Step 4: Speaker Diarization
        print("\n[4/4] Identifying speakers (Diarization)...")
        try:
            self.speaker_assignments = self._diarize_speakers()
            num_speakers = len(set(s['speaker'] for s in self.speaker_assignments))
            print(f"✓ Identified {num_speakers} speakers")
        except Exception as e:
            print(f"⚠ Diarization failed: {e}")
            # Continue without speaker info
            self.speaker_assignments = [{'speaker': 'Speaker 1'} for _ in self.transcripts]
        
        # Combine all results
        self.combined_results = self._combine_results()
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE")
        print("="*80)
        
        return self.combined_results
    
    def _detect_speech_segments(self) -> List[Dict]:
        """
        Detect speech segments using marblenet VAD.
        
        Returns:
            List[Dict]: Speech segments with start/end times
        """
        try:
            from nemo.collections.asr.models import EncDecClassificationModel
            import torch
            
            # Load MarbleNet model
            vad_model = EncDecClassificationModel.from_pretrained(model_name="vad_marblenet")
            vad_model.eval()
            
            # Normalize audio
            audio = self.audio.astype(np.float32)
            audio_max = np.abs(audio).max()
            if audio_max > 0:
                audio = audio / audio_max
            
            # Process in 1-second chunks (16000 samples at 16kHz)
            frame_size = 16000
            predictions_list = []
            
            for i in range(0, len(audio), frame_size):
                frame = audio[i:i + frame_size]
                
                # Pad if necessary
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
                
                # Convert to tensor
                frame_tensor = torch.from_numpy(frame).float().unsqueeze(0)
                signal_length = torch.tensor([frame_tensor.shape[-1]], dtype=torch.long)
                
                # Run VAD
                with torch.no_grad():
                    logits = vad_model(input_signal=frame_tensor, input_signal_length=signal_length)
                
                # Get predictions (speech=1, non-speech=0)
                prediction = torch.argmax(logits, dim=1).cpu().numpy().flatten()
                predictions_list.append(prediction)
            
            # Combine and smooth predictions
            predictions = np.concatenate(predictions_list)
            predictions = self._smooth_predictions(predictions)
            
            # Extract segments from predictions
            segments = self._extract_vad_segments(predictions, frame_size)
            
            # Merge close segments
            segments = self._merge_close_segments(segments, gap_threshold=1.0)
            
            return segments
        
        except Exception as e:
            logger.warning(f"NeMo VAD failed, using simple energy-based detection: {e}")
            return self._simple_vad()
    
    def _smooth_predictions(self, predictions: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Smooth predictions to reduce noise."""
        try:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(predictions.astype(float), size=window_size, mode='nearest')
            return (smoothed > 0.5).astype(int)
        except:
            return predictions
    
    def _extract_vad_segments(self, predictions: np.ndarray, frame_size: int = 16000) -> List[Dict]:
        """Extract speech segments from predictions."""
        frame_shift = frame_size / self.sr  # Time per frame in seconds
        segments = []
        current_label = None
        segment_start = 0
        
        for frame_idx in range(len(predictions)):
            label = predictions[frame_idx]
            
            if label != current_label:
                if current_label == 1:  # Was in speech region
                    start_time = segment_start * frame_shift
                    end_time = frame_idx * frame_shift
                    if end_time - start_time > 0.2:  # Min 200ms
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'duration': end_time - start_time
                        })
                
                current_label = label
                segment_start = frame_idx
        
        # Handle final segment
        if current_label == 1:
            start_time = segment_start * frame_shift
            end_time = len(predictions) * frame_shift
            if end_time - start_time > 0.2:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
        
        return segments
    
    def _simple_vad(self) -> List[Dict]:
        """
        Improved energy-based VAD with better segmentation and merging.
        Optimized for meeting transcription with 2-3+ speakers.
        """
        # Compute RMS energy with balanced frame sizes
        frame_length = int(0.04 * self.sr)  # 40ms frames (good balance)
        hop_length = int(0.01 * self.sr)       # 10ms hop (smooth detection)
        
        energy = []
        for i in range(0, len(self.audio) - frame_length, hop_length):
            frame = self.audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            energy.append(rms)
        
        # Adaptive threshold for better separation
        avg_energy = np.mean(energy)
        std_energy = np.std(energy)
        threshold = avg_energy + 0.5 * std_energy  # Better than fixed multiplier
        voice_frames = np.array(energy) > threshold
        
        # Group consecutive frames
        segments = []
        in_speech = False
        start_idx = 0
        
        for idx, is_voice in enumerate(voice_frames):
            if is_voice and not in_speech:
                start_idx = idx
                in_speech = True
            elif not is_voice and in_speech:
                start_time = (start_idx * hop_length) / self.sr
                end_time = (idx * hop_length) / self.sr
                duration = end_time - start_time
                
                # Keep segments >= 200ms (shorter for multi-speaker meetings)
                if duration >= 0.2:
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': duration
                    })
                in_speech = False
        
        # Handle final segment
        if in_speech:
            start_time = (start_idx * hop_length) / self.sr
            end_time = len(self.audio) / self.sr
            duration = end_time - start_time
            if duration >= 0.2:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
        
        # Merge segments that are very close (< 1 second gap) - more aggressive merging
        segments = self._merge_close_segments(segments, gap_threshold=1.0)
        
        return segments
    
    def _merge_close_segments(self, segments: List[Dict], gap_threshold: float = 0.3) -> List[Dict]:
        """
        Merge segments that are very close together (gaps < gap_threshold).
        Conservative merging to preserve content while reducing fragmentation.
        
        Args:
            segments: List of segments
            gap_threshold: Minimum gap in seconds to keep segments separate (default: 0.3 = 300ms)
        
        Returns:
            Merged segments list
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            gap = next_seg['start'] - current['end']
            
            if gap < gap_threshold:
                # Merge segments
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
            else:
                # Keep current, start new
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged
    
    def _diarize_speakers(self) -> List[Dict]:
        """
        Improved diarization using hierarchical clustering.
        Falls back to NeMo ClusteringDiarizer if clustering fails.
        
        Process:
        1. Try hierarchical clustering with TitaNet embeddings
        2. Fall back to NeMo ClusteringDiarizer if that doesn't work well
        """
        from pipeline.diarization import (
            SpeakerEmbeddingExtractor,
            SlidingWindowSegmenter,
            HierarchicalDiarizer,
            SpeakerSmoother,
            OverlapHandler,
            SpeakerRegistryMatcher
        )
        
        print("\n  [HIERARCHICAL DIARIZATION] Attempting improved clustering approach...")
        
        try:
            print("  [Stage 1] Sliding window segmentation...")
            windows = SlidingWindowSegmenter.create_windows(
                segments=self.vad_segments,
                window_length=1.2,
                overlap_ratio=0.5
            )
            print(f"    ✓ {len(windows)} windows created")
            
            if len(windows) < 2:
                raise ValueError("Too few windows for clustering")
            
            print("  [Stage 2] Extract speaker embeddings...")
            extractor = SpeakerEmbeddingExtractor(device="cpu")
            embeddings = extractor.extract_embeddings_batch(self.audio, windows, self.sr)
            print(f"    ✓ Embeddings extracted: {embeddings.shape}")
            
            print("  [Stage 3] Hierarchical clustering...")
            diarizer = HierarchicalDiarizer(
                distance_metric="cosine",
                linkage_method="ward",
                threshold_type="distance"
            )
            # Use lower threshold for better speaker separation
            labels = diarizer.cluster(embeddings, threshold=0.55)
            num_speakers = len(np.unique(labels[labels >= 0]))
            print(f"    ✓ {num_speakers} speakers identified")
            
            # Check if clustering produced reasonable results (at least 2 speakers if available)
            if num_speakers < 2 and self.num_speakers is None:
                print("  ⚠ Only 1 speaker detected, trying with more aggressive threshold...")
                labels = diarizer.cluster(embeddings, threshold=0.40)
                num_speakers = len(np.unique(labels[labels >= 0]))
                print(f"    ✓ Adjusted: {num_speakers} speakers identified")
            
            print("  [Stage 4] Smooth and collapse segments...")
            smoothed_labels = SpeakerSmoother.smooth_labels(labels, min_segment_length=5)
            diar_segments = SpeakerSmoother.collapse_windows_to_segments(
                windows, smoothed_labels, min_duration=0.5
            )
            print(f"    ✓ {len(diar_segments)} clean segments")
            
            print("  [Stage 5] Detect overlapping speech...")
            diar_segments = OverlapHandler.detect_potential_overlaps(self.audio, diar_segments, self.sr)
            diar_segments = OverlapHandler.handle_overlaps_strategy(diar_segments)
            
            print("  [Stage 6] Match to speaker registry...")
            matcher = SpeakerRegistryMatcher()
            if matcher.load_registry():
                cluster_means = []
                for cluster_id in sorted(np.unique(labels)):
                    if cluster_id >= 0:
                        mask = labels == cluster_id
                        if np.sum(mask) > 0:
                            cluster_means.append(np.mean(embeddings[mask], axis=0))
                        else:
                            cluster_means.append(np.zeros(192))
                
                if cluster_means:
                    cluster_embs = np.array(cluster_means)
                    speaker_map = matcher.match_clusters_to_speakers(
                        cluster_embs, labels, similarity_threshold=0.75
                    )
                    
                    for segment in diar_segments:
                        cluster_id = segment.get('cluster_id', -1)
                        if cluster_id in speaker_map:
                            segment['speaker'] = speaker_map[cluster_id]
            
            # Map diarization segments to transcripts
            assignments = []
            for transcript in self.transcripts:
                segment_center = (transcript['start'] + transcript['end']) / 2
                speaker = 'Speaker 1'
                
                for dia_seg in diar_segments:
                    if dia_seg['start'] <= segment_center <= dia_seg['end']:
                        speaker = dia_seg['speaker']
                        break
                
                assignments.append({'speaker': speaker})
            
            print(f"✓ Used hierarchical clustering diarization")
            return assignments
        
        except Exception as e:
            print(f"  ⚠ Hierarchical clustering failed: {e}")
            print("  [FALLBACK] Using NeMo ClusteringDiarizer...")
            
            try:
                from nemo.collections.asr.models import ClusteringDiarizer
                
                diarizer = ClusteringDiarizer.from_pretrained("diar_titanet_large")
                diar_output = diarizer.predict(self.audio, sr=self.sr)
                
                assignments = []
                for transcript in self.transcripts:
                    segment_center = (transcript['start'] + transcript['end']) / 2
                    speaker_id = 1
                    
                    for spk_info in diar_output:
                        if spk_info['start'] <= segment_center <= spk_info['end']:
                            speaker_id = spk_info['speaker']
                            break
                    
                    assignments.append({'speaker': f'Speaker {speaker_id}'})
                
                print(f"✓ Used NeMo ClusteringDiarizer fallback")
                return assignments
            
            except Exception as e2:
                logger.warning(f"NeMo also failed: {e2}, using simple alternating")
                return [{'speaker': f'Speaker {i % 2 + 1}'} for i in range(len(self.transcripts))]
    
    def _combine_results(self) -> Dict:
        """Combine all results into structured output."""
        segments = []
        
        for idx, transcript in enumerate(self.transcripts):
            # Get speaker assignment
            speaker = 'Speaker 1'
            if idx < len(self.speaker_assignments):
                speaker = self.speaker_assignments[idx]['speaker']
            
            segment = {
                'segment_id': idx + 1,
                'start_time': f"{transcript['start']:.2f}s",
                'end_time': f"{transcript['end']:.2f}s",
                'duration': round(transcript['duration'], 2),
                'speaker': speaker,
                'text': transcript['text'],
                'confidence': transcript['confidence'],
                'language': transcript['language']
            }
            segments.append(segment)
        
        # Speaker summary
        speaker_summary = {}
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in speaker_summary:
                speaker_summary[speaker] = {'duration': 0, 'segments': 0, 'words': 0}
            
            speaker_summary[speaker]['duration'] += segment['duration']
            speaker_summary[speaker]['segments'] += 1
            speaker_summary[speaker]['words'] += len(segment['text'].split())
        
        return {
            'audio_file': Path(self.file_path).name if self.file_path else 'unknown',
            'total_duration': len(self.audio) / self.sr,
            'num_segments': len(segments),
            'num_speakers': len(speaker_summary),
            'asr_model': f'Whisper-{self.whisper_model_size}',
            'language': self.whisper_asr.get_language(),
            'segments': segments,
            'speaker_summary': {k: v for k, v in sorted(speaker_summary.items())}
        }
    
    def save_results(self, output_path: str) -> bool:
        """Save results to JSON."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.combined_results, f, indent=2)
            
            print(f"\n✓ Results saved: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
            return False
    
    def print_results(self) -> None:
        """Print formatted results."""
        if not self.combined_results:
            print("No results available")
            return
        
        print("\n" + "="*80)
        print("TRANSCRIPTION RESULTS")
        print("="*80)
        print(f"File: {self.combined_results['audio_file']} ({self.combined_results['total_duration']:.2f}s)")
        print(f"Model: {self.combined_results['asr_model']} | Language: {self.combined_results['language']}")
        print(f"Speakers: {self.combined_results['num_speakers']} | Segments: {self.combined_results['num_segments']}")
        print("-"*80)
        
        for seg in self.combined_results['segments']:
            print(f"\n[{seg['segment_id']}] {seg['speaker']} ({seg['start_time']} → {seg['end_time']}) [conf: {seg['confidence']:.2f}]")
            print(f"    {seg['text']}")
        
        print("\n" + "="*80)
        print("SPEAKER SUMMARY")
        print("="*80)
        for speaker, stats in self.combined_results['speaker_summary'].items():
            print(f"{speaker}: {stats['duration']:.2f}s ({stats['segments']} segments, {stats['words']} words)")
