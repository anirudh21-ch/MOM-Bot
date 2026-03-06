"""Advanced Speaker Diarization Module

Improved diarization using:
1. Sliding window segmentation
2. TitaNet speaker embeddings
3. Agglomerative hierarchical clustering
4. Speaker smoothing & post-processing
5. Overlap detection
6. Speaker registry matching
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

logger = logging.getLogger(__name__)


class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings using NeMo TitaNet."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize TitaNet model.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        try:
            try:
                from nemo.collections.asr.models import EncDecSpeakerRecognitionModel
            except ImportError:
                # Try alternative import path
                from nemo.collections.speaker_recognition.models import EncDecSpeakerRecognitionModel
            
            import torch
            
            self.device = device
            self.model = EncDecSpeakerRecognitionModel.from_pretrained(
                model_name="titanet_large",
                map_location=device
            )
            self.model.eval()
            self.embedding_dim = 192  # TitaNet output dimension
            self.torch = torch
            self.is_initialized = True
            logger.info("✓ TitaNet model initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize TitaNet, using fallback: {e}")
            self.is_initialized = False
            self.embedding_dim = 192
    
    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding from audio segment.
        
        Args:
            audio: Audio waveform (mono)
            sr: Sample rate
        
        Returns:
            Embedding vector (192-dim)
        """
        if self.is_initialized:
            try:
                # Normalize audio
                audio = audio.astype(np.float32)
                audio_max = np.abs(audio).max()
                if audio_max > 0:
                    audio = audio / audio_max
                
                # Convert to tensor
                audio_tensor = self.torch.from_numpy(audio).float().unsqueeze(0)
                
                # Extract embedding
                with self.torch.no_grad():
                    embedding = self.model.get_embedding(audio_tensor)
                
                # Normalize embedding (L2)
                embedding = embedding.cpu().numpy()[0]
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                
                return embedding
            
            except Exception as e:
                logger.warning(f"TitaNet embedding failed: {e}, using fallback")
                return self._fallback_embedding(audio, sr)
        else:
            return self._fallback_embedding(audio, sr)
    
    def _fallback_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Fallback: Use MFCC-based features if TitaNet not available.
        This is less powerful than TitaNet but still useful for speaker clustering.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
        
        Returns:
            192-dim embedding using spectral features
        """
        try:
            import librosa
            
            # Extract MFCC features (13 coefficients + deltas = 26, repeat to 192)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=400, hop_length=160)
            
            # Compute statistics for each coefficient
            features = []
            for coeff in mfcc:
                features.extend([
                    np.mean(coeff), np.std(coeff),
                    np.min(coeff), np.max(coeff),
                    np.median(coeff)
                ])
            
            # Features: 13 coeffs * 5 stats = 65 features
            # Add spectral centroid, zero crossing rate, etc.
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            features.extend([
                np.mean(spectral_centroid), np.std(spectral_centroid),
                np.mean(zcr), np.std(zcr)
            ])
            
            # Pad or trim to 192-dim
            features = np.array(features)
            if len(features) < self.embedding_dim:
                features = np.pad(features, (0, self.embedding_dim - len(features)), mode='constant')
            else:
                features = features[:self.embedding_dim]
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-10)
            
            return features
        
        except Exception as e:
            logger.warning(f"Fallback embedding also failed: {e}, using zeros")
            return np.zeros(self.embedding_dim)
    
    def extract_embeddings_batch(self, 
                                 audio: np.ndarray,
                                 segments: List[Dict],
                                 sr: int = 16000) -> np.ndarray:
        """
        Extract embeddings for multiple segments.
        
        Args:
            audio: Full audio waveform
            segments: List of {'start': time, 'end': time} dicts
            sr: Sample rate
        
        Returns:
            Array of embeddings (N x 192)
        """
        embeddings = []
        
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) < sr // 10:  # Skip very short segments
                embeddings.append(np.zeros(self.embedding_dim))
            else:
                emb = self.extract_embedding(segment_audio, sr)
                embeddings.append(emb)
        
        return np.array(embeddings)


class SlidingWindowSegmenter:
    """Create overlapping segments from VAD output."""
    
    @staticmethod
    def create_windows(segments: List[Dict],
                      window_length: float = 1.2,
                      overlap_ratio: float = 0.5) -> List[Dict]:
        """
        Create sliding windows from VAD segments.
        
        Args:
            segments: VAD segments from pipeline
            window_length: Window length in seconds (1.0-1.5 recommended)
            overlap_ratio: Overlap percentage (0.5 = 50%)
        
        Returns:
            List of windowed segments with metadata
        """
        windows = []
        hop_length = window_length * (1 - overlap_ratio)
        window_id = 0
        
        for vad_idx, vad_segment in enumerate(segments):
            start = vad_segment['start']
            end = vad_segment['end']
            
            # Skip very short VAD segments
            if end - start < 0.5:
                continue
            
            # Create sliding windows
            current = start
            while current + window_length <= end:
                windows.append({
                    'window_id': window_id,
                    'start': current,
                    'end': min(current + window_length, end),
                    'duration': min(window_length, end - current),
                    'parent_vad_id': vad_idx
                })
                window_id += 1
                current += hop_length
            
            # Add final partial window if remainder > min length
            if current < end and end - current > 0.2:
                windows.append({
                    'window_id': window_id,
                    'start': current,
                    'end': end,
                    'duration': end - current,
                    'parent_vad_id': vad_idx
                })
                window_id += 1
        
        return windows


class HierarchicalDiarizer:
    """Hierarchical clustering-based speaker diarization."""
    
    def __init__(self,
                 distance_metric: str = "cosine",
                 linkage_method: str = "ward",
                 threshold_type: str = "distance"):
        """
        Initialize hierarchical diarizer.
        
        Args:
            distance_metric: 'cosine' or 'euclidean'
            linkage_method: 'ward', 'complete', 'average'
            threshold_type: 'distance' or 'maxclust'
        """
        self.distance_metric = distance_metric
        self.linkage_method = linkage_method
        self.threshold_type = threshold_type
        self.Z = None
        self.cluster_labels = None
    
    def cluster(self, embeddings: np.ndarray,
               threshold: float = 0.72,
               max_clusters: int = None) -> np.ndarray:
        """
        Perform hierarchical clustering on embeddings.
        
        Args:
            embeddings: Speaker embeddings (N x 192)
            threshold: Distance threshold for cutting dendrogram
            max_clusters: Maximum clusters (optional limit)
        
        Returns:
            Cluster labels (N,)
        """
        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int)
        
        # Compute distance matrix
        if self.distance_metric == "cosine":
            distances = cosine_distances(embeddings)
        else:
            distances = pdist(embeddings, metric=self.distance_metric)
        
        # Hierarchical clustering
        if isinstance(distances, np.ndarray) and distances.ndim == 2:
            distances = squareform(distances, checks=False)
        
        self.Z = linkage(distances, method=self.linkage_method)
        
        # Cut dendrogram
        if self.threshold_type == "distance":
            self.cluster_labels = fcluster(self.Z, t=threshold, criterion='distance')
        else:
            self.cluster_labels = fcluster(self.Z, t=max_clusters or 2, criterion='maxclust')
        
        # Relabel to 0-indexed
        self.cluster_labels -= 1
        
        return self.cluster_labels


class SpeakerSmoother:
    """Smooth and stabilize speaker labels."""
    
    @staticmethod
    def smooth_labels(cluster_labels: np.ndarray,
                     min_segment_length: int = 5) -> np.ndarray:
        """
        Smooth speaker labels using majority voting and merging.
        
        Args:
            cluster_labels: Raw cluster assignments from diarizer (N,)
            min_segment_length: Min windows for a speaker turn
        
        Returns:
            Smoothed labels (N,)
        """
        smoothed = cluster_labels.copy()
        
        # Merge adjacent short segments
        i = 0
        while i < len(smoothed):
            # Find run of same label
            j = i
            while j < len(smoothed) and smoothed[j] == smoothed[i]:
                j += 1
            
            run_length = j - i
            
            # If run too short, reassign to neighbor
            if 1 <= run_length < min_segment_length:
                if i > 0 and j < len(smoothed):
                    right_label = smoothed[j]
                    smoothed[i:j] = right_label
                elif i > 0:
                    smoothed[i:j] = smoothed[i - 1]
                elif j < len(smoothed):
                    smoothed[i:j] = smoothed[j]
            
            i = j
        
        return smoothed
    
    @staticmethod
    def collapse_windows_to_segments(windows: List[Dict],
                                     smoothed_labels: np.ndarray,
                                     min_duration: float = 0.5) -> List[Dict]:
        """
        Convert window-level labels back to continuous segments.
        
        Args:
            windows: Sliding windows from segmenter
            smoothed_labels: Smoothed cluster labels
            min_duration: Minimum segment duration
        
        Returns:
            Segments with speaker ID and timing
        """
        if len(windows) != len(smoothed_labels):
            raise ValueError(f"Mismatch: {len(windows)} windows vs {len(smoothed_labels)} labels")
        
        segments = []
        current_speaker = None
        segment_start = None
        segment_windows = []
        
        for idx, (window, label) in enumerate(zip(windows, smoothed_labels)):
            if label != current_speaker:
                # Save previous segment
                if current_speaker is not None and segment_windows:
                    duration = segment_windows[-1]['end'] - segment_start
                    if duration >= min_duration:
                        segments.append({
                            'speaker': f'Speaker {int(current_speaker) + 1}',
                            'cluster_id': int(current_speaker),
                            'start': segment_start,
                            'end': segment_windows[-1]['end'],
                            'duration': duration,
                            'num_windows': len(segment_windows),
                            'confidence': min(1.0, len(segment_windows) / 10.0)
                        })
                
                # Start new segment
                current_speaker = label
                segment_start = window['start']
                segment_windows = [window]
            else:
                segment_windows.append(window)
        
        # Handle final segment
        if current_speaker is not None and segment_windows:
            duration = segment_windows[-1]['end'] - segment_start
            if duration >= min_duration:
                segments.append({
                    'speaker': f'Speaker {int(current_speaker) + 1}',
                    'cluster_id': int(current_speaker),
                    'start': segment_start,
                    'end': segment_windows[-1]['end'],
                    'duration': duration,
                    'num_windows': len(segment_windows),
                    'confidence': min(1.0, len(segment_windows) / 10.0)
                })
        
        return segments


class OverlapHandler:
    """Detect and handle overlapping speech regions."""
    
    @staticmethod
    def detect_potential_overlaps(audio: np.ndarray,
                                  segments: List[Dict],
                                  sr: int = 16000,
                                  threshold_db: float = -20) -> List[Dict]:
        """
        Detect regions with high energy (potential overlaps).
        
        Args:
            audio: Audio waveform
            segments: Diarization segments
            sr: Sample rate
            threshold_db: Energy threshold above mean
        
        Returns:
            Segments flagged as potential overlaps
        """
        # Compute energy contour
        frame_length = 512
        hop_length = 160
        energy = []
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sqrt(np.mean(frame ** 2)))
        
        energy = np.array(energy)
        energy_db = 20 * np.log10(np.clip(energy, 1e-10, None))
        
        # Find high-energy regions
        mean_energy = np.mean(energy_db)
        
        for segment in segments:
            start_frame = int(segment['start'] * sr / hop_length)
            end_frame = int(segment['end'] * sr / hop_length)
            
            segment_energy = energy_db[max(0, start_frame):min(len(energy_db), end_frame)]
            peak_energy = np.max(segment_energy) if len(segment_energy) > 0 else 0
            
            # If peak energy very high relative to mean, flag as potential overlap
            if peak_energy - mean_energy > threshold_db:
                segment['potential_overlap'] = True
                segment['peak_energy_db'] = float(peak_energy)
            else:
                segment['potential_overlap'] = False
                segment['peak_energy_db'] = float(peak_energy)
        
        return segments
    
    @staticmethod
    def handle_overlaps_strategy(segments: List[Dict]) -> List[Dict]:
        """
        Apply practical overlap handling strategies.
        
        Returns:
            Enhanced segments with overlap metadata
        """
        enhanced = []
        
        for segment in segments:
            enhanced_seg = segment.copy()
            
            # If flagged as overlap, reduce confidence
            if enhanced_seg.get('potential_overlap', False):
                enhanced_seg['handling'] = 'overlap_region'
                enhanced_seg['confidence'] = enhanced_seg.get('confidence', 1.0) * 0.7
            else:
                enhanced_seg['handling'] = 'normal'
            
            enhanced.append(enhanced_seg)
        
        return enhanced


class SpeakerRegistryMatcher:
    """Match clusters to known speaker embeddings."""
    
    def __init__(self, registry_path: str = "config/speaker_registry.json"):
        """
        Initialize speaker registry.
        
        Args:
            registry_path: Path to speaker registry JSON
        """
        self.registry = {}
        self.registry_path = Path(registry_path)
    
    def register_speaker(self, name: str, embedding: np.ndarray) -> None:
        """
        Register a known speaker's embedding.
        
        Args:
            name: Speaker name
            embedding: Normalized speaker embedding (192-dim)
        """
        self.registry[name] = embedding.tolist()
        self._save_registry()
    
    def match_clusters_to_speakers(self,
                                   cluster_embeddings: np.ndarray,
                                   cluster_labels: np.ndarray,
                                   similarity_threshold: float = 0.75) -> Dict:
        """
        Match diarization clusters to registered speakers.
        
        Args:
            cluster_embeddings: Mean embedding per cluster (K x 192)
            cluster_labels: Cluster IDs (N,)
            similarity_threshold: Cosine similarity threshold
        
        Returns:
            Dict mapping cluster_id -> speaker_name
        """
        cluster_to_speaker = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id < 0 or cluster_id >= len(cluster_embeddings):
                cluster_to_speaker[int(cluster_id)] = f"Speaker {int(cluster_id) + 1}"
                continue
            
            cluster_emb = cluster_embeddings[cluster_id].reshape(1, -1)
            
            # Find best match in registry
            best_match = None
            best_score = 0
            
            for speaker_name, emb_list in self.registry.items():
                emb = np.array(emb_list).reshape(1, -1)
                similarity = cosine_similarity(cluster_emb, emb)[0][0]
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = speaker_name
            
            # Assign based on threshold
            if best_match and best_score >= similarity_threshold:
                cluster_to_speaker[int(cluster_id)] = best_match
                logger.info(f"✓ Cluster {cluster_id} -> {best_match} (sim: {best_score:.3f})")
            else:
                cluster_to_speaker[int(cluster_id)] = f"Speaker {int(cluster_id) + 1}"
        
        return cluster_to_speaker
    
    def load_registry(self) -> bool:
        """Load registry from JSON. Returns True if successful."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    self.registry = json.load(f)
                logger.info(f"✓ Loaded registry with {len(self.registry)} speakers")
                return True
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return False
        return False
    
    def _save_registry(self) -> None:
        """Save registry to JSON."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        logger.info(f"✓ Saved registry to {self.registry_path}")
