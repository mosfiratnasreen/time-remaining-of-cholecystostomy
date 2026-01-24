"""
Cholec80 Dataset for Task A: Remaining Time Prediction

This module predicts:
1. Remaining time in current surgical phase
2. Start and end times of all upcoming phases
3. Remaining surgery duration (RSD)

These predictions will then be used as inputs for Task B (tool anticipation).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import json


###################################################################################################################################
# constants
###################################################################################################################################

TOOLS = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']
NUM_TOOLS = len(TOOLS)

PHASES = [
    'Preparation',
    'CalotTriangleDissection',
    'ClippingCutting', 
    'GallbladderDissection',
    'GallbladderPackaging',
    'CleaningCoagulation',
    'GallbladderRetraction'
]
NUM_PHASES = len(PHASES)
PHASE_TO_IDX = {phase: idx for idx, phase in enumerate(PHASES)}
IDX_TO_PHASE = {idx: phase for phase, idx in PHASE_TO_IDX.items()}

# standard train/test split (following Cholec80 convention)
TRAIN_VIDEOS = list(range(1, 41))   # Videos 1-40
TEST_VIDEOS = list(range(41, 81))   # Videos 41-80


####################################################################################################################################
# data loading
####################################################################################################################################

def load_phase_annotations(phase_dir: Path, video_id: int) -> pd.DataFrame:
    """loads phase annotations for a video."""
    filename = phase_dir / f"video{video_id:02d}-phase.txt"
    df = pd.read_csv(filename, sep='\t')
    return df


def load_tool_annotations(tool_dir: Path, video_id: int) -> pd.DataFrame:
    """loads tool annotations for a video."""
    filename = tool_dir / f"video{video_id:02d}-tool.txt"
    df = pd.read_csv(filename, sep='\t')
    return df


def load_features(features_dir: Path, video_id: int) -> np.ndarray:
    """loads pre-extracted ResNet features for a video."""
    filename = features_dir / f"video{video_id:02d}.npy"
    return np.load(filename)


###################################################################################################################################
# statistics computed from training set 
###################################################################################################################################

@dataclass
class PhaseStatistics:
    """
    statistics for phase durations computed from training set.
    used for normalising predictions and computing pace signals.
    """
    mean_durations: Dict[str, float] = field(default_factory=dict)
    std_durations: Dict[str, float] = field(default_factory=dict)
    mean_surgery_duration: float = 0.0
    std_surgery_duration: float = 1.0
    
    @classmethod
    def compute_from_videos(cls, phase_dir: Path, video_ids: List[int]) -> 'PhaseStatistics':
        """computes phase duration statistics from a set of videos."""
        phase_durations = {phase: [] for phase in PHASES}
        surgery_durations = []
        
        for video_id in video_ids:
            df = load_phase_annotations(phase_dir, video_id)
            surgery_durations.append(len(df) / 25.0)  # total duration in seconds
            
            # finds phase segments
            phase_changes = df['Phase'].ne(df['Phase'].shift()).cumsum()
            segments = df.groupby(phase_changes).agg({
                'Phase': 'first',
                'Frame': 'count'
            })
            segments.columns = ['phase', 'duration']
            
            for _, row in segments.iterrows():
                phase = row['phase']
                duration_sec = row['duration'] / 25.0  # 25 fps
                if phase in phase_durations:
                    phase_durations[phase].append(duration_sec)
        
        mean_durations = {
            phase: np.mean(durations) if durations else 0.0
            for phase, durations in phase_durations.items()
        }
        std_durations = {
            phase: np.std(durations) if len(durations) > 1 else 1.0
            for phase, durations in phase_durations.items()
        }
        
        return cls(
            mean_durations=mean_durations,
            std_durations=std_durations,
            mean_surgery_duration=np.mean(surgery_durations),
            std_surgery_duration=np.std(surgery_durations)
        )
    
    def save(self, path: Path):
        """saves statistics to JSON file."""
        data = {
            'mean_durations': self.mean_durations,
            'std_durations': self.std_durations,
            'mean_surgery_duration': self.mean_surgery_duration,
            'std_surgery_duration': self.std_surgery_duration
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'PhaseStatistics':
        """loads statistics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


###################################################################################################################################
# video data container with phase timeline
###################################################################################################################################

@dataclass
class PhaseSegment:
    """represents a single phase segment in a surgery."""
    phase_idx: int
    phase_name: str
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds


class VideoData:
    """
    container for all data from a single video, aligned to 1fps.
    includes phase timeline for computing remaining times.
    """
    
    def __init__(
        self,
        video_id: int,
        features_dir: Path,
        phase_dir: Path,
        tool_dir: Path
    ):
        self.video_id = video_id
        
        # load features (already at 1fps)
        self.features = load_features(features_dir, video_id)
        self.duration = len(self.features)  # in seconds
        
        # load and process annotations
        self._load_annotations(phase_dir, tool_dir)
        self._build_phase_timeline(phase_dir)
    
    def _load_annotations(self, phase_dir: Path, tool_dir: Path):
        """loads and align annotations to 1fps."""
        phase_df = load_phase_annotations(phase_dir, self.video_id)
        tool_df = load_tool_annotations(tool_dir, self.video_id)
        
        T = self.duration
        
        # phase labels at 1fps
        phases = []
        for t in range(T):
            frame_idx = t * 25
            if frame_idx < len(phase_df):
                phase_name = phase_df.iloc[frame_idx]['Phase']
            else:
                phase_name = phase_df.iloc[-1]['Phase']
            phases.append(PHASE_TO_IDX.get(phase_name, 0))
        self.phases = np.array(phases, dtype=np.int64)
        
        # tool labels at 1fps
        tools = np.zeros((T, NUM_TOOLS), dtype=np.float32)
        for _, row in tool_df.iterrows():
            t = row['Frame'] // 25
            if t < T:
                for i, tool in enumerate(TOOLS):
                    tools[t, i] = row[tool]
        self.tools = tools
    
    def _build_phase_timeline(self, phase_dir: Path):
        """builds timeline of phase segments with start/end times."""
        phase_df = load_phase_annotations(phase_dir, self.video_id)
        
        self.phase_segments: List[PhaseSegment] = []
        
        # finds phase boundaries
        phase_changes = phase_df['Phase'].ne(phase_df['Phase'].shift()).cumsum()
        segments = phase_df.groupby(phase_changes).agg({
            'Phase': 'first',
            'Frame': ['min', 'max', 'count']
        })
        segments.columns = ['phase', 'start_frame', 'end_frame', 'duration_frames']
        
        for _, row in segments.iterrows():
            phase_name = row['phase']
            segment = PhaseSegment(
                phase_idx=PHASE_TO_IDX.get(phase_name, 0),
                phase_name=phase_name,
                start_time=row['start_frame'] / 25.0,
                end_time=(row['end_frame'] + 1) / 25.0,  # +1 for inclusive
                duration=row['duration_frames'] / 25.0
            )
            self.phase_segments.append(segment)
    
    def get_current_segment_idx(self, t: float) -> int:
        """get index of phase segment containing time t."""
        for i, seg in enumerate(self.phase_segments):
            if seg.start_time <= t < seg.end_time:
                return i
        return len(self.phase_segments) - 1  # Return last if past end
    
    def get_remaining_phase_time(self, t: int) -> float:
        """get remaining time in current phase at time t (seconds)."""
        seg_idx = self.get_current_segment_idx(t)
        segment = self.phase_segments[seg_idx]
        return max(0.0, segment.end_time - t)
    
    def get_remaining_surgery_time(self, t: int) -> float:
        """get remaining surgery duration at time t (seconds)."""
        return max(0.0, self.duration - t)
    
    def get_elapsed_phase_time(self, t: int) -> float:
        """get elapsed time in current phase at time t (seconds)."""
        seg_idx = self.get_current_segment_idx(t)
        segment = self.phase_segments[seg_idx]
        return t - segment.start_time
    
    def get_future_phase_times(self, t: int) -> Dict[str, Tuple[float, float]]:
        """
        get start and end times of all phases that haven't completed yet.
        returns dict mapping phase_name -> (start_time, end_time) relative to current time t.
        times are relative (0 = now).
        """
        current_seg_idx = self.get_current_segment_idx(t)
        future_times = {}
        
        for i, seg in enumerate(self.phase_segments):
            if i >= current_seg_idx:  # Current and future phases
                # Convert to time relative to t
                rel_start = max(0.0, seg.start_time - t)
                rel_end = max(0.0, seg.end_time - t)
                future_times[seg.phase_name] = (rel_start, rel_end)
        return future_times
    
    def get_upcoming_phases_tensor(self, t: int) -> np.ndarray:
        """
        get tensor representation of upcoming phase times.
        
        returns array of shape (NUM_PHASES, 3):
            - [:, 0]: will this phase occur? (0 or 1)
            - [:, 1]: relative start time (seconds from now, 0 if current/past)
            - [:, 2]: relative end time (seconds from now)
        
        this format allows predicting all future phase times in one output.
        """
        result = np.zeros((NUM_PHASES, 3), dtype=np.float32)
        current_seg_idx = self.get_current_segment_idx(t)
        
        for i, seg in enumerate(self.phase_segments):
            phase_idx = seg.phase_idx
            
            if i > current_seg_idx:  # Future phase
                result[phase_idx, 0] = 1.0  # Will occur
                result[phase_idx, 1] = seg.start_time - t  # Relative start
                result[phase_idx, 2] = seg.end_time - t    # Relative end
            elif i == current_seg_idx:  # Current phase
                result[phase_idx, 0] = 1.0  # Currently occurring
                result[phase_idx, 1] = 0.0  # Already started
                result[phase_idx, 2] = seg.end_time - t  # Relative end
            # Past phases remain zeros
        
        return result

###################################################################################################################################
# Task A Dataset: Remaining Time Prediction
# ============================================================================

class TaskADataset(Dataset):
    """
    Dataset for Task A: Remaining Time Prediction
    
    For each sample at time t:
    
    Inputs:
        - features: ResNet visual features (2048-dim)
        - phase: Current phase (one-hot or index)
        - elapsed_time: Time elapsed in current phase
        - surgery_progress: Fraction of surgery completed (proxy)
    
    Targets:
        - remaining_phase_time: Time until current phase ends
        - remaining_surgery_time: Time until surgery ends (RSD)
        - future_phases: Tensor of (will_occur, start_time, end_time) for each phase
    """
    
    def __init__(
        self,
        video_ids: List[int],
        features_dir: Path,
        phase_dir: Path,
        tool_dir: Path,
        phase_stats: PhaseStatistics,
        normalize_targets: bool = True,
        sequence_length: int = 10,  # For temporal context
    ):
        self.video_ids = video_ids
        self.features_dir = Path(features_dir)
        self.phase_dir = Path(phase_dir)
        self.tool_dir = Path(tool_dir)
        self.phase_stats = phase_stats
        self.normalize_targets = normalize_targets
        self.sequence_length = sequence_length
        
        # Load all videos
        self.videos: List[VideoData] = []
        print(f"Loading {len(video_ids)} videos...")
        for vid_id in video_ids:
            video = VideoData(vid_id, features_dir, phase_dir, tool_dir)
            self.videos.append(video)
        
        # Build sample index: (video_idx, time)
        self.index_map: List[Tuple[int, int]] = []
        for video_idx, video in enumerate(self.videos):
            # Valid time range
            start_t = self.sequence_length - 1
            end_t = video.duration - 1  # Need at least 1 second remaining
            for t in range(start_t, end_t):
                self.index_map.append((video_idx, t))
        
        print(f"Created Task A dataset: {len(self.index_map)} samples from {len(video_ids)} videos")
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, t = self.index_map[idx]
        video = self.videos[video_idx]
        
        # === INPUT FEATURES ===
        
        # visual features
        if self.sequence_length == 1:
            features = video.features[t]
        else:
            start = t - self.sequence_length + 1
            if start < 0:
                # pad at the beginning
                pad_length = -start
                start = 0
                seq = video.features[start:t+1]
                pad = np.zeros((pad_length, video.features.shape[1]), dtype=np.float32)
                seq = np.vstack([pad, seq])
            else:
                seq = video.features[start:t+1]
            features = seq.astype(np.float32)
        
        # Current phase (one-hot)
        phase_idx = video.phases[t]
        phase_onehot = np.zeros(NUM_PHASES, dtype=np.float32)
        phase_onehot[phase_idx] = 1.0
        
        # Elapsed time in current phase
        elapsed_phase = video.get_elapsed_phase_time(t)
        
        # Elapsed surgery time (proxy for progress since we don't know total duration at inference)
        elapsed_surgery = float(t)
        
        # === TARGET VALUES ===
        
        # Remaining time in current phase
        remaining_phase = video.get_remaining_phase_time(t)
        
        # Remaining surgery time (RSD)
        remaining_surgery = video.get_remaining_surgery_time(t)
        
        # Future phase times
        future_phases = video.get_upcoming_phases_tensor(t)
        
        # === NORMALIZATION ===
        
        if self.normalize_targets:
            # Normalize by reference statistics
            phase_name = IDX_TO_PHASE[phase_idx]
            mean_phase_dur = self.phase_stats.mean_durations.get(phase_name, 300.0)
            mean_surgery_dur = self.phase_stats.mean_surgery_duration
            
            # Normalize times (keep scale reasonable, e.g., divide by mean)
            elapsed_phase_norm = elapsed_phase / max(mean_phase_dur, 1.0)
            elapsed_surgery_norm = elapsed_surgery / max(mean_surgery_dur, 1.0)
            remaining_phase_norm = remaining_phase / max(mean_phase_dur, 1.0)
            remaining_surgery_norm = remaining_surgery / max(mean_surgery_dur, 1.0)
            
            # Normalize future phase times by mean surgery duration
            future_phases_norm = future_phases.copy()
            future_phases_norm[:, 1:] = future_phases[:, 1:] / max(mean_surgery_dur, 1.0)
        else:
            elapsed_phase_norm = elapsed_phase
            elapsed_surgery_norm = elapsed_surgery
            remaining_phase_norm = remaining_phase
            remaining_surgery_norm = remaining_surgery
            future_phases_norm = future_phases
        
        return {
            # Inputs
            'features': torch.tensor(features, dtype=torch.float32),
            'phase_onehot': torch.tensor(phase_onehot, dtype=torch.float32),
            'phase_idx': torch.tensor(phase_idx, dtype=torch.long),
            'elapsed_phase': torch.tensor(elapsed_phase_norm, dtype=torch.float32),
            'elapsed_surgery': torch.tensor(elapsed_surgery_norm, dtype=torch.float32),
            
            # Targets
            'remaining_phase': torch.tensor(remaining_phase_norm, dtype=torch.float32),
            'remaining_surgery': torch.tensor(remaining_surgery_norm, dtype=torch.float32),
            'future_phases': torch.tensor(future_phases_norm, dtype=torch.float32),
            
            # Raw values (for evaluation)
            'remaining_phase_raw': torch.tensor(remaining_phase, dtype=torch.float32),
            'remaining_surgery_raw': torch.tensor(remaining_surgery, dtype=torch.float32),
            
            # Metadata
            'video_id': video.video_id,
            'time': t,
        }


# ============================================================================
# Task B Dataset: Tool Anticipation (uses Task A predictions)
# ============================================================================

class TaskBDataset(Dataset):
    """
    Dataset for Task B: Tool Anticipation
    
    Two modes:
    1. Baseline: Only visual features → predict tools at t+horizon
    2. Timed: Visual features + timing signals → predict tools at t+horizon
    
    In the timed mode, timing signals come from:
    - During training: Ground truth timing (to learn the relationship)
    - During inference: Task A model predictions (to test if predictions help)
    """
    
    def __init__(
        self,
        video_ids: List[int],
        features_dir: Path,
        phase_dir: Path,
        tool_dir: Path,
        phase_stats: PhaseStatistics,
        horizon: int = 5,  # Predict tools 5 seconds ahead
        include_timing: bool = False,  # Whether to include timing signals
        sequence_length: int = 1,
    ):
        self.video_ids = video_ids
        self.features_dir = Path(features_dir)
        self.phase_dir = Path(phase_dir)
        self.tool_dir = Path(tool_dir)
        self.phase_stats = phase_stats
        self.horizon = horizon
        self.include_timing = include_timing
        self.sequence_length = sequence_length
        
        # Load all videos
        self.videos: List[VideoData] = []
        print(f"Loading {len(video_ids)} videos for Task B...")
        for vid_id in video_ids:
            video = VideoData(vid_id, features_dir, phase_dir, tool_dir)
            self.videos.append(video)
        
        # Build sample index
        self.index_map: List[Tuple[int, int]] = []
        for video_idx, video in enumerate(self.videos):
            start_t = self.sequence_length - 1
            end_t = video.duration - self.horizon
            for t in range(start_t, end_t):
                self.index_map.append((video_idx, t))
        
        mode = "timed" if include_timing else "baseline"
        print(f"Created Task B dataset ({mode}): {len(self.index_map)} samples")
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, t = self.index_map[idx]
        video = self.videos[video_idx]
        
        # Visual features
        if self.sequence_length == 1:
            features = video.features[t]
        else:
            start = max(0, t - self.sequence_length + 1)
            seq = video.features[start:t+1]
            if len(seq) < self.sequence_length:
                pad = np.zeros((self.sequence_length - len(seq), seq.shape[1]))
                seq = np.vstack([pad, seq])
            features = seq
        
        # Current phase
        phase_idx = video.phases[t]
        phase_onehot = np.zeros(NUM_PHASES, dtype=np.float32)
        phase_onehot[phase_idx] = 1.0
        
        # Target: tools at t + horizon
        target_tools = video.tools[t + self.horizon]
        
        # Current tools (useful context)
        current_tools = video.tools[t]
        
        sample = {
            'features': torch.tensor(features, dtype=torch.float32),
            'phase_onehot': torch.tensor(phase_onehot, dtype=torch.float32),
            'phase_idx': torch.tensor(phase_idx, dtype=torch.long),
            'current_tools': torch.tensor(current_tools, dtype=torch.float32),
            'target': torch.tensor(target_tools, dtype=torch.float32),
            'video_id': video.video_id,
            'time': t,
        }
        
        if self.include_timing:
            # Timing signals (ground truth during training)
            timing = self._get_timing_signals(video, t)
            sample['timing'] = torch.tensor(timing, dtype=torch.float32)
        
        return sample
    
    def _get_timing_signals(self, video: VideoData, t: int) -> np.ndarray:
        """
        get timing signals for the timed model.
        
        outputs that Task A predicts:
        1. remaining phase time (normalised)
        2. remaining surgery time (normalised) 
        3. progress in current phase (0-1)
        4. progress in surgery (0-1)
        5. pace signal (elapsed / expected)
        """
        phase_idx = video.phases[t]
        phase_name = IDX_TO_PHASE[phase_idx]
        mean_phase_dur = self.phase_stats.mean_durations.get(phase_name, 300.0)
        mean_surgery_dur = self.phase_stats.mean_surgery_duration
        
        elapsed_phase = video.get_elapsed_phase_time(t)
        remaining_phase = video.get_remaining_phase_time(t)
        remaining_surgery = video.get_remaining_surgery_time(t)
        
        signals = [
            remaining_phase / max(mean_phase_dur, 1.0),           # normalised remaining phase time
            remaining_surgery / max(mean_surgery_dur, 1.0),       # normalised RSD
            elapsed_phase / (elapsed_phase + remaining_phase + 1e-6),  # phase progress
            t / video.duration,                                    # surgery progress
            elapsed_phase / max(mean_phase_dur * 0.5, 1.0),       # pace signal
        ]
        return np.array(signals, dtype=np.float32)
    
    def get_class_weights(self) -> torch.Tensor:
        """compute inverse frequency weights for tool classes."""
        tool_counts = np.zeros(NUM_TOOLS, dtype=np.float32)
        total = 0
        
        for video in self.videos:
            tool_counts += video.tools.sum(axis=0)
            total += video.duration
        
        freq = tool_counts / total
        weights = 1.0 / (freq + 1e-6)
        weights = weights / weights.sum() * NUM_TOOLS
        
        return torch.tensor(weights, dtype=torch.float32)


###################################################################################################################################
# data module
###################################################################################################################################

class Cholec80DataModule:
    """
    manages data loading for both task A and B.
    """
    
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        num_workers: int = 4,
        val_videos: Optional[List[int]] = None,
        horizon: int = 5,  # For Task B
    ):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.phase_dir = self.data_dir / "cholec80" / "phase_annotations"
        self.tool_dir = self.data_dir / "cholec80" / "tool_annotations"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.horizon = horizon
        
        # data splits
        if val_videos is None:
            self.train_videos = list(range(1, 33))
            self.val_videos = list(range(33, 41))
        else:
            self.train_videos = [v for v in TRAIN_VIDEOS if v not in val_videos]
            self.val_videos = val_videos
        self.test_videos = TEST_VIDEOS
        
        # compute statistics from training set
        print("Computing phase statistics from training videos...")
        self.phase_stats = PhaseStatistics.compute_from_videos(
            self.phase_dir, self.train_videos
        )
        
        # print stats
        print(f"\nPhase duration statistics (from {len(self.train_videos)} training videos):")
        for phase in PHASES:
            mean = self.phase_stats.mean_durations[phase]
            std = self.phase_stats.std_durations[phase]
            print(f"  {phase}: {mean:.1f} ± {std:.1f} seconds")
        print(f"  Surgery duration: {self.phase_stats.mean_surgery_duration:.1f} ± {self.phase_stats.std_surgery_duration:.1f} seconds")
    
    # task A datasets
    def get_task_a_train(self, sequence_length: int = 10) -> TaskADataset:
        return TaskADataset(
            self.train_videos, self.features_dir, self.phase_dir, 
            self.tool_dir, self.phase_stats, sequence_length=sequence_length
        )
    
    def get_task_a_val(self, sequence_length: int = 10) -> TaskADataset:
        return TaskADataset(
            self.val_videos, self.features_dir, self.phase_dir,
            self.tool_dir, self.phase_stats, sequence_length=sequence_length
        )
    
    def get_task_a_test(self, sequence_length: int = 10) -> TaskADataset:
        return TaskADataset(
            self.test_videos, self.features_dir, self.phase_dir,
            self.tool_dir, self.phase_stats, sequence_length=sequence_length
        )
    
    # Task B datasets
    def get_task_b_train(self, include_timing: bool = False) -> TaskBDataset:
        return TaskBDataset(
            self.train_videos, self.features_dir, self.phase_dir,
            self.tool_dir, self.phase_stats, self.horizon, include_timing
        )
    
    def get_task_b_val(self, include_timing: bool = False) -> TaskBDataset:
        return TaskBDataset(
            self.val_videos, self.features_dir, self.phase_dir,
            self.tool_dir, self.phase_stats, self.horizon, include_timing
        )
    
    def get_task_b_test(self, include_timing: bool = False) -> TaskBDataset:
        return TaskBDataset(
            self.test_videos, self.features_dir, self.phase_dir,
            self.tool_dir, self.phase_stats, self.horizon, include_timing
        )
    
    # dataloaders
    def task_a_train_loader(self, sequence_length: int = 10) -> DataLoader:
        return DataLoader(
            self.get_task_a_train(sequence_length), batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
    
    def task_a_val_loader(self, sequence_length: int = 10) -> DataLoader:
        return DataLoader(
            self.get_task_a_val(sequence_length), batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
    
    def task_b_train_loader(self, include_timing: bool = False) -> DataLoader:
        return DataLoader(
            self.get_task_b_train(include_timing), batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
    
    def task_b_val_loader(self, include_timing: bool = False) -> DataLoader:
        return DataLoader(
            self.get_task_b_val(include_timing), batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )


###################################################################################################################################
# test
###################################################################################################################################

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    print("=" * 70)
    print("Testing Cholec80 Dataset for Task A and Task B")
    print("=" * 70)
    
    # create data module
    dm = Cholec80DataModule(data_dir, batch_size=32, num_workers=0)
    
    # test Task A
    print("\n" + "=" * 70)
    print("TASK A: Remaining Time Prediction")
    print("=" * 70)
    
    task_a_train = dm.get_task_a_train()
    print(f"\nTrain samples: {len(task_a_train)}")
    
    sample = task_a_train[5000]  # get sample from middle of a surgery
    print("\nSample contents:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if value.numel() < 10:
                print(f"       value={value}")
        else:
            print(f"  {key}: {value}")
    
    # test task B baseline
    print("\n" + "=" * 70)
    print("TASK B: Tool Anticipation (Baseline - no timing)")
    print("=" * 70)
    
    task_b_baseline = dm.get_task_b_train(include_timing=False)
    print(f"\nTrain samples: {len(task_b_baseline)}")
    
    sample_b = task_b_baseline[5000]
    print("\nSample contents:")
    for key, value in sample_b.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # test task B timed
    print("\n" + "=" * 70)
    print("TASK B: Tool Anticipation (Timed - with timing signals)")
    print("=" * 70)
    
    task_b_timed = dm.get_task_b_train(include_timing=True)
    sample_b_timed = task_b_timed[5000]
    print("\nSample contents:")
    for key, value in sample_b_timed.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}")
            if key == 'timing':
                print(f"       value={value}")
        else:
            print(f"  {key}: {value}")
    
    # test dataloader
    print("\n" + "=" * 70)
    print("Testing DataLoader")
    print("=" * 70)
    
    loader = dm.task_a_train_loader()
    batch = next(iter(loader))
    print("\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)