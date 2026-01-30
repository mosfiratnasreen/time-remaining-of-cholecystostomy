"""
task A - remaning time prediction model
1. baseline methods (statistical, heuristic, linear)
2. deep learning models (MLP, LSTM)
3. training loop
4. evaluation metrics

outputs:
- remaining time in current phase
- remaining surgery duration (RSD)
- future phase start/end times
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import (
    Cholec80DataModule, TaskADataset, PhaseStatistics,
    PHASES, NUM_PHASES, PHASE_TO_IDX, IDX_TO_PHASE
)


###################################################################################################################################
# baseline methods
###################################################################################################################################
# baseline 1
class StatisticalBaseline:
    """
    predict mean remaining time based on current phase.
    
    for remaining phase time: mean_phase_duration / 2 (assume halfway)
    for RSD: remaining_phase + sum of future phase means
    """
    
    def __init__(self, phase_stats: PhaseStatistics):
        self.phase_stats = phase_stats
        self.name = "Statistical Mean"
    
    def predict(self, phase_idx: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        args:
            phase_idx: array of phase indices (batch_size,)
        returns:
            dict with predictions
        """
        batch_size = len(phase_idx)
        
        remaining_phase = np.zeros(batch_size)
        remaining_surgery = np.zeros(batch_size)
        
        for i, p_idx in enumerate(phase_idx):
            phase_name = IDX_TO_PHASE[p_idx]
            
            # predict half of mean phase duration as remaining
            mean_dur = self.phase_stats.mean_durations[phase_name]
            remaining_phase[i] = mean_dur / 2
            
            # RSD = remaining in current + sum of future phases
            rsd = remaining_phase[i]
            for future_idx in range(p_idx + 1, NUM_PHASES):
                future_phase = IDX_TO_PHASE[future_idx]
                rsd += self.phase_stats.mean_durations[future_phase]
            remaining_surgery[i] = rsd
        
        return {
            'remaining_phase': remaining_phase,
            'remaining_surgery': remaining_surgery
        }

# baseline 2
class ElapsedHeuristicBaseline:
    """
    use elapsed time to estimate remaining time.
    
    remaining_phase = max(0, mean_duration - elapsed_in_phase)
    RSD = remaining_phase + sum of future phase means
    """
    
    def __init__(self, phase_stats: PhaseStatistics):
        self.phase_stats = phase_stats
        self.name = "Elapsed Heuristic"
    
    def predict(
        self, 
        phase_idx: np.ndarray, 
        elapsed_phase: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        args:
            phase_idx: array of phase indices (batch_size,)
            elapsed_phase: elapsed time in current phase in seconds (batch_size,)
        """
        batch_size = len(phase_idx)
        
        remaining_phase = np.zeros(batch_size)
        remaining_surgery = np.zeros(batch_size)
        
        for i, (p_idx, elapsed) in enumerate(zip(phase_idx, elapsed_phase)):
            phase_name = IDX_TO_PHASE[p_idx]
            mean_dur = self.phase_stats.mean_durations[phase_name]
            
            # remaining = mean - elapsed (clipped to 0)
            remaining_phase[i] = max(0, mean_dur - elapsed)
            
            # RSD = remaining in current + sum of future phases
            rsd = remaining_phase[i]
            for future_idx in range(p_idx + 1, NUM_PHASES):
                future_phase = IDX_TO_PHASE[future_idx]
                rsd += self.phase_stats.mean_durations[future_phase]
            remaining_surgery[i] = rsd
        
        return {
            'remaining_phase': remaining_phase,
            'remaining_surgery': remaining_surgery
        }

# baseline 3
class LinearRegressionBaseline(nn.Module):
    """
    linear regression on phase + elapsed time features.
    no visual features, simple linear combination.
    """
    
    def __init__(self):
        super().__init__()
        # input: phase_onehot (7) + elapsed_phase (1) + elapsed_surgery (1) = 9
        self.linear_phase = nn.Linear(9, 1)
        self.linear_surgery = nn.Linear(9, 1)
        self.name = "Linear Regression"
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # concatenates inputs (no visual features)
        x = torch.cat([
            batch['phase_onehot'],
            batch['elapsed_phase'].unsqueeze(-1),
            batch['elapsed_surgery'].unsqueeze(-1)
        ], dim=-1)
        
        remaining_phase = F.relu(self.linear_phase(x)).squeeze(-1)
        remaining_surgery = F.relu(self.linear_surgery(x)).squeeze(-1)
        
        return {
            'remaining_phase': remaining_phase,
            'remaining_surgery': remaining_surgery
        }


###################################################################################################################################
# deep learning models
###################################################################################################################################
# DL method 1 
class MLPModel(nn.Module):
    """
    MLP model: visual features + phase + elapsed time → predictions
    no temporal context (single frame input).
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        num_blocks: int = 4
    ):
        super().__init__()
        self.name = "MLP (deeper)"
        
        # input dimension: features + phase_onehot + elapsed_phase + elapsed_surgery
        input_dim = feature_dim + NUM_PHASES + 2

        self.input_proj = nn.Linear(input_dim, hidden_dim) #initial projection
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList() #multiple residual blocks
        self.layer_norms = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # output heads
        self.head_remaining_phase = nn.Linear(hidden_dim, 1)
        self.head_remaining_surgery = nn.Linear(hidden_dim, 1)
        self.head_future_phases = nn.Linear(hidden_dim, NUM_PHASES * 2)  # (start, end) for each
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # concatenate all inputs
        x = torch.cat([
            batch['features'],
            batch['phase_onehot'],
            batch['elapsed_phase'].unsqueeze(-1),
            batch['elapsed_surgery'].unsqueeze(-1)
        ], dim=-1)
        
        h = F.relu(self.ln1(self.input_proj(x))) #initial projection with norm layer

        for block, ln in zip(self.blocks, self.layer_norms):
            h = ln(h + block(h))
            h = F.relu(h)
        
        # predictions (ReLU ensures positive times)
        remaining_phase = F.relu(self.head_remaining_phase(h)).squeeze(-1)
        remaining_surgery = F.relu(self.head_remaining_surgery(h)).squeeze(-1)
        future_phases = self.head_future_phases(h).view(-1, NUM_PHASES, 2)
        future_phases = F.relu(future_phases)  # times should be positive
        
        return {
            'remaining_phase': remaining_phase,
            'remaining_surgery': remaining_surgery,
            'future_phases': future_phases
        }

# DL method 2
class LSTMModel(nn.Module):
    """
    LSTM model: temporal visual features + phase + elapsed time → predictions
    uses sequence of frames for temporal context.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True
    ):
        super().__init__()
        self.name = "LSTM (Temporal)"
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # feature projection (reduce dimension before LSTM)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # fusion layer (LSTM output + phase + elapsed times)
        fusion_dim = lstm_out_dim + NUM_PHASES + 2
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # output heads
        self.head_remaining_phase = nn.Linear(hidden_dim, 1)
        self.head_remaining_surgery = nn.Linear(hidden_dim, 1)
        self.head_future_phases = nn.Linear(hidden_dim, NUM_PHASES * 2)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = batch['features']
        
        # handle both single frame and sequence inputs
        if features.dim() == 2:
            # single frame: (batch, feature_dim) → (batch, 1, feature_dim)
            features = features.unsqueeze(1)
        
        # project features
        x = self.feature_proj(features)  # (batch, seq_len, hidden_dim)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # use last hidden state
        if self.bidirectional:
            # concatenate forward and backward final states
            temporal_features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            temporal_features = h_n[-1]  # (batch, hidden_dim)
        
        # fuse with phase and elapsed time
        fused = torch.cat([
            temporal_features,
            batch['phase_onehot'],
            batch['elapsed_phase'].unsqueeze(-1),
            batch['elapsed_surgery'].unsqueeze(-1)
        ], dim=-1)
        
        h = self.fusion(fused)
        
        # prediction
        remaining_phase = F.relu(self.head_remaining_phase(h)).squeeze(-1)
        remaining_surgery = F.relu(self.head_remaining_surgery(h)).squeeze(-1)
        future_phases = self.head_future_phases(h).view(-1, NUM_PHASES, 2)
        future_phases = F.relu(future_phases)
        
        return {
            'remaining_phase': remaining_phase,
            'remaining_surgery': remaining_surgery,
            'future_phases': future_phases
        }


###################################################################################################################################
# loss function
###################################################################################################################################
class TaskALoss(nn.Module):
    """
    multi-task loss for task A.
    combines losses for remaining phase time, RSD, and future phase times.
    """
    
    def __init__(
        self,
        lambda_phase: float = 0.5,
        lambda_surgery: float = 5.0,
        lambda_future: float = 0.3,
        use_huber: bool = True
    ):
        super().__init__()
        self.lambda_phase = lambda_phase
        self.lambda_surgery = lambda_surgery
        self.lambda_future = lambda_future
        
        if use_huber:
            self.criterion = nn.SmoothL1Loss()  # Huber loss, robust to outliers
        else:
            self.criterion = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        returns total loss and dictionary of individual losses for logging.
        """
        # original absolute loss
        loss_phase_abs = self.criterion(
            predictions['remaining_phase'],
            targets['remaining_phase']
        )

        loss_surgery_abs = self.criterion(
            predictions['remaining_surgery'],
            targets['remaining_surgery']
        )

        # percentage-aware component (scale-invariant)
        eps = 0.1  # avoid division by zero
        loss_phase_pct = torch.mean(
            torch.abs(predictions['remaining_phase'] - targets['remaining_phase']) / 
            (targets['remaining_phase'] + eps)
        )
        loss_surgery_pct = torch.mean(
            torch.abs(predictions['remaining_surgery'] - targets['remaining_surgery']) / 
            (targets['remaining_surgery'] + eps)
        )

        # combine: 70% absolute error + 30% percentage error
        loss_phase = 0.7 * loss_phase_abs + 0.3 * loss_phase_pct
        loss_surgery = 0.7 * loss_surgery_abs + 0.3 * loss_surgery_pct
        
        # future phases loss (only on the time values, columns 1 and 2)
        # column 0 is "will occur" flag which we don't predict
        if 'future_phases' in predictions:
            pred_future = predictions['future_phases']
            target_future = targets['future_phases'][:, :, 1:].contiguous()  # skip column 0
            loss_future = self.criterion(pred_future, target_future)
        else:
            loss_future = torch.tensor(0.0, device=loss_phase.device)
        
        total_loss = (
            self.lambda_phase * loss_phase +
            self.lambda_surgery * loss_surgery +
            self.lambda_future * loss_future
        )
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_phase': loss_phase.item(),
            'loss_surgery': loss_surgery.item(),
            'loss_future': loss_future.item()
        }
        
        return total_loss, loss_dict


###################################################################################################################################
# evaluation metrics
###################################################################################################################################

@dataclass
class EvaluationMetrics:
    """container for evaluation metrics."""
    mae_phase: float  # mean absolute error for remaining phase time
    mae_surgery: float  # MAE for RSD
    mae_phase_std: float
    mae_surgery_std: float
    
    # MAE at specific times before surgery end
    mae_at_30min: Optional[float] = None
    mae_at_20min: Optional[float] = None
    mae_at_10min: Optional[float] = None
    mae_at_5min: Optional[float] = None
    
    # percentage within thresholds
    pct_within_5min: Optional[float] = None
    pct_within_10min: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {k: float(v) for k, v in self.__dict__.items() if v is not None}
    
    def __str__(self) -> str:
        lines = [
            f"MAE (phase):   {self.mae_phase:.1f} ± {self.mae_phase_std:.1f} seconds ({self.mae_phase/60:.2f} min)",
            f"MAE (surgery): {self.mae_surgery:.1f} ± {self.mae_surgery_std:.1f} seconds ({self.mae_surgery/60:.2f} min)",
        ]
        if self.pct_within_5min is not None:
            lines.append(f"Within 5 min:  {self.pct_within_5min:.1f}%")
        if self.pct_within_10min is not None:
            lines.append(f"Within 10 min: {self.pct_within_10min:.1f}%")
        if self.mae_at_30min is not None:
            lines.append(f"MAE at t-30min: {self.mae_at_30min:.1f}s")
        if self.mae_at_20min is not None:
            lines.append(f"MAE at t-20min: {self.mae_at_20min:.1f}s")
        if self.mae_at_10min is not None:
            lines.append(f"MAE at t-10min: {self.mae_at_10min:.1f}s")
        return "\n".join(lines)


def evaluate_predictions(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    remaining_surgery_raw: Optional[np.ndarray] = None
) -> EvaluationMetrics:
    """
    compute evaluation metrics for time predictions.
    
    args:
        predictions: dict with 'remaining_phase' and 'remaining_surgery' arrays
        targets: dict with ground truth values (in seconds, raw scale)
        remaining_surgery_raw: raw remaining surgery time for stratified metrics
    """
    # basic MAE
    errors_phase = np.abs(predictions['remaining_phase'] - targets['remaining_phase'])
    errors_surgery = np.abs(predictions['remaining_surgery'] - targets['remaining_surgery'])
    
    mae_phase = np.mean(errors_phase)
    mae_surgery = np.mean(errors_surgery)
    mae_phase_std = np.std(errors_phase)
    mae_surgery_std = np.std(errors_surgery)
    
    # percentage within thresholds (for RSD)
    pct_within_5min = np.mean(errors_surgery < 300) * 100  # 5 min = 300s
    pct_within_10min = np.mean(errors_surgery < 600) * 100
    
    metrics = EvaluationMetrics(
        mae_phase=mae_phase,
        mae_surgery=mae_surgery,
        mae_phase_std=mae_phase_std,
        mae_surgery_std=mae_surgery_std,
        pct_within_5min=pct_within_5min,
        pct_within_10min=pct_within_10min
    )
    
    # stratified metrics (MAE at specific times before surgery end)
    if remaining_surgery_raw is not None:
        rsd = remaining_surgery_raw
        
        # at 30 minutes remaining
        mask_30 = (rsd >= 1750) & (rsd <= 1850)  # 29-31 min window
        if mask_30.sum() > 0:
            metrics.mae_at_30min = np.mean(errors_surgery[mask_30])
        
        # at 20 minutes remaining
        mask_20 = (rsd >= 1150) & (rsd <= 1250)
        if mask_20.sum() > 0:
            metrics.mae_at_20min = np.mean(errors_surgery[mask_20])
        
        # at 10 minutes remaining
        mask_10 = (rsd >= 550) & (rsd <= 650)
        if mask_10.sum() > 0:
            metrics.mae_at_10min = np.mean(errors_surgery[mask_10])
        
        # at 5 minutes remaining
        mask_5 = (rsd >= 250) & (rsd <= 350)
        if mask_5.sum() > 0:
            metrics.mae_at_5min = np.mean(errors_surgery[mask_5])
    return metrics


def analyse_per_phase_performance(
    model: nn.Module,
    data_module: Cholec80DataModule,
    device: torch.device,
    output_dir: Path
):
    """analyse model performance broken down by surgical phase."""
    
    model.eval()
    val_loader = data_module.task_a_val_loader(sequence_length=1)
    
    # collect predictions by phase
    phase_errors = {phase: {'phase': [], 'surgery': []} for phase in PHASES}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analysing per-phase"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            predictions = model(batch)
            
            # denormalise predictions
            pred_phase = predictions['remaining_phase'].cpu().numpy()
            pred_surgery = predictions['remaining_surgery'].cpu().numpy()
            phase_idx = batch['phase_idx'].cpu().numpy()
            
            for i, p_idx in enumerate(phase_idx):
                phase_name = IDX_TO_PHASE[p_idx]
                pred_phase[i] *= data_module.phase_stats.mean_durations[phase_name]
                pred_surgery[i] *= data_module.phase_stats.mean_surgery_duration
            
            true_phase = batch['remaining_phase_raw'].cpu().numpy()
            true_surgery = batch['remaining_surgery_raw'].cpu().numpy()
            
            # store errors by phase
            for i, p_idx in enumerate(phase_idx):
                phase_name = IDX_TO_PHASE[p_idx]
                phase_errors[phase_name]['phase'].append(abs(pred_phase[i] - true_phase[i]))
                phase_errors[phase_name]['surgery'].append(abs(pred_surgery[i] - true_surgery[i]))
    
    # print results
    print("\n" + "=" * 70)
    print("PER-PHASE PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    print(f"\n{'Phase':<30} {'MAE Phase (s)':>15} {'MAE Surgery (s)':>15} {'N samples':>12}")
    print("-" * 75)
    
    phase_mae_data = []
    for phase in PHASES:
        if phase_errors[phase]['phase']:
            mae_phase = np.mean(phase_errors[phase]['phase'])
            mae_surgery = np.mean(phase_errors[phase]['surgery'])
            n = len(phase_errors[phase]['phase'])
            phase_mae_data.append({
                'phase': phase,
                'mae_phase': mae_phase,
                'mae_surgery': mae_surgery,
                'n': n
            })
            print(f"{phase:<30} {mae_phase:>14.1f}s {mae_surgery:>14.1f}s {n:>12}")
    
    # identify hardest/easiest phases
    sorted_by_surgery = sorted(phase_mae_data, key=lambda x: x['mae_surgery'])
    
    print(f"\n  Easiest phase (lowest MAE): {sorted_by_surgery[0]['phase']} ({sorted_by_surgery[0]['mae_surgery']/60:.1f} min)")
    print(f"  Hardest phase (highest MAE): {sorted_by_surgery[-1]['phase']} ({sorted_by_surgery[-1]['mae_surgery']/60:.1f} min)")
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    phases_short = [p[:15] for p in PHASES]
    mae_values = [np.mean(phase_errors[p]['surgery'])/60 if phase_errors[p]['surgery'] else 0 for p in PHASES]
    
    colors = ['green' if m < 5 else 'orange' if m < 8 else 'red' for m in mae_values]
    bars = ax.bar(phases_short, mae_values, color=colors, edgecolor='black')
    
    ax.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Good (<5 min)')
    ax.axhline(y=8, color='orange', linestyle='--', linewidth=2, label='Acceptable (<8 min)')
    
    ax.set_xlabel('Surgical Phase')
    ax.set_ylabel('MAE (minutes)')
    ax.set_title('Remaining Surgery Duration Prediction Error by Phase')
    ax.set_xticklabels(phases_short, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_phase_mae.png', dpi=150)
    plt.close()
    
    print(f"\n  Per-phase analysis saved to {output_dir / 'per_phase_mae.png'}")
    return phase_mae_data



###################################################################################################################################
# training loop
###################################################################################################################################
class Trainer:
    """training manager for task A models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        phase_stats: PhaseStatistics,
        device: torch.device,
        learning_rate: float = 1e-4,
        output_dir: Path = Path("outputs/task_a")
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.phase_stats = phase_stats
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = TaskALoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # forward pass
            predictions = self.model(batch)
            
            # compute loss
            targets = {
                'remaining_phase': batch['remaining_phase'],
                'remaining_surgery': batch['remaining_surgery'],
                'future_phases': batch['future_phases']
            }
            loss, loss_dict = self.criterion(predictions, targets)
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # average losses
        avg_losses = {k: float(np.mean([d[k] for d in epoch_losses])) for k in epoch_losses[0]}
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], EvaluationMetrics]:
        """validate and compute metrics."""
        self.model.eval()
        
        all_pred_phase = []
        all_pred_surgery = []
        all_true_phase = []
        all_true_surgery = []
        all_rsd_raw = []
        epoch_losses = []
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            predictions = self.model(batch)
            
            targets = {
                'remaining_phase': batch['remaining_phase'],
                'remaining_surgery': batch['remaining_surgery'],
                'future_phases': batch['future_phases']
            }
            loss, loss_dict = self.criterion(predictions, targets)
            epoch_losses.append(loss_dict)
            
            # denormalise predictions for metrics
            # (predictions are normalised, we need raw seconds)
            pred_phase = predictions['remaining_phase'].cpu().numpy()
            pred_surgery = predictions['remaining_surgery'].cpu().numpy()
            
            # denormalise using phase stats
            phase_idx = batch['phase_idx'].cpu().numpy()
            for i, p_idx in enumerate(phase_idx):
                phase_name = IDX_TO_PHASE[p_idx]
                pred_phase[i] *= self.phase_stats.mean_durations[phase_name]
                pred_surgery[i] *= self.phase_stats.mean_surgery_duration
            
            all_pred_phase.extend(pred_phase)
            all_pred_surgery.extend(pred_surgery)
            all_true_phase.extend(batch['remaining_phase_raw'].cpu().numpy())
            all_true_surgery.extend(batch['remaining_surgery_raw'].cpu().numpy())
            all_rsd_raw.extend(batch['remaining_surgery_raw'].cpu().numpy())
        
        avg_losses = {k: float(np.mean([d[k] for d in epoch_losses])) for k in epoch_losses[0]}
        
        # compute metrics
        predictions_dict = {
            'remaining_phase': np.array(all_pred_phase),
            'remaining_surgery': np.array(all_pred_surgery)
        }
        targets_dict = {
            'remaining_phase': np.array(all_true_phase),
            'remaining_surgery': np.array(all_true_surgery)
        }
        
        metrics = evaluate_predictions(
            predictions_dict, targets_dict, np.array(all_rsd_raw)
        )
        return avg_losses, metrics

    def train(self, num_epochs: int = 50, early_stop_patience: int = 10):
        """full training loop."""
        print(f"\ntraining {self.model.name} for {num_epochs} epochs...")
        print(f"train samples: {len(self.train_loader.dataset)}")
        print(f"validation samples: {len(self.val_loader.dataset)}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses['loss_total'])
            
            # validate
            val_losses, metrics = self.validate()
            self.val_losses.append(val_losses['loss_total'])
            
            # learning rate scheduling
            self.scheduler.step(val_losses['loss_total'])
            
            # logging
            # print(f"\nepoch {epoch+1}/{num_epochs}")
            # print(f"  train Loss: {train_losses['loss_total']:.4f}")
            # print(f"  validation Loss:   {val_losses['loss_total']:.4f}")
            # print(f"  validation MAE (surgery): {metrics.mae_surgery/60:.2f} min")
            # print(f"  validation MAE (phase):   {metrics.mae_phase/60:.2f} min")
            
            # save best model
            if val_losses['loss_total'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss_total']
                self.save_checkpoint('best_model.pt')
                patience_counter = 0
                print(" new best model saved!")
            else:
                patience_counter += 1
            
            # early stopping
            if patience_counter >= early_stop_patience:
                print(f"\nearly stopping at epoch {epoch+1}")
                break
        
        # load best model for final evaluation
        self.load_checkpoint('best_model.pt')
        
        # plot training curves
        self.plot_training_curves()
        return self.validate()
    
    def save_checkpoint(self, filename: str):
        """save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, filename: str):
        """load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def plot_training_curves(self):
        """plot and save training curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {self.model.name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()


###################################################################################################################################
# evaluate baseline
###################################################################################################################################

def evaluate_baselines(
    data_module: Cholec80DataModule,
    device: torch.device
) -> Dict[str, EvaluationMetrics]:
    """evaluate all baseline methods on validation set."""
    
    results = {}
    val_dataset = data_module.get_task_a_val()
    
    # collect all validation data
    print("\ncollecting validation data for baseline evaluation...")
    all_phase_idx = []
    all_elapsed_phase = []
    all_true_phase = []
    all_true_surgery = []
    all_rsd_raw = []
    
    for i in tqdm(range(len(val_dataset)), desc="Loading"):
        sample = val_dataset[i]
        all_phase_idx.append(sample['phase_idx'].item())
        
        # denormalise elapsed time
        phase_name = IDX_TO_PHASE[sample['phase_idx'].item()]
        mean_dur = data_module.phase_stats.mean_durations[phase_name]
        elapsed = sample['elapsed_phase'].item() * mean_dur
        all_elapsed_phase.append(elapsed)
        
        all_true_phase.append(sample['remaining_phase_raw'].item())
        all_true_surgery.append(sample['remaining_surgery_raw'].item())
        all_rsd_raw.append(sample['remaining_surgery_raw'].item())
    
    all_phase_idx = np.array(all_phase_idx)
    all_elapsed_phase = np.array(all_elapsed_phase)
    all_true_phase = np.array(all_true_phase)
    all_true_surgery = np.array(all_true_surgery)
    all_rsd_raw = np.array(all_rsd_raw)
    
    targets = {
        'remaining_phase': all_true_phase,
        'remaining_surgery': all_true_surgery
    }
    
    # baseline 1: statistical mean
    print("\nevaluating Statistical Mean baseline...")
    baseline1 = StatisticalBaseline(data_module.phase_stats)
    pred1 = baseline1.predict(all_phase_idx)
    results[baseline1.name] = evaluate_predictions(pred1, targets, all_rsd_raw)
    
    # baseline 2: elapsed heuristic
    print("evaluating Elapsed Heuristic baseline...")
    baseline2 = ElapsedHeuristicBaseline(data_module.phase_stats)
    pred2 = baseline2.predict(all_phase_idx, all_elapsed_phase)
    results[baseline2.name] = evaluate_predictions(pred2, targets, all_rsd_raw)
    
    # baseline 3: linear regression
    print("training Linear Regression baseline...")
    linear_model = LinearRegressionBaseline().to(device)
    linear_optimizer = Adam(linear_model.parameters(), lr=1e-3)
    
    train_loader = data_module.task_a_train_loader()
    
    # quick training (10 epochs)
    linear_model.train()
    for epoch in range(10):
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            pred = linear_model(batch)
            loss = F.smooth_l1_loss(pred['remaining_phase'], batch['remaining_phase'])
            loss += F.smooth_l1_loss(pred['remaining_surgery'], batch['remaining_surgery'])
            linear_optimizer.zero_grad()
            loss.backward()
            linear_optimizer.step()
    
    # evaluate
    linear_model.eval()
    val_loader = data_module.task_a_val_loader()
    
    all_pred_phase = []
    all_pred_surgery = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            pred = linear_model(batch)
            
            # denormalise
            phase_idx = batch['phase_idx'].cpu().numpy()
            pred_phase = pred['remaining_phase'].cpu().numpy()
            pred_surgery = pred['remaining_surgery'].cpu().numpy()
            
            for i, p_idx in enumerate(phase_idx):
                phase_name = IDX_TO_PHASE[p_idx]
                pred_phase[i] *= data_module.phase_stats.mean_durations[phase_name]
                pred_surgery[i] *= data_module.phase_stats.mean_surgery_duration
            
            all_pred_phase.extend(pred_phase)
            all_pred_surgery.extend(pred_surgery)
    
    pred3 = {
        'remaining_phase': np.array(all_pred_phase),
        'remaining_surgery': np.array(all_pred_surgery)
    }
    results[linear_model.name] = evaluate_predictions(pred3, targets, all_rsd_raw)
    return results


###################################################################################################################################
# main training
###################################################################################################################################
def get_device():
    """get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    """main training and evaluation script."""
    
    # setup
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs" / "task_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"using device: {device}")
    
    # create data module
    print("\n" + "=" * 70)
    print("loading data")
    print("=" * 70)
    
    data_module = Cholec80DataModule(
        data_dir=data_dir,
        batch_size=64,
        num_workers=0  #0 for MPS compatibility
    )
    
    # evaluate baselines first
    print("\n" + "=" * 70)
    print("evaluating baselines")
    print("=" * 70)
    
    baseline_results = evaluate_baselines(data_module, device)
    
    print("\n--- Baseline Results ---")
    for name, metrics in baseline_results.items():
        print(f"\n{name}:")
        print(metrics)
    
    # train MLP model
    print("\n" + "=" * 70)
    print("training MLP model (no temporal context)")
    print("=" * 70)
    
    mlp_model = MLPModel(hidden_dim=1024, dropout=0.2)
    mlp_trainer = Trainer(
        model=mlp_model,
        train_loader=data_module.task_a_train_loader(sequence_length=1),
        val_loader=data_module.task_a_val_loader(sequence_length=1),
        phase_stats=data_module.phase_stats,
        device=device,
        learning_rate=5e-5,
        output_dir=output_dir / "mlp"
    )
    
    _, mlp_metrics = mlp_trainer.train(num_epochs=100, early_stop_patience=20)
    
    print("\n--- MLP Results ---")
    print(mlp_metrics)
    
    # train LSTM model
    print("\n" + "=" * 70)
    print("training LSTM model (temporal context)")
    print("=" * 70)
    
    lstm_model = LSTMModel(hidden_dim=512, lstm_layers=2, dropout=0.3, bidirectional=True)
    lstm_trainer = Trainer(
        model=lstm_model,
        train_loader=data_module.task_a_train_loader(sequence_length=60),
        val_loader=data_module.task_a_val_loader(sequence_length=60),
        phase_stats=data_module.phase_stats,
        device=device,
        learning_rate=5e-5,
        output_dir=output_dir / "lstm"
    )
    
    _, lstm_metrics = lstm_trainer.train(num_epochs=100, early_stop_patience=25)
    
    print("\n--- LSTM Results ---")
    print(lstm_metrics)
    
    # final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    all_results = {**baseline_results, mlp_model.name: mlp_metrics, lstm_model.name: lstm_metrics}
    
    # create comparison table
    print("\n{:<25} {:>15} {:>15} {:>12}".format(
        "Model", "MAE Phase (s)", "MAE Surgery (s)", "Within 5min"
    ))
    print("-" * 70)
    
    for name, metrics in all_results.items():
        print("{:<25} {:>15.1f} {:>15.1f} {:>11.1f}%".format(
            name,
            metrics.mae_phase,
            metrics.mae_surgery,
            metrics.pct_within_5min or 0
        ))

    # short vs long surgeries
    print("\n" + "=" * 70)
    print("STRATIFIED ANALYSIS (LSTM Model)")
    print("=" * 70)
    
    # best LSTM model and evaluate on test set
    test_loader = data_module.task_a_val_loader(sequence_length=60)
    
    lstm_model.eval()
    all_errors = []
    all_rsd_raw = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            pred = lstm_model(batch)
            
            # denormalise
            pred_surgery = pred['remaining_surgery'].cpu().numpy()
            phase_idx = batch['phase_idx'].cpu().numpy()
            for i, p_idx in enumerate(phase_idx):
                phase_name = IDX_TO_PHASE[p_idx]
                pred_surgery[i] *= data_module.phase_stats.mean_surgery_duration
            
            true_surgery = batch['remaining_surgery_raw'].cpu().numpy()
            errors = np.abs(pred_surgery - true_surgery)
            
            all_errors.extend(errors)
            all_rsd_raw.extend(true_surgery)
    
    all_errors = np.array(all_errors)
    all_rsd_raw = np.array(all_rsd_raw)
    
    # stratify by remaining time
    print("\nMAE by time remaining until surgery end:")
    for threshold, label in [(600, "< 10 min"), (1200, "10-20 min"), (1800, "20-30 min"), (3600, "30-60 min"), (float('inf'), "> 60 min")]:
        if threshold == 600:
            mask = all_rsd_raw < threshold
        elif threshold == float('inf'):
            mask = all_rsd_raw >= 3600
        else:
            prev_threshold = {1200: 600, 1800: 1200, 3600: 1800}[threshold]
            mask = (all_rsd_raw >= prev_threshold) & (all_rsd_raw < threshold)
        
        if mask.sum() > 0:
            mae = np.mean(all_errors[mask])
            print(f"  {label}: MAE = {mae:.1f}s ({mae/60:.1f} min), n={mask.sum()}")

    # save results
    results_dict = {name: metrics.to_dict() for name, metrics in all_results.items()}
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nresults saved to {output_dir}")

    #per phase analysis on best model MLP
    print("\n" + "=" * 70)
    print("PER-PHASE ANALYSIS (MLP Model)")
    print("=" * 70)
    
    phase_analysis = analyse_per_phase_performance(
        model=mlp_model,
        data_module=data_module,
        device=device,
        output_dir=output_dir / "mlp"
    )
    return all_results

###################################################################################################################################
###################################################################################################################################
if __name__ == "__main__":
    results = main()