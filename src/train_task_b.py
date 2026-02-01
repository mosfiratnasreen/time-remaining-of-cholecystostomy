"""
task B: tool anticipation with timing information

baseline model = visual features + phase + current tools → predict tools at t+horizon
timed model = visual features + phase + current tools + Task A timing → predict tools at t+horizon

hypothesise that timing from task a helps tool anticipation
"""

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
from sklearn.metrics import average_precision_score, f1_score
from dataset import VideoData
from train_task_a import MLPModel
from collections import defaultdict

from dataset import (
    Cholec80DataModule, TaskBDataset,
    TOOLS, NUM_TOOLS, PHASES, NUM_PHASES, IDX_TO_PHASE
)

###################################################################################################################################
# task B models
###################################################################################################################################
# weak baseline without current tools 
class ToolBaselineWeakModel(nn.Module):
    """
    NO current tools - predict future tools from visual + phase only.
    makes timing more valuable for comparison.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.4
    ):
        super().__init__()
        self.name = "Baseline-Weak (No Current Tools)"

        # input: visual features + phase one-hot (NO current tools)
        input_dim = feature_dim + NUM_PHASES

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_TOOLS)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch['features']

        if features.dim() == 3:
            features = features[:, -1, :]

        x = torch.cat([
            features,
            batch['phase_onehot'],
            # NO current_tools here!
        ], dim=-1)

        logits = self.network(x)
        return logits

# baseline without timing
class ToolBaselineModel(nn.Module):
    """
    baseline model for tool anticipation.
    only visual features + current phase + current tools.
    NO timing information.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.4
    ):
        super().__init__()
        self.name = "Baseline (No Timing)"

        # input: visual features + phase one-hot + current tools
        input_dim = feature_dim + NUM_PHASES + NUM_TOOLS

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_TOOLS)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch['features']

        # handle sequence input (take last frame)
        if features.dim() == 3:
            features = features[:, -1, :]

        x = torch.cat([
            features,
            batch['phase_onehot'],
            batch['current_tools']
        ], dim=-1)

        logits = self.network(x)
        return logits

# timed model with timing signals from task A
class ToolTimedModel(nn.Module):
    """
    timed model for tool anticipation.
    visual features + phase + current tools + timing signals.

    timing signals (5 values from task A):
    1. remaining phase time (normalised)
    2. remaining surgery time / RSD (normalised)
    3. phase progress (0-1)
    4. surgery progress (0-1)
    5. pace signal
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        timing_dim: int = 6,
        dropout: float = 0.4
    ):
        super().__init__()
        self.name = "timed (with task A)"

        # input: visual features + phase + current tools + timing signals
        visual_input_dim = feature_dim + NUM_PHASES + NUM_TOOLS
        self.visual_branch = nn.Sequential(
            nn.Linear(visual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128)
        )
        # dedicated timing branch
        self.timimg_branch = nn.Sequential(
            nn.Linear(timing_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)  # same scale as visual branch output portion
        )
        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_TOOLS)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch['features']

        # handle sequence input (take last frame)
        if features.dim() == 3:
            features = features[:, -1, :]

        visual_input = torch.cat([
            features,
            batch['phase_onehot'],
            batch['current_tools']
        ], dim=-1)
        visual_out = self.visual_branch(visual_input)

        timing_out = self.timimg_branch(
            batch['timing'])  # timing branch is separate
        fused = torch.cat([visual_out, timing_out], dim=1)
        logits = self.fusion(fused)
        return logits

# weak timed model without current tools
class ToolTimedWeakModel(nn.Module):
    """
    timed model without current tools - timing to compensate.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        timing_dim: int = 6,
        dropout: float = 0.4
    ):
        super().__init__()
        self.name = "Timed-Weak (No Current Tools)"

        # visual + phase branch (NO current tools)
        visual_input_dim = feature_dim + NUM_PHASES
        self.visual_branch = nn.Sequential(
            nn.Linear(visual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128)
        )

        # dedicated timing branch
        self.timing_branch = nn.Sequential(
            nn.Linear(timing_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_TOOLS)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch['features']

        if features.dim() == 3:
            features = features[:, -1, :]

        # visual branch (NO current tools)
        visual_input = torch.cat([
            features,
            batch['phase_onehot'],
        ], dim=-1)
        visual_out = self.visual_branch(visual_input)

        # timing branch
        timing_out = self.timing_branch(batch['timing'])

        # fusion
        fused = torch.cat([visual_out, timing_out], dim=-1)
        logits = self.fusion(fused)
        return logits


###################################################################################################################################
# loss function
###################################################################################################################################
class ToolAnticipationLoss(nn.Module):
    """
    BCE loss with class weighting for tool imbalance.

    rare tools (Scissors, Clipper) are weighted higher so the model
    doesn't just predict "always Grasper, never Scissors".
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=weights
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss


###################################################################################################################################
# evaluation metrics
###################################################################################################################################

@dataclass
class ToolMetrics:
    """container for tool anticipation metrics."""
    # per-tool metrics
    ap_per_tool: Dict[str, float]  # avg precision per tool
    f1_per_tool: Dict[str, float]  # F1 score per tool

    # overall metrics
    mAP: float  # mean avg precision
    macro_f1: float  # macro F1 (average across tools)
    micro_f1: float  # micro F1 (global)
    accuracy: float  # exact match accuracy

    def to_dict(self) -> Dict:
        return {
            'mAP': self.mAP,
            'macro_f1': self.macro_f1,
            'micro_f1': self.micro_f1,
            'accuracy': self.accuracy,
            'ap_per_tool': self.ap_per_tool,
            'f1_per_tool': self.f1_per_tool
        }

    def __str__(self) -> str:
        lines = [
            f"mAP:      {self.mAP:.4f}",
            f"Macro F1: {self.macro_f1:.4f}",
            f"Micro F1: {self.micro_f1:.4f}",
            f"Accuracy: {self.accuracy:.4f}",
            "",
            "Per-tool Average Precision:"
        ]
        for tool in TOOLS:
            lines.append(f"  {tool:<15}: {self.ap_per_tool[tool]:.4f}")
        return "\n".join(lines)


def compute_metrics(
    all_logits: np.ndarray,
    all_targets: np.ndarray,
    threshold: float = 0.5
) -> ToolMetrics:
    """
    comprehensive metrics for tool anticipation.

    args:
        all_logits: raw model outputs, shape (N, 7)
        all_targets: GT binary labels, shape (N, 7)
        threshold: classification threshold for F1/accuracy

    returns:
        ToolMetrics with mAP, F1, per-tool AP, etc.
    """
    # convert logits to probabilities
    probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid
    preds = (probs > threshold).astype(float)

    # per-tool Average Precision
    ap_per_tool = {}
    for i, tool in enumerate(TOOLS):
        if all_targets[:, i].sum() > 0:  # only if tool appears in targets
            ap_per_tool[tool] = average_precision_score(
                all_targets[:, i], probs[:, i])
        else:
            ap_per_tool[tool] = 0.0

    # per-tool F1 (with zero_division handling)
    f1_per_tool = {}
    for i, tool in enumerate(TOOLS):
        tp = ((preds[:, i] == 1) & (all_targets[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (all_targets[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (all_targets[:, i] == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_per_tool[tool] = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

    # mAP
    valid_aps = [ap for ap in ap_per_tool.values() if ap > 0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0

    # macro F1 (average across tools)
    macro_f1 = np.mean(list(f1_per_tool.values()))

    # micro F1 (treat all as one big binary classification)
    tp_total = ((preds == 1) & (all_targets == 1)).sum()
    fp_total = ((preds == 1) & (all_targets == 0)).sum()
    fn_total = ((preds == 0) & (all_targets == 1)).sum()

    precision_micro = tp_total / \
        (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall_micro = tp_total / \
        (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * precision_micro * recall_micro / \
        (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

    # exact match accuracy (all 7 tools correct)
    exact_matches = np.all(preds == all_targets, axis=1)
    accuracy = np.mean(exact_matches)

    return ToolMetrics(
        ap_per_tool=ap_per_tool,
        f1_per_tool=f1_per_tool,
        mAP=mAP,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        accuracy=accuracy
    )


###################################################################################################################################
# trainer
###################################################################################################################################
class TaskBTrainer:
    """training manager for task B models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        output_dir: Path = Path("outputs/task_b")
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5  # maximise mAP
        )
        self.criterion = ToolAnticipationLoss(class_weights)

        self.train_losses = []
        self.val_maps = []
        self.best_val_map = 0.0

    def train_epoch(self) -> float:
        """train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, ToolMetrics]:
        """validate and compute metrics."""
        self.model.eval()

        all_logits = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])

            all_logits.append(logits.cpu().numpy())
            all_targets.append(batch['target'].cpu().numpy())
            total_loss += loss.item()
            num_batches += 1

        all_logits = np.concatenate(all_logits, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_metrics(all_logits, all_targets)
        avg_loss = total_loss / num_batches

        return avg_loss, metrics

    def train(self, num_epochs: int = 50, early_stop_patience: int = 10) -> ToolMetrics:
        """full training loop."""
        print(f"\ntraining {self.model.name} for {num_epochs} epochs...")
        # print(f"Train samples: {len(self.train_loader.dataset)}")
        # print(f"Val samples: {len(self.val_loader.dataset)}")

        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            self.train_losses.append(train_loss)
            self.val_maps.append(metrics.mAP)

            # LR scheduling based on mAP
            self.scheduler.step(metrics.mAP)

            # print(f"\nEpoch {epoch+1}/{num_epochs}")
            # print(f"  Train Loss: {train_loss:.4f}")
            # print(f"  Val Loss:   {val_loss:.4f}")
            # print(f"  Val mAP:    {metrics.mAP:.4f}")
            # print(f"  Val F1:     {metrics.macro_f1:.4f}")

            # save best model
            if metrics.mAP > self.best_val_map:
                self.best_val_map = metrics.mAP
                self.save_checkpoint('best_model.pt')
                patience_counter = 0
                print("new best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"\nearly stopping at epoch {epoch+1}")
                break

        # load best model
        self.load_checkpoint('best_model.pt')
        _, final_metrics = self.validate()

        # plot training curves
        self.plot_training_curves()

        return final_metrics

    def save_checkpoint(self, filename: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_map': self.best_val_map,
            'train_losses': self.train_losses,
            'val_maps': self.val_maps
        }, self.output_dir / filename)

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(
            self.output_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.train_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model.name} - Training Loss')
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.val_maps)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title(f'{self.model.name} - Validation mAP')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()


###################################################################################################################################
# visualisation and analysis
###################################################################################################################################

def plot_comparison(baseline_metrics: ToolMetrics, timed_metrics: ToolMetrics, output_dir: Path):
    """create comparison visualizations."""
    output_dir = Path(output_dir)

    # bar chart comparing per-tool AP
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(TOOLS))
    width = 0.35

    baseline_aps = [baseline_metrics.ap_per_tool[tool] for tool in TOOLS]
    timed_aps = [timed_metrics.ap_per_tool[tool] for tool in TOOLS]

    bars1 = ax.bar(x - width/2, baseline_aps, width,
                   label='Baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, timed_aps, width,
                   label='Timed', color='darkorange')

    ax.set_xlabel('Tool')
    ax.set_ylabel('Average Precision')
    ax.set_title('Tool Anticipation: Baseline vs Timed Model')
    ax.set_xticks(x)
    ax.set_xticklabels(TOOLS, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'tool_ap_comparison.png', dpi=150)
    plt.close()

    # summary metrics comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics_names = ['mAP', 'Macro F1', 'Micro F1', 'Accuracy']
    baseline_vals = [baseline_metrics.mAP, baseline_metrics.macro_f1,
                     baseline_metrics.micro_f1, baseline_metrics.accuracy]
    timed_vals = [timed_metrics.mAP, timed_metrics.macro_f1,
                  timed_metrics.micro_f1, timed_metrics.accuracy]

    x = np.arange(len(metrics_names))

    bars1 = ax.bar(x - width/2, baseline_vals, width,
                   label='Baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, timed_vals, width,
                   label='Timed', color='darkorange')

    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics: Baseline vs Timed')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics_comparison.png', dpi=150)
    plt.close()
    # print(f"\nComparison plots saved to {output_dir}")


def analyse_timing_benefit(baseline_metrics: ToolMetrics, timed_metrics: ToolMetrics):
    """analyse where timing information helps most."""

    print("\n" + "=" * 70)
    print("ANALYSIS: where Does Timing Help?")
    print("=" * 70)

    # calculate improvements
    improvements = {}
    for tool in TOOLS:
        baseline_ap = baseline_metrics.ap_per_tool[tool]
        timed_ap = timed_metrics.ap_per_tool[tool]

        if baseline_ap > 0:
            relative_improvement = (timed_ap - baseline_ap) / baseline_ap * 100
        else:
            relative_improvement = 0

        improvements[tool] = {
            'baseline': baseline_ap,
            'timed': timed_ap,
            'absolute': timed_ap - baseline_ap,
            'relative': relative_improvement
        }

    # sort by improvement
    sorted_tools = sorted(
        improvements.keys(), key=lambda t: improvements[t]['absolute'], reverse=True)

    print("\nTools ranked by improvement from timing:")
    print(f"{'Tool':<18} {'Baseline':>10} {'Timed':>10} {'Δ AP':>10} {'% Improve':>12}")
    print("-" * 62)

    for tool in sorted_tools:
        imp = improvements[tool]
        print(f"{tool:<18} {imp['baseline']:>10.4f} {imp['timed']:>10.4f} "
              f"{imp['absolute']:>+10.4f} {imp['relative']:>+11.1f}%")

    # identify patterns
    print("\n" + "-" * 62)

    helped_tools = [t for t in TOOLS if improvements[t]['absolute'] > 0.01]
    hurt_tools = [t for t in TOOLS if improvements[t]['absolute'] < -0.01]

    if helped_tools:
        print(f"timing HELPED: {', '.join(helped_tools)}")
    if hurt_tools:
        print(f"timing HURT: {', '.join(hurt_tools)}")

    # clinical interpretation
    print("\nclinical interpretation:")

    # check if end-of-surgery tools improved
    end_tools = ['SpecimenBag', 'Irrigator']
    end_improvement = np.mean([improvements[t]['absolute'] for t in end_tools])

    # check if phase-specific tools improved
    phase_tools = ['Clipper', 'Scissors']  # ClippingCutting specific
    phase_improvement = np.mean(
        [improvements[t]['absolute'] for t in phase_tools])

    print(
        f"End-of-surgery tools (SpecimenBag, Irrigator): Δ = {end_improvement:+.4f}")
    print(
        f"Phase-specific tools (Clipper, Scissors): Δ = {phase_improvement:+.4f}")

    if end_improvement > 0:
        print("Timing helps predict tools that appear near surgery end")
    if phase_improvement > 0:
        print("Timing helps predict phase-specific tools")


def analyse_lead_time(
    model: nn.Module,
    data_module: Cholec80DataModule,
    device: torch.device,
    output_dir: Path,
    confidence_threshold: float = 0.3
):
    """
    analyse how much advance warning the model provides before tools are needed.

    for each tool appearance, find when the model first predicted it.
    measures clinical utility - can nurses prepare tools in time?
    """
    model.eval()
    output_dir = Path(output_dir)

    print("\n" + "=" * 70)
    print("lead time analysis:")
    print("=" * 70)

    # validation videos for sequential analysis
    val_videos = data_module.val_videos

    # lead time before each tool appearance
    lead_times = {tool: [] for tool in TOOLS}
    tool_appearances = {tool: 0 for tool in TOOLS}
    # predicted times
    predicted_appearances = {tool: 0 for tool in TOOLS}

    # print(f"\nanalysing lead times across {len(val_videos)} validation videos...")
    # print(f"confidence threshold: {confidence_threshold}")

    for video_id in tqdm(val_videos, desc="processing videos"):
        video = VideoData(
            video_id,
            data_module.features_dir,
            data_module.phase_dir,
            data_module.tool_dir
        )

        # track tool state changes
        prev_tools = set()

        # store predictions for lookback
        prediction_history = []  # list of (time, predictions_dict)

        # each timestep
        for t in range(video.duration - 1):
            current_tools = set(
                tool for i, tool in enumerate(TOOLS)
                if video.tools[t][i] > 0.5
            )

            # prediction at this timestep
            features = torch.tensor(
                video.features[t], dtype=torch.float32).unsqueeze(0)

            phase_idx = video.phases[t]
            phase_onehot = torch.zeros(NUM_PHASES)
            phase_onehot[phase_idx] = 1.0
            phase_onehot = phase_onehot.unsqueeze(0)

            current_tools_tensor = torch.tensor(
                video.tools[t], dtype=torch.float32).unsqueeze(0)

            # timing signals
            phase_name = IDX_TO_PHASE[phase_idx]
            mean_phase_dur = data_module.phase_stats.mean_durations.get(
                phase_name, 300.0)
            mean_surgery_dur = data_module.phase_stats.mean_surgery_duration

            elapsed_phase = video.get_elapsed_phase_time(t)
            remaining_phase = video.get_remaining_phase_time(t)
            remaining_surgery = video.get_remaining_surgery_time(t)

            near_phase_end = 1.0 if remaining_phase < 60 else 0.0
            near_surgery_end = 1.0 if remaining_surgery < 300 else 0.0
            phase_position = phase_idx / (NUM_PHASES - 1)

            timing = torch.tensor([
                remaining_phase / max(mean_phase_dur, 1.0),
                remaining_surgery / max(mean_surgery_dur, 1.0),
                elapsed_phase / (elapsed_phase + remaining_phase + 1e-6),
                t / video.duration,
                elapsed_phase / max(mean_phase_dur * 0.5, 1.0),
                near_phase_end,
                near_surgery_end,
                phase_position,
            ], dtype=torch.float32).unsqueeze(0)

            batch = {
                'features': features.to(device),
                'phase_onehot': phase_onehot.to(device),
                'current_tools': current_tools_tensor.to(device),
                'timing': timing.to(device)
            }

            with torch.no_grad():
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            predictions = {tool: probs[i] for i, tool in enumerate(TOOLS)}
            prediction_history.append((t, predictions))

            # check for new tool appearances
            new_tools = current_tools - prev_tools

            for tool in new_tools:
                tool_appearances[tool] += 1

                # find EARLIEST prediction above threshold
                earliest_prediction_time = None

                # look back up to 120 seconds
                lookback_limit = max(0, len(prediction_history) - 120)

                for i in range(len(prediction_history) - 1, lookback_limit - 1, -1):
                    hist_t, hist_preds = prediction_history[i]
                    if hist_preds[tool] >= confidence_threshold:
                        earliest_prediction_time = hist_t

                if earliest_prediction_time is not None:
                    lead_time = t - earliest_prediction_time
                    lead_times[tool].append(lead_time)
                    predicted_appearances[tool] += 1

            prev_tools = current_tools

    # statistics
    print("\n" + "-" * 70)
    print("LEAD TIME STATISTICS (seconds of warning before tool needed)")
    print("-" * 70)
    print(f"\n{'Tool':<15} {'Appearances':>12} {'Predicted':>12} {'Coverage':>10} {'Mean Lead':>12} {'Min':>8} {'Max':>8}")
    print("-" * 80)

    summary_data = []

    for tool in TOOLS:
        appearances = tool_appearances[tool]
        predicted = predicted_appearances[tool]
        coverage = predicted / appearances * 100 if appearances > 0 else 0

        if lead_times[tool]:
            mean_lead = np.mean(lead_times[tool])
            min_lead = np.min(lead_times[tool])
            max_lead = np.max(lead_times[tool])
            summary_data.append({
                'tool': tool,
                'appearances': appearances,
                'predicted': predicted,
                'coverage': coverage,
                'mean_lead': mean_lead,
                'min_lead': min_lead,
                'max_lead': max_lead
            })
            print(f"{tool:<15} {appearances:>12} {predicted:>12} {coverage:>9.1f}% {mean_lead:>11.1f}s {min_lead:>7.0f}s {max_lead:>7.0f}s")
        else:
            print(f"{tool:<15} {appearances:>12} {predicted:>12} {coverage:>9.1f}% {'N/A':>12} {'N/A':>8} {'N/A':>8}")


    print("\n" + "-" * 70)
    print("CLINICAL UTILITY ASSESSMENT")
    print("-" * 70)

    all_lead_times = []
    for tool in TOOLS:
        all_lead_times.extend(lead_times[tool])

    if all_lead_times:
        all_lead_times = np.array(all_lead_times)

        # categorise
        excellent = np.sum(all_lead_times >= 60)  # 60+ seconds
        good = np.sum((all_lead_times >= 30) & (all_lead_times < 60))  # 30-60 seconds
        marginal = np.sum((all_lead_times >= 10) & (all_lead_times < 30))  # 10-30 seconds
        insufficient = np.sum(all_lead_times < 10)  # <10 seconds

        total = len(all_lead_times)

        print(f"\n excellent (≥60s warning): {excellent:>5} ({excellent/total*100:>5.1f}%) - ample time to prepare")
        print(f"good (30-60s warning):{good:>5} ({good/total*100:>5.1f}%) - sufficient time")
        print(f"marginal (10-30s warning):  {marginal:>5} ({marginal/total*100:>5.1f}%) - tight, but usable")
        print(f"insufficient (<10s warning): {insufficient:>5} ({insufficient/total*100:>5.1f}%) - too late")

        print(f"\n overall Statistics:")
        print(f"Mean lead time: {np.mean(all_lead_times):.1f} seconds")
        print(f"Median lead time: {np.median(all_lead_times):.1f} seconds")
        print(f"Std dev: {np.std(all_lead_times):.1f} seconds")

        print(f"\n" + "-" * 70)
        print("CLINICAL RECOMMENDATION")
        print("-" * 70)

        useful_pct = (excellent + good) / total * 100
        if useful_pct >= 70:
            print(f"\n RECOMMENDED FOR CLINICAL USE")
            print(f"{useful_pct:.1f}% of predictions provide ≥30 seconds warning")
            print(f" average lead time of {np.mean(all_lead_times):.1f}s is clinically useful")
        elif useful_pct >= 50:
            print(f"\n POTENTIALLY USEFUL WITH IMPROVEMENTS")
            print(f" {useful_pct:.1f}% of predictions provide ≥30 seconds warning")
        else:
            print(f"\n NOT RECOMMENDED - INSUFFICIENT LEAD TIME")
            print(f" only {useful_pct:.1f}% of predictions provide ≥30 seconds warning")

    # plot
    if all_lead_times.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # histogram
        ax1 = axes[0]
        ax1.hist(all_lead_times, bins=30, edgecolor='black',
                 alpha=0.7, color='steelblue')
        ax1.axvline(x=30, color='orange', linestyle='--',
                    linewidth=2, label='30s threshold')
        ax1.axvline(x=60, color='green', linestyle='--',
                    linewidth=2, label='60s threshold')
        ax1.axvline(x=np.mean(all_lead_times), color='red', linestyle='-',
                    linewidth=2, label=f'Mean ({np.mean(all_lead_times):.1f}s)')
        ax1.set_xlabel('Lead Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Tool Anticipation Lead Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # per-tool mean lead times
        ax2 = axes[1]
        tools_with_data = [t for t in TOOLS if lead_times[t]]
        mean_leads = [np.mean(lead_times[t]) for t in tools_with_data]
        colors = ['green' if m >= 60 else 'orange' if m >=
                  30 else 'red' for m in mean_leads]

        bars = ax2.bar(tools_with_data, mean_leads,
                       color=colors, edgecolor='black')
        ax2.axhline(y=30, color='orange', linestyle='--',
                    linewidth=2, label='30s threshold')
        ax2.axhline(y=60, color='green', linestyle='--',
                    linewidth=2, label='60s threshold')
        ax2.set_xlabel('Tool')
        ax2.set_ylabel('Mean Lead Time (seconds)')
        ax2.set_title('Mean Lead Time by Tool')
        ax2.set_xticklabels(tools_with_data, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'lead_time_analysis.png', dpi=150)
        plt.close()

        print(f"\n lead time plots saved to {output_dir / 'lead_time_analysis.png'}")

    # save
    summary_data_clean = []
    for item in summary_data:
        clean_item = {}
        for k, v in item.items():
            if isinstance(v, (np.integer, np.int64)):
                clean_item[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                clean_item[k] = float(v)
            else:
                clean_item[k] = v
        summary_data_clean.append(clean_item)

    results = {
        'summary': summary_data_clean,
        'overall': {
            'mean_lead_time': float(np.mean(all_lead_times)) if all_lead_times.size > 0 else 0,
            'median_lead_time': float(np.median(all_lead_times)) if all_lead_times.size > 0 else 0,
            'excellent_pct': float(excellent / total * 100) if total > 0 else 0,
            'good_pct': float(good / total * 100) if total > 0 else 0,
            'marginal_pct': float(marginal / total * 100) if total > 0 else 0,
            'insufficient_pct': float(insufficient / total * 100) if total > 0 else 0,
        }
    }

    with open(output_dir / 'lead_time_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


###################################################################################################################################
# main
###################################################################################################################################
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    # setup
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs" / "task_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # hyperparameters
    HORIZON = 60  # predict tools 60 seconds ahead
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOP = 10

    print(f"\n" + "=" * 70)
    print("TASK B: Tool Anticipation")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Prediction horizon: {HORIZON} seconds")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {NUM_EPOCHS}")

    # print("\n" + "=" * 70)
    # print("Loading Data")
    # print("=" * 70)

    data_module = Cholec80DataModule(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=0,
        horizon=HORIZON
    )

    task_a_model = MLPModel(hidden_dim=1024, dropout=0.2, num_blocks=4)
    checkpoint = torch.load(base_dir / "outputs" / "task_a" / "mlp" / "best_model.pt",
                            map_location=device, weights_only=False)
    task_a_model.load_state_dict(checkpoint['model_state_dict'])
    task_a_model.to(device)
    task_a_model.eval()

    checkpoint_path = base_dir / "outputs" / "task_a" / "mlp" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"task A checkpoint not found at {checkpoint_path}. train task A first!")
    print(f"loaded task A MLP model")

    # class weights for imbalanced tools
    train_dataset_baseline = data_module.get_task_b_train(include_timing=False)
    class_weights = train_dataset_baseline.get_class_weights()

    # print(f"\ntool class weights (inverse frequency):")
    for tool, weight in zip(TOOLS, class_weights):
        print(f"  {tool:<15}: {weight:.2f}")

    # ###################################################################################################################################
    # train baseline model without timing
    # ###################################################################################################################################
    print("\n" + "=" * 70)
    print("training BASELINE Model (no timing information)")
    print("=" * 70)

    baseline_model = ToolBaselineModel(hidden_dim=512, dropout=0.4)
    # print(f"\nmodel: {baseline_model.name}")
    # print(f"input: Visual features (2048) + Phase (7) + Current tools (7) = 2062 dims")

    baseline_trainer = TaskBTrainer(
        model=baseline_model,
        train_loader=data_module.task_b_train_loader(include_timing=False),
        val_loader=data_module.task_b_val_loader(include_timing=False),
        device=device,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        output_dir=output_dir / "baseline"
    )

    baseline_metrics = baseline_trainer.train(
        num_epochs=NUM_EPOCHS, early_stop_patience=EARLY_STOP)

    print("\n" + "-" * 50)
    print("BASELINE MODEL - Final Results:")
    print("-" * 50)
    print(baseline_metrics)

    ###################################################################################################################################
    # train weak baseline (no current tools) - shows timing value more clearly
    ###################################################################################################################################
    print("\n" + "=" * 70)
    print("Training WEAK BASELINE Model (No Current Tools)")
    print("=" * 70)

    weak_baseline_model = ToolBaselineWeakModel(hidden_dim=512, dropout=0.4)
    # print(f"\nModel: {weak_baseline_model.name}")
    # print(f"Input: Visual features (2048) + Phase (7) = 2055 dims (NO current tools)")

    weak_baseline_trainer = TaskBTrainer(
        model=weak_baseline_model,
        train_loader=data_module.task_b_train_loader(include_timing=False),
        val_loader=data_module.task_b_val_loader(include_timing=False),
        device=device,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        output_dir=output_dir / "weak_baseline"
    )

    weak_baseline_metrics = weak_baseline_trainer.train(
        num_epochs=NUM_EPOCHS, early_stop_patience=EARLY_STOP)

    print("\n" + "-" * 50)
    print("WEAK BASELINE MODEL - Final Results:")
    print("-" * 50)
    print(weak_baseline_metrics)

    ###################################################################################################################################
    # train weak timed (no current tools + timing)
    ###################################################################################################################################
    print("\n" + "=" * 70)
    print("Training WEAK TIMED Model (No Current Tools + Timing)")
    print("=" * 70)

    weak_timed_model = ToolTimedWeakModel(hidden_dim=512, dropout=0.4)
    print(f"\nModel: {weak_timed_model.name}")
    print(f"Input: Visual (2048) + Phase (7) + Timing (8) = 2063 dims (NO current tools)")

    weak_timed_trainer = TaskBTrainer(
        model=weak_timed_model,
        train_loader=data_module.task_b_train_loader(include_timing=True),
        val_loader=data_module.task_b_val_loader(include_timing=True),
        device=device,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        output_dir=output_dir / "weak_timed"
    )

    weak_timed_metrics = weak_timed_trainer.train(
        num_epochs=NUM_EPOCHS, early_stop_patience=EARLY_STOP)

    print("\n" + "-" * 50)
    print("WEAK TIMED MODEL - Final Results:")
    print("-" * 50)
    print(weak_timed_metrics)

    ###################################################################################################################################
    # train timed model with GT timing information
    ###################################################################################################################################
    print("\n" + "=" * 70)
    print("Training TIMED Model (With GT Timing Information)")
    print("=" * 70)

    oracle_model = ToolTimedModel(hidden_dim=512, dropout=0.4)
    print(f"\nModel: {oracle_model.name}")
    print(f"Input: Visual (2048) + Phase (7) + Current tools (7) + Timing (5) = 2067 dims")

    oracle_trainer = TaskBTrainer(
        model=ToolTimedModel(hidden_dim=512, dropout=0.4),
        train_loader=data_module.task_b_train_loader(
            include_timing=True, task_a_model=None),
        val_loader=data_module.task_b_val_loader(
            include_timing=True, task_a_model=None),
        device=device,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        output_dir=output_dir / "oracle_timed"
    )

    oracle_metrics = oracle_trainer.train(
        num_epochs=NUM_EPOCHS, early_stop_patience=EARLY_STOP)

    print("\n" + "-" * 50)
    print("ORACLE MODEL - Final Results:")
    print("-" * 50)
    print(oracle_metrics)

    ###################################################################################################################################
    # train timed model with task A timing information
    ###################################################################################################################################
    print("\n" + "=" * 70)
    print("Training TIMED Model (With task A Timing Information)")
    print("=" * 70)

    task_a_timed_model = ToolTimedModel(hidden_dim=512, dropout=0.4)
    # task_a_timed_model.name = "task A timed (predicted)"
    # print(f"\nModel: {task_a_timed_model.name}")
    # print(f"Input: Visual (2048) + Phase (7) + Current tools (7) + Timing (8) = 2070 dims")

    task_a_timed_trainer = TaskBTrainer(
        model=task_a_timed_model,
        train_loader=data_module.task_b_train_loader(
            include_timing=True, task_a_model=task_a_model, device=device),
        val_loader=data_module.task_b_val_loader(
            include_timing=True, task_a_model=task_a_model, device=device),
        device=device,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        output_dir=output_dir / "task_a_timed"
    )

    task_a_timed_metrics = task_a_timed_trainer.train(
        num_epochs=NUM_EPOCHS, early_stop_patience=EARLY_STOP)

    print("\n" + "-" * 50)
    print("TASK A TIMED MODEL - Final Results:")
    print("-" * 50)
    print(task_a_timed_metrics)

    ###################################################################################################################################
    # final comparison for all models
    ###################################################################################################################################
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: all models")
    print("=" * 70)

    print(f"\n{'Model':<35} {'mAP':>10} {'Macro F1':>10} {'Accuracy':>10}")
    print("-" * 75)
    print(f"{'Baseline (no timing)':<40} {baseline_metrics.mAP:>10.4f} {baseline_metrics.macro_f1:>10.4f} {baseline_metrics.accuracy:>10.4f}")
    print(f"{'Weak Baseline (no current tools)':<40} {weak_baseline_metrics.mAP:>10.4f} {weak_baseline_metrics.macro_f1:>10.4f} {weak_baseline_metrics.accuracy:>10.4f}")
    print(f"{'Weak Timed (no current tools)':<40} {weak_timed_metrics.mAP:>10.4f} {weak_timed_metrics.macro_f1:>10.4f} {weak_timed_metrics.accuracy:>10.4f}")
    print(f"{'Oracle Timed (ground truth timing)':<40} {oracle_metrics.mAP:>10.4f} {oracle_metrics.macro_f1:>10.4f} {oracle_metrics.accuracy:>10.4f}")
    print(f"{'Task A Timed (predicted timing)':<40} {task_a_timed_metrics.mAP:>10.4f} {task_a_timed_metrics.macro_f1:>10.4f} {task_a_timed_metrics.accuracy:>10.4f}")
    
    # key comparisons
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    delta_task_a = task_a_timed_metrics.mAP - baseline_metrics.mAP
    delta_oracle = oracle_metrics.mAP - baseline_metrics.mAP
    delta_weak = weak_timed_metrics.mAP - weak_baseline_metrics.mAP

    print(f"\n1. Baseline → Task A Timed (predicted):")
    print(f"{baseline_metrics.mAP:.4f} → {task_a_timed_metrics.mAP:.4f} (Δ = {delta_task_a:+.4f})")

    print(f"\n2. Baseline → Oracle Timed (ground truth):")
    print(f"{baseline_metrics.mAP:.4f} → {oracle_metrics.mAP:.4f} (Δ = {delta_oracle:+.4f})")

    print(f"\n3. Task A Timed vs Oracle Timed (timing quality gap):")
    gap = oracle_metrics.mAP - task_a_timed_metrics.mAP
    print(f"{task_a_timed_metrics.mAP:.4f} vs {oracle_metrics.mAP:.4f} (gap = {gap:.4f})")

    print(f"\n4. Weak models (without current tools):")
    print(f"Baseline → Timed: {weak_baseline_metrics.mAP:.4f} → {weak_timed_metrics.mAP:.4f} (Δ = {delta_weak:+.4f})")

    # per-tool comparison: Baseline vs Task A Timed vs Oracle
    print(f"\n{'Tool':<18} {'Baseline':>10} {'Task A':>10} {'Oracle':>10} {'Δ(Task A)':>12}")
    print("-" * 62)

    for tool in TOOLS:
        b_ap = baseline_metrics.ap_per_tool[tool]
        t_ap = task_a_timed_metrics.ap_per_tool[tool]
        o_ap = oracle_metrics.ap_per_tool[tool]
        delta = t_ap - b_ap
        print(f"{tool:<18} {b_ap:>10.4f} {t_ap:>10.4f} {o_ap:>10.4f} {delta:>+12.4f}")

    # create visualisations
    plot_comparison(baseline_metrics, task_a_timed_metrics, output_dir)

    # detailed analysis
    analyse_timing_benefit(baseline_metrics, task_a_timed_metrics)

    # save results
    results = {
        'baseline': baseline_metrics.to_dict(),
        'task_a_timed': task_a_timed_metrics.to_dict(),
        'oracle_timed': oracle_metrics.to_dict(),
        'weak_baseline': weak_baseline_metrics.to_dict(),
        'weak_timed': weak_timed_metrics.to_dict(),
        'settings': {
            'horizon': HORIZON,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print(f"\n1. TASK A PREDICTIONS (real-world scenario):")
    if delta_task_a > 0.01:
        print(f"Task A timing HELPS: mAP improves by {delta_task_a:+.4f} ({delta_task_a/baseline_metrics.mAP*100:+.1f}%)")
    elif delta_task_a < -0.01:
        print(f"Task A timing HURTS: mAP decreases by {delta_task_a:.4f}")
    else:
        print(f"No significant difference (Δ = {delta_task_a:+.4f})")

    print(f"\n2. ORACLE TIMING:")
    if delta_oracle > 0.01:
        print(f"Oracle timing HELPS: mAP improves by {delta_oracle:+.4f} ({delta_oracle/baseline_metrics.mAP*100:+.1f}%)")
    else:
        print(f"Perfect timing doesn't help much (Δ = {delta_oracle:+.4f})")

    print(f"\n3. TIMING QUALITY IMPACT:")
    if gap < 0.01:
        print(f"Task A predictions nearly as good as ground truth (gap = {gap:.4f})")
        print(f"Task A model is sufficiently accurate for tool anticipation")
    elif gap < 0.03:
        print(f"Small gap between Task A and oracle (gap = {gap:.4f})")
        print(f"Room for improvement in Task A, but predictions are useful")
    else:
        print(f"Large gap between Task A and oracle (gap = {gap:.4f})")
        print(f"Task A predictions too noisy; need better time prediction")

    print(f"\n4. HYPOTHESIS VALIDATION:")
    if delta_task_a > 0.005:
        print(f"HYPOTHESIS SUPPORTED: Task A predictions improve Task B")
        print(f"Remaining time prediction is useful for tool anticipation")
    else:
        print(f"HYPOTHESIS NOT SUPPORTED in current setup")
        print(f"Current tools provide sufficient predictive signal")

    return baseline_metrics, weak_baseline_metrics, oracle_metrics, weak_timed_metrics, task_a_timed_metrics


###################################################################################################################################
###################################################################################################################################
if __name__ == "__main__":
    baseline_metrics, weak_baseline_metrics, oracle_metrics, weak_timed_metrics, task_a_timed_metrics = main()
