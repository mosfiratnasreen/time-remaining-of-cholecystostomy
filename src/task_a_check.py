"""
Diagnostic script to sanity-check Task A predictions.
Loads trained model from checkpoint and visualizes predictions vs actual.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import Cholec80DataModule, PHASES, IDX_TO_PHASE
from train_task_a import MLPModel, LSTMModel


def load_trained_model(checkpoint_path: Path, model_type: str = "mlp", device: torch.device = None):
    """load a trained model from checkpoint."""
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    if model_type == "mlp":
        model = MLPModel(hidden_dim=512, dropout=0.4)
    elif model_type == "lstm":
        model = LSTMModel(hidden_dim=256, lstm_layers=1, dropout=0.3, bidirectional=False)
    else:
        raise ValueError(f"unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"loaded {model_type.upper()} model from {checkpoint_path}")
    return model


def run_diagnostic(model, data_module, device, output_dir: Path, num_videos: int = 3):
    """
    sanity check predictions vs actual remaining times.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    val_dataset = data_module.get_task_a_val(sequence_length=1)
    
    # group samples by video
    print("\nGrouping samples by video...")
    video_data = {}
    
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        vid_id = sample['video_id']
        
        if vid_id not in video_data:
            video_data[vid_id] = []
        video_data[vid_id].append((i, sample['time']))
    
    video_ids = list(video_data.keys())[:num_videos]
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC CHECK: Predictions vs Actual")
    print("=" * 70)
    
    # create plots
    fig, axes = plt.subplots(num_videos, 2, figsize=(14, 4 * num_videos))
    if num_videos == 1:
        axes = axes.reshape(1, -1)
    
    for row, vid_id in enumerate(video_ids):
        print(f"\nProcessing Video {vid_id}...")
        
        # sort by time
        indices_times = sorted(video_data[vid_id], key=lambda x: x[1])
        
        times = []
        pred_phase = []
        true_phase = []
        pred_surgery = []
        true_surgery = []
        phases = []
        
        for idx, t in indices_times:
            sample = val_dataset[idx]
            
            # prep batch
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()}
            
            with torch.no_grad():
                pred = model(batch)
            
            # denormalise
            p_idx = sample['phase_idx'].item()
            phase_name = IDX_TO_PHASE[p_idx]
            mean_phase = data_module.phase_stats.mean_durations[phase_name]
            mean_surgery = data_module.phase_stats.mean_surgery_duration
            
            times.append(t)
            pred_phase.append(pred['remaining_phase'].item() * mean_phase)
            true_phase.append(sample['remaining_phase_raw'].item())
            pred_surgery.append(pred['remaining_surgery'].item() * mean_surgery)
            true_surgery.append(sample['remaining_surgery_raw'].item())
            phases.append(p_idx)
        
        times = np.array(times)
        pred_phase = np.array(pred_phase)
        true_phase = np.array(true_phase)
        pred_surgery = np.array(pred_surgery)
        true_surgery = np.array(true_surgery)
        phases = np.array(phases)
        
        # plot 1: remaining phase time
        ax1 = axes[row, 0]
        ax1.plot(times/60, true_phase/60, 'b-', label='Actual', linewidth=2)
        ax1.plot(times/60, pred_phase/60, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Surgery Time (min)')
        ax1.set_ylabel('Remaining Phase Time (min)')
        ax1.set_title(f'Video {vid_id}: Remaining Phase Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # plot 2: remaning surgery time
        ax2 = axes[row, 1]
        ax2.plot(times/60, true_surgery/60, 'b-', label='Actual', linewidth=2)
        ax2.plot(times/60, pred_surgery/60, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Surgery Time (min)')
        ax2.set_ylabel('Remaining Surgery Time (min)')
        ax2.set_title(f'Video {vid_id}: Remaining Surgery Time (RSD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # printn table of sample predictions
        print(f"\nVideo {vid_id} - Duration: {times[-1]/60:.1f} min")
        print(f"  {'Time':>8} {'Phase':<22} {'Pred RSD':>10} {'True RSD':>10} {'Error':>10}")
        print(f"  " + "-" * 65)
        
        # show at 5 checkpoints
        checkpoints = np.linspace(0, len(times)-1, 5, dtype=int)
        for idx in checkpoints:
            t = times[idx]
            phase_name = IDX_TO_PHASE[phases[idx]][:18]
            p_rsd = pred_surgery[idx]
            t_rsd = true_surgery[idx]
            err = p_rsd - t_rsd
            print(f"  {t/60:>7.1f}m {phase_name:<22} {p_rsd/60:>9.1f}m {t_rsd/60:>9.1f}m {err/60:>+9.1f}m")
    
    plt.tight_layout()
    plot_path = output_dir / 'diagnostic_predictions.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nplots saved to: {plot_path}")
    
    # per-phase analysis
    print("\n" + "=" * 70)
    print("PER-PHASE PREDICTION ACCURACY")
    print("=" * 70)
    
    phase_errors = {phase: [] for phase in PHASES}
    
    print("\nanalysing per-phase errors (sampling 5000 points)...")
    sample_indices = np.random.choice(len(val_dataset), min(5000, len(val_dataset)), replace=False)
    
    for i in sample_indices:
        sample = val_dataset[i]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()}
        
        with torch.no_grad():
            pred = model(batch)
        
        p_idx = sample['phase_idx'].item()
        phase_name = IDX_TO_PHASE[p_idx]
        mean_phase = data_module.phase_stats.mean_durations[phase_name]
        
        pred_p = pred['remaining_phase'].item() * mean_phase
        true_p = sample['remaining_phase_raw'].item()
        phase_errors[phase_name].append(abs(pred_p - true_p))
    
    print(f"\n{'Phase':<28} {'MAE (sec)':>10} {'MAE (min)':>10} {'Samples':>10}")
    print("-" * 62)
    for phase in PHASES:
        if phase_errors[phase]:
            mae = np.mean(phase_errors[phase])
            n = len(phase_errors[phase])
            print(f"{phase:<28} {mae:>10.1f} {mae/60:>10.2f} {n:>10}")
    
    # sanity checks
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    
    # do predictions decrease over time?
    print("\n1. Do RSD predictions decrease as surgery progresses?")
    for vid_id in video_ids[:1]:  # check first video
        indices_times = sorted(video_data[vid_id], key=lambda x: x[1])
        
        early_preds = []
        late_preds = []
        
        for idx, t in indices_times[:len(indices_times)//4]:  # 25%
            sample = val_dataset[idx]
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()}
            with torch.no_grad():
                pred = model(batch)
            early_preds.append(pred['remaining_surgery'].item() * data_module.phase_stats.mean_surgery_duration)
        
        for idx, t in indices_times[-len(indices_times)//4:]:  # last 25%
            sample = val_dataset[idx]
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()}
            with torch.no_grad():
                pred = model(batch)
            late_preds.append(pred['remaining_surgery'].item() * data_module.phase_stats.mean_surgery_duration)
        
        early_mean = np.mean(early_preds)
        late_mean = np.mean(late_preds)
        
        if early_mean > late_mean:
            print(f"YES - Early predictions ({early_mean/60:.1f} min) > Late predictions ({late_mean/60:.1f} min)")
        else:
            print(f"NO - Early predictions ({early_mean/60:.1f} min) <= Late predictions ({late_mean/60:.1f} min)")
    
    # are predictions in reasonable range?
    print("\n2. Are predictions in reasonable range?")
    all_preds = []
    for i in sample_indices[:1000]:
        sample = val_dataset[i]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()}
        with torch.no_grad():
            pred = model(batch)
        all_preds.append(pred['remaining_surgery'].item() * data_module.phase_stats.mean_surgery_duration)
    
    all_preds = np.array(all_preds)
    neg_count = np.sum(all_preds < 0)
    huge_count = np.sum(all_preds > 7200)  # > 2 hours
    
    print(f"Negative predictions: {neg_count} ({neg_count/len(all_preds)*100:.1f}%)")
    print(f"Predictions > 2 hours: {huge_count} ({huge_count/len(all_preds)*100:.1f}%)")
    
    if neg_count == 0 and huge_count < len(all_preds) * 0.05:
        print("predictions are in reasonable range")
    else:
        print("some predictions may be out of range")

###################################################################################################################################
def main():
    # setup
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs" / "task_a" / "diagnostic"
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"using device: {device}")
    
    print("\nloading data...")
    data_module = Cholec80DataModule(
        data_dir=data_dir,
        batch_size=64,
        num_workers=0
    )
    
    # trained MLP model
    mlp_checkpoint = base_dir / "outputs" / "task_a" / "mlp" / "best_model.pt"
    
    if not mlp_checkpoint.exists():
        print(f"ERROR: Checkpoint not found at {mlp_checkpoint}")
        print("Please run train_task_a.py first to train the model.")
        return
    
    model = load_trained_model(mlp_checkpoint, model_type="mlp", device=device)

    run_diagnostic(model, data_module, device, output_dir, num_videos=3)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

###################################################################################################################################
###################################################################################################################################
if __name__ == "__main__":
    main()