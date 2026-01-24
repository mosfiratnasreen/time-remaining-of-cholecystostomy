"""
Cholec80 data exploration script
analyses tool usage patterns, phase durations, and tool transitions to help design the tool anticipation task.

"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# config - adjust these paths to match your setup
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "cholec80"
PHASE_DIR = DATA_DIR / "phase_annotations"
TOOL_DIR = DATA_DIR / "tool_annotations"
OUTPUT_DIR = BASE_DIR / "outputs" / "exploration"

# tool and phase names
TOOLS = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']
PHASES = [
    'Preparation',
    'CalotTriangleDissection', 
    'ClippingCutting',
    'GallbladderDissection',
    'GallbladderPackaging',
    'CleaningCoagulation',
    'GallbladderRetraction'
]

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

###################################################################################################################################
def load_phase_annotations(video_id: int) -> pd.DataFrame:
    """loads phase annotations for a video (1 fps)."""
    filename = PHASE_DIR / f"video{video_id:02d}-phase.txt"
    df = pd.read_csv(filename, sep='\t')
    return df


def load_tool_annotations(video_id: int) -> pd.DataFrame:
    """loads tool annotations for a video (1 fps, every 25 frames)."""
    filename = TOOL_DIR / f"video{video_id:02d}-tool.txt"
    df = pd.read_csv(filename, sep='\t')
    return df


def align_annotations(video_id: int) -> pd.DataFrame:
    """
    aligns phase and tool annotations to 1fps.
    tool annotations are at frame 0, 25, 50... (1fps for 25fps video)
    phase annotations are at every frame.
    
    returns DataFrame with columns: [second, phase, tool_1, ..., tool_7]
    """
    phase_df = load_phase_annotations(video_id)
    tool_df = load_tool_annotations(video_id)
    
    # tool annotations are already at 1fps (every 25 frames)
    tool_df['second'] = tool_df['Frame'] // 25 # convert frame number to seconds
    
    # for phase, sample at the same seconds as tools
    # phase frame numbers correspond to the actual frame in video
    # at 25fps, second t corresponds to frame 25*t
    phase_at_tool_times = []
    for sec in tool_df['second']:
        frame_idx = sec * 25
        if frame_idx < len(phase_df):
            phase_at_tool_times.append(phase_df.iloc[frame_idx]['Phase'])
        else:
            # handle edge case - use last known phase
            phase_at_tool_times.append(phase_df.iloc[-1]['Phase'])
    
    # create aligned dataframe
    aligned = tool_df.copy()
    aligned['phase'] = phase_at_tool_times
    aligned = aligned.drop(columns=['Frame'])
    
    # reorder columns
    cols = ['second', 'phase'] + TOOLS
    aligned = aligned[cols]
    return aligned


def compute_tool_combination(row) -> str:
    """converts tool binary vector to string representation."""
    tools_present = [TOOLS[i] for i in range(len(TOOLS)) if row[TOOLS[i]] == 1]
    if not tools_present:
        return 'NoTool'
    return '+'.join(tools_present)


###################################################################################################################################
def analyse_single_video(video_id: int) -> dict:
    """analyses a single video and return statistics."""
    df = align_annotations(video_id)
    
    stats = {
        'video_id': video_id,
        'duration_seconds': len(df),
        'duration_minutes': len(df) / 60,
    }
    
    # extract phase durations
    phase_changes = df['phase'].ne(df['phase'].shift()).cumsum()
    phase_segments = df.groupby(phase_changes).agg({
        'phase': 'first',
        'second': ['min', 'max', 'count']
    })
    phase_segments.columns = ['phase', 'start', 'end', 'duration']
    
    stats['phase_sequence'] = phase_segments['phase'].tolist()
    stats['phase_durations'] = {
        phase: phase_segments[phase_segments['phase'] == phase]['duration'].sum()
        for phase in PHASES if phase in phase_segments['phase'].values
    }
    
    # tool usage per phase
    stats['tool_usage_by_phase'] = {}
    for phase in PHASES:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            tool_freq = {tool: phase_data[tool].mean() for tool in TOOLS}
            stats['tool_usage_by_phase'][phase] = tool_freq
    
    # tool combinations
    df['tool_combo'] = df.apply(compute_tool_combination, axis=1)
    stats['tool_combo_counts'] = df['tool_combo'].value_counts().to_dict()
    
    # tool transitions (what tool appears next)
    df['next_tools'] = df[TOOLS].shift(-1).values.tolist()
    return stats, df


def analyse_tool_transitions(all_dfs: list) -> pd.DataFrame:
    """
    analyses tool transitions across all videos.
    returns a transition matrix: P(tool_j at t+k | tool_i at t)
    """
    # analyses transitions at different horizons
    horizons = [1, 5, 10, 30]  # seconds ahead
    
    transition_stats = {}
    
    for horizon in horizons:
        # count transitions for each tool
        transitions = defaultdict(lambda: defaultdict(int))
        
        for df in all_dfs:
            for tool in TOOLS:
                # find frames where this tool is present
                tool_present = df[df[tool] == 1]
                
                for idx in tool_present.index:
                    future_idx = idx + horizon
                    if future_idx in df.index:
                        future_row = df.loc[future_idx]
                        for future_tool in TOOLS:
                            if future_row[future_tool] == 1:
                                transitions[tool][future_tool] += 1
        
        # convert to probability matrix
        trans_matrix = pd.DataFrame(0.0, index=TOOLS, columns=TOOLS)
        for tool_from in TOOLS:
            total = sum(transitions[tool_from].values())
            if total > 0:
                for tool_to in TOOLS:
                    trans_matrix.loc[tool_from, tool_to] = transitions[tool_from][tool_to] / total
        
        transition_stats[horizon] = trans_matrix
    return transition_stats


def analyse_new_tool_events(all_dfs: list) -> dict:
    """
    analyses when NEW tools appear (tool goes from 0 to 1) -- more helpful for anticipation rather than just presence
    """
    new_tool_events = defaultdict(list)
    
    for video_idx, df in enumerate(all_dfs):
        for tool in TOOLS:
            # find where tool transitions from 0 to 1
            tool_col = df[tool].values
            transitions = np.diff(tool_col)
            appear_indices = np.where(transitions == 1)[0] + 1  # +1 because diff shifts
            
            for idx in appear_indices:
                # record context: phase, other tools present, time in surgery
                row = df.iloc[idx]
                context = {
                    'video': video_idx + 1,
                    'second': row['second'],
                    'phase': row['phase'],
                    'progress': row['second'] / len(df),  # normalized time
                    'other_tools': [t for t in TOOLS if t != tool and row[t] == 1]
                }
                new_tool_events[tool].append(context)
    return new_tool_events


###################################################################################################################################
def plot_tool_usage_by_phase(all_stats: list):
    """creates heatmap of tool usage frequency by phase."""
    # agg across videos
    phase_tool_matrix = pd.DataFrame(0.0, index=PHASES, columns=TOOLS)
    phase_counts = defaultdict(int)
    
    for stats in all_stats:
        for phase, tool_freq in stats['tool_usage_by_phase'].items():
            for tool, freq in tool_freq.items():
                phase_tool_matrix.loc[phase, tool] += freq
            phase_counts[phase] += 1
    
    # normalise by number of videos that had each phase
    for phase in PHASES:
        if phase_counts[phase] > 0:
            phase_tool_matrix.loc[phase] /= phase_counts[phase]
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(phase_tool_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Tool Usage Frequency by Surgical Phase\n(Proportion of time tool is present)')
    ax.set_xlabel('Tool')
    ax.set_ylabel('Surgical Phase')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tool_usage_by_phase.png', dpi=150)
    plt.close()
    return phase_tool_matrix


def plot_phase_durations(all_stats: list):
    """plots distribution of phase durations."""
    phase_durations = defaultdict(list)
    
    for stats in all_stats:
        for phase, duration in stats['phase_durations'].items():
            phase_durations[phase].append(duration)
    
    # create box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = [phase_durations[phase] for phase in PHASES if phase in phase_durations]
    labels = [phase for phase in PHASES if phase in phase_durations]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # colour boxes
    colours = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colours):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Distribution of Phase Durations Across 80 Videos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase_durations.png', dpi=150)
    plt.close()
    
    # print stats
    print("\n=== Phase Duration Statistics (seconds) ===")
    for phase in PHASES:
        if phase in phase_durations:
            durations = phase_durations[phase]
            print(f"{phase}:")
            print(f"  Mean: {np.mean(durations):.1f}s, Std: {np.std(durations):.1f}s")
            print(f"  Min: {np.min(durations):.1f}s, Max: {np.max(durations):.1f}s")
    return phase_durations


def plot_surgery_duration_distribution(all_stats: list):
    """plots distribution of total surgery durations."""
    durations = [s['duration_minutes'] for s in all_stats]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.1f} min')
    ax.axvline(np.median(durations), color='green', linestyle='--', label=f'Median: {np.median(durations):.1f} min')
    ax.set_xlabel('Surgery Duration (minutes)')
    ax.set_ylabel('Number of Videos')
    ax.set_title('Distribution of Surgery Durations in Cholec80')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'surgery_durations.png', dpi=150)
    plt.close()
    
    print(f"\n=== Surgery Duration Statistics ===")
    print(f"Mean: {np.mean(durations):.1f} minutes")
    print(f"Std: {np.std(durations):.1f} minutes")
    print(f"Min: {np.min(durations):.1f} minutes")
    print(f"Max: {np.max(durations):.1f} minutes")


def plot_transition_matrices(transition_stats: dict):
    """plots tool transition matrices at different time horizons."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, (horizon, matrix) in zip(axes, transition_stats.items()):
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    vmin=0, vmax=1, cbar_kws={'label': 'Probability'})
        ax.set_title(f'Tool Transition Probabilities\n({horizon} seconds ahead)')
        ax.set_xlabel('Tool at t+{}'.format(horizon))
        ax.set_ylabel('Tool at t')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tool_transitions.png', dpi=150)
    plt.close()


###################################################################################################################################
def analyse_anticipation_task_design(new_tool_events: dict, all_dfs: list):
    """
    analyses what prediction task makes most sense. -- how often tools change, reasonable prediction, predict tool appearance or set?
    """
    print("\n=== Tool Anticipation Task Design Analysis ===")
    
    # how often new tools appear
    print("\n1. New Tool Appearance Events:")
    for tool in TOOLS:
        events = new_tool_events[tool]
        print(f"  {tool}: {len(events)} appearances across 80 videos")
        if events:
            phases = [e['phase'] for e in events]
            phase_counts = pd.Series(phases).value_counts()
            print(f"    Most common phases: {phase_counts.head(3).to_dict()}")
    
    # how often tools change
    print("\n2. Tool Set Change Frequency:")
    change_intervals = []
    for df in all_dfs:
        df['tool_combo'] = df.apply(compute_tool_combination, axis=1)
        changes = df['tool_combo'].ne(df['tool_combo'].shift())
        change_indices = df[changes].index.tolist()
        intervals = np.diff(change_indices)
        change_intervals.extend(intervals)
    
    print(f"  Mean interval between tool changes: {np.mean(change_intervals):.1f} seconds")
    print(f"  Median interval: {np.median(change_intervals):.1f} seconds")
    print(f"  25th percentile: {np.percentile(change_intervals, 25):.1f} seconds")
    print(f"  75th percentile: {np.percentile(change_intervals, 75):.1f} seconds")
    
    # prediction horizon recommendations (wip)
    print("\n3. Recommended Prediction Horizons:")
    print(f"  Short-term (5s): Good for immediate tool preparation")
    print(f"  Medium-term (15s): Reasonable for scrub nurse anticipation")
    print(f"  Long-term (30s): Challenging but clinically useful")
    return change_intervals


###################################################################################################################################
###################################################################################################################################
def main():
    print("=" * 60)
    print("Cholec80 Data Exploration")
    print("=" * 60)
    
    # load and analyse all videos
    all_stats = []
    all_dfs = []
    
    print("\nLoading and analysing videos...")
    for video_id in range(1, 81):
        try:
            stats, df = analyse_single_video(video_id)
            all_stats.append(stats)
            all_dfs.append(df)
            if video_id % 20 == 0:
                print(f"  Processed {video_id}/80 videos")
        except Exception as e:
            print(f"  Error processing video {video_id}: {e}")
    
    print(f"\nSuccessfully loaded {len(all_stats)} videos")
    
    # generate visualisation and statistics
    print("\n" + "=" * 60)
    print("Generating Statistics and Visualisations")
    print("=" * 60)
    
    # srugery duration distribution
    plot_surgery_duration_distribution(all_stats)
    
    # phase duration analysis
    phase_durations = plot_phase_durations(all_stats)
    
    # tool usage by phase
    tool_phase_matrix = plot_tool_usage_by_phase(all_stats)
    print("\n=== Tool Usage by Phase (proportion of time present) ===")
    print(tool_phase_matrix.round(2).to_string())
    
    # tool transitions
    print("\nAnalyzing tool transitions...")
    transition_stats = analyse_tool_transitions(all_dfs)
    plot_transition_matrices(transition_stats)
    
    # new tool events
    print("\nAnalyzing new tool appearance events...")
    new_tool_events = analyse_new_tool_events(all_dfs)
    
    # task design analysis
    change_intervals = analyse_anticipation_task_design(new_tool_events, all_dfs)
    
    # save summary statistics
    summary = {
        'n_videos': len(all_stats),
        'total_duration_hours': sum(s['duration_minutes'] for s in all_stats) / 60,
        'mean_duration_minutes': np.mean([s['duration_minutes'] for s in all_stats]),
        'tools': TOOLS,
        'phases': PHASES,
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total videos: {summary['n_videos']}")
    print(f"Total duration: {summary['total_duration_hours']:.1f} hours")
    print(f"Mean video duration: {summary['mean_duration_minutes']:.1f} minutes")
    print(f"\Visualisations saved to: {OUTPUT_DIR}")
    
    return all_stats, all_dfs, transition_stats, new_tool_events


if __name__ == "__main__":
    all_stats, all_dfs, transition_stats, new_tool_events = main()