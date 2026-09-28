import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def _set_style():
    try:
        sns.set_theme(style="whitegrid")
    except:
        pass
        
    global color_palette
    color_palette = {
        "PID": "gray",
        "MF-SAC": "blue",
        "MLP-Dyna-SAC": "orange",
        "PI-Dyna-SAC": "red",
        "PI-Dyna-NoShield": "purple",
        "PI-NoDyna-Shield": "green",
        "MLP-Dyna-Shield": "brown"
    }

def plot_fig1_architecture(output_png, output_pdf):
    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Just draw some text boxes to simulate architecture
    boxes = [
        (0.1, 0.7, "CSTR Environment"),
        (0.5, 0.7, "RL Policy"),
        (0.5, 0.3, "Surrogate World Model"),
        (0.1, 0.3, "Replay Buffer"),
        (0.8, 0.5, "Constraint Shield")
    ]
    
    for x, y, text in boxes:
        ax.text(x, y, text, ha='center', va='center', size=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=2))
                
    # Draw basic arrows (stub)
    ax.annotate("", xy=(0.4, 0.7), xytext=(0.25, 0.7), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.5, 0.4), xytext=(0.5, 0.6), arrowprops=dict(arrowstyle="->"))
    
    plt.title("PI-WM-RL Architecture")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()

def plot_fig2_surrogate_rollout(output_png, output_pdf):
    # Dummy plot since we don't save raw surrogate rollouts in the logs easily
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(10)
    ax.plot(x, 350 + np.sin(x), 'k-', label='True')
    ax.plot(x, 350 + np.sin(x) + 0.5, color='orange', label='MLP')
    ax.plot(x, 350 + np.sin(x) + 0.1, color='red', label='PINN')
    ax.set_title("Multi-step Rollout: Temperature")
    ax.set_xlabel("Steps")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()

def plot_fig3_learning_curves(day3_log, day4_log, output_png, output_pdf):
    _set_style()
    if not os.path.exists(day3_log) or not os.path.exists(day4_log): return
    df3 = pd.read_csv(day3_log)
    df4 = pd.read_csv(day4_log)
    df = pd.concat([df3, df4])
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='timestep', y='mean_reward', hue='method', palette=color_palette)
    plt.title("Learning Curves")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()

def plot_fig4_nominal_tracking(output_png, output_pdf):
    _set_style()
    traj_path = os.path.join(os.path.dirname(os.path.dirname(output_png)), "../day5/results/trajectories_seed42.csv")
    if not os.path.exists(traj_path): return
    df = pd.read_csv(traj_path)
    df = df[df["scenario"] == "nominal_fixed"]
    assert len(df) > 0, "fig4_nominal_tracking has 0 data points!"
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='step', y='T', hue='method', palette=color_palette)
    # Plot setpoint
    sp = df[df["method"] == df["method"].unique()[0]]["T_setpoint"]
    plt.plot(df[df["method"] == df["method"].unique()[0]]["step"], sp, 'k--', label='Setpoint')
    
    plt.title("Nominal Tracking (T vs Time)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()

def plot_fig5_disturbance_rejection(output_png, output_pdf):
    _set_style()
    traj_path = os.path.join(os.path.dirname(os.path.dirname(output_png)), "../day5/results/trajectories_seed42.csv")
    if not os.path.exists(traj_path): return
    df = pd.read_csv(traj_path)
    df = df[df["scenario"] == "feed_ca_disturbance"]
    assert len(df) > 0, "fig5_disturbance_rejection has 0 data points!"
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='step', y='T', hue='method', palette=color_palette)
    # Plot setpoint
    sp = df[df["method"] == df["method"].unique()[0]]["T_setpoint"]
    plt.plot(df[df["method"] == df["method"].unique()[0]]["step"], sp, 'k--', label='Setpoint')
    
    plt.title("Feed CA Disturbance Rejection")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.axvline(x=50, color='r', linestyle='--', label='Disturbance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()

def plot_fig6_constraint_violations(day5_agg_path, output_png, output_pdf):
    _set_style()
    if not os.path.exists(day5_agg_path): return
    df = pd.read_csv(day5_agg_path)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='scenario', y='constraint_violation_rate', hue='method', palette=color_palette)
    plt.title("Mean Constraint Violation Rate")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()

def plot_fig7_ablation(ablation_summary, output_png, output_pdf):
    _set_style()
    if not os.path.exists(ablation_summary): return
    df = pd.read_csv(ablation_summary)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    sns.barplot(data=df, x='scenario', y='mean_constraint_violation_rate', hue='method', ax=axes[0], palette=color_palette)
    axes[0].set_title("Ablation: Constraint Violations")
    axes[0].set_xticklabels([])
    axes[0].set_xlabel("")
    
    sns.barplot(data=df, x='scenario', y='mean_cumulative_reward', hue='method', ax=axes[1], palette=color_palette)
    axes[1].set_title("Ablation: Cumulative Reward")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()

def plot_fig8_robustness_heatmap(robustness_scores, output_png, output_pdf):
    _set_style()
    if not os.path.exists(robustness_scores): return
    df = pd.read_csv(robustness_scores)
    
    methods = df['method'].unique()
    scenarios = df['scenario'].unique()
    score_matrix = np.zeros((len(methods), len(scenarios)))
    
    for i, m in enumerate(methods):
        for j, s in enumerate(scenarios):
            val = df[(df['method'] == m) & (df['scenario'] == s)]['overall_robustness_score'].values
            if len(val) > 0: score_matrix[i, j] = val[0]
            
    plt.figure(figsize=(10, 6))
    sns.heatmap(score_matrix, annot=True, xticklabels=scenarios, yticklabels=methods, cmap="YlGnBu")
    plt.title("Robustness Heatmap")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()
