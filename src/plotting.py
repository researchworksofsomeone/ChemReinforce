import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_learning_curves(log_path, output_path):
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='timestep', y='mean_reward', hue='method')
    plt.title('Learning Curves')
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_constraint_violations(log_path, output_path):
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='timestep', y='shield_rate', hue='seed')
    plt.title('Constraint Shielding Rate over Time (PI-Dyna)')
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_tracking(results_df, output_path, scenario, title, show_disturbance=False, disturbance_step=50):
    df = results_df[(results_df['scenario'] == scenario) & (results_df['seed'] == 42)]
    if df.empty: return
    # Wait, episode_metrics does not contain full trajectories!
    # I need full trajectories for the tracking plot!
    pass

def plot_scenario_metrics(results_df, output_dir):
    agg = results_df.groupby(["method", "scenario"]).mean(numeric_only=True).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg, x='scenario', y='cumulative_reward', hue='method')
    plt.title('Mean Cumulative Reward by Scenario')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario_cumulative_reward.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg, x='scenario', y='constraint_violation_rate', hue='method')
    plt.title('Mean Constraint Violation Rate by Scenario')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario_constraint_violations.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg, x='scenario', y='iae_T', hue='method')
    plt.title('Mean IAE by Scenario')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario_iae.png"), dpi=300)
    plt.close()

def plot_robustness_heatmap(results_df, output_path):
    agg = results_df.groupby(["method", "scenario"]).mean(numeric_only=True).reset_index()
    
    # Simple score: norm(reward) - norm(iae) - norm(violation)
    methods = agg['method'].unique()
    scenarios = agg['scenario'].unique()
    score_matrix = np.zeros((len(methods), len(scenarios)))
    
    for j, sc in enumerate(scenarios):
        sc_data = agg[agg['scenario'] == sc]
        r = sc_data['cumulative_reward'].values
        i = sc_data['iae_T'].values
        v = sc_data['constraint_violation_rate'].values
        
        r_norm = (r - np.min(r)) / (np.max(r) - np.min(r) + 1e-6)
        i_norm = (np.max(i) - i) / (np.max(i) - np.min(i) + 1e-6)
        v_norm = (np.max(v) - v) / (np.max(v) - np.min(v) + 1e-6)
        
        score = r_norm + i_norm + v_norm
        
        for k, m in enumerate(methods):
            idx = sc_data['method'] == m
            if np.any(idx):
                score_matrix[k, j] = score[idx][0]
                
    plt.figure(figsize=(10, 6))
    sns.heatmap(score_matrix, annot=True, xticklabels=scenarios, yticklabels=methods, cmap="YlGnBu")
    plt.title("Robustness Heatmap (Higher is Better)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
