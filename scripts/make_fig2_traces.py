import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    checkpoints = [
        "outputs/day3/models/mf_sac_seed_42.zip",
        "outputs/day3/models/mlp_dyna_sac_seed_42.zip",
        "outputs/day4/models/pi_dyna_sac_seed_42.zip",
        "outputs/day3/models/pid_controller.json"
    ]
    missing = [c for c in checkpoints if not os.path.exists(c)]
    if missing:
        print("SKIP Item 2: Missing checkpoints: " + ", ".join(missing))
        sys.exit(0)
    
    try:
        df = pd.read_csv("outputs/day5/results/trajectories_seed42.csv")
        df_scenario = df[df['scenario'] == 'actuator_rate_limit']
        if df_scenario.empty:
            print("SKIP Item 2: No data for scenario 'actuator_rate_limit' found.")
            return
            
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = {
            'PID': '#999999',
            'MF-SAC': '#0072B2',
            'MLP-Dyna-SAC': '#E69F00',
            'PI-Dyna-SAC': '#D55E00'
        }
        
        for method, color in colors.items():
            # In the CSV, the method names might be just MLP-Dyna, etc.
            csv_method = method
            if csv_method not in df_scenario['method'].values:
                # try alternative names if needed
                if csv_method == 'MLP-Dyna-SAC': csv_method = 'MLP-Dyna'
                if csv_method == 'PI-Dyna-SAC': csv_method = 'PI-Dyna'
                
            mdf = df_scenario[df_scenario['method'] == csv_method]
            if mdf.empty: continue
            
            grouped = mdf.groupby('step')['T']
            mean_T = grouped.mean()
            min_T = grouped.min()
            max_T = grouped.max()
            
            steps = mean_T.index
            ax.plot(steps, mean_T, label=method, color=color, linewidth=1.5)
            ax.fill_between(steps, min_T, max_T, color=color, alpha=0.2)
            
        ax.axhline(370, color='black', linestyle='--', label='T_max')
        ax.fill_between([0, 200], 370, 450, color='red', alpha=0.1) 
        
        ax.set_xlim(0, 200)
        ax.set_xlabel('step')
        ax.set_ylabel('T [K]')
        ax.legend()
        
        out_dir = "outputs/paper_figures"
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "Fig2.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, "Fig2.png"), dpi=300, bbox_inches='tight')
        print(f"Generated {os.path.join(out_dir, 'Fig2.pdf')} and Fig2.png")
    except Exception as e:
        print(f"SKIP Item 2: Exception during plotting - {e}")

if __name__ == "__main__":
    main()
