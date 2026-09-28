import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def main():
    # 1328x531 pixels at 100 DPI
    fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 13.28)
    ax.set_ylim(0, 5.31)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    # Colors (Okabe-Ito)
    color_pinn = '#D55E00'  # vermillion
    color_ens = '#009E73'   # bluish green
    
    # Left: CSTR icon
    # Reactor body
    reactor = patches.Rectangle((1.0, 1.5), 2.0, 2.5, linewidth=2, edgecolor='black', facecolor='#E69F00', alpha=0.3)
    ax.add_patch(reactor)
    ax.text(2.0, 2.75, "CSTR", fontsize=14, ha='center', va='center', fontweight='bold')
    
    # Valve icon
    valve1 = patches.Polygon([[3.0, 3.2], [3.5, 3.4], [3.5, 3.0]], closed=True, edgecolor='black', facecolor='#999999')
    valve2 = patches.Polygon([[4.0, 3.2], [3.5, 3.4], [3.5, 3.0]], closed=True, edgecolor='black', facecolor='#999999')
    ax.add_patch(valve1)
    ax.add_patch(valve2)
    ax.text(3.5, 2.8, "rate limit + delay\n(unmodeled)", fontsize=11, ha='center', va='top')
    
    # Middle: Two branches
    # Branch 1 (Top): PINN Surrogate
    ax.annotate("", xy=(6.5, 4.0), xytext=(4.5, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.5)
    ax.text(6.8, 4.0, "PINN surrogate\n(ODE residual)", fontsize=12, ha='left', va='center', bbox=bbox_props)
    
    ax.annotate("", xy=(9.5, 4.0), xytext=(8.5, 4.0), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    ax.text(9.7, 4.0, "X  8.1% violations", fontsize=12, ha='left', va='center', color=color_pinn, fontweight='bold')

    # Branch 2 (Bottom): 3-head ensemble
    ax.annotate("", xy=(6.5, 2.0), xytext=(4.5, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    
    ax.text(6.8, 2.0, "3-head ensemble\n+ uncertainty gate", fontsize=12, ha='left', va='center', bbox=bbox_props)
    
    ax.annotate("", xy=(9.5, 2.0), xytext=(8.8, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    ax.text(9.7, 2.0, "V  1.1% violations", fontsize=12, ha='left', va='center', color=color_ens, fontweight='bold')

    # Right: Inset temperature traces
    ax_in = fig.add_axes([0.75, 0.3, 0.2, 0.4])
    import numpy as np
    x = np.linspace(0, 100, 100)
    # Crossing trace
    y_cross = 350 + 40 * np.sin(x/15) + x*0.2
    # Staying below trace
    y_stay = 350 + 15 * np.sin(x/15) + 10*(1 - np.exp(-x/10))
    
    ax_in.plot(x, y_cross, color=color_pinn, lw=2, label="PINN")
    ax_in.plot(x, y_stay, color=color_ens, lw=2, label="Ensemble")
    ax_in.axhline(370, color='black', linestyle='--', lw=1.5)
    
    ax_in.text(5, 372, "$T_{max}$", fontsize=11, fontfamily='sans-serif')
    
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_xlabel("Time", fontsize=10)
    ax_in.set_ylabel("T [K]", fontsize=10)
    ax_in.spines['top'].set_visible(False)
    ax_in.spines['right'].set_visible(False)
    
    out_dir = "outputs/paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    fig.savefig(os.path.join(out_dir, "graphical_abstract.png"), dpi=100)
    fig.savefig(os.path.join(out_dir, "graphical_abstract.pdf"), dpi=100)
    
    print(f"Generated {os.path.join(out_dir, 'graphical_abstract.png')} and .pdf")

if __name__ == "__main__":
    main()
