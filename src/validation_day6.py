import os

def validate_day6(project_root):
    out = os.path.join(project_root, "outputs/day6")
    reqs = [
        "analysis/consolidated_summary.json",
        "analysis/main_results.csv",
        "analysis/robustness_scores.csv",
        "analysis/statistical_tests.csv",
        "ablations/logs/ablation_report.json",
        "figures/fig1_architecture.png",
        "figures/fig2_surrogate_rollout.png",
        "figures/fig3_learning_curves.png",
        "figures/fig4_nominal_tracking.png",
        "figures/fig5_disturbance_rejection.png",
        "figures/fig6_constraint_violations.png",
        "figures/fig7_ablation.png",
        "figures/fig8_robustness_heatmap.png",
        "tables/table_main_results.csv",
        "tables/table_ablation.csv",
        "latex/table_main_results.tex",
        "logs/day6_report.json"
    ]
    for r in reqs:
        if not os.path.exists(os.path.join(out, r)):
            return False
            
    for seed in [42, 123, 2024]:
        for method in ["pi_dyna_no_shield", "pi_no_dyna_shield", "mlp_dyna_shield"]:
            if not os.path.exists(os.path.join(out, "ablations/models", f"{method}_seed_{seed}.zip")):
                return False
                
    return True
