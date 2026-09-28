import os

def validate_day7(project_root):
    out = os.path.join(project_root, "outputs/day7")
    reqs = [
        "paper/main.tex",
        "paper/main.md",
        "paper/references.bib",
        "zip/pi-wm-rl-repo.zip",
        "zip/pi-wm-rl-final-paper.zip",
        "logs/day7_report.json"
    ]
    for r in reqs:
        if not os.path.exists(os.path.join(out, r)):
            return False
            
    fig_dir = os.path.join(out, "paper/figures")
    if not os.path.exists(fig_dir) or len(os.listdir(fig_dir)) < 8:
        return False
        
    if not os.path.exists(os.path.join(project_root, "README.md")): return False
    if not os.path.exists(os.path.join(project_root, "LICENSE")): return False
    if not os.path.exists(os.path.join(project_root, "requirements.txt")): return False
    
    return True
