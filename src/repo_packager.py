import os
import shutil

def run_repo_packager(project_root):
    out_dir = os.path.join(project_root, "outputs/day7")
    os.makedirs(os.path.join(out_dir, "zip"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    
    # 1. Generate requirements.txt
    req_content = """torch
numpy
pandas
matplotlib
seaborn
gymnasium
stable-baselines3
pyarrow
scipy
kaggle"""
    with open(os.path.join(project_root, "requirements.txt"), "w") as f:
        f.write(req_content)
        
    # 2. Generate LICENSE
    license_content = "MIT License"
    with open(os.path.join(project_root, "LICENSE"), "w") as f:
        f.write(license_content)
        
    # 3. Generate README.md
    readme_content = """# PI-WM-RL CSTR Control

Physics-Informed World-Model Reinforcement Learning for Robust and Sample-Efficient Control of a Non-Isothermal CSTR.

## Repository Structure
- `src/`: Source code
- `scripts/`: Execution scripts
- `notebooks/`: Jupyter notebooks
- `outputs/`: Model artifacts, datasets, figures

## How to run Day 1 to Day 7
Use the provided `run_dayX.py` scripts inside the `scripts/` directory.

## Expected outputs
You should get trained models, plots, and a final paper.

## Hardware requirements
Kaggle T4 GPU recommended.

## Kaggle T4 notes
Ensure you use the GPU accelerator in Notebook settings.

## How to upload Kaggle datasets manually
Use `kaggle datasets create` with the provided `dataset-metadata.json`.

## How to generate the paper
Run `python scripts/run_day7.py`.

## Citation
Please cite our paper.

## License
MIT License
"""
    with open(os.path.join(project_root, "README.md"), "w") as f:
        f.write(readme_content)
        
    # 4. Zip repository
    # Shutil make_archive does not easily exclude folders by default without a custom function,
    # but we can copy the allowed files to a temporary folder, then zip it.
    temp_repo = os.path.join(project_root, "temp_repo")
    os.makedirs(temp_repo, exist_ok=True)
    
    # Copy allowed things
    for item in ["README.md", "LICENSE", "requirements.txt", "src", "scripts", "notebooks"]:
        src = os.path.join(project_root, item)
        dst = os.path.join(temp_repo, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.path.exists(src):
            shutil.copy(src, dst)
            
    os.makedirs(os.path.join(temp_repo, "outputs/day6/figures"), exist_ok=True)
    os.makedirs(os.path.join(temp_repo, "outputs/day6/tables"), exist_ok=True)
    os.makedirs(os.path.join(temp_repo, "outputs/day7/paper"), exist_ok=True)
    
    shutil.make_archive(os.path.join(out_dir, "zip", "pi-wm-rl-repo"), 'zip', temp_repo)
    shutil.rmtree(temp_repo)
    
    # 5. Zip paper
    shutil.make_archive(os.path.join(out_dir, "zip", "pi-wm-rl-final-paper"), 'zip', os.path.join(out_dir, "paper"))
