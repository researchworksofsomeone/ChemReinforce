import os
import json
import shutil
import subprocess

def push_to_kaggle(output_dir, dataset_slug, username=None):
    if not username:
        username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        return "MANUAL", "No KAGGLE_USERNAME"
        
    full_slug = f"{username}/{dataset_slug}"
    stage = "/kaggle/working/_sink_stage"
    
    try:
        if os.path.exists(stage):
            shutil.rmtree(stage)
        os.makedirs(stage, exist_ok=True)
        
        shutil.copytree(output_dir, os.path.join(stage, "outputs"))
        
        meta = {"title": dataset_slug, "id": full_slug, "licenses": [{"name": "CC0-1.0"}]}
        with open(os.path.join(stage, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
            
        res = subprocess.run(["kaggle", "datasets", "create", "-p", stage, "--dir-mode", "zip"], capture_output=True, text=True)
        if "already exists" in res.stdout.lower() or "already exists" in res.stderr.lower():
            res = subprocess.run(["kaggle", "datasets", "version", "-p", stage, "-m", "update", "--dir-mode", "zip"], capture_output=True, text=True)
            
        if res.returncode == 0:
            return "SUCCESS", full_slug
        else:
            return "FAILED", res.stdout + res.stderr
    except Exception as e:
        return "FAILED", str(e)
    finally:
        if os.path.exists(stage):
            shutil.rmtree(stage)
