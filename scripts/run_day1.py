import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import DATA_CONFIG
from src.utils import set_seed
from src.data_collector import collect_data
from src.validation_day1 import validate_day1
from src.kaggle_sink import push_to_kaggle
from src.config import DAY1_OUTPUT_DIR

def main():
    set_seed(42)
    print("Running Day 1...")
    
    collect_data(DATA_CONFIG["random_transitions"], DATA_CONFIG["pid_transitions"])
    
    if validate_day1():
        print("Day 1 Validation: PASS")
        res, msg = push_to_kaggle(DAY1_OUTPUT_DIR, "pi-wm-rl-day1-assets")
        print(f"Kaggle Sink: {res} - {msg}")
        return True
    else:
        print("Day 1 Validation: FAIL")
        return False

if __name__ == "__main__":
    main()
