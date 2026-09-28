import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.run_day1_and_day2 import run_all as run_d12
from scripts.run_day3_to_day5 import run_all as run_d35
from scripts.run_day6_to_day7 import run_all as run_d67

def run_everything():
    print("Running Day 1 to Day 2...")
    run_d12()
    print("Running Day 3 to Day 5...")
    run_d35()
    print("Running Day 6 to Day 7...")
    run_d67()

if __name__ == "__main__":
    run_everything()
