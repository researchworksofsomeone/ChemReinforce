import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Force python unbuffered
os.environ["PYTHONUNBUFFERED"] = "1"

print("--- REPAIR SPRINT STARTED ---")

print("\n>>> Running Day 2 (Surrogate Ensembles)...")
from scripts.run_day2 import main as run_d2
run_d2()

print("\n>>> Running Day 3-5 (RL Baselines & Evaluation)...")
from scripts.run_day3_to_day5 import run_all as run_d35
run_d35()

print("\n>>> Running Day 6 (Paper Generation 1)...")
from scripts.run_day6 import run_day6
run_day6()

print("\n>>> Running Day 7 (Paper Generation 2)...")
from scripts.run_day7 import run_day7
run_day7()

print("--- REPAIR SPRINT COMPLETED ---")
