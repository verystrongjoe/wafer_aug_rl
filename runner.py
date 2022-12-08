import os

NUM_REPEAT = 10
NUM_WORKERS = 100
for seed in range(NUM_REPEAT):
    os.system(f"python main.py --seed {seed} --num_workers {NUM_WORKERS}")