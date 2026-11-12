"""Shared project settings. Change values here, not inside other files."""
import os

DATA_DIR = os.environ.get("DATA_DIR", "/content/data")   # folder with val2014/, val2017/, annotations/, dataset_coco.json
RESULTS_DIR = "experiments/results"

# Blur types and their severity levels (blur extent in pixels; 0 = clean, run once and shared by all types)
BLUR_SETTINGS = {
    "linear": [0, 15, 30, 45],   # straight-line motion blur (camera moving in one direction)
    "shake":  [0, 15, 30, 45],   # curved, random-path motion blur (hand shake)
}

# Each seed sets a different blur direction or shake path
SEEDS = [0, 1, 2]
NUM_BEAMS = 3                   # beam search width, same for Florence-2 and BLIP

# One row per run in experiments/results/results.csv
RESULTS_COLUMNS = ["model", "task", "blur_type", "blur_px", "seed", "metric", "score", "n_images", "date", "notes"]
