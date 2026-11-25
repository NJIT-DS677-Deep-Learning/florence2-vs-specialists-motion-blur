# Shared project settings.
import os

# Folder with val2014/, val2017/, annotations/, dataset_coco.json
DATA_DIR = os.environ.get("DATA_DIR", "/content/data")
# Folder where result files are saved
RESULTS_DIR = "experiments/results"

# Blur types and their severity levels (blur extent in pixels; 0 = clean, run once and shared by all types)
# Changed from 15/30/45 after a 50-image pilot: both Florence-2 and DETR lost over 75% of their
# detection score at 45 px, so the higher levels could not tell the models apart
BLUR_SETTINGS = {
    # Straight-line motion blur (camera moving in one direction)
    "linear": [0, 5, 10, 20],
    # Curved, random-path motion blur (hand shake)
    "shake":  [0, 5, 10, 20]
}

# Each seed sets a different blur direction or shake path
SEEDS = [0, 1, 2]
# Beam search width, same for Florence-2 and BLIP
NUM_BEAMS = 3

# One row per run in experiments/results/results.csv
RESULTS_COLUMNS = ["model", "task", "blur_type", "blur_px", "seed", "metric", "score", "n_images", "date", "notes"]
