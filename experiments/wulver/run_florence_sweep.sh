#!/bin/bash -l
#SBATCH --job-name=florence_sweep
#SBATCH --partition=course_gpu
#SBATCH --account=2026-fall-ds-677-ikoutis-vrf5
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-33%3
#SBATCH --output=florence_sweep.%A_%a.log

set -euo pipefail

# Build the list of remaining experiments
runs=()

for task in captioning detection; do
    for blur_type in linear shake; do
        for px in 5 10 20; do
            for seed in 0 1 2; do

                # These two full runs already completed
                if [[ "$blur_type" == "linear" &&
                      "$px" == "5" &&
                      "$seed" == "0" ]]; then
                    continue
                fi

                runs+=("$task $blur_type $px $seed")
            done
        done
    done
done

# Select one experiment for this array task
read -r task blur_type px seed <<< "${runs[$SLURM_ARRAY_TASK_ID]}"

echo "Running: $task $blur_type ${px}px seed=$seed"

# Set up the Wulver environment
course_dir=/course/2026/fall/ds/677/ikoutis/vrf5
project_dir="$course_dir/florence2-vs-specialists-motion-blur"

module load CUDA
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$course_dir/envs/ds677"

export DATA_DIR="$course_dir/coco_data"
export HF_HOME="$course_dir/.cache/huggingface"

cd "$project_dir"

# Select the Florence-2 runner
case "$task" in
    captioning)
        module_name="src.run_florence_captioning"
        ;;
    detection)
        module_name="src.run_florence_detection"
        ;;
esac

# Unique output directory for every array task
output_dir="$course_dir/florence_runs/${task}_${blur_type}_${px}px_seed${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Execute the experiment
"$course_dir/envs/florence2/bin/python" -u -m "$module_name" \
    --output-dir "$output_dir" \
    --limit 5000 \
    --blur-type "$blur_type" \
    --blur-px "$px" \
    --seed "$seed"
