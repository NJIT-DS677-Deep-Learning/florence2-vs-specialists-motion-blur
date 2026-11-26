#!/bin/bash -l
#SBATCH --job-name=florence_clean
#SBATCH --partition=course_gpu
#SBATCH --account=2026-fall-ds-677-ikoutis-vrf5
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=florence_clean.%j.log

set -euo pipefail

course_dir=/course/2026/fall/ds/677/ikoutis/vrf5
project_dir="$course_dir/florence2-vs-specialists-motion-blur"

task="${1:?Specify captioning or detection}"
limit="${2:-}"
blur_type="${3:-linear}"
blur_px="${4:-0}"
seed="${5:-0}"

module load CUDA
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$course_dir/envs/ds677"

export DATA_DIR="$course_dir/coco_data"
export HF_HOME="$course_dir/.cache/huggingface"

cd "$project_dir"

case "$task" in
    captioning)
        module_name="src.run_florence_captioning"
        ;;
    detection)
        module_name="src.run_florence_detection"
        ;;
    blip_captioning)
        module_name="src.run_blip_captioning"
        ;;
    detr_detection)
        module_name="src.run_detr_detection"
        ;;
    *)
        echo "Unknown task: $task"
        exit 1
        ;;
esac

output_dir="$course_dir/florence_runs/${task}_${blur_type}_${blur_px}px_seed${seed}_${SLURM_JOB_ID}"

args=(
    --output-dir "$output_dir"
    --blur-type "$blur_type"
    --blur-px "$blur_px"
    --seed "$seed"
)

if [ -n "$limit" ]; then
    args+=(--limit "$limit")
fi

"$course_dir/envs/florence2/bin/python" -u -m "$module_name" "${args[@]}"
