#!/bin/bash
#SBATCH --job-name=icl_hint
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/vault/iwi5/iwi5359h/icl_input_ablation/logs/job_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/icl_input_ablation/logs/job_%j.err

source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/iclhint

export HF_HOME=/home/vault/iwi5/iwi5359h/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd /home/hpc/iwi5/iwi5359h/my_repos/introspective-interp/experiments/icl_input_ablation

# PILOT (uncomment for first run):
# python3 run_icl.py --ks 0,4,8 --seeds 0 --limit 300

# FULL:
python3 run_icl.py --ks 0,4,8 --seeds 0,1,2
python3 score.py
