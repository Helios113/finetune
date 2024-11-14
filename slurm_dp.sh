#!/bin/bash
#SBATCH -J LLM_finetune 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#! specify node
#SBATCH -w mauao
#SBATCH --output=slurm_out/%x.%j.ans
#SBATCH --error=slurm_out/err_%x.%j.ans

# source /nfs-share/pa511/developments/lm_memorisation/.vevn/bin/activate
# poetry shell
WANDB_MODE=disabled HYDRA_FULL_ERROR=1 srun poetry run python train_dp.py --config-name=config_pythia_dp
