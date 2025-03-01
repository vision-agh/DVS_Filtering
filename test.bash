#!/bin/bash -l
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=plgevent-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=out_test/test_%x_%j.out
#SBATCH --error=out_test/test_job-%j_%t.err

#SBATCH --array=0-3
 
# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

# create and activate the virtual environment
source /net/scratch/hscra/plgrid/plgjeziorek/dvs_fil/bin/activate

CONFIGS=("/net/scratch/hscra/plgrid/plgjeziorek/DVS_Filtering/checkpoints/snn_NCARS_filtered40000_noise_1_all_noise=False.ckpt" "/net/scratch/hscra/plgrid/plgjeziorek/DVS_Filtering/checkpoints/snn_NCARS_NCARS_dat_all_noise=False.ckpt" "/net/scratch/hscra/plgrid/plgjeziorek/DVS_Filtering/checkpoints/snn_NCARS_NCARS_dat_all_noise=True.ckpt" "/net/scratch/hscra/plgrid/plgjeziorek/DVS_Filtering/checkpoints/snn_NCARS_NCARS_filtered40000_all_noise=False.ckpt") 

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

python test.py --ckpt $CONFIG -m "snn" -cd "configs/dataset/ncars.yaml"