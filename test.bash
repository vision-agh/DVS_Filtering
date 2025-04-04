#!/bin/bash -l
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=plgevent-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=out_test/test_%x_%j.out
#SBATCH --error=out_test/test_job-%j_%t.err

#SBATCH --array=0-3
 
# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

# create and activate the virtual environment
source /net/scratch/hscra/plgrid/plgjeziorek/dvs_fil/bin/activate

CONFIGS=(
        "/net/scratch/hscra/plgrid/plgjeziorek/weights/ncaltech/CNN_NCaltech_best_model_filtered.ckpt" 
        "/net/scratch/hscra/plgrid/plgjeziorek/weights/ncaltech/CNN_NCaltech_best_model_noisy_1.ckpt" 
        "/net/scratch/hscra/plgrid/plgjeziorek/weights/ncaltech/CNN_NCaltech_best_model_noisy_all.ckpt" 
        "/net/scratch/hscra/plgrid/plgjeziorek/weights/ncaltech/CNN_NCaltech_best_model_original.ckpt"
) 

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

python test.py --ckpt $CONFIG -m "cnn" -cd "configs/dataset/ncaltech.yaml" -d "/net/storage/pr3/plgrid/plgg_dvs_phd/N-Caltech101_filtered6000_noise_DIFanalysis_filteredDIF/7249"