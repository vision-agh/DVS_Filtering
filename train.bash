#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=plgevent-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=out/job-%j.out
#SBATCH --error=out/job-%j.err
 

CONFIG_DATASET="/net/scratch/hscra/plgrid/plgjeziorek/DVS_Filtering/configs/dataset/nimagenet.yaml"
DATASET_PATH="/net/storage/pr3/plgrid/plgg_dvs_phd/N-miniImageNet_filtered40000"

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

# create and activate the virtual environment
source /net/scratch/hscra/plgrid/plgjeziorek/dvs_fil/bin/activate

export SSL_CERT_FILE=/net/scratch/hscra/plgrid/plgjeziorek/dvs_fil/lib/python3.11/site-packages/certifi/cacert.pem

python train.py -cd "$CONFIG_DATASET" -d "$DATASET_PATH" -m "vit"