import lightning as L
import argparse
import multiprocessing as mp
import wandb
import glob
import os

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf
from data.ncaltech101.dataset import NCaltech101
from data.ncars.dataset import NCars
from data.nimagenet.dataset import NImageNet
from model.recognition import LNRecognition

from configs.dirs_datasets import dirs_ncaltech, dirs_ncars, dirs_nimagenet

def main(args):

    print(args.ckpt_path)
    cfg_dataset = OmegaConf.load(args.config_data)
    cfg_dataset.train.all_noisy = False

    if 'cnn' in args.model:
        model_type = 'cnn'
        cfg_dataset.representation.type = 'event_frame'
    elif 'vit' in args.model:
        model_type = 'vit'
        cfg_dataset.representation.type = 'event_voxel'
    elif 'snn' in args.model:
        model_type = 'snn'
        cfg_dataset.representation.type = 'event_spikes'

    cfg_model = cfg_dataset.model[model_type]

    model = LNRecognition(cfg_dataset, cfg_model)

    # wandb_logger = WandbLogger(project=f'dvs_filtering', name=f'{cfg_dataset.name}_{cfg_model.stage.downsample.type}')
    # wandb_logger.watch(model)

    trainer = L.Trainer(max_epochs=100, 
                        log_every_n_steps=1, 
                        gradient_clip_val=0.0,
                        # logger=wandb_logger,
                        deterministic=True,
                        devices=1)

    model = LNRecognition.load_from_checkpoint(checkpoint_path=args.ckpt_path, 
                                                cfg_dataset=cfg_dataset, 
                                                cfg_model=cfg_model)

    
    if 'ncaltech' in args.config_data:
        dirs = dirs_ncaltech
    
    elif 'ncars' in args.config_data:
        dirs = dirs_ncars
        
    elif 'nimagenet' in args.config_data:
        dirs = dirs_nimagenet

    # For filtered noises use this:
    dirs = glob.glob(os.path.join(args.dataset, '*'))
    #

    for directory in dirs:
        print("#######################################################")
        print(directory)
        print("#######################################################")

        cfg_dataset.path = directory
        if 'ncaltech' in args.config_data:
            dm = NCaltech101(cfg_dataset, cfg_model)
        
        elif 'ncars' in args.config_data:
            dm = NCars(cfg_dataset, cfg_model)

        elif 'nimagenet' in args.config_data:
            dm = NImageNet(cfg_dataset, cfg_model)

        dm.setup()

        trainer.test(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--config_data', type=str, default='configs/dataset/ncaltech.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='cnn')
    parser.add_argument('-m', '--model', type=str, default='cnn')
    parser.add_argument('-ckpt', '--ckpt_path', type=str, default='/net/scratch/hscra/plgrid/plgjeziorek/DVS_Filtering/checkpoints/best_model_original.ckpt')

    args = parser.parse_args()
    main(args)