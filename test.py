import lightning as L
import argparse
import multiprocessing as mp
import wandb

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf
from data.ncaltech101.dataset import NCaltech101
from model.recognition import LNRecognition

def main(args):

    cfg_dataset = OmegaConf.load(args.config_data)

    if 'cnn' in args.config_model:
        cfg_dataset.representation.type = 'event_frame'
    elif 'vit' in args.config_model:
        cfg_dataset.representation.type = 'event_voxel'

    cfg_model = OmegaConf.load(args.config_model)

    model = LNRecognition(cfg_dataset, cfg_model)


    # wandb_logger = WandbLogger(project=f'dvs_filtering', name=f'{cfg_dataset.name}_{cfg_model.stage.downsample.type}')
    # wandb_logger.watch(model)

    trainer = L.Trainer(max_epochs=100, 
                        log_every_n_steps=1, 
                        gradient_clip_val=0.0,
                        # logger=wandb_logger,
                        deterministic=True,
                        devices=1)

    model = LNRecognition.load_from_checkpoint(checkpoint_path="/net/scratch/hscra/plgrid/plgjeziorek/DVS_Filtering/checkpoints/best_model_filtered.ckpt", 
                                                cfg_dataset=cfg_dataset, 
                                                cfg_model=cfg_model)

    dirs = ['/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_dat',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/0.1',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/0.01',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/0.5',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/0.05',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/0.25',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/0.75',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/1',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/1.5',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/2',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/2.5',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/3',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/4',
            '/net/scratch/hscra/plgrid/plgjeziorek/Datasets/N-Caltech/N-Caltech101_filtered6000_noise/5']

    for directory in dirs:
        print("#######################################################")
        print(directory)
        print("#######################################################")

        cfg_dataset.path = directory
        dm = NCaltech101(cfg_dataset, cfg_model)
        dm.setup()

        trainer.test(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--config_data', type=str, default='configs/dataset/ncaltech.yaml')
    parser.add_argument('-cm', '--config_model', type=str, default='configs/model/cnn.yaml')

    args = parser.parse_args()
    main(args)