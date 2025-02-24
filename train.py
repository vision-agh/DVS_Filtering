from omegaconf import OmegaConf
from data.ncaltech101.dataset import NCaltech101

cfg_dataset = OmegaConf.load('configs/dataset/ncaltech.yaml')
cfg_model = OmegaConf.load('configs/model/vit.yaml')

ds = NCaltech101(cfg_dataset, cfg_model)
ds.setup()

