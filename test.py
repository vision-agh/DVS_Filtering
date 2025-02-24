from omegaconf import OmegaConf


cfg = OmegaConf.load('configs/models/vit.yaml')
print(cfg.stage)