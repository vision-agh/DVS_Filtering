import math
import torch
from torch import nn

from omegaconf import DictConfig

from attention import (PartitionAttentionCl,
                    PartitionType,
                    nhwC_2_nChw,
                    get_downsample_layer_Cf2Cl)


class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim: int,
                 skip_first_norm: bool,
                 attention_cfg):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x
    

class MaxVitStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """
    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 stage_cfg: DictConfig):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        attention_cfg = stage_cfg.attention

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x)  # N H W C -> N C H W
        return x


class MaxVit(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        size = 48 # 32/48/64
        self.stage1 = MaxVitStage(dim_in=3, stage_dim=size*1, spatial_downsample_factor=4, num_blocks=1, stage_cfg=cfg)
        self.stage2 = MaxVitStage(dim_in=size*1, stage_dim=size*2, spatial_downsample_factor=2, num_blocks=1, stage_cfg=cfg)
        self.stage3 = MaxVitStage(dim_in=size*2, stage_dim=size*4, spatial_downsample_factor=2, num_blocks=1, stage_cfg=cfg)
        self.stage4 = MaxVitStage(dim_in=size*4, stage_dim=size*8, spatial_downsample_factor=2, num_blocks=1, stage_cfg=cfg)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(size*8, 10)

    def forward(self, x: torch.Tensor):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# make from this omegaconf config:
cfg = DictConfig({
    "attention": {
        "use_torch_mha": False,
        "partition_size": (8, 10),
        "dim_head": 32,
        "attention_bias": True,
        "mlp_activation": "gelu",
        "mlp_gated": False,
        "mlp_bias": True,
        "mlp_ratio": 4,
        "drop_mlp": 0,
        "drop_path": 0,
        "ls_init_value": 1e-5
    },
    "downsample": {
        "type": "patch",
        "overlap": True,
        "norm_affine": True
    }
})

model = MaxVit(cfg)

img = torch.randn(1, 3, 256, 320)
out = model(img)


print(out.shape)