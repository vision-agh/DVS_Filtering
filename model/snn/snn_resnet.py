import torch
import torch.nn as nn

from omegaconf import DictConfig
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.model import spiking_resnet

class SNNResNet(nn.Module):
    def __init__(self,
                    cfg,
                    num_classes=101):
        super().__init__()

        self.model = spiking_resnet.spiking_resnet18(pretrained=cfg.pretrained, 
                                                spiking_neuron=neuron.IFNode, 
                                                surrogate_function=surrogate.ATan(), 
                                                detach_reset=cfg.detach_reset,
                                                num_classes=num_classes)

        self.model.conv1 = layer.Conv2d(in_channels=cfg.channels, 
                                        out_channels=self.model.conv1.out_channels,
                                        kernel_size=self.model.conv1.kernel_size, 
                                        stride=self.model.conv1.stride, 
                                        padding=self.model.conv1.padding, 
                                        bias=self.model.conv1.bias,
                                        step_mode=self.model.conv1.step_mode)

    def forward(self, x: torch.Tensor):

        # Change from (B, T, C, H, W) to (T, B, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)
        
        functional.set_step_mode(self.model, 'm')
        x = self.model(x)
        x = x.mean(dim=0)
        functional.reset_net(self.model)
        return x

