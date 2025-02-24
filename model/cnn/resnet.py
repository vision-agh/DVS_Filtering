import numpy as np
import torch
import torchvision

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def get_resnet(model_name='resnet18'):
    if model_name == 'resnet18':
        model = resnet18(weights=None)
    elif model_name == 'resnet34':
        model = resnet34(weights=None)
    elif model_name == 'resnet50':
        model = resnet50(weights=None)
    elif model_name == 'resnet101':
        model = resnet101(weights=None)
    elif model_name == 'resnet152':
        model = resnet152(weights=None)
    else:
        raise ValueError('Invalid model name: {}'.format(model_name))
    return model

def get_resnet_weights(model_name='resnet18'):
    if model_name == 'resnet18':
        weight = torchvision.models.resnet18(weights=True).state_dict()
    elif model_name == 'resnet34':
        weight = torchvision.models.resnet34(weights=True).state_dict()
    elif model_name == 'resnet50':
        weight = torchvision.models.resnet50(weights=True).state_dict()
    elif model_name == 'resnet101':
        weight = torchvision.models.resnet101(weights=True).state_dict()
    elif model_name == 'resnet152':
        weight = torchvision.models.resnet152(weights=True).state_dict()
    else:
        raise ValueError('Invalid model name: {}'.format(model_name))
    return weight


class Backbone(torch.nn.Module):
    def __init__(self, 
                 input_channels,
                 model_name='resnet18',
                 weights=None):
        super(Backbone, self).__init__()

        self.model = get_resnet(model_name)

        if weights:
            self.model.load_state_dict(get_resnet_weights(model_name))
        
        if input_channels != self.model.conv1.in_channels:
            self.model.conv1 = torch.nn.Conv2d(input_channels, self.model.conv1.out_channels, 
                                               kernel_size=self.model.conv1.kernel_size, 
                                               stride=self.model.conv1.stride, 
                                               padding=self.model.conv1.padding, 
                                               bias=False)

        # Remove avgpool and fc layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        return self.model(x)
    
class Classifier(torch.nn.Module):
    def __init__(self, 
                 num_classes=10,
                 input_size=512):
        super(Classifier, self).__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNetModel(torch.nn.Module):
    def __init__(self, 
                 input_channels,
                 model_name='resnet18',
                 num_classes=10,
                 weights=None):
        super(ResNetModel, self).__init__()
        self.backbone = Backbone(input_channels, model_name, weights)
        self.classifier = Classifier(num_classes, input_size=512)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x