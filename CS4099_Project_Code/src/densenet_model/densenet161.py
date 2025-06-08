# densenet.py
import torch
import torch.nn as nn
from torch import amp
from torch import optim
from torchvision.models import densenet161, DenseNet161_Weights
from torch.utils.data import DataLoader
# import config as config


# TorchVision DensetNet121: https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html
def initialize_densenet(num_classes):
    model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1) # Pre-trained on ImageNet1K
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    # model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

