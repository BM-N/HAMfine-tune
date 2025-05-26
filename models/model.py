import torch
import torch.nn as nn
from torchvision.models import resnet50


def get_model(name: str, sub_layer= None):
    device = ('cuda' if torch.cuda.is_available() else "cpu")
    if name == "resnet50":
        model = resnet50(weights="IMAGENET1K_V2")
    if sub_layer is not None:
        model.fc = sub_layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True 
    return model.to(device)