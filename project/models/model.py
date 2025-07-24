import os
import torch
from torchvision.models import resnet50

def get_model(name: str = "resnet50", weight_path=None, new_head=None):
    if name == "resnet50":
        model=resnet50(weights=None)
        if not weight_path or not os.path.exists(weight_path):
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            model = resnet50(weights="IMAGENET1K_V2")
            torch.save(model.state_dict(), weight_path)
            model.load_state_dict(torch.load(weight_path, weights_only=False))
        else: 
            model.load_state_dict(torch.load(weight_path, weights_only=False))
        if new_head is not None:
            model.fc = new_head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model