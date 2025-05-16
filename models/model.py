import torch
import torch.nn as nn
from torchvision.models import resnet50


def get_model(name: str, remove_fc: bool = False, sub_layer: nn.Module = nn.Identity()):
    device = ('cuda' if torch.cuda.is_available() else "cpu")
    if name == "resnet50":
        model = resnet50(weights="IMAGENET1K_V2").to(device)
    if remove_fc:
        model.fc = sub_layer # type: ignore
    return model


# sequence = nn.Sequential(
#     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
# )

# model = get_model(name="resnet50", remove_fc=True, sub_layer=sequence)
# print(model)
