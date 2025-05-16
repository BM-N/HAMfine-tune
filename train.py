import os
import argparse

import torch
from torch import optim, nn
import torch.nn.functional as F

from models.model import get_model
from models.transforms import get_transforms
from data.datamodule import get_dataloader
from trainer.trainer import Trainer
from trainer.callbacks.base import LossLoggerCallback, WandbLogger, EarlyStoppingCallback
 

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--remove_fc", action="store_true", help="Remove the final fully connected layer from the model.")
args = parser.parse_args()

# Pass the new argument to get_model. Using keyword arguments for clarity.
model = get_model(name=args.model, remove_fc=args.remove_fc)

print("Training Configuration:")
print(f"Learning Rate: {args.lr}, Model: {args.model}, Batch Size: {args.bs}, Epochs: {args.epochs}, Remove FC: {args.remove_fc}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir1 = os.path.abspath("data/HAM10000_images_part_1/")
img_dir2 = os.path.abspath("data/HAM10000_images_part_2/")
train_file = os.path.abspath("data/train_set.csv")
val_file = os.path.abspath("data/val_set.csv")
test_file = os.path.abspath("data/test_set.csv")

train_transform = get_transforms()
val_transform = get_transforms(train=False)

train_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=train_file, transform=train_transform, shuffle=True)
val_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=val_file, transform=val_transform)
test_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=test_file, transform=val_transform)
opt = optim.Adam(model.parameters(), lr=args.lr)
loss_func = nn.CrossEntropyLoss()
callbacks = [WandbLogger(), LossLoggerCallback(), EarlyStoppingCallback()]


# terminar de implementar learning schedule, wandb (aqui) e fine-tuning.
trainer = Trainer(model, train_dls, val_dls, opt, loss_func, device, callbacks)