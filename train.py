import os
import shutil
import argparse
import torch
import wandb
import pandas as pd
# import torch
from torch import optim, nn
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

from models.model import get_model
from models.transforms import get_transforms
from data.datamodule import get_dataloader, get_loss_class_weights
from trainer.trainer import Trainer
from trainer.callbacks.base import LossLoggerCallback, WandbLogger, EarlyStoppingCallback, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.36057091203514374)
parser.add_argument("--gama", type=float, default=1.5)
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--fp16", action="store_true", help="Use mixed precision training.")
parser.add_argument('--artifact', type=str, default=None, help="Artifact's name to load the model from.")
args = parser.parse_args()

print("Training Configuration:")
print(f"Learning Rate: {args.lr}, Model: {args.model}, Batch Size: {args.bs}, Epochs: {args.epochs}, FP16: {args.fp16}")

WANDB_ENTITY = "bmnunes-universidade-federal-de-s-o-paulo-unifesp"
WANDB_PROJECT = "ham10000-resnet"
df = pd.read_csv(os.path.abspath("data/HAM10000_metadata.csv"))
img_dir1 = os.path.abspath("data/HAM10000_images_part_1/")
img_dir2 = os.path.abspath("data/HAM10000_images_part_2/")
train_file = os.path.abspath("data/train_set.csv")
val_file = os.path.abspath("data/val_set.csv")
test_file = os.path.abspath("data/test_set.csv")
class_names = sorted(df["dx"].unique())
class_weights, class_labels = get_loss_class_weights(train_file)

train_transform = get_transforms()
val_transform = get_transforms(train=False)

train_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=train_file,bs=args.bs, transform=train_transform, shuffle = True) # sampler=sampler)
val_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=val_file, bs=args.bs, transform=val_transform)
test_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=test_file, bs=args.bs, transform=val_transform)

new_head = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(args.dropout),
    nn.Linear(512, 7),
)
model = get_model(name=args.model, sub_layer=new_head)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# loss_func = nn.CrossEntropyLoss(weight=class_weights)
loss_func = torch.hub.load('adeelh/pytorch-multi-class-focal-loss',
                            model='FocalLoss',
                            alpha=class_weights,
                            gamma=1.5,
                            reduction='mean')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', patience=4, min_lr=1e-7)
callbacks = [
    WandbLogger(classes=class_names),
    ModelCheckpoint(monitor="f1_score", mode="max"),
    LossLoggerCallback(),
    EarlyStoppingCallback(patience=8, mode="max", min_delta=0.001),
    ]
average='none'
metrics = {
    "accuracy": Accuracy(task="multiclass", num_classes=7, average='macro'),
    "precision": Precision(task="multiclass", num_classes=7, average=average),
    "recall": Recall(task="multiclass", num_classes=7, average=average),
    "weighted_recall": Recall(task="multiclass", num_classes=7, average='weighted'),
    "f1_score": F1Score(task="multiclass", num_classes=7, average=average),
    "auroc": AUROC(task="multiclass", num_classes=7, average=average),
}
run = wandb.init(
    project="ham10000-resnet",
    config={
        "learning_rate": args.lr,
        "batch_size": args.bs,
        "epochs": args.epochs,
        "architecture": args.model,
        "fc_layer_dims": [2048, 512, 7],
        "optimizer": "Adam",
        "early_stopping": "True(f1_mel)",
        "loss": f"FocalLoss(weighted), gamma={args.gama}",
        "scheduler": "ReduceLROnPlateau",
        "batchnorm": "None",
        "dropout":f"{args.dropout}",
        "half_precision": args.fp16,
    }
)
config = wandb.config

if args.artifact:
    artifact_to_load = run.use_artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{args.artifact}')
    artifact_dir = artifact_to_load.download()
    model_path = os.path.join(artifact_dir, "model.pth")
    model.load_state_dict(torch.load(model_path))
    print('Successfuly loaded model from artifact')
    shutil.rmtree(artifact_dir)
    print('Successfuly deleted artifact from local storage')

trainer = Trainer(
    model=model,
    train_dataloader=train_dls,
    val_dataloader=val_dls,
    optimizer=optimizer,
    loss_func=loss_func,
    scheduler=scheduler,
    callbacks=callbacks,
    metrics=metrics,
    half_precision=args.fp16,
)

train_loss, train_metrics, val_loss, val_metrics = trainer.fit(epochs=args.epochs)
