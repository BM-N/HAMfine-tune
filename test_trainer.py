import torch
import wandb
from torch import nn, optim
from torchmetrics.classification import Accuracy

from models.model import get_model
from models.transforms import get_transforms
from data.datamodule import get_dataloader
from trainer.trainer import Trainer
from trainer.callbacks.base import ( LossLoggerCallback,
                                    WandbLogger,
                                    EarlyStoppingCallback)


# 1. Config
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir1 = "data/HAM10000_images_part_1"
img_dir2 = "data/HAM10000_images_part_2"
csv_file = "data/val_set.csv"  # small sample file is enough for this
# bs = 4  # keep it small for testing
train_transform = get_transforms()

metrics = {
    "accuracy": Accuracy(task="multiclass", num_classes=7)
}

wandb.init(
    project="ham10000-resnet",
    config={
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 2,
    "architecture": "resnet50",
    "remove_fc": False,
    },
    settings=wandb.Settings(console="wrap")
)
config = wandb.config

# 2. Setup
model = get_model(config.architecture, config.remove_fc) # .to(device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

train_dl = get_dataloader(img_dir1, img_dir2, csv_file=csv_file, transform=train_transform, bs=config.batch_size, shuffle=True)

# 3. Callback
callbacks = [WandbLogger(), LossLoggerCallback(), EarlyStoppingCallback()]

# 4. Trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dl,
    val_dataloader=train_dl,  # for testing, use train_dl as val_dl
    optimizer=optimizer,
    loss_func=criterion,
    # device=device,
    callbacks=callbacks 
)

# 5. Run a few epochs
train_loss, val_loss = trainer.fit(epochs=config.epochs)
print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# lr_scheduler
# metrics callback
# wandb setup
# fine-tune implementation