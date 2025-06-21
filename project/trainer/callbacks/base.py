import wandb
import torch
import os


class Callback:
    def on_epoch_start(self, trainer, *args, **kwargs):
        pass

    def on_epoch_end(self, trainer, *args, **kwargs):
        pass

    def on_batch_start(self, trainer, *args, **kwargs):
        pass

    def on_batch_end(self, trainer, *args, **kwargs):
        pass

    def on_train_start(self, trainer, *args, **kwargs):
        pass

    def on_train_end(self, trainer, *args, **kwargs):
        pass


class LossLoggerCallback(Callback):
    def on_epoch_start(self, trainer, epoch, *args, **kwargs):
        print(f"Epoch {epoch + 1} of {trainer.epochs} starting")

    def on_epoch_end(
        self,
        trainer,
        epoch,
        train_loss,
        val_loss,
        train_metrics,
        val_metrics,
        *args,
        **kwargs,
    ):
        print(
            f"Epoch {epoch + 1}/{trainer.epochs} - "
            f"Train loss: {train_loss:.4f} - "
            f"Val loss: {val_loss:.4f} - "
        )


class WandbLogger(Callback):
    def __init__(self, classes=None):
        self.classes = classes

    def on_epoch_end(
        self,
        trainer,
        epoch,
        train_loss,
        val_loss,
        train_metrics,
        val_metrics,
        *args,
        **kwargs,
    ):
        log_data = {}

        if self.classes:
            for name, value in train_metrics.items():
                if name not in ["accuracy", "weighted_recall"]:
                    log_data[f"train/{name}"] = {
                        cls: round(value[i].item(), 4)
                        for i, cls in enumerate(self.classes)
                    }
                else:
                    log_data[f"train/{name}"] = value.item()

            for name, value in val_metrics.items():
                if name not in ["accuracy", "weighted_recall"]:
                    log_data[f"val/{name}"] = {
                        cls: round(value[i].item(), 4)
                        for i, cls in enumerate(self.classes)
                    }
                else:
                    log_data[f"val/{name}"] = value.item()

        else:
            for name, value in train_metrics.items():
                log_data[f"train/{name}"] = value

            for name, value in val_metrics.items():
                log_data[f"val/{name}"] = value

        log_data["train/loss"] = train_loss
        log_data["val/loss"] = val_loss
        log_data["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]

        wandb.log(log_data, step=epoch)
        print(f"Data for Epoch {epoch + 1} of {trainer.epochs}")
        print(f"train/recall: {log_data['train/recall']}", "\n")
        print(f"val/recall: {log_data['val/recall']}", "\n")
        print(f"train/precision: {log_data['train/precision']}", "\n")
        print(f"val/precision: {log_data['val/precision']}", "\n")
        print(f"train/f1_score: {log_data['train/f1_score']}", "\n")
        print(f"val/f1_score: {log_data['val/f1_score']}", "\n")
        print(f"train/accuracy: {log_data['train/accuracy']}", "\n")
        print(f"val/accuracy: {log_data['val/accuracy']}", "\n")
        print(f"train/auroc: {log_data['train/auroc']}", "\n")
        print(f"val/auroc: {log_data['val/auroc']}", "\n")
        print(f"train/weighted_recall: {log_data['train/weighted_recall']}", "\n")
        print(f"val/weighted_recall: {log_data['val/weighted_recall']}", "\n")


class EarlyStoppingCallback(Callback):
    def __init__(self, patience=10, mode="min", min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        if self.mode == "min":
            self.best_loss = float("inf")
        if self.mode == "max":
            self.best_value = float("-inf")
        self.should_stop = False
        self.counter = 0

    def on_epoch_end(self, trainer, *args, **kwargs):
        if self.mode == "min":
            val_loss = trainer.val_loss
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        if self.mode == "max":
            val_f1_mel = kwargs["val_metrics"]["f1_score"][4]
            if val_f1_mel > self.best_value + self.min_delta:
                self.best_value = val_f1_mel
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class ModelCheckpoint(Callback):
    def __init__(self, monitor="val_loss", mode="min", save_best_only=True):
        self.monitor = monitor
        self.best_score = float("inf")
        self.save_best_only = save_best_only
        self.mode = mode
        if self.mode == "min":
            self.best_score = float("inf")
        if self.mode == "max":
            self.best_score = float("-inf")

    def on_epoch_end(self, trainer, epoch, val_loss, val_metrics, *args, **kwargs):
        should_save = False
        if not self.save_best_only:
            should_save = True
        if self.mode == "min":
            current_score = val_loss
            if current_score < self.best_score:
                self.best_score = current_score
                should_save = True
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved. Saving model...")
        if self.mode == "max":
            metric_score = val_metrics[self.monitor]
            print(metric_score)
            current_score = metric_score[4]
            print(current_score)
            if current_score > self.best_score:
                self.best_score = current_score
                should_save = True
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved. Saving model...")
        if should_save:
            aliases = ["latest", f"epoch_{epoch + 1}"]
            if self.save_best_only:
                aliases.append("best")
            # artifact logging
            model_artifact = wandb.Artifact(
                name="my-model",
                type="model",
                description=f"Model from epoch {epoch + 1} with {self.monitor} of {current_score:.4f}",
                metadata={
                    "epoch": epoch + 1,
                    self.monitor: self.best_score,
                    **val_metrics,
                },
            )

            # create a temporary file to save the checkpoint
            checkpoint_path = "model.pth"
            torch.save(trainer.model.state_dict(), checkpoint_path)

            # add the file to the artifact and log it
            model_artifact.add_file(checkpoint_path)

            wandb.log_artifact(model_artifact, aliases=aliases)

            # clean up the local file
            os.remove(checkpoint_path)
