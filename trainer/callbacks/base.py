class Callback:
    def on_epoch_start(self): pass
    def on_epoch_end(self): pass
    def on_batch_start(self): pass
    def on_batch_end(self): pass
    def on_train_start(self): pass
    def on_train_end(self): pass
    
class LossLoggerCallback(Callback):
    def on_epoch_start(self, trainer, epoch):
        print(f"Epoch {epoch+1} of {trainer.epochs} starting")
    
    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        print(f'Epoch {epoch+1}/{trainer.epochs} - '
              f'Train loss: {train_loss:.4f} - '
              f'Val loss: {val_loss:.4f}')
        
class WandbLogger(Callback):
    def __init__(self): self.wandb = __import__('wandb')
        
    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        self.wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
class EarlyStoppingCallback(Callback):
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.should_stop = False
        self.counter = 0
    
    def on_epoch_end(self, trainer, *args):
        val_loss = trainer.val_loss
        for self.counter in range(self.patience):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
class MetricLoggerCallback(Callback):
    def __init__(self, metrics: dict):
        self.metrics = metrics
        
    def on_epoch_end(self, trainer, y_pred, y_true):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(y_pred, y_true)