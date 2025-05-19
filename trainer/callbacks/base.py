import wandb

class Callback:
    def on_epoch_start(self, trainer, *args, **kwargs): pass
    def on_epoch_end(self, trainer, *args, **kwargs): pass
    def on_batch_start(self, trainer, *args, **kwargs): pass
    def on_batch_end(self, trainer, *args, **kwargs): pass
    def on_train_start(self, trainer, *args, **kwargs): pass
    def on_train_end(self, trainer, *args, **kwargs): pass
    
class LossLoggerCallback(Callback, ):
    def on_epoch_start(self, trainer, epoch,*args, **kwargs):
        print(f"Epoch {epoch+1} of {trainer.epochs} starting")

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, *args, **kwargs):
        print(f'Epoch {epoch+1}/{trainer.epochs} - '
              f'Train loss: {train_loss:.4f} - '
              f'Val loss: {val_loss:.4f}')
        
class WandbLogger(Callback):  
    def on_epoch_end(self, trainer, train_metrics, val_metrics, *args, **kwargs):
        results= {
            'train': train_metrics,
            'val': val_metrics
        }
        wandb.log(results)
class EarlyStoppingCallback(Callback):
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.should_stop = False
        self.counter = 0
    
    def on_epoch_end(self, trainer, *args, **kwargs):
        val_loss = trainer.val_loss
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

# class MetricLoggerCallback(Callback):
#     def __init__(self, metrics: dict):
#         self.metrics = metrics
        
#     def on_epoch_end(self, trainer, y_pred, y_true):
#         results = {}
#         for name, metric in self.metrics.items():
#             results[name] = metric(y_pred, y_true)
#         return results