import torch
from torchmetrics import MetricCollection
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_func, scheduler=None, metrics = {}, callbacks = None, half_precision=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.callbacks = callbacks or []
        self.half_precision = half_precision
        if scheduler is not None:
            self.scheduler = scheduler
        if self.half_precision:
            self.scaler = GradScaler()
        self.metrics = metrics
        self.other_metrics = MetricCollection({})
        for name, metric in self.metrics.items():
            if name == "auroc":
                self.auroc = MetricCollection({name: metric})
                # self.metrics.pop(name)
            else:
                self.other_metrics[name] = metric
                # self.other_metrics = MetricCollection(self.other_metrics)
    
    def _one_epoch(self):
        self.model.train()
        total_loss = 0.
        auroc = self.auroc.clone().to(self.device)
        other_metrics = self.other_metrics.clone().to(self.device)
        for batch in self.train_dataloader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if self.half_precision:
                with autocast(str(self.device)): # just for type error warning supression
                    y_pred = self.model(X)
                    loss = self.loss_func(y_pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                y_pred = self.model(X)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            probs = torch.softmax(y_pred, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            auroc.update(probs, y)
            other_metrics.update(preds, y)
        avg_loss = total_loss / len(self.train_dataloader)
        final_auroc = auroc.compute()
        final_other_metrics = other_metrics.compute()
        final_metrics = {**final_other_metrics, **final_auroc}
        auroc.reset()
        other_metrics.reset()
        return avg_loss, final_metrics
            
    def _validation(self):        
        self.model.eval()
        total_loss = 0.
        auroc = self.auroc.clone().to(self.device)
        other_metrics = self.other_metrics.clone().to(self.device)
        with torch.no_grad():
            for batch in self.val_dataloader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                if self.half_precision:
                    with autocast(str(self.device)): # just for type error correction
                        y_pred = self.model(X)
                        loss = self.loss_func(y_pred, y)
                else:
                    y_pred = self.model(X)
                    loss = self.loss_func(y_pred, y)
                total_loss += loss.item()
                
                probs = torch.softmax(y_pred, dim=1)
                preds = torch.argmax(probs, dim=1)
                auroc.update(probs, y)
                other_metrics.update(preds, y)
            
            avg_loss = total_loss / len(self.val_dataloader)
            probs = torch.softmax(y_pred, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            auroc.update(probs, y)
            other_metrics.update(preds, y)
        avg_loss = total_loss / len(self.train_dataloader)
        final_auroc = auroc.compute()
        final_other_metrics = other_metrics.compute()
        final_metrics = { **final_other_metrics, **final_auroc}
        auroc.reset()
        other_metrics.reset()
        return avg_loss, final_metrics
    
    def fit(self, epochs):
        self.epochs = epochs
        for epoch in range(self.epochs):
            self._call_callbacks('on_epoch_start', epoch=epoch)
            train_loss, train_metrics = self._one_epoch()
            # print("Train metrics:", train_metrics)
            val_loss, val_metrics = self._validation()
            self.val_loss = val_loss
            self._call_callbacks('on_epoch_end',
                                 epoch=epoch,
                                 train_loss=train_loss,
                                 val_loss=val_loss,
                                 train_metrics=train_metrics,
                                 val_metrics=val_metrics,
                                 )
            if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                print("Early stopping training")
                break
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.scheduler.mode == "min":
                    self.scheduler.step(self.val_loss)
                else:
                    self.scheduler.step(val_metrics['f1_score'][4])
            else:
                self.scheduler.step()
        return train_loss, train_metrics, val_loss, val_metrics
                
    def _call_callbacks(self, cb_name,*args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, cb_name, None)
            if callable(method):
                method(self,*args, **kwargs)