import torch
from torchmetrics import MetricCollection
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler




class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_func, scheduler=None, metrics = {}, callbacks = None, half_precision=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.callbacks = callbacks or []
        self.metrics = MetricCollection(metrics)
        self.half_precision = half_precision
        if self.half_precision:
            self.scaler = GradScaler()
    
    def _one_epoch(self):
        self.model.train()
        total_loss = 0.
        train_metrics = self.metrics.clone().to(self.device)
        
        for batch in self.train_dataloader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if self.half_precision:
                with autocast(str(self.device)): # just for type error correction
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
            train_metrics.update(y_pred, y)
        avg_loss = total_loss / len(self.train_dataloader)
        final_metrics = train_metrics.compute()
        train_metrics.reset()
        return avg_loss, final_metrics
            
    def _validation(self):        
        self.model.eval()
        total_loss = 0.
        val_metrics = self.metrics.clone().to(self.device)
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
                val_metrics.update(y_pred, y)
            avg_loss = total_loss / len(self.val_dataloader)
            final_metrics = val_metrics.compute()
            val_metrics.reset()
            return avg_loss, final_metrics
    
    def fit(self, epochs):
        self.epochs = epochs
        for epoch in range(epochs):
            self._call_callbacks('on_epoch_start', epoch=epoch)
            train_loss, train_metrics = self._one_epoch()
            val_loss, val_metrics = self._validation()
            self.val_loss = val_loss
            # colocar named arguments
            self._call_callbacks('on_epoch_end',
                                 epoch=epoch,
                                 train_loss=train_loss,
                                 val_loss=val_loss,
                                 train_metrics=train_metrics,
                                 val_metrics=val_metrics)
                                #  y_train_pred=y_train_pred,
                                #  y_train=y_train,
                                #  y_val_pred=y_val_pred,
                                #  y_val=y_val)
            if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                break
            if self.scheduler:
                self.scheduler.step(self.val_loss)
        return train_loss, val_loss
                
    def _call_callbacks(self, cb_name,*args, **kwargs):
        for callback in self.callbacks:
            # if hasattr(callback,'should_stop') and callback.should_stop:
            #     break
            method = getattr(callback, cb_name, None)
            if callable(method):
                method(self,*args, **kwargs)
                
                
### Procurar otimizacao de memoria para treino do fastai