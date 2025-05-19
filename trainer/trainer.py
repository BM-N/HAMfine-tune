import torch

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_func, device, scheduler=None, metrics = None, callbacks = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.callbacks = callbacks or []
        self.metrics = metrics or {}
    
    def _one_epoch(self):
        self.model.train()
        total_loss = 0.
        y_epoch = []
        y_epoch_pred = []
        for batch in self.train_dataloader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_func(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            y_epoch += [y.cpu()]
            y_epoch_pred += [y_pred.cpu()]
        y_epoch = torch.cat(y_epoch)
        y_epoch_pred = torch.cat(y_epoch_pred)  
        avg_loss = total_loss / len(self.train_dataloader)
        # torch.cuda.empty_cache()
        return avg_loss, y_epoch, y_epoch_pred
            
    def _validation(self):        
        self.model.eval()
        total_loss = 0.
        y_epoch = []
        y_epoch_pred = []
        with torch.no_grad():
            for batch in self.val_dataloader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_func(y_pred, y)
                total_loss += loss.item()
                y_epoch += [y.cpu()]
                y_epoch_pred += [y_pred.cpu()]
            y_epoch = torch.cat(y_epoch)
            y_epoch_pred = torch.cat(y_epoch_pred)  
            avg_loss = total_loss / len(self.val_dataloader)
            # torch.cuda.empty_cache()
            return avg_loss, y_epoch, y_epoch_pred
    
    def fit(self, epochs):
        self.epochs = epochs
        for epoch in range(epochs):
            self._call_callbacks('on_epoch_start', epoch=epoch)
            train_loss, y_train, y_train_pred = self._one_epoch()
            val_loss, y_val, y_val_pred = self._validation()
            self.val_loss = val_loss
            # colocar named arguments
            self._call_callbacks('on_epoch_end',
                                 epoch=epoch,
                                 train_loss=train_loss,
                                 val_loss=val_loss,
                                 y_train_pred=y_train_pred,
                                 y_train=y_train,
                                 y_val_pred=y_val_pred,
                                 y_val=y_val)
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
                
                
### Procurar otimizacao de memoria para treino do fastapi