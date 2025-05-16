# import torch

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_func, device, callbacks = None):
        self.torch = __import__("torch")
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.callbacks = callbacks or []
    
    def _one_epoch(self):
        self.model.train()
        total_loss = 0.
        for batch in self.train_dataloader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_func(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(self.train_dataloader)
            
    def _validation(self):        
        self.model.eval()
        total_loss = 0.
        with self.torch.no_grad():
            for batch in self.val_dataloader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_func(y_pred, y)
                total_loss += loss.item()
            return total_loss / len(self.val_dataloader)
    
    def fit(self, epochs):
        self.epochs = epochs
        for epoch in range(epochs):
            self._call_callbacks('on_epoch_start', epoch)
            train_loss = self._one_epoch()
            val_loss = self._validation()
            self.val_loss = val_loss
            self._call_callbacks('on_epoch_end', epoch, train_loss, val_loss)
        return train_loss, val_loss
                
    def _call_callbacks(self, cb_name, *args):
        for callback in self.callbacks:
            if hasattr(callback,'should_stop') and callback.should_stop:
                break
            method = getattr(callback, cb_name, None)
            if callable(method):
                method(self, *args)