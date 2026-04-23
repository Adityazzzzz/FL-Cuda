import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    """Random initialization of the model's parameters"""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, counter_info=True, is_save=True, early=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter_info = counter_info
        self.best_val_acc = 0
        self.best_epoch = 0
        self.is_save = is_save
        self.early = early

    def __call__(self, val_loss, model, val_acc, epoch, global_epochs):
        if self.early:
            score = -val_loss
            if self.best_score is None:
                self._save_best(score, val_loss, val_acc, epoch, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter_info:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self._save_best(score, val_loss, val_acc, epoch, model)
                self.counter = 0
        else: 
            # If early stopping is off, just save the final epoch's model
            if epoch == global_epochs - 1:
                self.best_epoch = epoch
                self.best_val_acc = val_acc
                if self.is_save:
                    torch.save(model.state_dict(), self.path)

    def _save_best(self, score, val_loss, val_acc, epoch, model):
        """Helper function to save the model when validation loss decreases."""
        self.best_score = score
        self.best_val_acc = val_acc
        self.best_epoch = epoch
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss
        if self.is_save:
            torch.save(model.state_dict(), self.path)