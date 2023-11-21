'''
Source: VLADI software
https://github.com/GillesOrban/VLADI
'''
import logging
import json
from datetime import datetime, timedelta
import numpy as np
import torch
from collections import OrderedDict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool()
        else:
            return super(NpEncoder, self).default(obj)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=20, eps_factor=0.01, model_path='./model.pth'):
        """
        Parameters
        ----------
            patience: int
                How many epochs to wait since the last validation loss improvement.
                Default: 20
            eps_factor: float
                Factor to apply to the best loss to define the minimum change in the loss to qualify as an improvement.
                Default: 0.01
            model_path: str
                Path to the directory where the model's weights are to be saved.
                Default: './model.pth'
        """
        self.patience = patience
        self.eps_factor = eps_factor
        self.model_path = model_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_model(val_loss, model)
        elif val_loss > (self.best_loss - (self.eps_factor * self.best_loss)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_model(val_loss, model)
            self.counter = 0

    def save_model(self, val_loss, model):
        """Saves the model if the validation loss has decreased.
        """

        # Save the model if improvement:
        torch.save(model.state_dict(), self.model_path)


def get_time(time_seconds):
    sec = timedelta(seconds=int(time_seconds))
    d = datetime(1, 1, 1) + sec

    return d.day - 1, d.hour, d.minute, d.second


def set_logger(_log, log_path):
    """
    Set the logger to log info in terminal and file `log_path`.

    Args:
        log_path: (string) where to log
    """

    _log.setLevel(logging.INFO)
    if not _log.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s'))
        _log.addHandler(file_handler)


def get_lr(optimizer):
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_state_dict(state_dict, remove_str_size):
    """Load and prepare the state dict containing pre-trained weights

    Parameters
    ----------
        weights_path: str
            Path to the model containing the weights to load
    Returns
    ----------
        new_state_dict: dict
            State dictionary prepared
    """

    if remove_str_size > 0:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[remove_str_size:]
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


