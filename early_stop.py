
import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=True, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0

            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.verbose = verbose
        self.early_stop = False
        self.trace_func = trace_func
        self.save_model = False

    def __call__(self, dice_score):

        #score = val_loss
        score = dice_score

        if self.best_score is None:
            self.best_score = score
            self.save_model = True
        elif score < self.best_score:
            self.counter += 1
            self.save_model = False
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_model = True

