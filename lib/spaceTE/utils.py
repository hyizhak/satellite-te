import sys
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def weight_initialization(module):
    """Initialize weights in nn module"""

    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


def uni_rand(low=-1, high=1):
    """Uniform random variable [low, high)"""
    return (high - low) * np.random.rand() + low


def print_(*args, file=None):
    """print out *args to file"""
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()

def smoothing(y, window_size=100):
    ns = pd.Series(y)

    # Apply rolling window smoothing with window size 100
    windows = ns.rolling(window=window_size)
    moving_averages = windows.mean()

    # Convert the result to a list and remove the NaN values
    ys = moving_averages.tolist()
    ys = ys[window_size-1:]  # Removing the first NaN values

    return ys
