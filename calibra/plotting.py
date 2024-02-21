
import pandas as pd 
import numpy as np 
from typing import Union

from utils import bin_probabilities, reshape_y_pred, _get_bin_weight


def _get_classwise_bin_weights(bins: dict, num_bins: int, num_samples: int, num_classes: int) -> np.ndarray:
    """
    Calculate the 
    Args:
        bins (dict):
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and occurences of the given class.                    
        num_bins (int):
            Number of equal-width bins the interval [0, 1] is divided into.
        num_samples (int):
            Number of data points.
        num_classes (int):
            Number of classes.

    Returns:
        np.ndarray:
            Numpy array of shape (num_classes, num_bins) whose ijth element represents the proportion of the overall dataset whose predictions for class i lie in bin j.    
    """    
    weights = [
        [
            _get_bin_weight(bins[i][b], num_samples) for b in range(num_bins)
        ]
        for i in range(num_classes)
    ]
    
    return np.asarray(weights)
