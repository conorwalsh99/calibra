import pandas as pd 
import numpy as np 
from typing import Union

from calibra.utils import bin_probabilities, _reshape_y_pred, _get_bin_weight, validate_input


@validate_input
def classwise_ece(
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        num_bins: int = 20, 
        method: str = 'width', 
        return_classwise_errors: bool = False
        ) -> Union[float, tuple]:
    """ 
    Calculate the class-wise Expected Calibration Error (ECE) of a set of predictions.

    Args:
        y_pred (ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
            Alternatively may be of shape (num_samples,) for binary classification where i position is predicted probability of data point i belonging to class 1 (positive class).
        y_true (ndarray):
            This 1-D array of length num_samples contains the true label for each data point.        
        num_bins (int):
            Number of bins the interval [0, 1] is divided into. Exact if method='width', approximate if method='frequency'.
        method (str):
            Method of splitting interval [0, 1] into bins. If set to 'width' divides the interval [0, 1] into num_bins bins of equal width.
            If set to 'frequency' divides the interval [0, 1] into approximately num_bins bins, each containing approximately num_samples/num_bins data points. 
            Defaults to 'width'.      
        return_classwise_errors (bool):
            If True returns a numpy array of shape (num_classes,) containing the contribution of each class to the class-wise expected calibration error. Defaults to False.

    Returns:    
        float:
            Value of the class-wise expected calibration error. This is an element of [0,1].

        tuple:
            Tuple containing a float and a numpy array.
                float:
                    Value of the class-wise expected calibration error. This is an element of [0,1].
                np.ndarray:
                    Numpy array of shape (num_classes) whose ith element represents the contribution of class i to the class-wise expected calibration error. 
    """
    y_pred = _reshape_y_pred(y_pred)
    num_samples, num_classes = y_pred.shape

    bins = bin_probabilities(y_pred, y_true, num_bins, method)     
    classwise_errors = _get_classwise_errors(bins, num_bins, num_samples, num_classes)
    overall_error = classwise_errors.mean()

    return (overall_error, classwise_errors) if return_classwise_errors else overall_error


def _get_classwise_errors(bins: dict, num_bins: int, num_samples: int, num_classes: int) -> np.ndarray:
    """
    Calculate the error for each class.

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
            Numpy array of shape (num_classes) whose ith element represents the contribution of class i to the class-wise expected calibration error.                      
    """
    errors = np.zeros(num_classes)

    for i in range(num_classes):
        for b in range(num_bins):
            bin_weight = _get_bin_weight(bins[i][b], num_samples)
            if bin_weight > 0:      
                bin_deviation = _calculate_bin_deviation(bins[i][b])
                weighted_bin_deviation = bin_weight * bin_deviation
                errors[i] += weighted_bin_deviation 
    
    return errors

      
def _calculate_bin_deviation(bin: dict) -> float:
    """
    Calculate the deviation between the expected rate of occurrence and the actual rate of occurrence, for a given bin.

    Args:
        bin (dict):
            Dictionary containing the predicted probabilities for a given class and the number of occurrences of that class, for a given bin.

    Returns:
        float 
    """
    probs, num_occurrences = bin['probs'], bin['num_occurrences']
    num_trials = len(probs)
    expected_num_occurrences = sum(probs)
    actual_num_occurrences = num_occurrences
    expected_occurrence_rate = expected_num_occurrences / num_trials
    actual_occurrence_rate = actual_num_occurrences / num_trials
    deviation = abs(expected_occurrence_rate - actual_occurrence_rate)
    return deviation
