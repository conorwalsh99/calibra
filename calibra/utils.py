import math
import pandas as pd
import numpy as np


def bin_probabilities(y_pred: np.ndarray, y_true: np.ndarray, num_bins: int, method: str = 'width') -> dict:
    """
    Group predictions into bins, along with corresponding true labels.
    
    Args:
        y_pred (ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
            Alternatively may be of shape (num_samples,) where i position is predicted probability of data point i belonging to class 1 (positive class).
        y_true (ndarray):
            This 1-D array of length num_samples contains the true label for each data point.        
        num_bins (int):
            Number of bins the interval [0, 1] is divided into. Exact if method='width', approximate if method='frequency'.
        method (str):
            Method of splitting interval [0, 1] into bins. If set to 'width' divides the interval [0, 1] into num_bins bins of equal width.
            If set to 'frequency' divides the interval [0, 1] into approximately num_bins bins, each containing approximately num_samples/num_bins data points. 
            Defaults to 'width'.      

    Returns:
        dict
    """
    if np.asarray(y_pred).ndim == 1:
        y_pred = np.asarray(
            [
            [1 - prob, prob] for prob in y_pred
            ]
        )
    num_samples, num_classes = y_pred.shape

    y_pred = pd.DataFrame(data=y_pred, columns=[i for i in range(num_classes)]) 
    
    bins = {
        i: {
            j: {
                'probs': [],
                'num_occurrences': 0,
                }
                for j in range(num_bins)
            } 
            for i in range(num_classes)
        }

    if method == 'width':
        bins = get_equal_width_bins(y_pred, y_true, num_classes, num_samples, num_bins, bins)
    elif method == 'frequency':
        bins = get_equal_frequency_bins(y_pred, y_true, num_classes, num_samples, num_bins, bins)
    else:
        raise ValueError("Method must be 'width' or 'frequency'")

    return bins


def get_equal_width_bins(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int, num_samples: int, num_bins: int, bins: dict) -> dict:
    """
    Group predictions into bins of equal width.

    Args:
        y_pred (ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
        y_true (ndarray):
            This 1-D array of length num_samples contains the true label for each data point.
        num_classes (int):
            Number of classes.        
        num_samples (int):
            Number of data points.                    
        num_bins (int):
            Number of equal-width bins the interval [0, 1] is divided into.
        bins (dict):
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and number of occurences of the given class.
    
    Returns:
        dict
    """
    for i in range(num_classes):
        y_pred_class_i = y_pred[i].to_list()
        y_true_class_i = list(y_true == i) 
        bin_index = list(
            map(
                lambda x: (num_bins-1) if x==1 else math.floor(
                    (100 * x) / (100 / num_bins)
                    ), 
                    y_pred_class_i
                )
                ) 
        # returns num_bins-1 if p==1, else returns m for p in [(m-1)/num_bins, m/num_bins] 
        for j in range(num_samples): 
            bins[i][bin_index[j]]['probs'].append(y_pred_class_i[j]) # group predicted probabilities into the bins
            bins[i][bin_index[j]]['num_occurrences'] += y_true_class_i[j] # keep track of the number of occurrences of class i in each bin
            
    return bins


def get_equal_frequency_bins(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int, num_samples: int, num_bins: int, bins: dict) -> dict:
    """
    Group predictions into bins containing equal numbers of data points.

    Args:
        y_pred (ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
        y_true (ndarray):
            This 1-D array of length num_samples contains the true label for each data point.
        num_classes (int):
            Number of classes.        
        num_samples (int):
            Number of data points.                    
        num_bins (int):
            Approximate number of bins the interval [0, 1] is divided into, each containing approximately the same number of data points.
        bins (dict):
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and number of occurences of the given class.
    
    Returns:
        dict
    """    
    points_per_bin = math.ceil(num_samples / num_bins)

    for i in range(num_classes):
        y_pred_class_i = y_pred[i].to_list()
        y_true_class_i = list(y_true == i)
        y_pred_class_i_sorted, y_true_class_i_sorted = sort_predictions(y_pred_class_i, y_true_class_i)
        for b in range(num_bins):
            probs = []
            num_occurrences = 0
            while len(probs) < points_per_bin:
                probs.append(y_pred_class_i_sorted.pop(0)) # moving first item of sorted list to end of probs. Check this does what you expect it to.
                num_occurrences += y_true_class_i_sorted.pop(0)
            bins[i][b]['probs'] = probs
            bins[i][b]['num_occurrences'] = num_occurrences

    return bins


def sort_predictions(y_pred_class_i: list, y_true_class_i: list) -> tuple:
    """
    Sort the predicted probabilities in ascending order. Sort the true labels so they correspond to the sorted predictions.

    Args:
        y_pred_class_i (ndarray):
            Array-like object of shape (num_samples,) where i position is predicted probability of data point i belonging to given class.
        y_true_class_i (ndarray):
            1-D array of length num_samples, where i position is 1 if data point i belongs to given class, 0 otherwise.
    
    Returns:
        tuple
    """
    zipped_list = list(zip(y_pred_class_i, y_true_class_i))
    sorted_zipped_list = sorted(zipped_list)
    y_pred_sorted, y_true_sorted = zip(*sorted_zipped_list)

    return list(y_pred_sorted), list(y_true_sorted)
