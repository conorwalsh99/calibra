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
    y_pred = _reshape_y_pred(y_pred=y_pred)
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


def _reshape_y_pred(y_pred: np.ndarray) -> np.ndarray:
    """
    If y_pred is a 1D array, reshape to (num_samples, num_classes)

    Args:
        y_pred (ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
            Alternatively may be of shape (num_samples,) where i position is predicted probability of data point i belonging to class 1 (positive class).

    Returns:
        ndarray
    """
    if np.asarray(y_pred).ndim == 1:
        y_pred = np.asarray(
            [
            [1 - prob, prob] for prob in y_pred
            ]
        )

    return y_pred


def _get_bin_weight(bin: dict, num_samples: int) -> float:
    """
    Calculate the proportion of the overall dataset whose predictions lie in the given bin.

    Args:
        bin (dict):
            Dictionary containing the predicted probabilities for a given class and the number of occurrences of that class, for a given bin.
        num_samples (int):
            Number of data points.

    Returns:
        float            
    """
    probs = bin['probs']
    num_trials = len(probs)
    weight = num_trials / num_samples
    return weight


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
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and occurences of the given class.
    
    Returns:
        dict
    """
    for i in range(num_classes):
        y_pred_class_i = y_pred[i].to_list()
        y_true_class_i = list(y_true == i) 
        bin_index = list(
            map(
                lambda x: (num_bins-1) if x==1 else math.floor(num_bins * x), 
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
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and occurences of the given class.
    
    Returns:
        dict
    """    
    lower_bound_freq = math.floor(num_samples / num_bins)
    upper_bound_freq = math.ceil(num_samples / num_bins)

    for i in range(num_classes):
        y_pred_class_i = y_pred[i].to_list()
        y_true_class_i = list(y_true == i)
        y_pred_class_i_sorted, y_true_class_i_sorted = sort_predictions(y_pred_class_i, y_true_class_i)
        # forward-fill bins up to upper bound 
        bins = forward_fill_equal_frequency_bins(class_label=i, 
                                                num_bins=num_bins, 
                                                bins=bins, 
                                                upper_bound_freq=upper_bound_freq, 
                                                y_pred_class_i_sorted=y_pred_class_i_sorted, 
                                                y_true_class_i_sorted=y_true_class_i_sorted)     
        # back-fill bins above lower bound
        bins = back_fill_equal_frequency_bins(class_label=i,
                                            num_bins=num_bins,
                                            bins=bins,
                                            lower_bound_freq=lower_bound_freq)
        # calculate num_occurrences once bins filled satisfactorily
        bins = sum_occurrences(class_label=i, num_bins=num_bins, bins=bins)
    
    return bins


def forward_fill_equal_frequency_bins(class_label: str, num_bins: int, bins: dict, upper_bound_freq: int, y_pred_class_i_sorted: list, y_true_class_i_sorted: list) -> dict:
    """
    For each bin in the given class, assign predictions until the upper limit of points per bin is reached.    

    Args:
        class_label (str):
            The class under consideration.
        num_bins (int):
            Number of bins the interval [0, 1] is divided into.
        bins (dict):
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and occurences of the given class.
        upper_bound_freq (int):
            Maximum number of points a given bin may contain.
        y_pred_class_i_sorted (list):
            List of predicted probabilities (in ascending order) for the given class.
        y_true_class_i_sorted (list):
            List of true labels for the given class, sorted according to the predicted probabilities (in ascending order).

    Returns:
        dict
    """
    i = class_label
    for b in range(num_bins):
        bins[i][b]['occurrences'] = [] # need to track the occurrence associated with each prediction because these may be redistributed to different bins later
        while len(bins[i][b]['probs']) < upper_bound_freq and len(y_pred_class_i_sorted) > 0:
            bins[i][b]['probs'].append(y_pred_class_i_sorted.pop(0))
            bins[i][b]['occurrences'].append(y_true_class_i_sorted.pop(0))
    return bins

def back_fill_equal_frequency_bins(class_label: str, num_bins: int, bins: dict, lower_bound_freq: int) -> dict:
    """
    Redistribute predictions between bins so that each bin contains at least a specified minimum number of samples.

    For a given class, iterate backwards through the bins, continuously reassigning the final sample in the preceding bin to the given bin, 
    until the given bin contains at least a specified minimum number of points. If a bin already has at least this number of minimum points upon first inspection, 
    exit the loop.

    Args:
        class_label (str):
            The class under consideration.
        num_bins (int):
            Number of bins the interval [0, 1] is divided into.
        bins (dict):
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and occurences of the given class.
        lower_bound_freq (int):
            Minimum number of points a given bin must contain.
        y_pred_class_i_sorted (list):
            List of predicted probabilities (in ascending order) for the given class.
        y_true_class_i_sorted (list):
            List of true labels for the given class, sorted according to the predicted probabilities (in ascending order).

    Returns:
        dict
    """
    i = class_label
    for b in range(num_bins-1, 0, -1):
        if len(bins[i][b]['probs']) >= lower_bound_freq:
            return bins
        while len(bins[i][b]['probs']) < lower_bound_freq:
            bins[i][b]['probs'].insert(0, bins[i][b-1]['probs'].pop(-1))
            bins[i][b]['occurrences'].insert(0, bins[i][b-1]['occurrences'].pop(-1))
    return bins

def sum_occurrences(class_label: str, num_bins: int, bins: dict) -> dict:
    """
    For a given class, loop through the bins and find the number of occurrences of the given class in each bin.

    Once this sum of occurrences is found, delete the (now irrelevant) list tracking these occurrences.

    Args:
        class_label (str):
            The class under consideration.
        num_bins (int):
            Number of bins the interval [0, 1] is divided into.
        bins (dict):
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and occurences of the given class.    
    """
    i = class_label
    for b in range(num_bins):
        bins[i][b]['num_occurrences'] = sum(bins[i][b]['occurrences'])
        del bins[i][b]['occurrences']    
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
