import math
import pandas as pd
import numpy as np
from typing import Callable, Any
from functools import wraps


def validate_input(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Validate the input parameters to ensure they are in the correct format. Convert to correct format where possible, otherwise, raise error.

    Args:
        func: The method for which we validate the input.

    The decorator performs the following validations:
    1. y_pred must be a 1D or 2D array with values between 0 and 1. It is converted to a NumPy array.
    2. y_true must be a list or 1D array of non-negative whole numbers. It is converted to a NumPy array.
    3. y_true and y_pred must have the same length.
    4. num_bins must be a positive whole number.
    5. method must be one of two specific string values ('width' or 'frequency').

    Raises:
      ValueError: If any of the inputs do not meet the validation criteria.

    Returns:
      The original function's return value, if all inputs are valid.
    """    
    @wraps(func)
    def wrapper(y_pred, y_true, num_bins: int = 20, method: str = 'width', *args, **kwargs):
        
        y_pred = np.asarray(y_pred)
        if y_pred.ndim > 2 or np.any(y_pred < 0) or np.any(y_pred > 1):
            raise ValueError("y_pred must be a 1D or 2D array with values between 0 and 1.")
        
        if isinstance(y_true, list):
            if not all(label % 1 == 0 and label >= 0 for label in y_true):
                raise ValueError("y_true must only contain non-negative whole numbers.")
            y_true = np.array(y_true)

        elif isinstance(y_true, np.ndarray):
            if y_true.ndim != 1 or np.any(y_true % 1 != 0) or np.any(y_true < 0):
                raise ValueError("y_true must be a list or 1D array of non-negative whole numbers.")
        else:
            raise TypeError("y_true must be either a list or a 1D numpy array.")

        if y_true.size != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        
        if num_bins % 1 != 0 or num_bins <= 0:
            raise ValueError("num_bins must be a positive whole number.")
        num_bins = int(num_bins)
        
        if method not in ['width', 'frequency']:
            raise ValueError("Method must be either 'width' or 'frequency'")

        return func(y_pred, y_true, num_bins, method, *args, **kwargs)
    return wrapper


@validate_input
def bin_probabilities(y_pred: np.ndarray, y_true: np.ndarray, num_bins: int = 20, method: str = 'width') -> dict:
    """
    Group predictions into bins, along with corresponding true labels.
    
    Args:
        y_pred (np.ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
            Alternatively may be of shape (num_samples,) for binary classification where i position is predicted probability of data point i belonging to class 1 (positive class).
        y_true (np.ndarray):
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
        bins = _get_equal_width_bins(y_pred, y_true, num_classes, num_samples, num_bins, bins)
    elif method == 'frequency':
        bins = _get_equal_frequency_bins(y_pred, y_true, num_classes, num_samples, num_bins, bins)

    return bins

def _reshape_y_pred(y_pred: np.ndarray) -> np.ndarray:
    """
    If y_pred is a 1D array, reshape to (num_samples, num_classes)

    Args:
        y_pred (np.ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
            Alternatively may be of shape (num_samples,) for binary classification where i position is predicted probability of data point i belonging to class 1 (positive class).

    Returns:
        np.ndarray
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


def _get_equal_width_bins(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int, num_samples: int, num_bins: int, bins: dict) -> dict:
    """
    Group predictions into bins of equal width.

    Args:
        y_pred (np.ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
        y_true (np.ndarray):
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
        # returns num_bins-1 if p==1, else returns m for p in [m/num_bins, (m+1)/num_bins) 
        for j in range(num_samples): 
            bins[i][bin_index[j]]['probs'].append(y_pred_class_i[j]) 
            bins[i][bin_index[j]]['num_occurrences'] += y_true_class_i[j] 
            
    return bins

def _get_equal_frequency_bins(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int, num_samples: int, num_bins: int, bins: dict) -> dict:
    """
    Group predictions into bins containing equal numbers of data points.

    Args:
        y_pred (np.ndarray):
            Array-like object of shape (num_samples, num_classes) where ij position is predicted probability of data point i belonging to class j.
        y_true (np.ndarray):
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
        y_pred_class_i_sorted, y_true_class_i_sorted = _sort_predictions(y_pred_class_i, y_true_class_i)
        # forward-fill bins up to upper bound 
        bins = _forward_fill_equal_frequency_bins(class_label=i, 
                                                num_bins=num_bins, 
                                                bins=bins, 
                                                upper_bound_freq=upper_bound_freq, 
                                                y_pred_class_i_sorted=y_pred_class_i_sorted, 
                                                y_true_class_i_sorted=y_true_class_i_sorted)     
        # back-fill bins above lower bound
        bins = _back_fill_equal_frequency_bins(class_label=i,
                                            num_bins=num_bins,
                                            bins=bins,
                                            lower_bound_freq=lower_bound_freq)
        # calculate num_occurrences once bins filled satisfactorily
        bins = _sum_occurrences(class_label=i, num_bins=num_bins, bins=bins)
    
    return bins


def _forward_fill_equal_frequency_bins(class_label: str, num_bins: int, bins: dict, upper_bound_freq: int, y_pred_class_i_sorted: list, y_true_class_i_sorted: list) -> dict:
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

def _back_fill_equal_frequency_bins(class_label: str, num_bins: int, bins: dict, lower_bound_freq: int) -> dict:
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

def _sum_occurrences(class_label: str, num_bins: int, bins: dict) -> dict:
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


def _sort_predictions(y_pred_class_i: list, y_true_class_i: list) -> tuple:
    """
    Sort the predicted probabilities in ascending order. Sort the true labels so they correspond to the sorted predictions.

    Args:
        y_pred_class_i (np.ndarray):
            Array-like object of shape (num_samples,) where i position is predicted probability of data point i belonging to given class.
        y_true_class_i (np.ndarray):
            1-D array of length num_samples, where i position is 1 if data point i belongs to given class, 0 otherwise.
    
    Returns:
        tuple
    """
    zipped_list = list(zip(y_pred_class_i, y_true_class_i))
    sorted_zipped_list = sorted(zipped_list)
    y_pred_sorted, y_true_sorted = zip(*sorted_zipped_list)

    return list(y_pred_sorted), list(y_true_sorted)


def validate_bins(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Validate the input parameters to ensure they are in the correct format. Otherwise, raise error.

    Args:
        func: The method for which we validate the input.

    Raises:
      ValueError: If any of the inputs do not meet the validation criteria.

    Returns:
      The original function's return value, if all inputs are valid.
    """    
    @wraps(func)
    def wrapper(bins: dict,  *args, **kwargs):
        assert isinstance(bins, dict) and len(bins) > 0, "'bins' must be a non-empty dict"
        assert all(isinstance(key, int) for key in bins.keys()), "'bins' keys must be integers"

        for class_i_bins in bins.values():            
            assert isinstance(class_i_bins, dict) and len(class_i_bins) > 0, "'bins' first-level values must themselves be non-empty dictionaries."
            assert all(isinstance(key, int) for key in class_i_bins.keys()), "'bins' first-level values must themselves be non-empty dictionaries with integer keys."

            for class_i_bin_j in class_i_bins.values():            
                assert isinstance(class_i_bin_j, dict) and "probs" in class_i_bin_j, "'bins' second-level values must themselves be dictionaries with keys: {'probs', 'num_occurrences'}"            
                                
        return func(bins, *args, **kwargs)

    return wrapper

@validate_bins
def get_classwise_bin_weights(bins: dict) -> np.ndarray:
    """
    Calculate the weight of each bin across all classes.

    Args:
        bins (dict):
            Dictionary containing, for each class, each bin, itself containing the predicted probabilities and occurences of the given class.                    

    Returns:
        np.ndarray:
            Numpy array of shape (num_classes, num_bins) whose ijth element represents the proportion of the overall dataset whose predictions for class i lie in bin j.    
    """
    num_classes = len(bins)
    num_bins = len(bins[0])
    num_samples = sum(
        [len(bins[0][i]['probs']) for i in range(num_bins)]
    )

    weights = [
        [
            _get_bin_weight(bins[i][b], num_samples) for b in range(num_bins)
        ]
        for i in range(num_classes)
    ]
    
    return np.asarray(weights)
