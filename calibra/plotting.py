
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from typing import Union, Dict, Any, Tuple, List 

from utils import bin_probabilities, get_classwise_bin_weights


class CalibrationCurve:
    """
    Class to generate a calibration curve (AKA reliability curve) given a list of predictions and corresponding true labels.

    Uses matplotlib to generate plots. User can specify any matplotlib kwargs and interact directly with the matplotlib figure
    object to customise calibration curve graph to their desire.

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

    Attributes:
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
        bins (Dict[int, Dict[int, Dict[str, Union[list, int]]]]):
            Dictionary containing, for each class, a dictionary for each bin, itself a dictionary containing the predicted probabilities and number of occurrences of the given class.
    """
    def __init__(self, y_pred: np.ndarray, y_true: np.ndarray, num_bins: int = 20, method: str = 'width'):
        self.y_pred = y_pred
        self.y_true = y_true
        self.num_bins = num_bins
        self.method = method   
        self.bins = bin_probabilities(self.y_pred, self.y_true, self.num_bins, self.method)     
                
    def _segment_curve_data(self, x: List[float], y: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate continuous segments of the non-empty bins to plot in the calibration curve.

        Args:
            x (list): 
                List containing expected rate of occurrence for each bin. Null values represent empty bins.
            y (list): 
                List containing actual rate of occurrence for each bin. Null values represent empty bins. 

        Returns:
            List[list]
        """
        segments = []
        current_segment = {'x': [], 'y': []}

        for xi, yi in zip(x, y):
            if xi is None or yi is None:
                if current_segment['x'] and current_segment['y']:
                    segments.append(current_segment)
                    current_segment = {'x': [], 'y': []}
            else:
                current_segment['x'].append(xi)
                current_segment['y'].append(yi)

        # Add the last segment if not empty
        if current_segment['x'] and current_segment['y']:
            segments.append(current_segment)

        x = [
            segment['x'] for segment in segments
        ]
        y = [
            segment['y'] for segment in segments
        ]
        
        return x, y
    
    def _generate_x_y(self, class_label: int = 0) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Generates the x and y values for the calibration curve based on provided predictions, true labels, and class.

        Args:
            class_label (int):
                Label of the class whose calibration curve is plotted. Defaults to 0 (the first class). 
        
        Returns:
            Tuple[List[List[float]], List[List[float]]]:
                Lists of lists of x and y values to be plotted. Need to use sublists in case curve is discontinuous (and therefore plotted in individual segments).
        """        
        class_i_bins = self.bins[class_label]
        x, y = [], []
        for bin in class_i_bins.values():
            expected_occurrence_rate = sum(bin['probs']) / len(bin['probs']) if bin['probs'] else None
            actual_occurrence_rate = bin['num_occurrences'] / len(bin['probs']) if bin['probs'] else None
            x.append(expected_occurrence_rate)
            y.append(actual_occurrence_rate)

        if None not in x:
            return [x], [y]
        else:
            return self._segment_curve_data(x, y)

    def plot(self, class_label: int = 0, **kwargs: Dict[str, Any]):
        """
        Generates and plots the calibration curve for the specified class. Accepts matplotlib.pyplot.plot keyword arguments for customisation.

        Args:
            class_label (int):
                Label of the class whose calibration curve is plotted. Defaults to 0 (the first class). 
            **kwargs (Dict[str, Any]): 
                matplotlib keyword arguments for customising the plot.
        """
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label='Perfectly Calibrated')
        
        x_segments, y_segments = self._generate_x_y(class_label)
        for x, y in zip(x_segments, y_segments):
            ax.plot(x, y, **kwargs)

        return fig, ax
    