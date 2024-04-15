import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, Any, Tuple, List
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap

from utils import bin_probabilities, get_classwise_bin_weights, _reshape_y_pred


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
            If set to 'frequency' divides the interval [0, 1] into approximately num_bins bins, each containing approximately (num_samples/num_bins) data points.
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
        num_samples (int):
            Number of data points.
        num_classes (int):
            Number of classes.
        bins (Dict[int, Dict[int, Dict[str, Union[list, int]]]]):
            Dictionary containing, for each class, a dictionary for each bin, itself a dictionary containing the predicted probabilities and number of occurrences of the given class.
        classwise_bin_weights (np.ndarray):
            Numpy array of shape (num_classes, num_bins) whose ijth element represents the proportion of the overall dataset whose predictions for class i lie in bin j.
        bin_boundaries (list):
            Ordered list of length num_bins + 1 containing start and end point of each bin on interval [0, 1].
        LIGHT_BLUE (np.ndarray):
            Normalised rgb value for light blue. Used as lightest shade of blue for density-based color mapping (i.e. for bins with extremely low density)
        DARK_BLUE (np.ndarray):
            Normalised rgb value for dark blue. Used as darkest shade of blue for density-based color mapping (i.e. for bins with extremely high density)
        COLORMAP (LinearSegmentedColormap):
            Color map which determines shade of blue bin segments are plotted with (depending on weight of bin).
        NORM (Normalize):
            Object that linearly normalises weights of bins for density-based color mapping. Here, mapping is set to [0.001, 1] -> [0, 1].
    """

    LIGHT_BLUE = np.array([173, 216, 230]) / 255.0
    DARK_BLUE = np.array([4, 2, 115]) / 255.0
    COLORMAP = LinearSegmentedColormap.from_list(
        "light_to_dark_blue", [LIGHT_BLUE, DARK_BLUE]
    )
    NORM = Normalize(vmin=0.001, vmax=1)

    def __init__(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        num_bins: int = 20,
        method: str = "width",
    ):
        self.y_pred = _reshape_y_pred(y_pred)
        self.y_true = y_true
        self.num_bins = num_bins
        self.method = method
        self.num_samples, self.num_classes = self.y_pred.shape
        self.bins = bin_probabilities(
            self.y_pred, self.y_true, self.num_bins, self.method
        )
        self.classwise_bin_weights = get_classwise_bin_weights(
            self.bins, self.num_bins, self.num_samples, self.num_classes
        )
        self.bin_boundaries = list(np.linspace(0, 1, self.num_bins + 1))

    def _segment_curve_data(
        self, x: List[float], y: List[float], weights: np.ndarray
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """Generate continuous segments of the non-empty bins to plot in the calibration curve.

        Args:
            x (list):
                List containing expected rate of occurrence for each bin. Null values represent empty bins.
            y (list):
                List containing actual rate of occurrence for each bin. Null values represent empty bins.
            weights (list):
                List containing proportion of predictions residing in each bin.

        Returns:
            Tuple[List[List[float]], List[List[float]], List[List[float]]]
        """
        segments = []
        current_segment = {"x": [], "y": [], "weight": []}

        for xi, yi, weight in zip(x, y, weights):
            if xi is None or yi is None:
                if current_segment["x"] and current_segment["y"]:
                    segments.append(current_segment)
                    current_segment = {"x": [], "y": [], "weight": []}
            else:
                current_segment["x"].append(xi)
                current_segment["y"].append(yi)
                current_segment["weight"].append(weight)

        # Add the last segment if not empty
        if current_segment["x"] and current_segment["y"]:
            segments.append(current_segment)

        x = [segment["x"] for segment in segments]
        y = [segment["y"] for segment in segments]
        weights = [segment["weight"] for segment in segments]

        return x, y, weights

    def _generate_x_y(
        self, class_label: int = 0
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """
        Generate x and y values to be plotted as calibration curve based on provided predictions, true labels, and class.
        Also return weights of each bin, in case calibration curve is plotted with density-based color mapping.

        Args:
            class_label (int):
                Label of the class whose calibration curve is plotted. Defaults to 0 (the first class).

        Returns:
            Tuple[List[List[float]], List[List[float]], List[List[float]]]:
                Lists of lists of x and y values to be plotted along with corresponding weights for each bin. Need to use sublists in case curve is discontinuous (and therefore plotted in individual segments).
        """
        class_i_bins = self.bins[class_label]
        x, y = [], []
        for bin in class_i_bins.values():
            expected_occurrence_rate = (
                sum(bin["probs"]) / len(bin["probs"]) if bin["probs"] else None
            )
            actual_occurrence_rate = (
                bin["num_occurrences"] / len(bin["probs"]) if bin["probs"] else None
            )
            x.append(expected_occurrence_rate)
            y.append(actual_occurrence_rate)

        weights = self.classwise_bin_weights[class_label]
        if None not in x:
            return [x], [y], [weights]
        else:
            return self._segment_curve_data(x, y, weights)

    def _add_bin_boundaries_to_x(
        self, x: List[float], weights: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        For x segment to be plotted, (where each value represents the expected frequency or 'centre' of a given bin), get x value at end boundary of each bin and add to list.
        Consecutive pairs of values in updated x list now represent half of a bin (i.e. start to centre, or centre to end).
        Get new array of weights so that values correspond to new x values list i.e. each full bin in x should have its weight entered twice (consecutively) in weights, for half bins.

        PROBLEM:
            When show_density == False we plot the curve using the bin 'centres' (expected frequencies) as the x values, and the actual frequencies as the y values.
            When show_density == True, we want to plot this same curve, but color each bin segment according to its weight. If we plotted this using the same x and y values as before,
            each segment would go from the centre of one bin to the centre of the next. Coloring the segments by the corresponding bin weight, the first half of each bin
            would be colored by the weight of the previous bin, and only the second half would be correctly colored according to the weight of the bin.

        To ensure each bin segment is colored correctly, we need to plot two segments for each bin - from the start to the centre, and from the centre to the end.
        We also need to ensure the bin's weight is associated with both segments.

        Therefore, we will create a new array of x values (x_new) containing values for the start, centre and end of each bin, and a new array  of the weights (w_new) containing
        two weight entries for each full bin.

        The only exceptions to this are the first bin, for which we only need to include its centre and endpoint in the x_new, (as the original calibration curve plot begins from this point)
        and the final bin, for which we only need to include the start and centre (as the original calibration curve plot finishes at its centre).
        Similarly, the weights of the first and final bins only need to be entered in w_new once.

        The only other exception is for whose 'centre' (expected frequency) coincides with their start boundary. In this case we only include this point once
        in x_new, and similarly, only include its weight once in the w_new (as we can simply plot the whole bin in one segment, start to finish).

        pseduo-code:

        -> if only one point, return as is

        -> else:
            -> add centre, weight of first bin to x_new, w_new
            -> add end boundary of first bin to x_new
            -> for remaining bins up to but excluding final bin:
                -> check if centre already in x_new (if centre coincides with start boundary - end boundary of previous bin - this will be the case)
                    -> if not, add centre, weight to x_new, w_new
                -> add end boundary, weight to x_new, w_new
            -> add centre of final bin to x_new

        Args:
            x (List[float]):
                List of x values for given segment to be plotted as part of the calibration curve.
                Each value represents the 'centre' of a bin (expected frequency) along the calibration curve.
            weights (List[float]):
                List of weights corresponding to x values. Each value represents proportion of predictions residing in a given each bin.

        Returns:
            Tuple[List[float], List[float]]
        """
        if len(x) == 1:
            return x, weights

        x_new, w_new = [x[0]], [weights[0]]
        first_bin_end_boundary = list(
            filter(lambda boundary: boundary > x[0], self.bin_boundaries)
        )[0]
        x_new.append(first_bin_end_boundary)

        for bin_expected_freq, bin_weight in zip(x[1:-1], weights[1:-1]):
            if bin_expected_freq not in x_new:
                x_new.append(bin_expected_freq)
                w_new.append(bin_weight)
            bin_end_boundary = list(
                filter(
                    lambda boundary: boundary > bin_expected_freq, self.bin_boundaries
                )
            )[0]
            x_new.append(bin_end_boundary)
            w_new.append(bin_weight)
        x_new.append(x[-1])

        return x_new, w_new

    def _get_y_at_bin_centre(
        self, x: List[float], x_new: List[float], y: List[float], i: int
    ) -> float:
        """Get actual frequency for bin in question (y value at this bin's 'centre') for plotting purposes.

        Args:
            x (List[float]): List of expected frequencies of each bin ('centre' of bin) along given segment of the calibration curve.
            x_new (List[float]): Ordered list containing expected frequencies and boundary values of each bin along given segment of the calibration curve.
            y (List[float]): List of actual frequencies of each bin along given segment of the calibration curve.
            i (int): position in iteration through segment of the calibration curve.

        Returns:
            float: Actual frequency of given bin.
        """
        bin_expected_freq = x_new[i]
        bin_expected_freq_original_index = x.index(bin_expected_freq)
        return y[bin_expected_freq_original_index]

    def _get_y_at_bin_boundary(
        self, x: List[float], x_new: List[float], y: List[float], i: int
    ) -> float:
        """Get interpolated frequency at end boundary of bin in question.

        Get x = [current bin expected frequency, next bin expected frequency] and y = [current bin actual frequency, next bin actual frequency]
        and interpolate for y value at x = current bin boundary.

        If iterable i corresponds to end of list, return (as final bin is end of the curve we do not interpolate beyond this point).

        Args:
            x (List[float]): List of expected frequencies of each bin ('centre' of bin) along given segment of the calibration curve.
            x_new (List[float]): Ordered list containing expected frequencies and boundary values of each bin along given segment of the calibration curve.
            y (List[float]): List of actual frequencies of each bin along given segment of the calibration curve.
            i (int): position in iteration through segment of the calibration curve.

        Returns:
            float: Interpolated frequency at boundary of given bin.
        """
        if i < len(x_new) - 1:
            bin_expected_freq = x_new[i - 1]
            next_bin_expected_freq = x_new[i + 1]
            bin_expected_freq_original_index = x.index(bin_expected_freq)
            next_bin_expected_freq_original_index = x.index(next_bin_expected_freq)
            bin_actual_freq = y[bin_expected_freq_original_index]
            next_bin_actual_freq = y[next_bin_expected_freq_original_index]
            bin_boundary = x_new[i]
            return np.interp(
                bin_boundary,
                [bin_expected_freq, next_bin_expected_freq],
                [bin_actual_freq, next_bin_actual_freq],
            )
        return

    def _generate_x_y_with_bin_boundaries(
        self, x: List[float], y: List[float], weights: List[float]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate new x, y, and weights arrays where x values include expected frequencies at bin centres and boundaries, y values include actual frequencies
        for these points (interpolating to get values at bin boundaries) and weights correspond to bins as before.

        Args:
            x (List[float]): List of expected frequencies of each bin ('centre' of bin) along given segment of the calibration curve.
            y (List[float]): List of actual frequencies of each bin along given segment of the calibration curve.
            weights (List[float]): List of weights corresponding to each bin along given segment of the calibration curve.

        Returns:
            Tuple[List[List[float]], List[List[float]]]: Updated x, y, weights lists that include boundary points of each bin along given segment of the calibration curve.
        """
        x_new, weights_new = self._add_bin_boundaries_to_x(x, weights)
        y_new = []

        for i in range(len(x_new)):
            if x_new[i] in x:
                bin_actual_freq = self._get_y_at_bin_centre(x, x_new, y, i)
                y_new.append(bin_actual_freq)
            else:
                bin_boundary_interpolated_freq = self._get_y_at_bin_boundary(
                    x, x_new, y, i
                )
                y_new.append(bin_boundary_interpolated_freq)

        return x_new, y_new, weights_new

    def _get_density_based_line_collection(
        self, x: List[float], y: List[float], weights: List[float]
    ) -> LineCollection:
        """Get collections of lines to plot calibration curve. Each line in the collection represents either half a bin, or a full bin
        along the calibration curve. The color of each line represents the density (weight) of the given bin.

        Args:
            x (List[float]): List of expected frequencies of each bin ('centre' of bin) along given segment of the calibration curve.
            y (List[float]): List of actual frequencies of each bin along given segment of the calibration curve.
            weights (List[float]): List of weights corresponding to each bin along given segment of the calibration curve.

        Returns:
            LineCollection:
                Collection of lines where each line represents a segment of the calibration curve.
                Each segment is either half a bin or a full bin and the color is determined by the weight of the given bin.
        """
        adjusted_x, adjusted_y, adjusted_weights = (
            self._generate_x_y_with_bin_boundaries(x, y, weights)
        )
        segments = np.array(
            [
                [[adjusted_x[i], adjusted_y[i]], [adjusted_x[i + 1], adjusted_y[i + 1]]]
                for i in range(len(adjusted_x) - 1)
            ]
        )
        return LineCollection(
            segments,
            cmap=CalibrationCurve.COLORMAP,
            norm=CalibrationCurve.NORM,
            array=adjusted_weights,
            linewidth=2,
        )

    def plot(
        self, class_label: int = 0, show_density: bool = False, **kwargs: Dict[str, Any]
    ) -> Tuple[Figure, Axes]:
        """
        Generate and plot the calibration curve for the specified class. Accepts matplotlib.pyplot.plot keyword arguments for customisation.

        Args:
            class_label (int):
                Label of the class whose calibration curve is plotted. Defaults to 0 (the first class).
            show_density (bool):
                If True, the color strength of the curve is a function of the bin density at each point.
                Defaults to False.
            **kwargs (Dict[str, Any]):
                matplotlib keyword arguments for customising the plot.
        """
        if self.method == "frequency" and show_density:
            show_density = False
            warnings.warn(
                """Density-based color mapping not available when method=='frequency', as bins do not have well-defined boundaries.  
                          In any case, equal frequency bins have equal density, by definition. Setting show_density='False'."""
            )

        fig, ax = plt.subplots()
        ax.plot(
            [0, 1], [0, 1], color="black", linestyle="--", label="Perfectly Calibrated"
        )
        x_segments, y_segments, weight_segments = self._generate_x_y(class_label)

        for x, y, weights in zip(x_segments, y_segments, weight_segments):
            if show_density and len(x) > 1:
                lc = self._get_density_based_line_collection(x, y, weights)
                ax.add_collection(lc)
            elif show_density and len(x) == 1:
                ax.plot(
                    x,
                    y,
                    "o",
                    color=CalibrationCurve.COLORMAP(CalibrationCurve.NORM(weights[0])),
                )
            else:
                ax.plot(x, y, **kwargs)

        return fig, ax
