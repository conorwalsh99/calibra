import pytest
import pandas as pd
import numpy as np


###### test_utils.py ######


@pytest.fixture
def sort_predictions_good_input():
    """
    Good input that should not be troublesome for sort_predictions.
    """
    y_pred_class_i = [0.5, 0.3, 0.6]
    y_true_class_i = [1, 1, 0]
    return y_pred_class_i, y_true_class_i


@pytest.fixture
def expected_sort_predictions_good_input():
    """
    Expected result after sorting of good input.
    """
    y_pred_class_i_sorted = [0.3, 0.5, 0.6]
    y_true_class_i_sorted = [1, 1, 0]
    return y_pred_class_i_sorted, y_true_class_i_sorted


@pytest.fixture
def sort_predictions_duplicate_predictions():
    """
    Input containing multiple predictions with different labels. Could cause trouble depending on sort_predictions algorithm.
    """
    y_pred_class_i = [0.9, 0.4, 0.4, 0.1]
    y_true_class_i = [1, 1, 0, 0]
    return y_pred_class_i, y_true_class_i


@pytest.fixture
def expected_sort_predictions_duplicate_predictions():
    """
    Expected result after sorting of input with duplicate predictions with different labels.
    """
    y_pred_class_i_sorted = [0.1, 0.4, 0.4, 0.9]
    y_true_class_i_sorted_acceptable = [[0, 1, 0, 1], [0, 0, 1, 1]]
    return y_pred_class_i_sorted, y_true_class_i_sorted_acceptable


@pytest.fixture
def get_bins_good_input():
    """
    A set of predictions and corresponding labels that should not cause trouble for either method of binning probabilities.
    """
    y_pred = pd.DataFrame(
        {
            0: [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            1: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        }
    )
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 10, 5
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }

    return y_pred, y_true, num_classes, num_samples, num_bins, bins


@pytest.fixture
def get_bins_y_pred_greater_than_1_input():
    """
    A set of predictions and corresponding labels with bad y_pred that should cause the validate_input decorator to raise an error.
    """
    y_pred = pd.DataFrame(
        {
            0: [0.95, 1.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            1: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        }
    )
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 10, 5
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }

    return y_pred, y_true, num_classes, num_samples, num_bins, bins


@pytest.fixture
def get_bins_y_pred_less_than_0_input():
    """
    A set of predictions and corresponding labels with bad y_pred that should cause the validate_input decorator to raise an error.
    """
    y_pred = pd.DataFrame(
        {
            0: [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            1: [0.05, -0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        }
    )
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 10, 5
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }

    return y_pred, y_true, num_classes, num_samples, num_bins, bins


@pytest.fixture
def get_bins_y_pred_3_dimensional_input():
    """
    A set of predictions and corresponding labels with bad y_pred that should cause the validate_input decorator to raise an error.
    """
    num_classes, num_samples, num_bins = 2, 10, 5
    y_pred = np.zeros((2, num_samples, num_classes))
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }

    return y_pred, y_true, num_classes, num_samples, num_bins, bins


@pytest.fixture
def get_bins_y_true_fractional_input():
    """
    A set of predictions and corresponding labels with bad y_true that should cause the validate_input decorator to raise an error.
    """
    y_pred = pd.DataFrame(
        {
            0: [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            1: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        }
    )
    y_true = np.asarray([0, 0, 0, 0, 0.5, 1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 10, 5
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }

    return y_pred, y_true, num_classes, num_samples, num_bins, bins


@pytest.fixture
def get_bins_negative_y_true_input():
    """
    A set of predictions and corresponding labels with bad y_true that should cause the validate_input decorator to raise an error.
    """
    y_pred = pd.DataFrame(
        {
            0: [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            1: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        }
    )
    y_true = np.asarray([0, 0, 0, 0, 0, -1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 10, 5
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }

    return y_pred, y_true, num_classes, num_samples, num_bins, bins


@pytest.fixture
def expected_get_equal_width_bins_good_input():
    return {
        0: {
            0: {
                "probs": [0.15, 0.05],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.35, 0.25],
                "num_occurrences": 0,
            },
            2: {
                "probs": [0.55, 0.45],
                "num_occurrences": 1,
            },
            3: {
                "probs": [0.75, 0.65],
                "num_occurrences": 2,
            },
            4: {
                "probs": [0.95, 0.85],
                "num_occurrences": 2,
            },
        },
        1: {
            0: {
                "probs": [0.05, 0.15],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.25, 0.35],
                "num_occurrences": 0,
            },
            2: {
                "probs": [0.45, 0.55],
                "num_occurrences": 1,
            },
            3: {
                "probs": [0.65, 0.75],
                "num_occurrences": 2,
            },
            4: {
                "probs": [0.85, 0.95],
                "num_occurrences": 2,
            },
        },
    }


# as get_equal_frequency_bins uses sort_predictions to sort by the size of probability, we see that in the resulting bins, the probabilities are always in ascending order e.g. [0.05, 0.15].
# this is not necessarily true for get_equal_width_bins, where we loop through the predictions as they are presented to us, hence the difference in the resulting bins.


@pytest.fixture
def expected_get_equal_frequency_bins_good_input():
    return {
        0: {
            0: {
                "probs": [0.05, 0.15],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.25, 0.35],
                "num_occurrences": 0,
            },
            2: {
                "probs": [0.45, 0.55],
                "num_occurrences": 1,
            },
            3: {
                "probs": [0.65, 0.75],
                "num_occurrences": 2,
            },
            4: {
                "probs": [0.85, 0.95],
                "num_occurrences": 2,
            },
        },
        1: {
            0: {
                "probs": [0.05, 0.15],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.25, 0.35],
                "num_occurrences": 0,
            },
            2: {
                "probs": [0.45, 0.55],
                "num_occurrences": 1,
            },
            3: {
                "probs": [0.65, 0.75],
                "num_occurrences": 2,
            },
            4: {
                "probs": [0.85, 0.95],
                "num_occurrences": 2,
            },
        },
    }


@pytest.fixture
def bin_probabilities_good_input_width_1d_y_pred():
    y_pred = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = "width"
    return y_pred, y_true, num_bins, method


@pytest.fixture
def bin_probabilities_good_input_width_2d_y_pred():
    y_pred = np.asarray(
        [
            [0.95, 0.05],
            [0.85, 0.15],
            [0.75, 0.25],
            [0.65, 0.35],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.35, 0.65],
            [0.25, 0.75],
            [0.15, 0.85],
            [0.05, 0.95],
        ]
    )
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = "width"
    return y_pred, y_true, num_bins, method


@pytest.fixture
def bin_probabilities_good_input_frequency_1d_y_pred():
    y_pred = np.asarray(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    )
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = "frequency"
    return y_pred, y_true, num_bins, method


@pytest.fixture
def bin_probabilities_good_input_frequency_2d_y_pred():
    y_pred = np.asarray(
        [
            [0.95, 0.05],
            [0.85, 0.15],
            [0.75, 0.25],
            [0.65, 0.35],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.35, 0.65],
            [0.25, 0.75],
            [0.15, 0.85],
            [0.05, 0.95],
        ]
    )
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = "frequency"
    return y_pred, y_true, num_bins, method


@pytest.fixture
def get_equal_frequency_bins_non_integer_freq():
    y_pred = pd.DataFrame(
        {
            0: [0.95, 0.9, 0.85, 0.8, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
            1: [0.05, 0.1, 0.15, 0.2, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        }
    )
    y_true = np.asarray([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 13, 4
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }

    return y_pred, y_true, num_classes, num_samples, num_bins, bins


@pytest.fixture
def expected_get_equal_frequency_bins_non_integer_freq():
    return {
        0: {
            0: {
                "probs": [0.35, 0.4, 0.45, 0.5],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.55, 0.6, 0.65],
                "num_occurrences": 0,
            },
            2: {
                "probs": [0.7, 0.8, 0.8],
                "num_occurrences": 1,
            },
            3: {
                "probs": [0.85, 0.9, 0.95],
                "num_occurrences": 0,
            },
        },
        1: {
            0: {
                "probs": [0.05, 0.1, 0.15, 0.2],
                "num_occurrences": 3,
            },
            1: {
                "probs": [0.2, 0.3, 0.35],
                "num_occurrences": 3,
            },
            2: {
                "probs": [0.4, 0.45, 0.5],
                "num_occurrences": 3,
            },
            3: {
                "probs": [0.55, 0.6, 0.65],
                "num_occurrences": 3,
            },
        },
    }


@pytest.fixture
def _reshape_y_pred_input():
    return [1, 1, 1, 1]


@pytest.fixture
def _reshape_y_pred_expected():
    y_pred = np.ones((4, 2))
    y_pred[:, 0] = 0
    return y_pred


@pytest.fixture
def _get_bin_weight_input():
    bin = {"probs": [0.01, 0.02, 0.03], "num_occurrences": 0}
    num_samples = 100
    return bin, num_samples


@pytest.fixture
def _get_bin_weight_expected():
    return 0.03


@pytest.fixture
def _sum_occurrences_input():
    class_label = 0
    num_bins = 2
    bins = {
        0: {
            0: {
                "probs": [0.01, 0.02],
                "occurrences": [0, 0],
            },
            1: {
                "probs": [0.5, 0.55],
                "occurrences": [1, 1],
            },
        },
        1: {
            0: {
                "probs": [0.45],
                "occurrences": [0],
            },
            1: {
                "probs": [0.5, 0.99, 0.98],
                "occurrences": [0, 1, 1],
            },
        },
    }

    return class_label, num_bins, bins


@pytest.fixture
def _sum_occurrences_expected():
    return {
        0: {
            0: {
                "probs": [0.01, 0.02],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.5, 0.55],
                "num_occurrences": 2,
            },
        },
        1: {
            0: {
                "probs": [0.45],
                "occurrences": [0],
            },
            1: {
                "probs": [0.5, 0.99, 0.98],
                "occurrences": [0, 1, 1],
            },
        },
    }


@pytest.fixture
def _forward_fill_equal_frequency_bins_input():
    num_classes = 2
    class_label = 0
    num_bins = 4
    upper_bound_freq = 4
    bins = {
        i: {
            j: {
                "probs": [],
                "num_occurrences": 0,
            }
            for j in range(num_bins)
        }
        for i in range(num_classes)
    }
    y_pred_class_i_sorted = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_true_class_i_sorted = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    return (
        class_label,
        num_bins,
        bins,
        upper_bound_freq,
        y_pred_class_i_sorted,
        y_true_class_i_sorted,
    )


@pytest.fixture
def _forward_fill_equal_frequency_bins_expected():
    return {
        0: {
            0: {
                "probs": [0, 0, 0, 0],
                "num_occurrences": 0,
                "occurrences": [0, 0, 0, 1],
            },
            1: {
                "probs": [0, 0, 0, 1],
                "num_occurrences": 0,
                "occurrences": [0, 0, 0, 1],
            },
            2: {
                "probs": [1, 1, 1, 1],
                "num_occurrences": 0,
                "occurrences": [1, 1, 0, 1],
            },
            3: {"probs": [1], "num_occurrences": 0, "occurrences": [1]},
        },
        1: {
            0: {"probs": [], "num_occurrences": 0},
            1: {"probs": [], "num_occurrences": 0},
            2: {"probs": [], "num_occurrences": 0},
            3: {"probs": [], "num_occurrences": 0},
        },
    }


@pytest.fixture
def _back_fill_equal_frequency_bins_input():
    """
    For the sake of this unit test, we provide a bins object with just the first class forward-filled.

    In reality, all classes would be forward-filled before back-filling begins.
    """
    num_classes = 2
    class_label = 0
    num_bins = 4
    lower_bound_freq = 3
    bins = {
        0: {
            0: {
                "probs": [0, 0, 0, 0],
                "num_occurrences": 0,
                "occurrences": [0, 0, 0, 1],
            },
            1: {
                "probs": [0, 0, 0, 1],
                "num_occurrences": 0,
                "occurrences": [0, 0, 0, 1],
            },
            2: {
                "probs": [1, 1, 1, 1],
                "num_occurrences": 0,
                "occurrences": [1, 1, 0, 1],
            },
            3: {"probs": [1], "num_occurrences": 0, "occurrences": [1]},
        },
        1: {
            0: {"probs": [], "num_occurrences": 0},
            1: {"probs": [], "num_occurrences": 0},
            2: {"probs": [], "num_occurrences": 0},
            3: {"probs": [], "num_occurrences": 0},
        },
    }
    return class_label, num_bins, bins, lower_bound_freq


@pytest.fixture
def _back_fill_equal_frequency_bins_expected():
    return {
        0: {
            0: {
                "probs": [0, 0, 0, 0],
                "num_occurrences": 0,
                "occurrences": [0, 0, 0, 1],
            },
            1: {"probs": [0, 0, 0], "num_occurrences": 0, "occurrences": [0, 0, 0]},
            2: {"probs": [1, 1, 1], "num_occurrences": 0, "occurrences": [1, 1, 1]},
            3: {"probs": [1, 1, 1], "num_occurrences": 0, "occurrences": [0, 1, 1]},
        },
        1: {
            0: {"probs": [], "num_occurrences": 0},
            1: {"probs": [], "num_occurrences": 0},
            2: {"probs": [], "num_occurrences": 0},
            3: {"probs": [], "num_occurrences": 0},
        },
    }


@pytest.fixture
def get_classwise_bin_weights_input():
    bins = {
        0: {
            0: {
                "probs": [0.05, 0.15],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.25, 0.35],
                "num_occurrences": 0,
            },
            2: {
                "probs": [0.45, 0.55],
                "num_occurrences": 1,
            },
            3: {
                "probs": [0.65, 0.75],
                "num_occurrences": 2,
            },
            4: {
                "probs": [0.85, 0.95],
                "num_occurrences": 2,
            },
        },
        1: {
            0: {
                "probs": [0.05, 0.15],
                "num_occurrences": 0,
            },
            1: {
                "probs": [0.25, 0.35],
                "num_occurrences": 0,
            },
            2: {
                "probs": [0.45, 0.55],
                "num_occurrences": 1,
            },
            3: {
                "probs": [0.65, 0.75],
                "num_occurrences": 2,
            },
            4: {
                "probs": [0.85, 0.95],
                "num_occurrences": 2,
            },
        },
    }
    return bins


@pytest.fixture
def get_classwise_bin_weights_expected():
    return np.asarray([[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]])


@pytest.fixture
def get_classwise_bin_weights_bad_input():
    return {}


###### test_errors.py ######


@pytest.fixture
def classwise_ece_perfectly_calibrated_input():
    """
    Input that should result in perfect calibration.
    """
    y_pred = np.asarray([0, 0, 1])
    y_true = np.asarray([0, 0, 1])
    return y_pred, y_true


@pytest.fixture
def classwise_ece_perfectly_uncalibrated_input():
    """
    Input that should result in perfect calibration.
    """
    y_pred = np.asarray([0, 0, 1, 1])
    y_true = np.asarray([1, 1, 0, 0])
    return y_pred, y_true


@pytest.fixture
def classwise_ece_good_input():
    """
    Good Input.
    """
    y_pred = np.asarray(
        [0.4, 0.685, 0.625, 0.466, 0.254, 0.393, 0.455, 0.223, 0.986, 0.413]
    )
    y_true = np.asarray([1, 0, 1, 1, 1, 1, 1, 0, 1, 1])
    return y_pred, y_true


@pytest.fixture
def classwise_ece_3_class_good_input():
    y_pred = np.array(
        [
            [0.821, 0.133, 0.046],
            [0.712, 0.018, 0.27],
            [0.417, 0.258, 0.325],
            [0.867, 0.039, 0.094],
            [0.952, 0.018, 0.03],
            [0.484, 0.063, 0.453],
            [0.378, 0.455, 0.167],
            [0.486, 0.344, 0.17],
            [0.18, 0.244, 0.576],
            [0.169, 0.171, 0.66],
        ]
    )
    y_true = np.asarray([0, 2, 2, 1, 1, 0, 2, 1, 1, 0])
    return y_pred, y_true


@pytest.fixture
def classwise_ece_3_class_good_input_expected():
    """
    Good input for the 3 class problem. Expected errors calculated by hand.
    """
    overall_error = 0.382333333333333
    classwise_errors = np.asarray([0.3768, 0.3775, 0.3927])
    return (overall_error, classwise_errors)


@pytest.fixture
def good_bin_input():
    return {"probs": [0.55, 0.56, 0.57, 0.58, 0.59], "num_occurrences": 3}


@pytest.fixture
def good_bin_input_calculate_bin_deviation_expected():
    """
    Deviation calculated by hand.
    """
    return 0.03


@pytest.fixture
def _get_classwise_errors_3_class_good_input():
    bins = {
        0: {
            0: {"probs": [], "num_occurrences": 0},
            1: {"probs": [0.18, 0.169], "num_occurrences": 1},
            2: {"probs": [], "num_occurrences": 0},
            3: {"probs": [0.378], "num_occurrences": 0},
            4: {"probs": [0.417, 0.484, 0.486], "num_occurrences": 1},
            5: {"probs": [], "num_occurrences": 0},
            6: {"probs": [], "num_occurrences": 0},
            7: {"probs": [0.712], "num_occurrences": 0},
            8: {"probs": [0.821, 0.867], "num_occurrences": 1},
            9: {"probs": [0.952], "num_occurrences": 0},
        },
        1: {
            0: {"probs": [0.018, 0.039, 0.018, 0.063], "num_occurrences": 2},
            1: {"probs": [0.133, 0.171], "num_occurrences": 0},
            2: {"probs": [0.258, 0.244], "num_occurrences": 1},
            3: {"probs": [0.344], "num_occurrences": 1},
            4: {"probs": [0.455], "num_occurrences": 0},
            5: {"probs": [], "num_occurrences": 0},
            6: {"probs": [], "num_occurrences": 0},
            7: {"probs": [], "num_occurrences": 0},
            8: {"probs": [], "num_occurrences": 0},
            9: {"probs": [], "num_occurrences": 0},
        },
        2: {
            0: {"probs": [0.046, 0.094, 0.03], "num_occurrences": 0},
            1: {"probs": [0.167, 0.17], "num_occurrences": 1},
            2: {"probs": [0.27], "num_occurrences": 1},
            3: {"probs": [0.325], "num_occurrences": 1},
            4: {"probs": [0.453], "num_occurrences": 0},
            5: {"probs": [0.576], "num_occurrences": 0},
            6: {"probs": [0.66], "num_occurrences": 0},
            7: {"probs": [], "num_occurrences": 0},
            8: {"probs": [], "num_occurrences": 0},
            9: {"probs": [], "num_occurrences": 0},
        },
    }
    num_bins = 10
    num_samples = 10
    num_classes = 3
    return bins, num_bins, num_samples, num_classes


@pytest.fixture
def _get_classwise_errors_3_class_good_input_expected():
    """
    Good input for the 3 class problem. Expected errors calculated by hand.
    """
    classwise_errors = np.asarray([0.3768, 0.3775, 0.3927])
    return classwise_errors


@pytest.fixture
def classwise_ece_mismatching_lengths_input():
    """
    Input that should result in perfect calibration.
    """
    y_pred = np.asarray([0, 0, 1, 1])
    y_true = np.asarray([0, 0, 1])
    return y_pred, y_true


@pytest.fixture
def classwise_ece_perfectly_calibrated_y_true_list_input():
    """
    Input that should result in perfect calibration.
    """
    y_pred = np.asarray([0, 0, 1])
    y_true = [0, 0, 1]
    return y_pred, y_true


###### test_plotting.py ######


@pytest.fixture
def continuous_calibration_curve_input():
    y_pred = [0.04, 0.12, 0.27, 0.33, 0.45, 0.55, 0.68, 0.73, 0.84, 0.96]
    y_true = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    num_bins = 10
    method = "width"
    return y_pred, y_true, num_bins, method


@pytest.fixture
def continuous_calibration_curve_generate_x_y_expected():
    x = [0.04, 0.12, 0.27, 0.33, 0.45, 0.55, 0.68, 0.73, 0.84, 0.96]
    y = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    return [x], [y], [weights]


@pytest.fixture
def discontinuous_calibration_curve_input():
    y_pred = [0.04, 0.12, 0.27, 0.33, 0.68, 0.73, 0.84, 0.96]
    y_true = [0, 1, 0, 0, 1, 1, 1, 0]
    num_bins = 10
    method = "width"
    return y_pred, y_true, num_bins, method


@pytest.fixture
def discontinuous_calibration_curve_generate_x_y_expected():
    x = [[0.04, 0.12, 0.27, 0.33], [0.68, 0.73, 0.84, 0.96]]
    y = [[0, 1, 0, 0], [1, 1, 1, 0]]
    weights = [[1 / 8, 1 / 8, 1 / 8, 1 / 8], [1 / 8, 1 / 8, 1 / 8, 1 / 8]]
    return x, y, weights


@pytest.fixture
def _segment_curve_data_two_continuous_curves_input():
    x = [0.04, 0.12, 0.27, 0.33, None, None, 0.68, 0.73, 0.84, 0.96]
    y = [0, 1, 0, 0, None, None, 1, 1, 1, 0]
    weights = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 0, 0, 1 / 8, 1 / 8, 1 / 8, 1 / 8]
    return x, y, weights


@pytest.fixture
def _segment_curve_data_two_continuous_curves_expected():
    x = [[0.04, 0.12, 0.27, 0.33], [0.68, 0.73, 0.84, 0.96]]
    y = [[0, 1, 0, 0], [1, 1, 1, 0]]
    weights = [[1 / 8, 1 / 8, 1 / 8, 1 / 8], [1 / 8, 1 / 8, 1 / 8, 1 / 8]]
    return x, y, weights


@pytest.fixture
def _segment_curve_data_one_continuous_curve_one_point_input():
    x = [0.04, 0.12, 0.27, 0.33, None, None, 0.68, None, None, None]
    y = [0, 1, 0, 0, None, None, 1, None, None, None]
    weights = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 0, 0, 1 / 5, 0, 0, 0]
    return x, y, weights


@pytest.fixture
def _segment_curve_data_one_continuous_curve_one_point_expected():
    x = [[0.04, 0.12, 0.27, 0.33], [0.68]]
    y = [[0, 1, 0, 0], [1]]
    weights = [[1 / 5, 1 / 5, 1 / 5, 1 / 5], [1 / 5, 1 / 5, 1 / 5, 1 / 5]]
    return x, y, weights


@pytest.fixture
def _segment_curve_data_single_point_input():
    x = [0.04, None, None, None, None, None, None, None, None, None]
    y = [0, None, None, None, None, None, None, None, None, None]
    weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return x, y, weights


@pytest.fixture
def _segment_curve_data_single_point_expected():
    x = [[0.04]]
    y = [[0]]
    weights = [[1]]
    return x, y, weights


@pytest.fixture
def _add_bin_boundaries_to_x_single_point_input():
    x = [0.78]
    weights = [0.05]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_single_point_expected():
    x = [0.78]
    weights = [0.05]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_normal_input():
    """
    This will be used in conjunction with discontinuous_calibration_curve_input (as we need a CalibrationCurve object to test this method).
    However we will manually adjust the weights here to ensure the method behaves as desired, even though these weights are not the true weights for the bins
    given the input discontinuous_calibration_curve_input.
    """
    x = [0.04, 0.12, 0.27, 0.33]
    weights = [2 / 9, 1 / 9, 4 / 9, 1 / 9]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_normal_expected():
    x = [0.04, 0.1, 0.12, 0.2, 0.27, 0.3, 0.33]
    weights = [2 / 9, 1 / 9, 1 / 9, 4 / 9, 4 / 9, 1 / 9]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_two_values_normal_input():
    """
    This will be used in conjunction with discontinuous_calibration_curve_input (as we need a CalibrationCurve object to test this method).
    However we will manually adjust the weights here to ensure the method behaves as desired, even though these weights are not the true weights for the bins
    given the input discontinuous_calibration_curve_input.
    """
    x = [0.04, 0.12]
    weights = [2 / 9, 1 / 9]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_two_values_normal_expected():
    x = [0.04, 0.1, 0.12]
    weights = [2 / 9, 1 / 9]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_two_values_special_case_input():
    """
    This will be used in conjunction with discontinuous_calibration_curve_input (as we need a CalibrationCurve object to test this method).
    However we will manually adjust the weights here to ensure the method behaves as desired, even though these weights are not the true weights for the bins
    given the input discontinuous_calibration_curve_input.

        # Special case:
        # 1. Only two values in x and second is at start of bin
    """
    x = [0.04, 0.1]
    weights = [2 / 9, 1 / 9]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_two_values_special_case_expected():
    x = [0.04, 0.1]
    weights = [2 / 9]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_all_centres_coincide_with_start_special_case_input():
    """
    This will be used in conjunction with continuous_calibration_curve_input (as we need a CalibrationCurve object to test this method).
    However we will manually adjust the weights here to ensure the method behaves as desired, even though these weights are not the true weights for the bins
    given the input continuous_calibration_curve_input (and the weights may not even add up to 1).

        # Special case:
        # 2. All values in x happen to be the start points of the respective bins - should return same x, weights.
    """
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weights = [0.1, 0.11, 0.12, 0.23, 0.4, 0.9, 0.3, 0.4, 0.8, 0.23]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_all_centres_coincide_with_start_special_case_expected():
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weights = [0.1, 0.11, 0.12, 0.23, 0.4, 0.9, 0.3, 0.4, 0.8]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_some_centres_coincide_with_start_special_case_input():
    """
    This will be used in conjunction with continuous_calibration_curve_input (as we need a CalibrationCurve object to test this method).
    However we will manually adjust the weights here to ensure the method behaves as desired, even though these weights are not the true weights for the bins
    given the input continuous_calibration_curve_input (and the weights may not even add up to 1).

        # Special case:
        # 2. All values in x happen to be the start points of the respective bins - should return same x, weights.
    """
    x = [0.02, 0.1, 0.25, 0.3, 0.42, 0.53, 0.66, 0.73, 0.84, 0.96]
    weights = [0.1, 0.11, 0.12, 0.23, 0.4, 0.9, 0.3, 0.4, 0.8, 0.23]
    return x, weights


@pytest.fixture
def _add_bin_boundaries_to_x_some_centres_coincide_with_start_special_case_expected():
    x = [
        0.02,
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.42,
        0.5,
        0.53,
        0.6,
        0.66,
        0.7,
        0.73,
        0.8,
        0.84,
        0.9,
        0.96,
    ]
    weights = [
        0.1,
        0.11,
        0.12,
        0.12,
        0.23,
        0.4,
        0.4,
        0.9,
        0.9,
        0.3,
        0.3,
        0.4,
        0.4,
        0.8,
        0.8,
        0.23,
    ]
    return x, weights


@pytest.fixture
def _get_y_at_bin_centre_input():
    x = [0.12, 0.23, 0.34]
    x_new = [0.12, 0.2, 0.23, 0.3, 0.34]
    y = [0.1, 0.25, 0.45]
    i = 4
    return x, x_new, y, i


@pytest.fixture
def _get_y_at_bin_centre_expected():
    return 0.45


@pytest.fixture
def _get_y_at_bin_boundary_input():
    x = [0.12, 0.23, 0.34]
    x_new = [0.12, 0.2, 0.23, 0.3, 0.34]
    y = [0.1, 0.25, 0.45]
    i = 3
    return x, x_new, y, i


@pytest.fixture
def _get_y_at_bin_boundary_expected():
    """Interpolated by hand"""
    return 83 / 220


@pytest.fixture
def _get_y_at_bin_boundary_bad_case_input():
    x = [0.12, 0.23, 0.34]
    x_new = [0.12, 0.2, 0.23, 0.3, 0.34]
    y = [0.1, 0.25, 0.45]
    i = 4
    return x, x_new, y, i


@pytest.fixture
def _get_y_at_bin_boundary_bad_case_expected():
    return


@pytest.fixture
def _generate_x_y_with_bin_boundaries_normal_input():
    x = [0.15, 0.25, 0.35]
    y = [0.1, 0.24, 0.4]
    weights = [0.1, 0.2, 0.3]
    return x, y, weights


@pytest.fixture
def _generate_x_y_with_bin_boundaries_normal_expected():
    """Interpolated by hand"""
    x_new = [0.15, 0.2, 0.25, 0.3, 0.35]
    y_new = [0.1, 0.17, 0.24, 0.32, 0.4]
    weights_new = [0.1, 0.2, 0.2, 0.3]
    return x_new, y_new, weights_new


@pytest.fixture
def _generate_x_y_with_bin_boundaries_two_values_input():
    x = [0.15, 0.2]
    y = [0.1, 0.24]
    weights = [0.1, 0.2]
    return x, y, weights


@pytest.fixture
def _generate_x_y_with_bin_boundaries_two_values_expected():
    """Interpolated by hand"""
    x_new = [0.15, 0.2]
    y_new = [0.1, 0.24]
    weights_new = [0.1]
    return x_new, y_new, weights_new


@pytest.fixture
def _get_density_based_line_collection_segment_start_boundaries():
    """Rather than generate an expected LineCollection object, it is easier to ensure the segments are as we expect them to be in terms of (half-) bin boundaries and associated weights"""
    return [
        0.04,
        0.1,
        0.12,
        0.2,
        0.27,
        0.3,
        0.33,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.68,
        0.7,
        0.73,
        0.8,
        0.84,
        0.9,
    ]


@pytest.fixture
def _get_density_based_line_collection_segment_end_boundaries():
    """Rather than generate an expected LineCollection object, it is easier to ensure the segments are as we expect them to be in terms of (half-) bin boundaries and associated weights"""
    return [
        0.1,
        0.12,
        0.2,
        0.27,
        0.3,
        0.33,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.68,
        0.7,
        0.73,
        0.8,
        0.84,
        0.9,
        0.96,
    ]


@pytest.fixture
def _get_density_based_line_collection_segment_weights():
    """Rather than generate an expected LineCollection object, it is easier to ensure the segments are as we expect them to be in terms of (half-) bin boundaries and associated weights"""
    return [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
