import pytest
import pandas as pd 
import numpy as np 

from calibra.errors import classwise_ece, _get_classwise_errors, _calculate_bin_deviation


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
    y_pred = np.asarray([0.4, 0.685, 0.625, 0.466, 0.254, 0.393, 0.455, 0.223, 0.986, 0.413])
    y_true = np.asarray([1, 0, 1, 1, 1, 1, 1, 0, 1, 1])
    return y_pred, y_true

@pytest.fixture
def classwise_ece_3_class_good_input():
    y_pred = np.array(
        [ 
        [0.821, 0.133, 0.046],
        [0.712, 0.018, 0.27 ],
        [0.417, 0.258, 0.325],
        [0.867, 0.039, 0.094],
        [0.952, 0.018, 0.03 ],
        [0.484, 0.063, 0.453],
        [0.378, 0.455, 0.167],
        [0.486, 0.344, 0.17 ],
        [0.18 , 0.244, 0.576],
        [0.169, 0.171, 0.66 ]
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
    return {
        'probs': [0.55, 0.56, 0.57, 0.58, 0.59],
        'num_occurrences': 3
    }

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
            0: {
                'probs': [],
                'num_occurrences': 0
            },
            1: {
                'probs': [0.18, 0.169],
                'num_occurrences': 1
            },
            2: {
                'probs': [],
                'num_occurrences': 0
            },
            3: {
                'probs': [0.378],
                'num_occurrences': 0
            },
            4: {
                'probs': [0.417, 0.484, 0.486],
                'num_occurrences': 1
            },
            5: {
                'probs': [],
                'num_occurrences': 0
            },
            6: {
                'probs': [],
                'num_occurrences': 0
            },
            7: {
                'probs': [0.712],
                'num_occurrences': 0
            },
            8: {
                'probs': [0.821, 0.867],
                'num_occurrences': 1
            },
            9: {
                'probs': [0.952],
                'num_occurrences': 0
            },                                                                                                                                    
        },
        1: {
            0: {
                'probs': [0.018, 0.039, 0.018, 0.063],
                'num_occurrences': 2
            },
            1: {
                'probs': [0.133, 0.171],
                'num_occurrences': 0
            },
            2: {
                'probs': [0.258, 0.244],
                'num_occurrences': 1
            },
            3: {
                'probs': [0.344],
                'num_occurrences': 1
            },
            4: {
                'probs': [0.455],
                'num_occurrences': 0
            },
            5: {
                'probs': [],
                'num_occurrences': 0
            },
            6: {
                'probs': [],
                'num_occurrences': 0
            },
            7: {
                'probs': [],
                'num_occurrences': 0
            },
            8: {
                'probs': [],
                'num_occurrences': 0
            },
            9: {
                'probs': [],
                'num_occurrences': 0
            },                                                                                                                                    
        },
        2: {
            0: {
                'probs': [0.046, 0.094, 0.03],
                'num_occurrences': 0
            },
            1: {
                'probs': [0.167, 0.17],
                'num_occurrences': 1
            },
            2: {
                'probs': [0.27],
                'num_occurrences': 1
            },
            3: {
                'probs': [0.325],
                'num_occurrences': 1
            },
            4: {
                'probs': [0.453],
                'num_occurrences': 0
            },
            5: {
                'probs': [0.576],
                'num_occurrences': 0
            },
            6: {
                'probs': [0.66],
                'num_occurrences': 0
            },
            7: {
                'probs': [],
                'num_occurrences': 0
            },
            8: {
                'probs': [],
                'num_occurrences': 0
            },
            9: {
                'probs': [],
                'num_occurrences': 0
            },                                                                                                                                    
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


def test_classwise_ece_perfectly_calibrated_input(classwise_ece_perfectly_calibrated_input):
    y_pred, y_true = classwise_ece_perfectly_calibrated_input
    result = classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=20, method='width', return_classwise_errors=False)
    expected = 0
    assert result == expected, f'classwise_ece returned {result} instead of {expected}.'

def test_classwise_ece_perfectly_uncalibrated__input(classwise_ece_perfectly_uncalibrated_input):
    y_pred, y_true = classwise_ece_perfectly_uncalibrated_input
    result = classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=20, method='width', return_classwise_errors=False)
    expected = 1
    assert result == expected, f'classwise_ece returned {result} instead of {expected}.'

def test_classwise_ece_good_input_10_samples_10_bins(classwise_ece_good_input):
    y_pred, y_true = classwise_ece_good_input
    overall_error_result = classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=10,  method='width', return_classwise_errors=False)
    overall_error_expected = 0.372 # calculated by hand
    assert overall_error_result == overall_error_expected, f'classwise_ece returned {overall_error_result} instead of {overall_error_expected}'

def test_classwise_ece_3_class_good_input(classwise_ece_3_class_good_input, classwise_ece_3_class_good_input_expected):
    y_pred, y_true = classwise_ece_3_class_good_input
    overall_error_result, classwise_errors_result = classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=10,  method='width', return_classwise_errors=True)
    overall_error_expected, classwise_errors_expected = classwise_ece_3_class_good_input_expected
    assert np.isclose(overall_error_result, overall_error_expected), f'classwise_ece returned {overall_error_result} instead of {overall_error_expected}'
    assert np.allclose(classwise_errors_result, classwise_errors_expected), f'classwise_ece returned {classwise_errors_result} instead of {classwise_errors_expected}'

def test_calculate_bin_deviation(good_bin_input, good_bin_input_calculate_bin_deviation_expected):
    result = _calculate_bin_deviation(good_bin_input)
    expected = good_bin_input_calculate_bin_deviation_expected
    assert np.isclose(result, expected), f'_calculate_bin_deviation returned {result} instead of {expected}'

def test_get_classwise_errors_3_class_good_input(_get_classwise_errors_3_class_good_input, _get_classwise_errors_3_class_good_input_expected):
    bins, num_bins, num_samples, num_classes = _get_classwise_errors_3_class_good_input
    result = _get_classwise_errors(bins, num_bins, num_samples, num_classes)
    expected = _get_classwise_errors_3_class_good_input_expected
    assert np.allclose(result, expected), f'_get_classwise_errors returned {result} instead of {expected}'
