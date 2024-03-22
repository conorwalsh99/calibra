import pytest
import pandas as pd 
import numpy as np 

from calibra.errors import classwise_ece, _get_classwise_errors, _calculate_bin_deviation


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

def test_classwise_ece_perfectly_calibrated_y_true_list_input(classwise_ece_perfectly_calibrated_y_true_list_input):
    y_pred, y_true = classwise_ece_perfectly_calibrated_y_true_list_input
    result = classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=20, method='width', return_classwise_errors=False)
    expected = 0
    assert result == expected, f'classwise_ece returned {result} instead of {expected}.'

def test_classwise_ece_negative_num_bins(classwise_ece_perfectly_calibrated_input):
    """
    Error should be raised as num_bins is invalid.
    """
    y_pred, y_true = classwise_ece_perfectly_calibrated_input
    with pytest.raises(ValueError):
        classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=-20, method='width', return_classwise_errors=False)

def test_classwise_ece_fractional_num_bins(classwise_ece_perfectly_calibrated_input):
    """
    Error should be raised as num_bins is invalid.
    """
    y_pred, y_true = classwise_ece_perfectly_calibrated_input
    with pytest.raises(ValueError):
        classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=16.5, method='width', return_classwise_errors=False)

def test_classwise_ece_0_num_bins(classwise_ece_perfectly_calibrated_input):
    """
    Error should be raised as num_bins is invalid.
    """
    y_pred, y_true = classwise_ece_perfectly_calibrated_input
    with pytest.raises(ValueError):
        classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=0, method='width', return_classwise_errors=False)

def test_classwise_ece_invalid_method(classwise_ece_perfectly_calibrated_input):
    """
    Error should be raised as method is invalid.
    """
    y_pred, y_true = classwise_ece_perfectly_calibrated_input
    with pytest.raises(ValueError):
        classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=20, method='length', return_classwise_errors=False)

def test_classwise_ece_mismatching_lengths(classwise_ece_mismatching_lengths_input):
    """
    Error should be raised as y_pred and y_true lengths don't match.
    """
    y_pred, y_true = classwise_ece_mismatching_lengths_input
    with pytest.raises(ValueError):
        classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=20, method='width', return_classwise_errors=False)
