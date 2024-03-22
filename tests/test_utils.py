import pytest
import pandas as pd 
import numpy as np 

from calibra.utils import (bin_probabilities, 
                        _get_equal_width_bins, 
                        _get_equal_frequency_bins, 
                        _sort_predictions, 
                        _reshape_y_pred,
                        _get_bin_weight,
                        _sum_occurrences,
                        _back_fill_equal_frequency_bins,
                        _forward_fill_equal_frequency_bins,
                    )
    

def test_sort_predictions_good_input(sort_predictions_good_input, expected_sort_predictions_good_input):
    y_pred_class_i, y_true_class_i = sort_predictions_good_input
    result_y_pred_class_i_sorted, result_y_true_class_i_sorted = _sort_predictions(y_pred_class_i, y_true_class_i)
    expected_y_pred_class_i_sorted, expected_y_true_class_i_sorted = expected_sort_predictions_good_input
    
    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_pred_class_i_sorted} instead of {expected_y_pred_class_i_sorted}"
    assert result_y_pred_class_i_sorted == expected_y_pred_class_i_sorted, message

    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_true_class_i_sorted} instead of {expected_y_true_class_i_sorted}"
    assert result_y_true_class_i_sorted == expected_y_true_class_i_sorted, message


def test_sort_predictions_duplicate_predictions(sort_predictions_duplicate_predictions, expected_sort_predictions_duplicate_predictions):
    y_pred_class_i, y_true_class_i = sort_predictions_duplicate_predictions
    result_y_pred_class_i_sorted, result_y_true_class_i_sorted = _sort_predictions(y_pred_class_i, y_true_class_i)
    expected_y_pred_class_i_sorted, expected_y_true_class_i_sorted = expected_sort_predictions_duplicate_predictions

    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_pred_class_i_sorted} instead of {expected_y_pred_class_i_sorted}"
    assert result_y_pred_class_i_sorted == expected_y_pred_class_i_sorted, message

    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_true_class_i_sorted} instead of {expected_y_true_class_i_sorted}"
    assert result_y_true_class_i_sorted == expected_y_true_class_i_sorted[0] or result_y_true_class_i_sorted == expected_y_true_class_i_sorted[1], message


def test_get_equal_width_bins(get_bins_good_input, expected_get_equal_width_bins_good_input):
    result = _get_equal_width_bins(*get_bins_good_input)
    expected = expected_get_equal_width_bins_good_input
    message = f"get_equal_width_bins returned {result}, instead of {expected}"
    assert result == expected, message 
    
def test_get_equal_frequency_bins(get_bins_good_input, expected_get_equal_frequency_bins_good_input):
    result = _get_equal_frequency_bins(*get_bins_good_input)
    expected = expected_get_equal_frequency_bins_good_input
    message = f"get_equal_frequency_bins returned {result}, instead of {expected}"

def test_bin_probabilities_good_input_width_1d_ypred(bin_probabilities_good_input_width_1d_y_pred, expected_get_equal_width_bins_good_input):
    result = bin_probabilities(*bin_probabilities_good_input_width_1d_y_pred)
    expected = expected_get_equal_width_bins_good_input
    message = f"bin_probabilities returned {result}, instead of {expected}"
    dicts_equal = True
    try:
        for class_key in expected:
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict:
                expected_bin_level_dict = expected_class_level_dict[bin_key]
                result_bin_level_dict = result_class_level_dict[bin_key]
                expected_probs = expected_bin_level_dict['probs']
                result_probs = result_bin_level_dict['probs']
                expected_num_occurrences = expected_bin_level_dict['num_occurrences']
                result_num_occurrences = result_bin_level_dict['num_occurrences']

                if expected_num_occurrences != result_num_occurrences:
                    dicts_equal = False
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif len(expected_probs) != len(result_probs):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif not all(abs(a - b) < 1e-8 for a, b in zip(expected_probs, result_probs)):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
    except:
        dicts_equal = False

    assert dicts_equal, message 


def test_bin_probabilities_good_input_width_2d_ypred(bin_probabilities_good_input_width_2d_y_pred, expected_get_equal_width_bins_good_input):
    result = bin_probabilities(*bin_probabilities_good_input_width_2d_y_pred)
    expected = expected_get_equal_width_bins_good_input
    message = f"bin_probabilities returned {result}, instead of {expected}"
    dicts_equal = True
    try:
        for class_key in expected:
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict:
                expected_bin_level_dict = expected_class_level_dict[bin_key]
                result_bin_level_dict = result_class_level_dict[bin_key]
                expected_probs = expected_bin_level_dict['probs']
                result_probs = result_bin_level_dict['probs']
                expected_num_occurrences = expected_bin_level_dict['num_occurrences']
                result_num_occurrences = result_bin_level_dict['num_occurrences']

                if expected_num_occurrences != result_num_occurrences:
                    dicts_equal = False
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif len(expected_probs) != len(result_probs):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif not all(abs(a - b) < 1e-8 for a, b in zip(expected_probs, result_probs)):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
    except:
        dicts_equal = False

    assert dicts_equal, message 


def test_bin_probabilities_good_input_frequency_1d_ypred(bin_probabilities_good_input_frequency_1d_y_pred, expected_get_equal_frequency_bins_good_input):
    result = bin_probabilities(*bin_probabilities_good_input_frequency_1d_y_pred)
    expected = expected_get_equal_frequency_bins_good_input
    message = f"bin_probabilities returned {result}, instead of {expected}"
    dicts_equal = True
    try:
        for class_key in expected:
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict:
                expected_bin_level_dict = expected_class_level_dict[bin_key]
                result_bin_level_dict = result_class_level_dict[bin_key]
                expected_probs = expected_bin_level_dict['probs']
                result_probs = result_bin_level_dict['probs']
                expected_num_occurrences = expected_bin_level_dict['num_occurrences']
                result_num_occurrences = result_bin_level_dict['num_occurrences']

                if expected_num_occurrences != result_num_occurrences:
                    dicts_equal = False
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif len(expected_probs) != len(result_probs):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif not all(abs(a - b) < 1e-8 for a, b in zip(expected_probs, result_probs)):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
    except:
        dicts_equal = False

    assert dicts_equal, message 


def test_bin_probabilities_good_input_frequency_2d_ypred(bin_probabilities_good_input_frequency_2d_y_pred, expected_get_equal_frequency_bins_good_input):
    result = bin_probabilities(*bin_probabilities_good_input_frequency_2d_y_pred)
    expected = expected_get_equal_frequency_bins_good_input
    message = f"bin_probabilities returned {result}, instead of {expected}"
    dicts_equal = True
    try:
        for class_key in expected:
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict:
                expected_bin_level_dict = expected_class_level_dict[bin_key]
                result_bin_level_dict = result_class_level_dict[bin_key]
                expected_probs = expected_bin_level_dict['probs']
                result_probs = result_bin_level_dict['probs']
                expected_num_occurrences = expected_bin_level_dict['num_occurrences']
                result_num_occurrences = result_bin_level_dict['num_occurrences']

                if expected_num_occurrences != result_num_occurrences:
                    dicts_equal = False
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif len(expected_probs) != len(result_probs):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif not all(abs(a - b) < 1e-8 for a, b in zip(expected_probs, result_probs)):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
    except:
        dicts_equal = False

    assert dicts_equal, message

    
def test_get_equal_frequency_bins_non_integer_freq(get_equal_frequency_bins_non_integer_freq, expected_get_equal_frequency_bins_non_integer_freq):
    result = _get_equal_frequency_bins(*get_equal_frequency_bins_non_integer_freq)
    expected = expected_get_equal_frequency_bins_non_integer_freq

    message = f"Bins do not match. Returned {result} instead of {expected}."

    dicts_equal = True
    try:
        for class_key in expected:
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict:
                expected_bin_level_dict = expected_class_level_dict[bin_key]
                result_bin_level_dict = result_class_level_dict[bin_key]
                expected_probs = expected_bin_level_dict['probs']
                result_probs = result_bin_level_dict['probs']
                expected_num_occurrences = expected_bin_level_dict['num_occurrences']
                result_num_occurrences = result_bin_level_dict['num_occurrences']

                if expected_num_occurrences != result_num_occurrences:
                    dicts_equal = False
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif len(expected_probs) != len(result_probs):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
                elif not all(abs(a - b) < 1e-8 for a, b in zip(expected_probs, result_probs)):
                    print(f'Class: {class_key}, bin: {bin_key}, expected num_occurrences: {expected_num_occurrences}, result num_occurrences: {result_num_occurrences}')
                    raise AssertionError('Dictionaries not equal')
    except:
        dicts_equal = False

    assert dicts_equal, message

def test_bin_probabilities_y_pred_greater_than_1(get_bins_y_pred_greater_than_1_input):
    """
    We expect an error to be raised as the input is bad.
    """
    with pytest.raises(ValueError):
        bin_probabilities(*get_bins_y_pred_greater_than_1_input)
    
def test_bin_probabilities_y_pred_less_than_0(get_bins_y_pred_less_than_0_input):
    """
    We expect an error to be raised as the input is bad.
    """
    with pytest.raises(ValueError):
        bin_probabilities(*get_bins_y_pred_less_than_0_input)

def test_bin_probabilities_y_pred_3_dimensional(get_bins_y_pred_3_dimensional_input):
    """
    We expect an error to be raised as the input is bad.
    """
    with pytest.raises(ValueError):
        bin_probabilities(*get_bins_y_pred_3_dimensional_input)    

def test_bin_probabilities_y_true_fractional(get_bins_y_true_fractional_input):
    """
    We expect an error to be raised as the input is bad.
    """
    with pytest.raises(ValueError):
        bin_probabilities(*get_bins_y_true_fractional_input)

def test_bin_probabilities_negative_y_true(get_bins_negative_y_true_input):
    """
    We expect an error to be raised as the input is bad.
    """
    with pytest.raises(ValueError):
        bin_probabilities(*get_bins_negative_y_true_input)


def test_reshape_y_pred(_reshape_y_pred_input, _reshape_y_pred_expected):
    expected = _reshape_y_pred_expected
    result = _reshape_y_pred(_reshape_y_pred_input)
    assert np.allclose(result, expected), f'_reshape_y_pred returned {result} instead of {expected}.'

def test_get_bin_weight(_get_bin_weight_input, _get_bin_weight_expected):
    bin, num_samples = _get_bin_weight_input
    result = _get_bin_weight(bin, num_samples)
    expected = _get_bin_weight_expected
    assert result == expected, f'_get_bin_weight returned {result} instead of {expected}.'

def test_sum_occurrences(_sum_occurrences_input, _sum_occurrences_expected):
    result = _sum_occurrences(*_sum_occurrences_input)
    expected = _sum_occurrences_expected
    assert result == expected, f'_sum_occurrences returned {result} instead of {expected}.'

def test_forward_fill_equal_frequency_bins(_forward_fill_equal_frequency_bins_input, _forward_fill_equal_frequency_bins_expected):
    result = _forward_fill_equal_frequency_bins(*_forward_fill_equal_frequency_bins_input)
    expected = _forward_fill_equal_frequency_bins_expected
    assert result == expected, f'_forward_fill_equal_frequency_bins returned {result} instead of {expected}.'

def test_back_fill_equal_frequency_bins(_back_fill_equal_frequency_bins_input, _back_fill_equal_frequency_bins_expected):
    result = _back_fill_equal_frequency_bins(*_back_fill_equal_frequency_bins_input)
    expected = _back_fill_equal_frequency_bins_expected
    assert result == expected, f'_back_fill_equal_frequency_bins returned {result} instead of {expected}.'

