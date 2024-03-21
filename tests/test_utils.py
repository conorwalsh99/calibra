import pytest
import pandas as pd 
import numpy as np 

from calibra.utils import bin_probabilities, _get_equal_width_bins, _get_equal_frequency_bins, _sort_predictions
    

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
        for class_key in expected.keys():
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict.keys():
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
        for class_key in expected.keys():
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict.keys():
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
        for class_key in expected.keys():
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict.keys():
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
        for class_key in expected.keys():
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict.keys():
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
        for class_key in expected.keys():
            expected_class_level_dict = expected[class_key]
            result_class_level_dict = result[class_key]
            for bin_key in expected_class_level_dict.keys():
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
    