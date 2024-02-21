import pytest
import pandas as pd 
import numpy as np 


from calibra.utils import bin_probabilities, get_equal_width_bins, get_equal_frequency_bins, sort_predictions


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
            1: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        }
    ) 
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 10, 5
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

    return y_pred, y_true, num_classes, num_samples, num_bins, bins

@pytest.fixture
def expected_get_equal_width_bins_good_input():
    return {
        0: {
            0: {
                'probs': [0.15, 0.05], 
                'num_occurrences': 0,
            },

            1: {
                'probs': [0.35, 0.25], 
                'num_occurrences': 0,
            },

            2: {
                'probs': [0.55, 0.45], 
                'num_occurrences': 1,
            },
            
            3: {
                'probs': [0.75, 0.65], 
                'num_occurrences': 2,
            },

            4: {
                'probs': [0.95, 0.85], 
                'num_occurrences': 2,
            },                                
        },

        1: {
            0: {
                'probs': [0.05, 0.15], 
                'num_occurrences': 0,
            },

            1: {
                'probs': [0.25, 0.35], 
                'num_occurrences': 0,
            },

            2: {
                'probs': [0.45, 0.55], 
                'num_occurrences': 1,
            },
            
            3: {
                'probs': [0.65, 0.75], 
                'num_occurrences': 2,
            },

            4: {
                'probs': [0.85, 0.95], 
                'num_occurrences': 2,
            }            
        }
    }

# as get_equal_frequency_bins uses sort_predictions to sort by the size of probability, we see that in the resulting bins, the probabilities are always in ascending order e.g. [0.05, 0.15].
# this is not necessarily true for get_equal_width_bins, where we loop through the predictions as they are presented to us, hence the difference in the resulting bins.
@pytest.fixture
def expected_get_equal_frequency_bins_good_input():
    return {
        0: {
            0: {
                'probs': [0.05, 0.15], 
                'num_occurrences': 0,
            },

            1: {
                'probs': [0.25, 0.35], 
                'num_occurrences': 0,
            },

            2: {
                'probs': [0.45, 0.55], 
                'num_occurrences': 1,
            },
            
            3: {
                'probs': [0.65, 0.75], 
                'num_occurrences': 2,
            },

            4: {
                'probs': [0.85, 0.95], 
                'num_occurrences': 2,
            },                                
        },

        1: {
            0: {
                'probs': [0.05, 0.15], 
                'num_occurrences': 0,
            },

            1: {
                'probs': [0.25, 0.35], 
                'num_occurrences': 0,
            },

            2: {
                'probs': [0.45, 0.55], 
                'num_occurrences': 1,
            },
            
            3: {
                'probs': [0.65, 0.75], 
                'num_occurrences': 2,
            },

            4: {
                'probs': [0.85, 0.95], 
                'num_occurrences': 2,
            }            
        }
    }

@pytest.fixture
def bin_probabilities_good_input_width_1d_y_pred():
    y_pred = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]) 
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = 'width'
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
            [0.05, 0.95]
        ]
    ) 
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = 'width'
    return y_pred, y_true, num_bins, method     
                
@pytest.fixture
def bin_probabilities_good_input_frequency_1d_y_pred():
    y_pred = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],) 
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = 'frequency'
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
            [0.05, 0.95]
        ]
    )  
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    num_bins = 5
    method = 'frequency'
    return y_pred, y_true, num_bins, method 


@pytest.fixture
def get_equal_frequency_bins_non_integer_freq():
    y_pred = pd.DataFrame(
        {
            0: [0.95, 0.9, 0.85, 0.8, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
            1: [0.05, 0.1, 0.15, 0.2, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        }
    )
    y_true = np.asarray([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    num_classes, num_samples, num_bins = 2, 13, 4
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

    return y_pred, y_true, num_classes, num_samples, num_bins, bins

@pytest.fixture
def expected_get_equal_frequency_bins_non_integer_freq():
    return {        
            0: {
                0: {
                    'probs': [0.35, 0.4, 0.45, 0.5],
                    'num_occurrences': 0,
                },

                1: {
                    'probs': [0.55, 0.6, 0.65],
                    'num_occurrences': 0,
                },

                2: {
                    'probs': [0.7, 0.8, 0.8],
                    'num_occurrences': 1,
                },

                3: {
                    'probs': [0.85, 0.9, 0.95],
                    'num_occurrences': 0,
                }                         
            },

            1: {
                0: {
                    'probs': [0.05, 0.1, 0.15, 0.2],
                    'num_occurrences': 3,
                },

                1: {
                    'probs': [0.2, 0.3, 0.35],
                    'num_occurrences': 3,
                },

                2: {
                    'probs': [0.4, 0.45, 0.5],
                    'num_occurrences': 3,
                },

                3: {
                    'probs': [0.55, 0.6, 0.65],
                    'num_occurrences': 3,
                }
            }
        }
    


def test_sort_predictions_good_input(sort_predictions_good_input, expected_sort_predictions_good_input):
    y_pred_class_i, y_true_class_i = sort_predictions_good_input
    result_y_pred_class_i_sorted, result_y_true_class_i_sorted = sort_predictions(y_pred_class_i, y_true_class_i)
    expected_y_pred_class_i_sorted, expected_y_true_class_i_sorted = expected_sort_predictions_good_input
    
    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_pred_class_i_sorted} instead of {expected_y_pred_class_i_sorted}"
    assert result_y_pred_class_i_sorted == expected_y_pred_class_i_sorted, message

    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_true_class_i_sorted} instead of {expected_y_true_class_i_sorted}"
    assert result_y_true_class_i_sorted == expected_y_true_class_i_sorted, message


def test_sort_predictions_duplicate_predictions(sort_predictions_duplicate_predictions, expected_sort_predictions_duplicate_predictions):
    y_pred_class_i, y_true_class_i = sort_predictions_duplicate_predictions
    result_y_pred_class_i_sorted, result_y_true_class_i_sorted = sort_predictions(y_pred_class_i, y_true_class_i)
    expected_y_pred_class_i_sorted, expected_y_true_class_i_sorted = expected_sort_predictions_duplicate_predictions

    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_pred_class_i_sorted} instead of {expected_y_pred_class_i_sorted}"
    assert result_y_pred_class_i_sorted == expected_y_pred_class_i_sorted, message

    message = f"sort_predictions did not sort predictions correctly. Returned: {result_y_true_class_i_sorted} instead of {expected_y_true_class_i_sorted}"
    assert result_y_true_class_i_sorted == expected_y_true_class_i_sorted[0] or result_y_true_class_i_sorted == expected_y_true_class_i_sorted[1], message


def test_get_equal_width_bins(get_bins_good_input, expected_get_equal_width_bins_good_input):
    result = get_equal_width_bins(*get_bins_good_input)
    expected = expected_get_equal_width_bins_good_input
    message = f"get_equal_width_bins returned {result}, instead of {expected}"
    assert result == expected, message 
    
def test_get_equal_frequency_bins(get_bins_good_input, expected_get_equal_frequency_bins_good_input):
    result = get_equal_frequency_bins(*get_bins_good_input)
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
    result = get_equal_frequency_bins(*get_equal_frequency_bins_non_integer_freq)
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
    