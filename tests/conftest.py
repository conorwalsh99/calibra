import pytest
import pandas as pd
import numpy as np 


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


@pytest.fixture
def classwise_ece_perfectly_calibrated_input():
    """
     Input that should result in perfect calibration.
    """
    y_pred = np.asarray([0, 0, 1])
    y_true = [0, 0, 1]
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

