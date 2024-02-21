import pytest
import pandas as pd 
import numpy as np 

from calibra.errors import classwise_ece, _get_classwise_errors, _calculate_bin_deviation


@pytest.fixture
def classwise_ece_perfect_input():
    """
     Input that should result in perfect calibration.
    """
    y_pred = np.asarray([0, 0, 1])
    y_true = np.asarray([0, 0, 1])
    return y_pred, y_true


def test_classwise_ece_perfect_input(classwise_ece_perfect_input):
    y_pred, y_true = classwise_ece_perfect_input
    result = classwise_ece(y_pred=y_pred, y_true=y_true, num_bins=20, method='width', return_classwise_errors=False)
    expected = 0
    assert result == expected, f'classwise_ece returned {result} instead of {expected}.'
