import pytest
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from calibra.plotting import CalibrationCurve


class TestCalibrationCurve:
    def test_segment_curve_data_two_continuous_curves(
        self,
        _segment_curve_data_two_continuous_curves_input,
        _segment_curve_data_two_continuous_curves_expected,
    ):
        x, y, weights = _segment_curve_data_two_continuous_curves_input
        x_result, y_result, weights_result = CalibrationCurve._segment_curve_data(
            x, y, weights
        )
        x_expected, y_expected, weights_expected = (
            _segment_curve_data_two_continuous_curves_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_segment_curve_data returned {x_result} instead of {x_expected}"
        assert np.allclose(
            y_result, y_expected
        ), f"_segment_curve_data returned {y_result} instead of {y_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_segment_curve_data returned {weights_result} instead of {weights_expected}"

    def test_segment_curve_data_one_continuous_curve_one_point(
        self,
        _segment_curve_data_one_continuous_curve_one_point_input,
        _segment_curve_data_one_continuous_curve_one_point_expected,
    ):
        x, y, weights = _segment_curve_data_one_continuous_curve_one_point_input
        x_result, y_result, weights_result = CalibrationCurve._segment_curve_data(
            x, y, weights
        )
        x_expected, y_expected, weights_expected = (
            _segment_curve_data_one_continuous_curve_one_point_expected
        )
        for segment_x_result, segment_x_expected in zip(x_result, x_expected):
            assert np.allclose(
                segment_x_result, segment_x_expected
            ), f"_segment_curve_data returned {x_result} instead of {x_expected}"
        for segment_y_result, segment_y_expected in zip(y_result, y_expected):
            assert np.allclose(
                segment_y_result, segment_y_expected
            ), f"_segment_curve_data returned {y_result} instead of {y_expected}"
        for segment_weights_result, segment_weights_expected in zip(
            weights_result, weights_expected
        ):
            assert np.allclose(
                segment_weights_result, segment_weights_expected
            ), f"_segment_curve_data returned {weights_result} instead of {weights_expected}"

    def test_segment_curve_data_single_point(
        self,
        _segment_curve_data_single_point_input,
        _segment_curve_data_single_point_expected,
    ):
        x, y, weights = _segment_curve_data_single_point_input
        x_result, y_result, weights_result = CalibrationCurve._segment_curve_data(
            x, y, weights
        )
        x_expected, y_expected, weights_expected = (
            _segment_curve_data_single_point_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_segment_curve_data returned {x_result} instead of {x_expected}"
        assert np.allclose(
            y_result, y_expected
        ), f"_segment_curve_data returned {y_result} instead of {y_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_segment_curve_data returned {weights_result} instead of {weights_expected}"

    def test_generate_x_y_single_continuous_curve(
        self,
        continuous_calibration_curve_input,
        continuous_calibration_curve_generate_x_y_expected,
    ):
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x_result, y_result, weights_result = calibration_curve._generate_x_y(
            class_label=1
        )
        x_expected, y_expected, weights_expected = (
            continuous_calibration_curve_generate_x_y_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_generate_x_y returned {x_result} instead of {x_expected}"
        assert np.allclose(
            y_result, y_expected
        ), f"_generate_x_y returned {y_result} instead of {y_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_generate_x_y returned {weights_result} instead of {weights_expected}"

    def test_generate_x_y_two_continuous_curves(
        self,
        discontinuous_calibration_curve_input,
        discontinuous_calibration_curve_generate_x_y_expected,
    ):
        y_pred, y_true, num_bins, method = discontinuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x_result, y_result, weights_result = calibration_curve._generate_x_y(
            class_label=1
        )
        x_expected, y_expected, weights_expected = (
            discontinuous_calibration_curve_generate_x_y_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_generate_x_y returned {x_result} instead of {x_expected}"
        assert np.allclose(
            y_result, y_expected
        ), f"_generate_x_y returned {y_result} instead of {y_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_generate_x_y returned {weights_result} instead of {weights_expected}"

    def test_add_bin_boundaries_to_x_single_point(
        self,
        continuous_calibration_curve_input,
        _add_bin_boundaries_to_x_single_point_input,
        _add_bin_boundaries_to_x_single_point_expected,
    ):
        """
        We need to initialise a CalibrationCurve() object to call _add_bin_boundaries_to_x. Note that the input to the CalibrationCurve object
        is not consistent with the x, weights input to _add_bin_boundaries_to_x but that is okay since it will not affect it in the bcase of a single point in x.
        """
        # Initialise CalibrationCurve with arbitrary y_pred, y_true.
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, weights = _add_bin_boundaries_to_x_single_point_input
        x_result, weights_result = calibration_curve._add_bin_boundaries_to_x(
            x, weights
        )
        x_expected, weights_expected = _add_bin_boundaries_to_x_single_point_expected
        assert np.allclose(
            x_result, x_expected
        ), f"_add_bin_boundaries_to_x returned {x_result} instead of {x_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_add_bin_boundaries_to_x returned {weights_result} instead of {weights_expected}"

    def test_add_bin_boundaries_to_x_normal(
        self,
        discontinuous_calibration_curve_input,
        _add_bin_boundaries_to_x_normal_input,
        _add_bin_boundaries_to_x_normal_expected,
    ):
        """
        While we need discontinuous_calibration_curve_input to initialise the CalibrationCurve object and calculate self.bin_boundaries,
        we will be artificially adjusting the weights in _add_bin_boundaries_to_x_normal_input to test if our method behaves as desired
        (the weights are not equal to thoise calculated for discontinuous_calibration_curve_input).
        """
        y_pred, y_true, num_bins, method = discontinuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, weights = _add_bin_boundaries_to_x_normal_input
        x_result, weights_result = calibration_curve._add_bin_boundaries_to_x(
            x, weights
        )
        x_expected, weights_expected = _add_bin_boundaries_to_x_normal_expected
        assert np.allclose(
            x_result, x_expected
        ), f"_add_bin_boundaries_to_x returned {x_result} instead of {x_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_add_bin_boundaries_to_x returned {weights_result} instead of {weights_expected}"

    def test_add_bin_boundaries_to_x_two_values_normal(
        self,
        discontinuous_calibration_curve_input,
        _add_bin_boundaries_to_x_two_values_normal_input,
        _add_bin_boundaries_to_x_two_values_normal_expected,
    ):
        """
        While we need discontinuous_calibration_curve_input to initialise the CalibrationCurve object and calculate self.bin_boundaries,
        we will be artificially adjusting the weights in _add_bin_boundaries_to_x_normal_input to test if our method behaves as desired
        (the weights are not equal to thoise calculated for discontinuous_calibration_curve_input).
        """
        y_pred, y_true, num_bins, method = discontinuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, weights = _add_bin_boundaries_to_x_two_values_normal_input
        x_result, weights_result = calibration_curve._add_bin_boundaries_to_x(
            x, weights
        )
        x_expected, weights_expected = (
            _add_bin_boundaries_to_x_two_values_normal_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_add_bin_boundaries_to_x returned {x_result} instead of {x_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_add_bin_boundaries_to_x returned {weights_result} instead of {weights_expected}"

    def test_add_bin_boundaries_to_x_two_values_special(
        self,
        discontinuous_calibration_curve_input,
        _add_bin_boundaries_to_x_two_values_special_case_input,
        _add_bin_boundaries_to_x_two_values_special_case_expected,
    ):
        """
        While we need discontinuous_calibration_curve_input to initialise the CalibrationCurve object and calculate self.bin_boundaries,
        we will be artificially adjusting the weights in _add_bin_boundaries_to_x_normal_input to test if our method behaves as desired
        (the weights are not equal to those calculated for discontinuous_calibration_curve_input).
        """
        y_pred, y_true, num_bins, method = discontinuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, weights = _add_bin_boundaries_to_x_two_values_special_case_input
        x_result, weights_result = calibration_curve._add_bin_boundaries_to_x(
            x, weights
        )
        x_expected, weights_expected = (
            _add_bin_boundaries_to_x_two_values_special_case_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_add_bin_boundaries_to_x returned {x_result} instead of {x_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_add_bin_boundaries_to_x returned {weights_result} instead of {weights_expected}"

    def test_add_bin_boundaries_to_x_all_centres_coincide_with_start_special(
        self,
        continuous_calibration_curve_input,
        _add_bin_boundaries_to_x_all_centres_coincide_with_start_special_case_input,
        _add_bin_boundaries_to_x_all_centres_coincide_with_start_special_case_expected,
    ):
        """
        While we need continuous_calibration_curve_input to initialise the CalibrationCurve object and calculate self.bin_boundaries,
        we will be artificially adjusting the weights in _add_bin_boundaries_to_x_normal_input to test if our method behaves as desired
        (the weights are not equal to those calculated for continuous_calibration_curve_input and don't even need to add up to 1).
        """
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, weights = (
            _add_bin_boundaries_to_x_all_centres_coincide_with_start_special_case_input
        )
        x_result, weights_result = calibration_curve._add_bin_boundaries_to_x(
            x, weights
        )
        x_expected, weights_expected = (
            _add_bin_boundaries_to_x_all_centres_coincide_with_start_special_case_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_add_bin_boundaries_to_x returned {x_result} instead of {x_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_add_bin_boundaries_to_x returned {weights_result} instead of {weights_expected}"

    def test_add_bin_boundaries_to_x_some_centres_coincide_with_start_special(
        self,
        continuous_calibration_curve_input,
        _add_bin_boundaries_to_x_some_centres_coincide_with_start_special_case_input,
        _add_bin_boundaries_to_x_some_centres_coincide_with_start_special_case_expected,
    ):
        """
        While we need continuous_calibration_curve_input to initialise the CalibrationCurve object and calculate self.bin_boundaries,
        we will be artificially adjusting the weights in _add_bin_boundaries_to_x_normal_input to test if our method behaves as desired
        (the weights are not equal to those calculated for continuous_calibration_curve_input and don't even need to add up to 1).
        """
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, weights = (
            _add_bin_boundaries_to_x_some_centres_coincide_with_start_special_case_input
        )
        x_result, weights_result = calibration_curve._add_bin_boundaries_to_x(
            x, weights
        )
        x_expected, weights_expected = (
            _add_bin_boundaries_to_x_some_centres_coincide_with_start_special_case_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_add_bin_boundaries_to_x returned {x_result} instead of {x_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_add_bin_boundaries_to_x returned {weights_result} instead of {weights_expected}"

    def test_get_y_at_bin_centre(
        self, _get_y_at_bin_centre_input, _get_y_at_bin_centre_expected
    ):
        x, x_new, y, i = _get_y_at_bin_centre_input
        y_result = CalibrationCurve._get_y_at_bin_centre(x, x_new, y, i)
        y_expected = _get_y_at_bin_centre_expected
        assert (
            y_result == y_expected
        ), f"_get_y_at_bin_centre returned {y_result} instead of {y_expected}"

    def test_get_y_at_bin_boundary(
        self, _get_y_at_bin_boundary_input, _get_y_at_bin_boundary_expected
    ):
        x, x_new, y, i = _get_y_at_bin_boundary_input
        y_result = CalibrationCurve._get_y_at_bin_boundary(x, x_new, y, i)
        y_expected = _get_y_at_bin_boundary_expected
        assert np.isclose(
            y_result, y_expected
        ), f"_get_y_at_bin_boundary returned {y_result} instead of {y_expected}"

    def test_get_y_at_bin_boundary_bad_case(
        self,
        _get_y_at_bin_boundary_bad_case_input,
        _get_y_at_bin_boundary_bad_case_expected,
    ):
        x, x_new, y, i = _get_y_at_bin_boundary_bad_case_input
        y_result = CalibrationCurve._get_y_at_bin_boundary(x, x_new, y, i)
        y_expected = _get_y_at_bin_boundary_bad_case_expected
        assert (
            y_result == y_expected
        ), f"_get_y_at_bin_boundary returned {y_result} instead of {y_expected}"

    def test_generate_x_y_with_bin_boundaries_normal(
        self,
        continuous_calibration_curve_input,
        _generate_x_y_with_bin_boundaries_normal_input,
        _generate_x_y_with_bin_boundaries_normal_expected,
    ):
        """Once again we need an input to initialise the CalibrationCurve object with appropriate boundaries,
        but then we will be artificially modifying the weights and x and y values to test our method.

        Args:
            _generate_x_y_with_bin_boundaries_normal_input (_type_): _description_
            _generate_x_y_with_bin_boundaries_normal_expected (_type_): _description_
        """
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, y, weights = _generate_x_y_with_bin_boundaries_normal_input
        x_result, y_result, weights_result = (
            calibration_curve._generate_x_y_with_bin_boundaries(x, y, weights)
        )
        x_expected, y_expected, weights_expected = (
            _generate_x_y_with_bin_boundaries_normal_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_generate_x_y_with_bin_boundaries returned {x_result} instead of {x_expected}"
        assert np.allclose(
            y_result, y_expected
        ), f"_generate_x_y_with_bin_boundaries returned {y_result} instead of {y_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_generate_x_y_with_bin_boundaries returned {weights_result} instead of {weights_expected}"

    def test_generate_x_y_with_bin_boundaries_two_values(
        self,
        continuous_calibration_curve_input,
        _generate_x_y_with_bin_boundaries_two_values_input,
        _generate_x_y_with_bin_boundaries_two_values_expected,
    ):
        """Once again we need an input to initialise the CalibrationCurve object with appropriate boundaries,
        but then we will be artificially modifying the weights and x and y values to test our method.

        Args:
            _generate_x_y_with_bin_boundaries_normal_input (_type_): _description_
            _generate_x_y_with_bin_boundaries_normal_expected (_type_): _description_
        """
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        x, y, weights = _generate_x_y_with_bin_boundaries_two_values_input
        x_result, y_result, weights_result = (
            calibration_curve._generate_x_y_with_bin_boundaries(x, y, weights)
        )
        x_expected, y_expected, weights_expected = (
            _generate_x_y_with_bin_boundaries_two_values_expected
        )
        assert np.allclose(
            x_result, x_expected
        ), f"_generate_x_y_with_bin_boundaries returned {x_result} instead of {x_expected}"
        assert np.allclose(
            y_result, y_expected
        ), f"_generate_x_y_with_bin_boundaries returned {y_result} instead of {y_expected}"
        assert np.allclose(
            weights_result, weights_expected
        ), f"_generate_x_y_with_bin_boundaries returned {weights_result} instead of {weights_expected}"

    def test_get_density_based_line_collection(self, 
                                            continuous_calibration_curve_input, 
                                            continuous_calibration_curve_generate_x_y_expected, 
                                            _get_density_based_line_collection_segment_start_boundaries, 
                                            _get_density_based_line_collection_segment_end_boundaries, 
                                            _get_density_based_line_collection_segment_weights):
        """
        Rather than generate an expected LineCollection object, it is easier to ensure the segments are as we expect them to be in terms of (half-) bin boundaries and associated weights
        """
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        [x], [y], [weights] = continuous_calibration_curve_generate_x_y_expected
        normalizer = LogNorm(vmin=0.01, vmax=1)
        result = calibration_curve._get_density_based_line_collection(x, y, weights, normalizer)

        segments = result.get_segments()
        segment_start_boundaries_result = [segment[0][0] for segment in segments]
        segment_end_boundaries_result = [segment[1][0] for segment in segments]
        segment_weights_result = result.get_array()

        segment_start_boundaries_expected = _get_density_based_line_collection_segment_start_boundaries
        segment_end_boundaries_expected = _get_density_based_line_collection_segment_end_boundaries
        segment_weights_expected = _get_density_based_line_collection_segment_weights
        
        assert type(result) == LineCollection, f'_get_density_based_line_collection() failed to return LineCollection object.'
        assert np.allclose(segment_start_boundaries_result, segment_start_boundaries_expected), f'Segment start boundaries different. Expected {segment_start_boundaries_expected}, got {segment_start_boundaries_result}'
        assert np.allclose(segment_end_boundaries_result, segment_end_boundaries_expected), f'Segment end boundaries different. Expected {segment_end_boundaries_expected}, got {segment_end_boundaries_result}'
        assert np.allclose(segment_weights_result, segment_weights_expected), f'Segment weights different. Expected {segment_weights_expected}, got {segment_weights_result}'                

    def test_configure_show_density_settings_normal(self, continuous_calibration_curve_input):
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)        
        show_density = True
        normalization_type = 'log'
        vmin, vmax = 0.01, 1
        show_density_result, normalizer_result = calibration_curve._configure_show_density_settings(show_density, normalization_type, vmin, vmax)        
        assert show_density_result == True, f'Failed to return correct show_density. Got {show_density_result} instead of {True}.'
        assert type(normalizer_result) == LogNorm, f'Failed to return correct normalizer. Got {type(normalizer_result)} instead of {LogNorm}.'

    def test_configure_show_density_settings_bad_normalizer(self, continuous_calibration_curve_input):
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)        
        show_density = True
        normalization_type = 'power'
        vmin, vmax = 0.01, 1
        with pytest.raises(ValueError):
            show_density_result, normalizer_result = calibration_curve._configure_show_density_settings(show_density, normalization_type, vmin, vmax)
    
    def test_configure_show_density_settings_bad_vmin_vmax(self, continuous_calibration_curve_input):
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)        
        show_density = True
        normalization_type = 'linear'
        vmin, vmax = 2, -2
        with pytest.raises(ValueError):
            show_density_result, normalizer_result = calibration_curve._configure_show_density_settings(show_density, normalization_type, vmin, vmax)

    def test_configure_show_density_settings_method_is_frequency(self, continuous_calibration_curve_input):
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method='frequency')        
        show_density = True
        normalization_type = 'log'
        vmin, vmax = 0.1, 1
        with pytest.warns(UserWarning):
            show_density_result, normalizer_result = calibration_curve._configure_show_density_settings(show_density, normalization_type, vmin, vmax)
        assert show_density_result == False, f'Failed to set show_density=False.'
        assert type(normalizer_result) == LogNorm, f'Failed to return correct normalizer. Got {normalizer_result} instead of {LogNorm}'

    @pytest.mark.mpl_image_compare
    def test_plot(self, continuous_calibration_curve_input):
        y_pred, y_true, num_bins, method = continuous_calibration_curve_input
        calibration_curve = CalibrationCurve(y_pred, y_true, num_bins, method)
        fig_result, ax_result = calibration_curve.plot()
        return fig_result 
