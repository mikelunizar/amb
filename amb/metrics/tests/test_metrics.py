import numpy as np
import pytest
from amb.metrics import rrmse_inf, rmse

# Sample data for testing
data_ground_truth = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
data_predicted = np.array([[[0.9, 1.8], [2.9, 4.1]], [[5.1, 5.9], [7.2, 8.3]]])


@pytest.mark.parametrize(
    "data_ground_truth, data_predicted, expected_result",

    [
        (data_ground_truth, data_predicted, 0.020395982694686593),
    ])
def test_rrmse_inf(data_ground_truth, data_predicted, expected_result):
    result = rrmse_inf(data_ground_truth, data_predicted)
    assert np.isclose(result, expected_result)


@pytest.mark.parametrize(
    "data_ground_truth, data_predicted, expected_result",
    [
        (np.array([1, 2, 3, 4]), np.array([0.9, 2.1, 3.2, 3.9]), 0.13228756555322962),
    ])
def test_rmse(data_ground_truth, data_predicted, expected_result):
    result = rmse(data_ground_truth, data_predicted)
    assert np.isclose(result, expected_result)


if __name__ == "__main__":
    pytest.main()
