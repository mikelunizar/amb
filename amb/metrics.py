import numpy as np
import torch


def rrmse_inf(data_ground_truth, data_predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) normalized by the infinity norm.

    Args:
        data_ground_truth (numpy.ndarray): Ground truth data with shape (snapshot, nodes, variable).
                                           In case of (nodes, variable), it expands snapshot dim to 1.
        data_predicted (numpy.ndarray): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variable), it expands snapshot dim to 1.

    Returns:
        float: RMSE normalized by the infinity norm.
    """

    x, y = data_ground_truth, data_predicted

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)
        y = np.asarray(y)

    # Infinity norm calculation
    se_inf = []
    for i in range(x.shape[0]):
        x_snap = x[i]
        error = x[i] - y[i]
        l2_norm_se = np.linalg.norm(error, ord=2) ** 2 / x.shape[-1]
        infinite_norm_se = np.linalg.norm(x_snap, ord=np.inf) ** 2
        se_inf.append(l2_norm_se / infinite_norm_se)
    # Cast into array
    mse_inf = np.mean(np.array(se_inf))
    # Square root of the mean squared error
    rmse_inf = np.sqrt(mse_inf)

    return rmse_inf


def rmse(x, y):
    """
    Calculates the Root Mean Squared Error (RMSE) between two arrays x and y.

    Parameters:
    x (array-like): Array of true values.
    y (array-like): Array of predicted values.

    Returns:
    float: The Root Mean Squared Error (RMSE) between x and y.
    """
    # Calculate squared error
    se = (x - y) ** 2
    # Calculate mean squared error
    mse = np.mean(se)

    return np.sqrt(mse)
