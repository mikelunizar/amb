import numpy as np
import torch


def rmse_inf(data_ground_truth, data_predicted):
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
    # Expand dimensions if necessary
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

    # Calculate squared error
    se = (x - y) ** 2
    # Mean across the number of variables
    se = np.mean(se, axis=-1)
    # Infinity norm calculation
    se_inf = []
    for i in range(x.shape[0]):
        x_snap = x[i]
        se_snap = se[i]
        infinite_norm_se = np.max(x_snap**2)
        se_inf.append(se_snap / infinite_norm_se)
    # Cast into array
    se_inf = np.array(se_inf)
    # Mean across all nodes
    mse_inf = np.mean(se_inf)
    # Square root of the mean squared error
    rmse_inf = np.sqrt(mse_inf)

    return rmse_inf


def rmse_inf_tensor(data_ground_truth, data_predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) normalized by the infinity norm.
    Args:
        data_ground_truth (torch.Tensor): Ground truth data with shape (snapshot, nodes, variable).
                                           In case of (nodes, variable), it expands snapshot dim to 1.
        data_predicted (torch.Tensor): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variable), it expands snapshot dim to 1.
    Returns:
        float: RMSE normalized by the infinity norm.
    """
    if data_ground_truth.ndim == 2:
        data_ground_truth = data_ground_truth[None, :, :]
        data_predicted = data_predicted[None, :, :]
    # squared error
    se = (data_ground_truth - data_predicted)**2
    # mean across number of variables
    se = torch.mean(se, axis=-1)
    # inifnite norm
    se_inf = []
    for i in range(data_ground_truth.shape[0]):
        x_snap = data_ground_truth[i]
        se_snap = se[i]
        infinite_norm_se = torch.max((x_snap**2))
        se_inf.append(se_snap / infinite_norm_se)
    # se_inf = torch.array(se_inf)
    se_inf = torch.cat(se_inf, dim=0)
    # mean across all nodes
    mse_inf = torch.mean(se_inf)
    # squared root of the mse
    rmse_inf = torch.sqrt(mse_inf)

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
    # Calculate squared root of the mean squared error
    rmse = np.sqrt(mse)

    return rmse
