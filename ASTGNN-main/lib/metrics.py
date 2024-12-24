# -*- coding:utf-8 -*-
import numpy as np

def masked_mape_np(y_true, y_pred, null_val=0, epsilon=1e-6, min_val=1e-2):
    """
    Masked MAPE calculation with handling for small values.
    :param y_true: true values
    :param y_pred: predicted values
    :param null_val: value to mask (usually np.nan or 0)
    :param epsilon: small number to avoid division by zero
    :param min_val: threshold to replace very small values of y_true to avoid large MAPE
    :return: masked MAPE value
    """
    # Masking for NaNs or null values in y_true
    mask = np.not_equal(y_true, null_val)
    
    # Avoid division by very small values in y_true
    y_true_safe = np.maximum(y_true, min_val)  # Replace small y_true values with a threshold (min_val)
    
    # Calculate the absolute percentage error
    mape = np.abs((y_pred - y_true_safe) / y_true_safe)  # Avoid division by zero
    
    # Apply the mask to the MAPE
    mape = np.nan_to_num(mask * mape)
    
    # Return the mean of MAPE, multiplied by 100 to get the percentage
    return np.mean(mape) * 100