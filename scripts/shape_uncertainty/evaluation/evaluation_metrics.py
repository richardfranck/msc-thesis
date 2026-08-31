import numpy as np

def compute_RMSE(predicted, reference):
    """Compute the root mean squared error (RMSE) 

    Compute the RMSE between all predicted and reference (noise-free/observed) 
    trajectories for all samples in a dataset. 

    Args:
        predicted: np.ndarray of shape (M,); predicted trajectory values.
        reference: np.ndarray of shape (M,); reference trajectory values.

    Returns:
        rmse (float): the root mean squared error over the M entries.
    """
    mse = np.mean((predicted - reference) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def compute_individual_RMSE(predicted, reference):
    """Compute the root mean squared error (RMSE) of each individual.

    The trajectories of different individuals may operate on different
    scales. An aggregate RMSE may hide poor fit by some individuals.
    This function therefore computes the RMSE of each individual based 
    on their own observation times. 
    For individual i with N_i observations,

        RMSE_i = sqrt( (1 / N_i) * sum_n (predicted_in - reference_in)^2 ).

    That is,`compute_RMSE` restricted to the residuals of a single 
    individual.

    Args:
        predicted: list of length D; element i is an np.ndarray of shape
            (N_i,) holding the predicted values for individual i.
        reference: list of length D; element i is an np.ndarray of shape (N_i,)
            holding the reference (noise-free or observed) values for individual i.

    Returns:
        np.ndarray of shape (D,); element i is the RMSE of individual i,
            in the order the individuals are held in the dataset.
    """
    return np.array([
        compute_RMSE(np.asarray(p), np.asarray(y))
        for p, y in zip(predicted, reference)
    ])

def compute_R_squared(predicted, reference):
    """Compute the R-squared value for the predicted trajectory.

    Recall that R^2 = 1 - RSS/TSS

    Args:
        predicted: np.ndarray of shape (M,); predicted trajectory values.
        reference: np.ndarray of shape (M,); reference (noise-free or observed) 
        trajectory values.

    Returns:
        r_squared (float): r_squared value. 
    """
    rss = np.sum((reference - predicted) ** 2)
    tss = np.sum((reference - reference.mean()) ** 2)
    r_squared = 1 - (rss/tss)
    return r_squared

def compute_individual_R_squared(predicted, reference):
    """Compute the R-squared value of each individual.

    The trajectories of different individuals may operate on different
    scales. An aggregate R-squared may hide poor fit by some individuals.
    This function therefore computes the R-squared of each individual based 
    on their own observation times. 
    For individual i with N_i observations,

        R^2_i = 1 - sum_n (reference_in - predicted_in)^2
                    / sum_n (reference_in - mean(reference_i))^2,

    with mean(reference_i) = (1 / N_i) * sum_n reference_in.

    Note:
    An individual whose referene trajectory is constant over its observation
    times R-squared is undefined due to a zero denominator. In that case, 
    we report np.nan, to prevent corrupting the computation of an average
    R-square taken over individuals.

    Args:
        predicted: list of length D; element i is an np.ndarray of shape
            (N_i,) holding the predicted values for individual i.
        reference: list of length D; element i is an np.ndarray of shape (N_i,)
            holding the reference (noise-free or observed) values for individual i.

    Returns:
        np.ndarray of shape (D,); element i is the R-squared of individual
            i, or np.nan where that individual's trajectory is constant.
    """
    r_squared = []
    for p, y in zip(predicted, reference):
        p, y = np.asarray(p), np.asarray(y)
        tss = np.sum((y - y.mean()) ** 2)
        r_squared.append(compute_R_squared(p, y) if tss > 0 else np.nan)
    return np.array(r_squared)
