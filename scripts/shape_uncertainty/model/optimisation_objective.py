import numpy as np
import torch

def negative_log_likelihood(model, batch):
    """Compute the summed negative marginal log-likelihood over the batch.

    Args:
        model: a RandomEffectsModel (e.g. GaussianModel); supplies
            marginal_log_likelihood(y_i, Phi_i, x_i).
        batch: a Batch of prepared training data.

    Returns:
        scalar torch.Tensor; the total negative marginal log-likelihood.
    """
    total = 0.0
    for i in range(batch.D):
        # Subtract each individual's marginal log-likelihood (sum of NLLs over the batch)
        total = total - model.marginal_log_likelihood(
            batch.y_list[i], batch.Phi_list[i], batch.x_list[i])
    return total


def wiggle_penalty(model, Omega, X, lambda_mean, lambda_re):
    """Evaluate the model's two-component smoothness penalty.

    Args:
        model: a RandomEffectsModel
        Omega: torch.Tensor of shape (B, B); the spline penalty matrix
            (integral of phi''(t) phi''(t)^T dt).
        X: torch.Tensor of shape (D, M); the normalised batch covariates.
        lambda_mean: float, weight on the mean-trajectory roughness.
        lambda_re: float, weight on the random-effects roughness tr(Omega Sigma).

    Returns:
        scalar torch.Tensor; the total weighted smoothness penalty.
    """
    return model.penalty(Omega, X, lambda_mean, lambda_re)


def objective(model, batch, Omega, lambda_mean, lambda_re):
    """Compute the total training objective to minimise.

    Returns the summed negative marginal log-likelihood plus the weighted
    two-component wiggle penalty, i.e. L = NLL + penalty.

    Args:
        model: a RandomEffectsModel (e.g. GaussianModel).
        batch: a Batch of prepared training data.
        Omega: torch.Tensor of shape (B, B); the spline penalty matrix.
        lambda_mean: float, weight on the mean-trajectory roughness.
        lambda_re: float, weight on the random-effects roughness.

    Returns:
        scalar torch.Tensor; the total objective L(theta, Sigma, sigma^2).
    """
    return (negative_log_likelihood(model, batch)
            + wiggle_penalty(model, Omega, batch.X, lambda_mean, lambda_re))