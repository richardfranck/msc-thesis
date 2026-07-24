import numpy as np
import torch

def negative_log_likelihood(model, batch):
    """Compute the average (per-individual) negative marginal log-likelihood over the batch.

    We compute the average negative log likelihood where we average
    over individuals (not observations). This is key so in the overall
    loss function the the likelihood/penalty balance is invariant to the
    number of individuals D and to the mini-batch size.
    
    When every individual in the batch has the same observation count N, 
    we use the vectorised model.batch_log_likelihood. When the counts
    differ, the per-individual model.marginal_log_likelihood loop is used
    instead.

    Args:
        model: a RandomEffectsModel (e.g. GaussianModel); supplies
            marginal_log_likelihood(y_i, Phi_i, x_i) or batch_log_likelihood
        batch: a Batch of prepared training data.

    Returns:
        scalar torch.Tensor; the average negative marginal log-likelihood.
    """
    # Case 1: If all individuals in the batch have the same observation count
    if len({y.shape[0] for y in batch.y_list}) == 1:
        return -model.batch_log_likelihood(batch).sum() / batch.D

    # Case 2: If individuals in the batch do not have the same observation count
    total = 0.0
    for i in range(batch.D):
        # Subtract each individual's marginal log-likelihood (sum of NLLs over the batch)
        total = total - model.marginal_log_likelihood(
            batch.y_list[i], batch.Phi_list[i], batch.x_list[i])
    average = total / batch.D
    return average


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