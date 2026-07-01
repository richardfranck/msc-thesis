import numpy as np

from scripts.shape_uncertainty.shape_extraction.shape_summary import merge_adjacent

def _compute_true_curve_amplitude(size, g, d, rho, T, t_star):
    """Compute exact peak-to-peak amplitude of the convex Wilkerson curve on [0, T].

    Since y is strictly convex, it has at most one interior minimum at t_star.
    If t_star lies in (0, T) the amplitude is the larger endpoint value minus the
    value at the minimum; otherwise the curve is monotone on [0, T] and the
    amplitude is the difference between the two endpoint values. The amplitude is
    used to convert the relative slope/curvature thresholds into absolute ones,
    consistently with the shape extractor.

    Args:
        size: scalar, initial tumour size S (the multiplicative scale, y(0)=size).
        g: scalar, growth rate of the resistant fraction.
        d: scalar, decay rate of the sensitive fraction.
        rho: scalar, treatment-sensitive fraction in (0, 1).
        T: scalar, time horizon so the curve is considered on [0, T].
        t_star: scalar, location of the interior minimum (root of y'); used only
            to decide whether the minimum falls within (0, T).

    Returns:
        A: scalar, the peak-to-peak amplitude max(y) - min(y) over [0, T].
    """
    # Step 1: Get a np.array of dimension (D,) of lists of candidate points.
    #           The candidate points are both endpoints, plus the interior minimum if applicable.
    ts = [0.0, T]
    if 0.0 < t_star < T:
        ts.append(t_star)

    # Step 2: Compute the Wilkerson function value at all canididate points. 
    values = [size * (rho * np.exp(-d * t) + (1 - rho) * np.exp(g * t)) for t in ts]

    # Step 3: Compute the amplitudes for the given sample
    A = max(values) - min(values) # scalar

    return A

def _get_analytic_piece_flags(size, g, d, rho, T, t_star, upsilon_1):
    """Flag whether each monotone piece of the convex curve is slope-flat.

    The Wilkerson curve is strictly convex (y'' > 0), so y' is strictly
    increasing and each piece has the steepest slope at its outer
    endpoint. 
    That is, the decreasing piece [0, t_star] is steepest at t = 0, and
    the increasing piece [t_star, T] is steepest at t = T. 
    A piece is slope-flat when its steepest |y'| falls below the 
    absolute threshold upsilon_1, matching the piece-level flat rule
    specfied in shape_summary.py

    Args:
        size, g, d, rho: scalar Wilkerson parameters for one individual.
        T: scalar time horizon.
        t_star: scalar location of the interior minimum (root of y').
        upsilon_1: scalar absolute slope threshold (upsilon_rel_1 * A / T).

    Returns:
        (dec_flat, inc_flat): tuple of booleans. 
          - dec_flat is True if the decreasing piece [0, t_star] is 
            slope-flat (only meaningful when t_star > 0); 
          - inc_flat is True if the increasing piece [t_star, T] is 
            slope-flat (only meaningful when t_star < T).
          - For a monotone curve (t_star outside (0, T)) the single
            existing piece's flag is the relevant one.
    """
    # Step 1: Define. helper to compute the trajectory slope at a given t
    def y_prime(t):
        return size * (-rho * d * np.exp(-d * t) + (1.0 - rho) * g * np.exp(g * t))

    # Step 2: Compute the steepest absolute slope of each piece is at its outer endpoint
    # The descending piece is steepest at t=0
    dec_flat = abs(y_prime(0.0)) < upsilon_1 # True or False 
    #  The ascending piece steepest at t=T
    inc_flat = abs(y_prime(T)) < upsilon_1 # True or False 

    return dec_flat, inc_flat

def get_analytic_shape_summary(dataset, upsilon_rel_1, upsilon_rel_2):
    """Compute the Wilkerson ground-truth shape summary from analytic derivatives.

    The second derivative of the Wilkerson trajectory is strictly positive,
    y''(t) > 0, provided S > 0 and 0 <= rho <= 1, since
            y(t)   = S * [rho * exp(-d * t) + (1 - rho) * exp(g * t)],
            y'(t)  = S * [-rho * d * exp(-d * t) + (1 - rho) * g * exp(g * t)],
            y''(t) = S * [rho * d^2 * exp(-d * t) + (1 - rho) * g^2 * exp(g * t)],
    and y'' is a sum of non-negative terms. The curve is therefore strictly
    convex, so to assign shape states (convex_increasing or convex_decreasing)
    we only need to determine where the first derivative changes sign.

    A strictly convex function has one stationary point (a minimum) at
            t* = ln( rho * d / ((1 - rho) * g) ) / (d + g).
    If t* lies inside of (0, T), the trajectory has one minimum, which
    splits the trajectory into convex_decreasing then convex_increasing
    Otherwise, the trajectory is monotone on [0, T].

    Because the shape extractor (in shape_summary.py) uses relative thresholds 
    (a slope below upsilon_1 or a curvature below upsilon_2 is classified as 
    flat), the ground-truth shape here is classified with the same relative 
    thresholds, so that the estimate and the ground truth are expressed 
    in the same vocabulary.

    Args:
        dataset: An instance of the SimulatedDataset class. Reads the
            per-individual true parameters (g, d, rho), initial size, and
            horizon T.
        upsilon_rel_1 (float): scalar relative slope significance threshold.
        upsilon_rel_2 (float): scalar relative curvature significance threshold.

    Returns:
        shape_summaries: A length-D list of analytically derived shape
            summaries (format: [(state, start_time), ...]), one per sample.
    """
    # Step 1.1: Get per individual true parameters
    g = dataset.params["g"]      # (D,)
    d = dataset.params["d"]
    rho = dataset.params["rho"]

    # Step 1.2: Get the time horizonm, initial tumour size and sample size
    T = dataset.T
    size = dataset.X["size"]     # (D,)
    D = dataset.D

    # Step 2: Compute the candidate minimum location
    # The candidate minimum is at:
    #  y'(t)  = 0 
    #       0 = S * [-rho * d * exp(-d * t) + (1 - rho) * g * exp(g * t)]
    # exp([d+g] * t)] = d*rho / g*(1 - rho)  
    #          t_cand = ln[d*rho / g*(1 - rho)]/ (d+g)
    t_star = np.log((d * rho) / (g * (1.0 - rho))) / (d + g)   # (D,)

    # Step 3: Obtain per-individual shape summary
    shape_summaries = []
    for i in range(D):
        ti = t_star[i]

        # Absolute slope threshold from the exact amplitude, and piece-flat flags
        A_i = _compute_true_curve_amplitude(size[i], g[i], d[i], rho[i], T, ti)
        upsilon_1 = upsilon_rel_1 * A_i / T
        dec_flat, inc_flat = _get_analytic_piece_flags(
            size[i], g[i], d[i], rho[i], T, ti, upsilon_1
        )

        # Case 1: Minimum is at or before the start so increasing throughout [0, T]
        if ti <= 0.0:
            # Increasing throughout
            summary = [("convex_increasing", 0.0)]
            # Overwrite summary if inc_flat is True
            if inc_flat:
                summary = [("constant", 0.0)]

        # Case 2: Minimum is at or after the end so decreasing throughout [0, T]
        elif ti >= T:
            # Decreasing throughout
            summary = [("convex_decreasing", 0.0)]
            # Overwrite summary if dec_flat is True
            if dec_flat:
                summary = [("constant", 0.0)]

        # Case 3: Minimum is in (0, T) so decreasing then increasing. 
        else:
            # Interior minimum
            left_piece = "constant" if dec_flat else "convex_decreasing"
            right_piece = "constant" if inc_flat else "convex_increasing"
            summary = [(left_piece, 0.0), (right_piece, ti)]

        shape_summaries.append(summary)

    # Step 4: Collapse any adjacent duplicates
    #       If e.g. constant, constant this is collapesed to constant constant)
    shape_summaries = [merge_adjacent(summary) for summary in shape_summaries]

    return shape_summaries


def estimate_true_random_effects(covariate_values, hyperparams, model, T,
                           n_mc=20000, rng=None, n_dense=2000):
    """Estimate the ground-truth aleatoric covariance Sigma at a given
        covariate vector (NORMAL random effects).

    Estimate the true aleatoric covariance Sigma at a single covariate vector
    (covariate_values). Sigma is the unexplained between-individual variation in
    the spline coefficients induced by the unobserved heterogeneity (z_g, z_d) at
    that covariate point. We draw a large number (n_mc) of latent pairs (z_g, z_d),
    holding the covariates fixed; each draw yields one noise-free Wilkerson curve.
    Each curve is then represented in the model's spline coefficient space, and 
    Sigma is the empirical covariance of the resulting coefficient vectors.

    The representation uses the SAME spline basis / coefficient space the model.
    Estimation is then performed under idealised conditions:
        - noise-free curves,
        - a large number of dense observations,
    The resulting observed coefficient spread is then pure aleatoric variation the
    latents induce at this covariate point -- the ground-truth Sigma. We can
    compare the model's estimated Sigma against this ground truth.

    Two objects are returned: 
    1. The empirical sample of coefficient vectors (distributional ground truth) 
    2. Its covariance Sigma

    Note that this function returns Sigma(x). Considering whether this varies
    will likely show that the model assumption of homoscedasticity of Sigma is
    false. We are interested in testing whether this is an issue. Furthermore
    we are interested in exploring whether the assumptions about the distributional
    form of Sigma is an issue. 

    Args:
        covariate_values: dict of scalar covariate values (size, age, weight,
            dosage); the covariate point at which the random effects are evaluated.
        hyperparams: dict of generative hyperparameters (for compute_parameters).
        model: the trained model providing the spline coefficient representation.
        T: scalar time horizon.
        n_mc (int): number of latent draws (z_g, z_d) in the Monte-Carlo estimate.
        rng: np.random.Generator for the latent draws.
        n_dense (int): number of dense, noise-free observation times per curve.

    Returns:
        coeffs: np.ndarray of shape (n_mc, B); the sampled spline coefficient
            vectors, i.e. the assumption-free distributional ground truth.
        Sigma: np.ndarray of shape (B, B); the empirical covariance of coeffs.
    """
    pass


# The following functions will probably want to be moved to evaluation:

def evaluate_random_effects_at_point(coeffs, model, covariate_values):
    """Characterise the shape of the ground-truth random-effects distribution.

    Diagnoses the distributional SHAPE of the Monte-Carlo coefficient sample
    returned by estimate_true_random_effects, to judge whether a given modelling
    assumption (normal / Gaussian mixture / non-parametric) is appropriate. Because
    the covariance alone is shape-blind, this inspects higher-order structure of the
    sample -- e.g. unimodality vs. multimodality, symmetry/skewness, tail behaviour,
    and goodness-of-fit of the assumed family -- and may compare the empirical
    distribution against the distribution the model assumes.

    This is where the question "is the normality assumption reasonable, or are GMM /
    NPEB needed?" is actually answered, on the assumption-free ground-truth sample.

    PLACEHOLDER: not implemented yet; to be specified alongside the random-effects
    model variants. Listed here to fix the interface.

    Args:
        coeffs: np.ndarray of shape (n_mc, B); the ground-truth coefficient sample
            from estimate_true_random_effects.
        model: optional; the model whose assumed random-effects family is being
            checked against the empirical sample.

    Returns:
        A characterisation of the sample's distributional shape (e.g. modality,
        skewness/kurtosis, and goodness-of-fit of the assumed family). Exact return
        type to be fixed when the random-effects variants are specified.
    """
    pass


def evaluate_random_effects_homoscedasticity(estimates):
    pass