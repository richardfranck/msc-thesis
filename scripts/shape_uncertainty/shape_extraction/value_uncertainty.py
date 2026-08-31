import numpy as np

def _build_distance_matrix(values):
    """Compute the matrix of absolute value-space distances across a cloud of trajectories.

    This function is the value-space counterpart of _build_distance_matrix 
    in shape_uncertainty.py which computes the distance over the shape-state vocabulary.
    Here, we compute the matrix of absolute value-space distances across a 
    cloud of trajectories. Entry (m, m') is |values[m] - values[m']|, the 
    distance between the values two cloud members predict at one point in time. 

    Args:
        values: np.ndarray of shape (M,); the value every cloud member predicts
            at one time point, in original y units.

    Returns:
        np.ndarray of shape (M, M) and dtype float; symmetric, with a zero
            diagonal.
    """
    # M = values.shape[0]
    # distance_matrix = np.zeros((M, M), dtype=float)
    # for m, value_a in enumerate(values):
    #     for m_prime, value_b in enumerate(values):
    #         distance_matrix[m, m_prime] = abs(value_a - value_b)
    # return distance_matrix
    
    # Compute |values[m] - values[m']| pair
    return np.abs(values[:, None] - values[None, :])


def _sum_pairwise_distances(curves):
    """Sum the value-space distances over every ordered pair, per time.

    This is the implementation of the double sum needed for Q_k below.

    Args:
        curves: np.ndarray of shape (M, N); the trajectories of M cloud members
            on a shared grid of N times, in original y units.

    Returns:
        np.ndarray of shape (N,) and dtype float; the pair sum at each time, in
            original y units.
    """
    nr_times = curves.shape[1]
    total_pairwise_distance = np.zeros(nr_times, dtype=float)

    # One distance matrix per time point
    for n in range(nr_times):
        distance_matrix = _build_distance_matrix(curves[:, n])
        total_pairwise_distance[n] = distance_matrix.sum()

    return total_pairwise_distance


def _compute_pointwise_uncertainties(curves):
    """Return the value-space uncertainty Q(t) at every time on the grid.

    This is the value-space counterpart of _compute_region_uncertainties in 
    shape_uncertainty.py, onlt that here we compute the metric at every point
    in time. Each pointwise uncertainty is computed as

        Q(t) = 1 / (M (M - 1)) * sum_m sum_m' |y_m(t) - y_m'(t)|,

    which is the Q_k of shape_uncertainty with the absolute value-space distance
    in place of the shape distance, and a time point in place of a region.

    Args:
        curves: np.ndarray of shape (M, N); the trajectories of M cloud members
            on a shared grid of N times, in original y units.

    Returns:
        np.ndarray of shape (N,) and dtype float; the Q(t), in original y units.
    """
    M = curves.shape[0]
    return _sum_pairwise_distances(curves) / (M * (M - 1))


def _normalise_by_amplitude(uncertainty, amplitude):
    """Express a profile in y units as a fraction of the trajectory amplitude.

    Q(t) is a mean distance between trajectories and is in y units. 
    Since shape_uncertainty.py's Q_k is unitless, we divided the above derived
    Q(t) by the trajectory's peak-to-peak amplitude A. 
    This way Q(t) is expressed as a fraction of the amplitude.

        V(t) = Q(t) / A.

    A cloud whose members disagree by a tenth of the amplitude scores V = 0.1

    Args:
        uncertainty: np.ndarray of shape (N,); a profile in original y units.
        amplitude: float; the peak-to-peak amplitude A of the reference
            trajectory, in original y units.

    Returns:
        np.ndarray of shape (N,) and dtype float; the dimensionless V(t).
    """
    if not amplitude > 0.0:
        raise ValueError("The amplitude must be positive.")

    return np.clip(uncertainty, 0.0, None) / amplitude # Clipping due to floating point precision errors. 


def _average_over_horizon(value_uncertainty):
    """Reduce a profile to one number by averaging it over the horizon.

    This computes V, which is the value-space counterpart to U. 
    U is computed as the duration weighted average of the Q_k values,
    normalised by T, because the regions have unequal lengths. 
    V is computed as the simple average as it is computed from an evenly    
    spaced grid
                 V = 1 / N * sum_n V(t_n),
    Args:
        value_uncertainty: np.ndarray of shape (N,); the profile to reduce,
            sampled on an evenly spaced grid spanning [0, T].

    Returns:
        float; the mean of the profile over the horizon.
    """
    return float(np.mean(value_uncertainty))


def compute_value_uncertainty(curves, times, reference):
    """Compute the value-space uncertainty of a cloud of trajectories over [0, T].

    Value-space uncertainty is intended as the value-space counterpart of shape uncertainty.
    We therefore compute it analogously to shape uncertainty. Specifically, as
    the average pairwise distance across a cloud of M equally plausible 
    trajectories, expressed as a fraction of the trajectory amplitude. 

    Step 1: At each time on the grid, build the matrix of value-space distances
            between members and sum it into a pointwise uncertainty Q(t).
    Step 2: Divide by the amplitude A of the reference trajectory, giving a
            dimensionless profile V(t).
    Step 3: Compute the value-space uncertainty as the mean of V(t) over the grid.

    Note:
        - The cloud of trajectories can be an ensemble or aleatoric draws.
        - Where the cloud is nested, decompose_value_uncertainty below  
        splits the result into its two parts.

    Args:
        curves: np.ndarray of shape (M, N); one individual's cloud of M
            trajectories, evaluated on a shared grid of N times and in original
            y units.
        times: np.ndarray of shape (N,); the evaluation grid on [0, T].
        reference: int; index for the medoid trajectory whose shape summary is
            reported alongside it.

    Returns:
        tuple (V, profile) where
            - V is a float, zero for a cloud whose members agree exactly, and
              otherwise the fraction of the amplitude the cloud spans on average
            - profile is a dict with
            profile = {
                'times':                np.ndarray (N,)   -- the evaluation grid
                'absolute_uncertainty': np.ndarray (N,)   -- Q(t), y units
                'value_uncertainty':    np.ndarray (N,)   -- V(t), dimensionless
                'curves':               np.ndarray (M, N) -- the cloud itself,
                                        kept so a nested cloud can be decomposed
                'reference':            int               -- the row A came from
                'amplitude':            float             -- the A divided by
            }
    """
    # Step 1: Compute the pointwise uncertainties Q(t)
    absolute_uncertainty = _compute_pointwise_uncertainties(curves)

    # Step 2: Express them as a fraction of the amplitude
    amplitude = float(np.ptp(curves[reference]))
    value_uncertainty = _normalise_by_amplitude(absolute_uncertainty, amplitude)

    # Step 3: Compute the cloud's value-space uncertainty
    V = _average_over_horizon(value_uncertainty)

    # Step 4: Construct the uncertainty profile
    profile = {
        'times': times,
        'absolute_uncertainty': absolute_uncertainty,
        'value_uncertainty': value_uncertainty,
        'curves': curves,
        'reference': int(reference),
        'amplitude': amplitude,
    }

    return V, profile


def _group_pair_sums(curves, group_ids):
    """Sum the distances within each group of a nested cloud, per time.

    Args:
        curves: np.ndarray of shape (E * S, N); the pooled cloud.
        group_ids: np.ndarray of shape (E * S,) of int; the member behind each
            row of curves.

    Returns:
        np.ndarray of shape (E, N) and dtype float; row e is the pair sum over
            the S draws member e contributed, in original y units.
    """
    pass


def decompose_value_uncertainty(profile, group_ids):
    """Split one cloud's value-space uncertainty into aleatoric and epistemic parts.

    This is the value-space counterpart of decompose_shape_uncertainty.
    We obtin a cloud buil by nesting S aleatoric draws inside each of 
    E ensemble members. Writing
    g(t) = sum_{m, m' in g} |y_m(t) - y_m'(t)| for the pair sum over a group
    g, and g_e for the draws member e contributed,

        Q^ale(t)   = mean_e <g_e>(t) / (S (S - 1))
        Q^cross(t) = (<all>(t) - sum_e <g_e>(t)) / (E (E - 1) S^2)
        Q^epi(t)   = Q^cross(t) - Q^ale(t)

    Q^ale averages pairs drawn from the SAME member, which share that member's
    mean trajectory and differ only by the random effect, so it is pure
    aleatoric spread. Q^cross averages pairs from DIFFERENT members, which carry
    both the model gap and that same spread; subtracting cancels the spread and
    leaves model disagreement alone.

    The subtraction is not merely plausible. Q^epi(t) is half the mean pairwise
    ENERGY DISTANCE between the members' predictive distributions at t, the
    energy distance between P and P' being

        D(P, P') = 2 E|X - X'| - E|X - X~| - E|X' - X'~|.

    The absolute distance is of negative type on the line, so D is non-negative
    and vanishes exactly when the two members agree in distribution. Both terms
    are unbiased U-statistics of their population counterparts, so no correction
    for finite S is needed: the aleatoric spread that inflates Q^cross is the
    same spread Q^ale measures, and it cancels in expectation. Sampling noise
    can still push the estimate slightly below zero where the epistemic part is
    genuinely nil, which _normalise_by_amplitude clips.

    Because no square root is taken anywhere, the components add in the units
    that are plotted: Q^ale + Q^epi = Q^cross by construction, and dividing
    through by A preserves it, so V^ale(t) + V^epi(t) = V^cross(t). A figure may
    stack the two parts directly.

    Against the POOLED uncertainty the identity is
    Q^tot = Q^ale + ((E - 1) S / (E S - 1)) Q^epi, since the pooled cloud counts
    within-member pairs alongside cross pairs. The factor is 0.990 at
    E = S = 100. It is pure pair counting and holds whatever the distance, which
    is why the same factor appears on the shape side.

    Args:
        profile: dict as returned by compute_value_uncertainty for the POOLED
            cloud, supplying "curves", "times" and "amplitude". Its curves are
            ordered to match group_ids.
        T: float, the right endpoint of the forecasting horizon.
        group_ids: np.ndarray of shape (E * S,) of int; the member behind each
            cloud position, aligned with the rows of profile["curves"].

    Returns:
        dict with
            "V_aleatoric", "V_epistemic", "V_cross": floats; the components
                averaged over the horizon, on the same dimensionless scale as
                the V that compute_value_uncertainty returns.
            "value_uncertainty_aleatoric", "value_uncertainty_epistemic",
                "value_uncertainty_cross": np.ndarray (N,); the dimensionless
                profiles, the first two summing to the third.
            "reducible_fraction": float, V_epi / (V_ale + V_epi); the share of
                the spread that more data could remove.
    """
    pass
