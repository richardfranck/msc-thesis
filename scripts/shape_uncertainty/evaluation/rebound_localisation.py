
import numpy as np


def resample_profile_on_grid(profile, times):
    """Read a piecewise-constant shape uncertainty profile at given times.

    The profile is constant on each region of the transition grid, so each time
    is scored by the region containing it.

    Args:
        profile: dict; as compute_shape_uncertainty returns, carrying "regions"
            of shape (n, 2) and "region_uncertainty" of shape (n,).
        times: np.ndarray of shape (M,); the times to read, within [0, T].

    Returns:
        np.ndarray of shape (M,); the profile at those times.
    """
    regions = profile["regions"]
    region_uncertainty = profile["region_uncertainty"]

    # The region whose start is the last one at or before each queried time
    position = np.searchsorted(regions[:, 0], times, side="right") - 1
    position = np.clip(position, 0, len(region_uncertainty) - 1)

    return region_uncertainty[position]


def align_profile_on_offsets(values, times, t_star, offsets, T):
    """Re-express one profile as time since its own turning point.

    Args:
        values: np.ndarray of shape (M,); the profile on the grid.
        times: np.ndarray of shape (M,); that grid, over [0, T].
        t_star: float; the individual's own turning point.
        offsets: np.ndarray of shape (K,); the aligned axis, t - t*.
        T: float; the right endpoint of the horizon.

    Returns:
        np.ndarray of shape (K,); the profile at t* + offsets, np.nan at any
            offset falling outside [0, T].
    """
    aligned = np.full(len(offsets), np.nan)

    queried = t_star + offsets
    inside = (queried >= 0.0) & (queried <= T)
    aligned[inside] = np.interp(queried[inside], times, values)

    return aligned


def score_localisation_contrast(values, times, t_star, away):
    """Score how far a profile stands above itself at the turning point.

    Args:
        values: np.ndarray of shape (M,); the profile on the grid.
        times: np.ndarray of shape (M,); that grid.
        t_star: float; the individual's own turning point.
        away: float; how far from t* a time must lie to count as away from it.

    Returns:
        float; the profile at t* over its mean away from t*. Above one where it
            localises on the turn, np.nan where nothing lies away from t* or
            the profile vanishes there.
    """
    at_turn = float(np.interp(t_star, times, values))

    distant = np.abs(times - t_star) > away
    if not distant.any():
        return np.nan

    baseline = float(np.mean(values[distant]))
    if baseline <= 0.0:
        return np.nan

    return at_turn / baseline
