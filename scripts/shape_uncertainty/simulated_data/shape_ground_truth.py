import numpy as np
from scipy.optimize import brentq

from scripts.shape_uncertainty.shape_extraction.shape_summary import absolute_thresholds
from scripts.shape_uncertainty.shape_extraction.shape_summary import construct_regions
from scripts.shape_uncertainty.shape_extraction.shape_summary import classify_signs
from scripts.shape_uncertainty.shape_extraction.shape_summary import compute_episodes
from scripts.shape_uncertainty.shape_extraction.shape_summary import _absorb_lowest_scoring_episode
from scripts.shape_uncertainty.shape_extraction.shape_summary import merge_adjacent


def _evaluate_wilkerson(t, size, g, d, rho):
    """Evaluate the Wilkerson curve.

    The wilkerson curve is y(t) = S[rho e^{-dt} + (1 - rho) e^{gt}]. 

    Args:
        t: scalar or np.ndarray of evaluation times.
        size: scalar initial size S, the multiplicative scale, so that
            y(0) = size.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).

    Returns:
        np.ndarray or float: y(t), matching the shape of t.
    """
    return size * (rho * np.exp(-d * t) + (1.0 - rho) * np.exp(g * t))


def _evaluate_wilkerson_prime(t, size, g, d, rho):
    """Evaluate the first derivative of the Wilkerson curve.

    y'(t) = S[-rho d e^{-dt} + (1 - rho) g e^{gt}].

    Args:
        t: scalar or np.ndarray of evaluation times.
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).

    Returns:
        np.ndarray or float: y'(t), matching the shape of t.
    """
    return size * (-rho * d * np.exp(-d * t) + (1.0 - rho) * g * np.exp(g * t))


def _evaluate_wilkerson_double_prime(t, size, g, d, rho):
    """Evaluate the second derivative of the Wilkerson curve.

    y''(t) = S[rho d^2 e^{-dt} + (1 - rho) g^2 e^{gt}].

    Args:
        t: scalar or np.ndarray of evaluation times.
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).

    Returns:
        np.ndarray or float: y''(t), matching the shape of t.
    """
    return size * (rho * d**2 * np.exp(-d * t) + (1.0 - rho) * g**2 * np.exp(g * t))



def _compute_critical_point(g, d, rho):
    """Return the stationary point t* of the Wilkerson curve.

    The wilkerson curve is strictly convex, so y' has at most one root at
                          0 = y'(t)
                          0 = S[-rho d e^{-dt} + (1 - rho) g e^{gt}]
                          0 = + (1 - rho) g e^{gt} 
              rho d e^{-dt} = (1 - rho) g e^{gt} 
    rho d / ((1 - rho) g) ) = e^{(g+d) t} 
                         t* = ln( rho d / ((1 - rho) g) ) / (d + g).
    This function returns the root irrespective of whether or not it lies 
    in [0, T]

    Args:
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).

    Returns:
        float: The stationary point t*, or np.nan if no real stationary point
            exists.
    """
    # Step 1: Compute numerator and denominator the ln-component of the root.
    numerator = rho * d
    denominator = (1.0 - rho) * g

    # Step 2: Check whether the logarithm is well defined. 
    if denominator == 0.0 or numerator / denominator <= 0.0:
        return np.nan

    # Step 3: Cmpute the root. 
    return float(np.log(numerator / denominator) / (d + g))

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
        A: float, the peak-to-peak amplitude max(y) - min(y) over [0, T].
    """
    # Step 1: Get a np.array of dimension (D,) of lists of candidate points.
    #           The candidate points are both endpoints, plus the interior minimum if applicable.
    ts = [0.0, T]
    if 0.0 < t_star < T:
        ts.append(t_star)

    # Step 2: Compute the Wilkerson function value at all canididate points. 
    values = [_evaluate_wilkerson(t, size, g, d, rho) for t in ts]

    # Step 3: Compute the amplitudes for the given sample
    A = max(values) - min(values) # scalar

    return float(A)

def _compute_slope_flat_band(size, g, d, rho, T, upsilon_1):
    """Return the sub-interval of [0, T] on which the slope is negligible.

    This function utilises that because the wilkerson curve is strictly convex,
    y' is strictly increasing, so the set B1 = {t in [0, T] : |y'(t)| < upsilon_1}
    is a single interval. 
    The left endpoint is the solution of y'(t) = -upsilon_1. 
    The right endpoint is the solution of y'(t) = +upsilon_1. 
    We clip each to [0, T]. 
    We solve for the two solutions numerically. 

    Args:
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).
        T: scalar time horizon.
        upsilon_1: scalar absolute slope threshold, upsilon_rel_1 * A / T.

    Returns:
        tuple or None: (a, b) with 0 <= a <= b <= T giving the slope-flat band,
            or None if no point of [0, T] is slope flat.
    """
    # Step 1: Confirm that a slope-flat band can exists (requires upsilon>0)
    if upsilon_1 <= 0.0:
        return None

    # Step 2: Determine whether a slope flat band exists on [0, T]
    # The flat band is where  -upsilon_1 <  slope < +upsilon_1. If the slope is
    # less than  -upsilon_1 everywhere on [0, T] or larger +upsilon_1 everywhere
    # on [0, T], there is no slope flat band.
    # Since y'' > 0, y' is strictly increasing and the minimum slope is at t=0 and the maximum at t=T
    minimum_slope = _evaluate_wilkerson_prime(0.0, size, g, d, rho) # Minimum slope is reached at slope_start
    maximum_slope = _evaluate_wilkerson_prime(T, size, g, d, rho)  # Maximum slope is reached at slope_end
    if maximum_slope <= -upsilon_1 or minimum_slope >= upsilon_1: 
        return None

    # Step 3: Given that a slope flat band exists on [0, T], find its left boundary "a"
    if minimum_slope >= -upsilon_1:
        a = 0.0 
    else:
        a = brentq(lambda t: _evaluate_wilkerson_prime(t, size, g, d, rho) + upsilon_1, 0.0, T)

    # Step 4: Given that a slope flat band exists on [0, T], find its right boundary "b"
    if maximum_slope <= upsilon_1:
        b = T 
    else:
        b = brentq(lambda t: _evaluate_wilkerson_prime(t, size, g, d, rho)  - upsilon_1, 0.0, T)

    # Step 5: Return the endpoints of the slope flat bands as floats
    return (float(a), float(b))


def _compute_curvature_flat_band(size, g, d, rho, T, upsilon_2):
    """Return the sub-interval of [0, T] on which the curvature is negligible.

    This function utilises that because  econd, y''''(t) > 0, y'''(t) is strictly
    increasing and crosses zero at most once such that y''(t) is either monotone 
    or U-shaped on [0, T]. It follows that y''(t) crosses the positive threshold 
    value upsilon_2 at most twice such that any region 
    B2 = {t in [0, T] : y''(t) < upsilon_2} where the curvature is flat is one 
    single continuous block of time. 
    The left endpoint is the solution of y''(t) = -upsilon_2. 
    The right endpoint is the solution of y''(t) = +upsilon_2. 
    We clip each to [0, T]. 
   
    Args:
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).
        T: scalar time horizon.
        upsilon_2: scalar absolute curvature threshold,
            2 * upsilon_rel_2 * A / T^2.

    Returns:
        tuple or None: (a, b) with 0 <= a <= b <= T giving the curvature-flat
            band, or None if no point of [0, T] is curvature flat.
    """
    # Step 1: Confirm that a curvature-flat band can exists (requires upsilon2>0)
    if upsilon_2 <= 0.0:
        return None

    # Step 2: Determine whether a curvature flat band exists on [0, T]. For this the minimum
    # curvature must fall below upsilon2.

    # Case 1: y''(t) is U-shaped
    # Locate the minimum curvature of y on [0, T], as the root of y''' clipped to the horizon. 
    #                            y''' = 0 
    #                               0 = S[-rho d^3 e^{-dt} + (1 - rho) g^3 e^{gt}]
    #                 rho d^3 e^{-dt} = (1 - rho) g^3 e^{gt}
    #     (rho d^3) / [(1 - rho) g^3] =  e^{(d+g) t}
    # t^** = ln{(rho d^3) / [(1 - rho) g^3]} / (d+g)
    numerator = rho * d**3
    denominator = (1.0 - rho) * g**3
    if denominator != 0.0 and numerator / denominator > 0.0:
        t_min = np.log(numerator / denominator) / (d + g)
        t_min = float(np.clip(t_min, 0.0, T))

    # Case 2: y''(t) is monotone on [0, T] (no interior stationary point)
    else:
        # The minimum curvature is reached either at the interval start or end
        curvature_start = _evaluate_wilkerson_double_prime(0.0, size, g, d, rho) 
        curvature_end = _evaluate_wilkerson_double_prime(T, size, g, d, rho) 
        if curvature_start <= curvature_end:
            t_min = 0.0 
        else:
            t_min = T

    # Evaluate whether the minimum curvature exceeds upsilon2, i.e. no curvature flat region
    if _evaluate_wilkerson_double_prime(t_min, size, g, d, rho) >= upsilon_2:
        return None

    # Step 3: Given that a curvature flat region exists find its left boundary "a"
    if _evaluate_wilkerson_double_prime(0.0, size, g, d, rho) < upsilon_2:
        band_start = 0.0
    else:
        band_start = brentq(lambda t: _evaluate_wilkerson_double_prime(t, size, g, d, rho) - upsilon_2, 0.0, t_min)

    # Step 4: Given that a curvature flat region exists, find its right boundary "b"
    if _y_double_prime(T, size, g, d, rho) < upsilon_2:
        band_end = T
    else:
        band_end = brentq(lambda t: _y_double_prime(t, size, g, d, rho) - upsilon_2, t_min, T)

    # Step 5: Return the endpoints of the curvature flat bands as floats
    return (float(band_start), float(band_end))


def _build_gamma(T, t_star, slope_band, curvature_band):
    """Assemble and sort the analytic candidate transitions.

    This function assembles the array of candidate transitions Gamma. 
    This is given as the union of 
        - the domain boundaries,
        - the endpoints of the slope-flat interval,
        - the endpoints of the curvature-flat interval,
        - the stationary point t* if it lies strictly inside (0, T).
    We add the stationary point for the case where upsilon1/2==0.

    Note: No zeta de-duplication is needed. Unlike the numerical extractor, whose
    candidates are discovered by root finding and can near-coincide by
    floating-point accident, these candidates are computed in closed form, so
    only exact duplicates arise and np.unique removes them.

    Args:
        T: scalar time horizon.
        t_star: scalar stationary point, possibly np.nan or outside [0, T].
        slope_band: tuple (a, b) or None, as returned by _slope_flat_band.
        curvature_band: tuple (a, b) or None, as returned by
            _curvature_flat_band.

    Returns:
        np.ndarray: Sorted array of candidate transitions, with first entry 0.0
            and last entry T.
    """
    gamma = [0.0, T]

    # The sign change of y' is a transition whether or not a band surrounds it
    if 0.0 < t_star < T:
        gamma.append(t_star)

    for band in (slope_band, curvature_band):
        if band is not None:
            gamma.extend(band)

    # Convert into np.array, sort and remove dublicates
    return np.unique(np.asarray(gamma, dtype=float))


def _check_points_in_interval(points, interval):
    """Return boolean flags for points in a given given interval [a, b].

    Args:
        points: A 1D NumPy array of shape (M - 1,). 
        band: A tuple of (start, end) floats defining an interval,
            or None.

    Returns:
        np.ndarray: A 1D boolean array of shape (M - 1,). 
            - An element is True if the point falls strictly inside the band interval
            - and False otherwise. 
    """
    # Case 1: If there is no flat zone we mark every region as False
    if interval is None:
        return np.zeros_like(points, dtype=bool)
            
    # Case 2: Create flags for midpoints that sit inside the zone boundaries
    zone_start, zone_end = interval
    is_after_start = (points > zone_start)
    is_before_end  = (points < zone_end)
    in_interval = is_after_start & is_before_end
    return in_interval


def _compute_region_signs_analytic(regions, size, g, d, rho, slope_band, curvature_band):
    """Return the sign of the first- and second-order derivative over each region. 

    Args:
        regions: np.ndarray of shape (M - 1, 2) of region boundaries [a, b] 
            (M = nr of split points, so M-1 = nr of regions)
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).
        slope_band: tuple (a, b) or None, as returned by _slope_flat_band.
        curvature_band: tuple (a, b) or None, as returned by
            _curvature_flat_band.

    Returns:
        tuple: (s1, s2) of np.ndarray of shape (M - 1,) and dtype int. Entries of
            s1 lie in {-1, 0, 1} and entries of s2 lie in {0, 1}.
    """
    # Step 1: Construct region start points, end points and mid points. 
    t_start = regions[:, 0]
    t_end = regions[:, 1]
    midpoints = (t_start + t_end) / 2.0

    # Step 2: Generate flags for whether a regions is slope or curvature flat
    slope_flat = _check_points_in_interval(midpoints, slope_band)
    curvature_flat = _check_points_in_interval(midpoints, curvature_band)

    # Step 3: Determine first derivative sign s1; overwrite if slope_flat
    derivative1 = _evaluate_wilkerson_prime(midpoints, size, g, d, rho)
    s1 = np.sign(derivative1)
    s1 = np.where(slope_flat, 0, s1)

    # Step 4: Determine second derivative sign s2; overwrite if curvature_flat
    # Since the curve is strictly convex the sign is either 1 or overwritten to zero.
    s2 = np.ones(midpoints.shape[0])
    s2 = np.where(curvature_flat, 0, s2)
    s2 = np.where(slope_flat, 0, s2) # Slope flat is set to imply curvature flat. 

    return s1.astype(int), s2.astype(int)


def compute_analytic_episode_score(mode, episode, size, g, d, rho, n_grid=513):
    """Return the significance score of a slope or curvature episode.

    We distinguish between two modes: 
        - slope episodes are mode = 1,
        - curvature episodes are mode = 2. 

    For slope episodes the score of the epsiode is defined as the total
    vertical change of y over the epsiode. Since y is monotone for a slope
    episode this change is given as the difference between the two 
    endpoints. 

    For curvature episodes, the score is the maximum absolute deviation
    of y from the secant through the episode endpoints. Unlike the spline case,
    the Wilkerson deviation y - l is a sum of two exponentials minus a line and
    has no elementary stationary point, so the maximum is taken over a dense
    uniform grid spanning the episode.

    Args:
        mode: int, 1 for slope pruning or 2 for curvature pruning.
        episode: dict including keys 't_L' and 't_R' for the episode endpoints.
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).
        n_grid: int, number of grid points spanning the episode in mode 2.

    Returns:
        float: The episode score, in the units of y.
    """
    if mode not in (1, 2):
        raise ValueError(f"Mode must be 1 or 2.")

    # Compute the curve value at all epsiode boundaries
    t_episode_L = episode['t_L']
    t_episode_R = episode['t_R']
    if t_episode_R <= t_episode_L:
        return 0.0

    # -------------------------------------------------------------------------
    # Mode 1: Slope significance
    # -------------------------------------------------------------------------
    if mode == 1:
        y_episode_L = _evaluate_wilkerson(t_episode_L, size, g, d, rho)
        y_episode_R = _evaluate_wilkerson(t_episode_R, size, g, d, rho)
        return float(np.abs(y_episode_R - y_episode_L))

    # -------------------------------------------------------------------------
    # Mode 2: Curvature significance
    # The curvature significance is the maximum vertical distance between
    # the curve y(t) and the secant line l(t) = m*t + b passing 
    # through the episode's start and end points.
    #
    # We determine this maximum deviation as follows:
    # 1. We evaluate y on a dense uniform grid spanning the episode.
    # 2. We build the secant through the episode endpoints.
    # 3. We take the maximum absolute deviation over the grid.
    # -------------------------------------------------------------------------
    # Step 1: Evaluate the curve across the episode
    ts = np.linspace(t_episode_L, t_episode_R, n_grid)
    ys = _evaluate_wilkerson(ts, size, g, d, rho)

    # Step 2: Build the secant l(t) = intercept slope*t that passes through
    # the episode endpoints
    secant_slope = (ys[-1] - ys[0]) / (t_episode_R - t_episode_L)
    secant = ys[0] + secant_slope * (ts - t_episode_L)

    # Step 3: Determine the maximum distance over the episode
    max_deviation = float(np.max(np.abs(ys - secant)))

    return max_deviation


def _select_lowest_scoring_analytic_episode(mode, episodes, size, g, d, rho,
                                            upsilon_prune):
    """Select the lowest-scoring signed episode below the pruning threshold.

    Iterate over the list of episodes and select the lowest scoring episode 
    (leftmost if tied). The score is computed according to compute_epsiode_score
    depending on the selected mode. Flat episodes are not considered.

    Args:
        mode: int, 1 for slope pruning or 2 for curvature pruning.
        episodes: list of episode dicts, as returned by episodes_from_signs.
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).
        upsilon_prune: float, absolute significance threshold in the units of y.

    Returns:
        int or None: The index of the selected episode, or None if every signed
            episode scores at or above the threshold.
    """
    selected = None        
    best_score = np.inf

    for i, episode in enumerate(episodes):
        # We do not consider flat epsiodes for pruning (explained in pruning doc string)
        if episode['sign'] == 0:
            continue
        
        # Compute the episode score
        score = compute_analytic_episode_score(mode, episode, size, g, d, rho)

        # Note that the strict inequality on the score keeps the leftmost episode 
        # if there is a score tie
        if score < upsilon_prune and score < best_score:
            selected = i
            best_score = score

    # Return the inex of the selected episode
    return selected


def prune_analytic(mode, s1, s2, transitions, size, g, d, rho, upsilon_prune):
    """Iteratively absorb insignificant signed episodes into their neighbours.

    The analytic counterpart of prune, applied to the Wilkerson curve rather than
    to a fitted spline. Only the scoring differs; the episode construction and
    absorption hierarchy are shared with the spline implementation.

    We distinguish two modes: mode = 1 prunes slope episodes, mode = 2 prunes
    curvature episodes. We then describe the curve a sequence of sign/curvature
    epsiodes. Pruning operates as follows:

    While any *signed* episode scores below upsilon_prune (score_episode):
        1. select the lowest-scoring such episode (leftmost if tied);
        2. resolve it:
            - if two signed neighbours: overwrite episode with neighbour sign.
            - if any neighbour flat or no neighbours: flatten (sign <- 0),
                extending the adjacent flat / collapsing the entire summary to flat;
            - exactly one neighbour and it is signed (domain boundary):
                overwrite with that sign (to ensure boundary does not generate
                spurious flat state;
        3. re-form episodes and re-score (absorption changes adjacency).

    Notes:
        (1) Flat episodes are not scored or pruned. This is because flatness
        is established by a vertical movement limit holding over the entire
        region classed as flat. In contrast, signed states have not faced a
        significance test at this stage.

        (2) When mode = 1 flattens an episode (s1 <- 0), it also writes
        s2 <- 0 on those regions. This ensures  s1 == 0 => s2 == 0. 
        This is needed such that the state (0, +-1) cannot be reached.
        Since mode 1 can  rewrite s2 (while mode 2 never modifies s1)
        slope pruning must run before curvature pruning. 

    Args:
        mode: int, 1 for slope pruning or 2 for curvature pruning.
        s1: np.ndarray of shape (M - 1,) of first-derivative signs.
        s2: np.ndarray of shape (M - 1,) of second-derivative signs.
        transitions: np.ndarray of shape (M,) of de-duplicated candidate
                transitions, sorted ascending.
        size: scalar initial size S.
        g: scalar growth rate of the resistant fraction.
        d: scalar decay rate of the sensitive fraction.
        rho: scalar treatment-sensitive fraction in (0, 1).
        upsilon_prune: float, absolute significance threshold in units of y.

    Returns:
        tuple: (s1, s2) pruned sign arrays (copied)
    """
    if mode not in (1, 2):
        raise ValueError(f"Mode must be 1 or 2; got {mode}.")

    # Create copies of the signs so we do not modify the arrays in place. 
    s1 = s1.copy()
    s2 = s2.copy()

    while True:
        # Compute epsiodes of the approrpiate type
        active = s1 if mode == 1 else s2
        episodes = compute_episodes(active, transitions)

        # Find the lowest scoring episode (among those scored insignificant)
        selected = _select_lowest_scoring_analytic_episode(
                            mode, 
                            episodes, 
                            size, 
                            g, 
                            d, 
                            rho, 
                            upsilon_prune,
            )

        # End the loop if no epsiode scored insignificant remains
        if selected is None:
            break

        # Absorb the lowest scoring episode into its neighbour
        _absorb_lowest_scoring_episode(mode, episodes, selected, s1, s2)

    # Return the pruned derviative sign arrays 
    return s1, s2


def get_analytic_shape_summary_single(size, g, d, rho, T, upsilon_rel_1=0.0,
                                      upsilon_rel_2=0.0, upsilon_rel_prune=0.0,
                                      do_prune=False):
    """Compute the Wilkerson ground-truth shape summary from analytic derivatives.

    Args:
        size: float initial size S.
        g: float growth rate of the resistant fraction.
        d: float decay rate of the sensitive fraction.
        rho: float treatment-sensitive fraction in (0, 1).
        upsilon_rel_1: float, relative slope threshold
        upsilon_rel_2: float, relative curvature significance threshold.
        upsilon_rel_prune: float, relative pruning significance threshold.
        do_prune: bool, whether to apply slope then curvature pruning. 

    Returns:
        list: A length-D list of shape summaries, each a chronological list of
            (state, start_time) tuples.
    """
    # Step 1: Compute the critical point
    t_star = _compute_critical_point(g, d, rho)

    # Step 2: Compute the amplitude and the absolute thresholds
    A = _compute_true_curve_amplitude(size, g, d, rho, T, t_star)
    upsilon_1, upsilon_2 = absolute_thresholds(upsilon_rel_1, upsilon_rel_2, A, T)
    upsilon_prune = upsilon_rel_prune * A

    # Step 3: Identify flat bands B1 and B2l
    slope_band = _compute_slope_flat_band(size, g, d, rho, T, upsilon_1)
    curvature_band = _compute_curvature_flat_band(size, g, d, rho, T, upsilon_2)

    # Step 4: Assemble the candidate transitions and construct the regions
    gamma = _build_gamma(T, t_star, slope_band, curvature_band)
    regions = construct_regions(gamma)

    # Step 5: Assign derivative signs to each region
    s1, s2 = _compute_region_signs_analytic(regions, size, g, d, rho, slope_band, curvature_band)

    # Step 6: Prune insignificant episodes, slope first since it rewrites s2
    if do_prune:
        s1, s2 = prune_analytic(1, s1, s2, gamma, size, g, d, rho, upsilon_prune)
        s1, s2 = prune_analytic(2, s1, s2, gamma, size, g, d, rho, upsilon_prune)

    # Step 7: Map sign pairs to shape states and merge adjacent duplicates
    candidate_states = [
        (classify_signs(s1[j], s2[j]), gamma[j]) for j in range(len(s1))
    ]

    return merge_adjacent(candidate_states)


def get_analytic_shape_summary(dataset, upsilon_rel_1, upsilon_rel_2,
                               upsilon_rel_prune, do_prune):
    """Compute ground-truth shape summaries for every individual in a dataset.

      Args:
        dataset: A SimulatedDataset instance.
        upsilon_rel_1: float, relative slope significance threshold.
        upsilon_rel_2: float, relative curvature significance threshold.
        upsilon_rel_prune: float, relative pruning significance threshold.
        do_prune: bool, whether to apply slope then curvature pruning.

    Returns:
        list: A length-D list of shape summaries, each a chronological list of
            (state, start_time) tuples.
    """
    return [
        get_analytic_shape_summary_single(
            dataset.X["size"][i], dataset.params["g"][i],
            dataset.params["d"][i], dataset.params["rho"][i],
            dataset.T, upsilon_rel_1, upsilon_rel_2,
            upsilon_rel_prune, do_prune,
        ) 
        for i in range(dataset.D)
    ]



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