import numpy as np

# Class tags for candidate transitions. The retention preference is END > KNOT > ROOT 
CLASS_ROOT = 0
CLASS_KNOT = 1
CLASS_END  = 2

# Possible shape state derivative sign tuples and the associated lablels
SHAPE_STATES = { 
        (+1, +1): "convex_increasing",
        (-1, +1): "convex_decreasing",
        (+1, -1): "concave_increasing",
        (-1, -1): "concave_decreasing",
        (+1,  0): "linear_increasing",
        (-1,  0): "linear_decreasing", 
        ( 0,  0): "constant"
    }
    
def piece_coefficients(w, C):
    """Return monomial coefficents for each knot-interval.
    
    Given the monomial conversion matrix C and the spline
    coefficent vector w, return the (K+1, 4) matrix of 
    monomial coefficients for each knot-interval k. 

    Args:
        w: np.ndarray of shape (B, 1) of spline coefficents
        C: np.ndarray of shape (K + 1, 4, B), where C[k] is the
            conversion matrix for the k-th knot interval, ordered left
            to right over the K + 1 intervals of [0, T].
    Returns:
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficents
    """
    coeffs = (C @ w).squeeze(-1) 
    return coeffs


def candidate_transitions(coeffs, breakpoints):
    """Assemble all candidate transition points with retention-class tags.

    This function builds Gamma. That is, the unition of the domain boundaries, 
    the interior knots, and all real roots of the first and second derivatives
    of each piecewise cubic that lie strictly strictly inside each open knot 
    interval (xi_k, xi_{k+1}).
    Note that we search for roots on open intervals only, since y''' is 
    discontinuous at the knots.

    Args:
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficents.
            Each row is [c0, c1, c2, c3]
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus
            boundaries.

    Returns:
        tuple: A tuple (gamma, classes) where gamma is a sorted np.ndarray of
            shape (M,) of candidate transition points, and classes is an
            np.ndarray of shape (M,) of integer class tags drawn from
            {CLASS_ROOT, CLASS_KNOT, CLASS_END}. Exact duplicates are removed,
            retaining the highest class.
    """
    unique_knots = np.unique(breakpoints)
    t_start = unique_knots[:-1] # left knots xi_k; shape (K + 1,)
    t_end = unique_knots[1:] # right knots xi_{k+1}; shape (K + 1,)

    c1 = coeffs[:, 1]
    c2 = coeffs[:, 2]
    c3 = coeffs[:, 3]
    
    derivative_roots = []

    # -------------------------------------------------------------------------
    # 1. Roots of the second derivative 
    # -------------------------------------------------------------------------
    # The second derivative is: p''(t) =  6*c3*t + 2*c2
    #                                0 =  3*c3*t + c2
    # Find roots while preventing zero division error
    safe_c3 = np.where(c3 == 0, 1.0, c3)
    roots2 = -c2/(3*safe_c3)

    # We find the valid roots of the second derivative of a piece under the 
    # condition that c3 != 0, and that 
    # the roots that are strictly inside the open interval of a piece.
    derivative2_mask = (c3 != 0) & (t_start < roots2) & (roots2 < t_end)

    derivative_roots.extend(roots2[derivative2_mask])

    # -------------------------------------------------------------------------
    # 2. Roots of the first derivative 
    # -------------------------------------------------------------------------
    # The first derivative is: p'(t) = 3*c3*t^2 + 2*c2*t + c1
    #                             0  = 3*c3*t^2 + 2*c2*t + c1
    # The abc formula solves this as:
    #   a t^2 + b t + c = 0
    #   Solution 1: t_1 = [-b + \\sqrt(b^2 - 4ac)] / 2a 
    #   Solution 2: t_2 = [-b - \\sqrt(b^2 - 4ac)] / 2a 
    a = 3*c3
    b = 2*c2
    c = c1
    discriminant = b**2 - 4*a*c

    # We calculate the two roots safely
    safe_a = np.where(a==0, 1.0, a) # Preventing zero division error
    safe_discriminant = np.where(discriminant<0, 0.0, discriminant) # Preventing sqrt of a -ve.
    sqrt_discriminant = np.sqrt(safe_discriminant)

    root1_plus = (-b + sqrt_discriminant) / (2 * safe_a)
    roots1_minus = (-b - sqrt_discriminant) / (2 * safe_a)

    # Retain roots where (1) the quadratic is really quadratic (a != 0),
    # (2) the discriminant is non-negative, and (3) the root lies strictly
    # inside the open interval of a piece.
    deriv1_mask_r1_plus = (a != 0) & (discriminant >= 0) & (t_start < root1_plus) & (root1_plus < t_end)
    deriv1_mask_r1_minus =(a != 0) & (discriminant >= 0) & (t_start < roots1_minus) & (roots1_minus < t_end)

    derivative_roots.extend(root1_plus[deriv1_mask_r1_plus])
    derivative_roots.extend(roots1_minus[deriv1_mask_r1_minus])

    # Handle the case where we had a == 0 (must have b != 0)
    safe_b = np.where(b == 0, 1.0, b)
    linear_root_candidates = -c / safe_b
    deriv_mask_lin = (a == 0) & (b != 0) & (t_start < linear_root_candidates) & (linear_root_candidates < t_end)
    derivative_roots.extend(linear_root_candidates[deriv_mask_lin])

    # --------------------------------------------------------------------------
    # 3. Add roots from the interior knots and boundary knots and add class tags
    # --------------------------------------------------------------------------
    # Convert existing derivative roots into a clean numpy array
    derivative_roots = np.asarray(derivative_roots, dtype=float)

    # Add all interior knots and boundary knots
    boundaries = unique_knots[[0, -1]]
    interior_knots = unique_knots[1:-1]

    # Create an np.array for all roots and an np.array for all root lables (0, 1, 2)
    gamma = np.concatenate([boundaries, interior_knots, derivative_roots])
    classes = np.concatenate([
        np.full(boundaries.shape[0], CLASS_END, dtype=int),
        np.full(interior_knots.shape[0], CLASS_KNOT, dtype=int),
        np.full(derivative_roots.shape[0], CLASS_ROOT, dtype=int),
    ])

    # Perform a primary sort of gamma in algebraical order (from lowest 
    # to highest, i.e. chronological) and a secondary sort according to classes 
    # (from highest to lowest class) as a tiebreaker
    order = np.lexsort((-classes, gamma)) # Generate array of sorting keys
    gamma = gamma[order] # Select items of gamma in order of 'order'
    classes = classes[order]

    # Create a boolean flag that is True if two adjacent knots are not equal. 
    # We will only retain those, i.e. delete all exact duplicates. 
    keep = np.ones(gamma.shape[0], dtype=bool)
    keep[1:] = gamma[1:] != gamma[:-1]

    # Return the sorted candidate transition points and their class labels. 
    return gamma[keep], classes[keep]


def evaluate_spline(coeffs, breakpoints, t):
    """Return the spline function value at an aribtrary times in [0, T].

    Args:
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficients.
            Each row is [c0, c1, c2, c3].
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus
            boundaries.
        t: np.ndarray of shape (M,) of evaluation times, assumed to lie within
            [0, T].

    Returns:
        np.ndarray: Array of shape (M,) of spline values y(t).
    """
    # Step 1: Determine the index the knot interval, in which t falls:
    unique_knots = np.unique(breakpoints)
    indices = np.searchsorted(unique_knots, t, side='right') - 1

    # For the boundaries t=0 and t=1 we need to clip the indices
    n_pieces = coeffs.shape[0]
    indices = np.clip(indices, 0, n_pieces - 1)

    # Step 2: Pull the coefficents for the monomials at the determined time(s)
    c0 = coeffs[indices, 0]
    c1 = coeffs[indices, 1]
    c2 = coeffs[indices, 2]
    c3 = coeffs[indices, 3]

    # Step 3: Evaluate the polynomial at the pulled indices
    return c0 + t * (c1 + t * (c2 + t * c3))


def compute_amplitude(coeffs, breakpoints, gamma):
    """Return the peak-to-peak amplitude of the spline of the spline over [0, T].

    This function returns the peak-to-peak amplitude of the spline. This is the
    distance between the global minimum and the global maximum. Rather than
    recomputing the candidate extrema, we utilise Gamma which  contains every 
    point at which y can attain a local or global extremum. 

    Note: This function must be called on the gamma before de-duplication. 

    Args:
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficients.
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus
            boundaries.
        gamma: np.ndarray of shape (M,) of candidate transition points, prior
            to de-duplication.

    Returns:
        float: The peak-to-peak amplitude A of the spline.
    """
    # Step 1: Compute the spline values at all candidate transition time points
    values = evaluate_spline(coeffs, breakpoints, gamma)

    # Step 2: Compute the peak-to-peak amplitude
    return float(np.max(values) - np.min(values))


def absolute_thresholds(upsilon_rel_1, upsilon_rel_2, A, T):
    """Compute the shape significance thresholds upsilon_1 and upsilon_2. 

    Args:
        upsilon_rel_1 (float): scalar reltative slope significance threshold
        upsilon_rel_2 (float): scalar relative curvature significance threshold
        A: (float): amplitude.
        T: Forecasting horizon.
    
    Returns:
        upsilon_1 (float): scalar absolute slope significance threshold
        upsilon_2 (float): scalar absolute curvature significance threshold
    """
    upsilon_1 = upsilon_rel_1*A/T
    upsilon_2 = 2*upsilon_rel_2*A/(T**2)
    return upsilon_1, upsilon_2


def deduplicate(gamma, classes, zeta):
    """Merge candidate transitions that lie closer together than zeta.

    Group candidate transition points in gamma into clusters. A new cluster is
    started whenever the gap to the previous transition point candidate is at
    least the threshold zeta. 
    From the generated set of clusters, we retain only a single point per 
    cluster. Retention follows a retention preference ordering based on the
    type of transition point END > KNOT > ROOT. If multiple points of the
    same class are present we retain the leftmost piint. 

    Note: Setting zeta = 0 disables de-duplication.

    Args:
        gamma: np.ndarray of shape (M,) of sorted candidate transitions.
        classes: np.ndarray of shape (M,) of INTEGER class tags, from
            {CLASS_ROOT, CLASS_KNOT, CLASS_END}.
        zeta: float, absolute minimum transition point distance threshold.

    Returns:
        np.ndarray: Sorted array of de-duplicated candidate transitions.
    """
    # Step 1: We want to compare neighbours in Gamma. This only works if there 
    # are more than one candidate transition points in Gamma. 
    if gamma.shape[0] <= 1:
        return gamma.copy()

    # Step 2: Compute Gamma star
    # 2.1 Indicate where the distance between adjacent points exceeds zeta
    is_new_cluster = np.diff(gamma) >= zeta # boolean flags; (M-1,)
    is_new_cluster = np.concatenate([[False], is_new_cluster]) # add back flag for first point; (M,)
    cluster_ids = np.cumsum(is_new_cluster) # Generate cluster ids for all transition points; (M,)

    # 2.2 Obtain a sorted gamma. We sort such that we have cluster_id ascending,
    # transition point class descending, left to right
    order = np.lexsort((gamma, -classes, cluster_ids))
    sorted_gamma = gamma[order]
    sorted_cluster_ids = cluster_ids[order]

    # 2.3 Retain only the first element of each cluster
    is_first_cluster_element = sorted_cluster_ids[1:] != sorted_cluster_ids[:-1] # boolean flags; (M-1,)
    is_first_cluster_element = np.concatenate([[True], is_first_cluster_element]) # add back flag for first point; (M,)
    gamma_star = sorted_gamma[is_first_cluster_element] # select the first element of each cluster only

    return gamma_star


def construct_regions(gamma_star):
    """Construct the evaluation regions bounded by consecutive de-duplicated candidate transitions.

    Given M de-duplicated candidate transitions, return the M-1 regions
    R_i = [gamma_{i-1}, gamma_i]. Over each of these regions the signs of
    the first and second derivatives are constant, since every point at 
    which a derivative can change sign is itself a candidate transition.

    Args:
        gamma_star: np.ndarray of shape (M,) of de-duplicated candidate
            transitions, sorted ascending. Must contain at least two points.

    Returns:
        np.ndarray: Array of shape (M - 1, 2), where row i is the pair of
            boundaries [a, b] of region R_i.
    """
    return np.stack([gamma_star[:-1], gamma_star[1:]], axis=1)


def generate_region_flags(regions, coeffs, breakpoints, upsilon_1, upsilon_2):
    """Return boolean flags for slope_flat and curvature_flat for each region.

    A region is:
        - slope flat if the maximum of abs(y') over the region falls below upsilon_1;
        - curvature flat if the maximum of abs(y'') falls below upsilon_2. 
    The second derivative y'' is linear on each piece so its max is attained at 
    the region endpoints.
    The first derivative y' is quadratic on each piece. It attains its max at either
    the boundaties or at its extremum if this falls within the region. However, the
    extremum is attained where y''=0 which is the boundary to a next region. Hence, 
    the max is attained at the region endpoints.

    Args:
        regions: np.ndarray of shape (M - 1, 2) of region boundaries [a, b],
            derived from consecutive de-duplicated candidate transitions.
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficients.
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus
            boundaries.
        upsilon_1: float, absolute slope significance threshold.
        upsilon_2: float, absolute curvature significance threshold.

    Returns:
        tuple: A tuple (slope_flat, curvature_flat) of np.ndarray of shape
            (M - 1,) and dtype bool.
    """
    unique_knots = np.unique(breakpoints)
    t_start = regions[:, 0] # left boundary of each region, shape (M - 1,)
    t_end = regions[:, 1] # right boundary of each region, shape (M - 1,)

    # -------------------------------------------------------------------------
    # 1. Map each region to its knot interval (i.e. polynomial)
    # -------------------------------------------------------------------------
    # Assign each region to its knot interval according to the region's mid-point
    midpoints = (t_start + t_end) / 2.0
    indices = np.searchsorted(unique_knots, midpoints, side='right') - 1

    # For the boundaries t=0 and t=1 we need to clip the indices
    n_pieces = coeffs.shape[0]
    indices = np.clip(indices, 0, n_pieces - 1)

    # Find the coefficents for each region based on the regions's assgined knot interval
    c1 = coeffs[indices, 1]
    c2 = coeffs[indices, 2]
    c3 = coeffs[indices, 3]

    # --------------------------------------------------------------------------
    # Step 1: Determine whether a region is curvature flat
    # --------------------------------------------------------------------------
    # A region is curvature flat over a region when the maximum curvature,
    # if it were sustained over the entire region, moves by less than upsilon_2.  
    # We determine the maximum curvature as the (absolute) maximum of the 
    # second-order derivative. Since the second order derivative of a cubic is 
    # linear, we know that this maximum is achieved at a boundary. 
    
    # The second-order derivative is 6*c3*t + 2*c2 
    derivative2_start = 6*c3*t_start + 2*c2 
    derivative2_end = 6*c3*t_end + 2*c2 

    # Obtain a (K+1,) np.array with the max absolute curvature for each region
    max_derivative2 =np.maximum(np.abs(derivative2_start), np.abs(derivative2_end)) 

    # Obtain boolean flag for curvature flag for each region
    curvature_flat  = max_derivative2 < upsilon_2

    # --------------------------------------------------------------------------
    # Step 2: Determine whether a region is slope flat
    # --------------------------------------------------------------------------
    # A region is slope flat over a region when the maximum slope, if it 
    # were sustained over the entire region, moves by less than upsilon_1. 
    # We determine the maximum slope as the (absolute) maximum of the first-order 
    # derivative. Since the spline is cubic the first order derivative is a square
    # so its maximum is obtained at either of the endpoints or at the extremum
    # of the quadratic, if this falls within the region. However, here the extremum
    # can never fall into a region so evaluating the boundaries is sufficent. 

    # The first-order derivative is 3*c3*t^2 + 2*c2*t + c1
    derivative1_start = 3*c3*(t_start**2) + 2*c2*t_start + c1
    derivative1_end = 3*c3*(t_end**2) + 2*c2*t_end + c1

    # Obtain a (K+1,) np.array with the max absolute slope for each region (at boundaries)
    max_derivative1 = np.maximum(np.abs(derivative1_start), np.abs(derivative1_end))

    # Obtain boolean flag for slope flag for each region
    slope_flat = max_derivative1 < upsilon_1

    return slope_flat, curvature_flat


def compute_region_signs(regions, coeffs, breakpoints, slope_flat, curvature_flat):
    """Return the sign of the first- and second-order derivative over each region. 

    Args:
        regions: np.ndarray of shape (M - 1, 2) of region boundaries [a, b].
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficents.
            Each row is [c0, c1, c2, c3]
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus boundaries.
        slope_flat: np.ndarray of shape (M - 1, ) of boolean flags, one per region.
        curvature_flat: np.ndarray of shape (M - 1, ) of boolean flags, one per region.

    Returns:
        (s1, s2): A tuple of two np.ndarray of shape (M - 1,), containing the signs 
            (-1, 0, or 1) of the first- and second-order derivatives over each 
            interval spanned by consecutive candidate roots.
    """
    # --------------------------------------------------------------------------
    # 1. Map each midpoint to its owning polynomial piece
    # --------------------------------------------------------------------------
    
    # Calculate midpoaints
    midpoints = (regions[:, 0] + regions[:, 1]) / 2.0
    unique_knots = np.unique(breakpoints)

    # Map each midpoint to its polynomial interval index [0, K]
    indices = np.searchsorted(unique_knots, midpoints, side='right') - 1

    # Ensure that indices are in [0, K]
    n_pieces = coeffs.shape[0]
    indices = np.clip(indices, 0, n_pieces - 1)

    # Get the monomial coefficents for each region
    c1 = coeffs[indices, 1]
    c2 = coeffs[indices, 2]
    c3 = coeffs[indices, 3]
    
    # --------------------------------------------------------------------------
    # 2. Compute first derivative (slope) signs at midpoints
    # --------------------------------------------------------------------------
    deriv1 = 3 * c3 * (midpoints**2) + 2 * c2 * midpoints + c1
    # Get the sign of the first derivative of the relevant monomial to each midpoint. 
    s1 = np.sign(deriv1)
    # If the region is slope_flat, overwrite the sign to 0.
    s1 = np.where(slope_flat, 0, s1)

    # --------------------------------------------------------------------------
    # 3. Compute second derivative (curvature) signs at midpoints
    # --------------------------------------------------------------------------
    deriv2 = 6 * c3 * midpoints + 2 * c2
    # Get the sign of the second derivative of the relevant monomial to each midpoint. 
    s2 = np.sign(deriv2)
    # If the interval is curvature_flat, overwrite the sign to 0.
    s2 = np.where(curvature_flat, 0, s2)
    # If the interval is slope_flat overwrite to be curvature flat. 
    s2 = np.where(slope_flat, 0, s2)
    
    return s1.astype(int), s2.astype(int)


def compute_episodes(s, transitions):
    """Group consecutive regions of equal sign into episodes.

    An episode is a maximal contiguous run of regions sharing the same sign, 
    with flat regions (sign zero) form episodes of their own class.
    We distinguish between slope and curvature episodes, depending on whether
    we are handed s1 or s2 for s.

    Args:
        s: np.ndarray of shape (M - 1,) of signs in {-1, 0, 1}, one per region.
        transitions: np.ndarray of shape (M,) of de-duplicated candidate
            transitions bounding the regions.

    Returns:
        list: A list of dicts, one dict per epsiode:
    
        episode =  {
            'sign': int -- sign identifying the epsiode 
            't_L': float -- transition point identifying the episode start 
            't_R': float -- transition point identifying the episode end
            'start': int -- the region index where the epsiode starts
            'stop': int -- the region index where the epsiode ends, plus one
        }

        Note:
        We use the dictionary elements as follows:
        - 'sign': used for absorption hierarchy in pruning stage,  
        - 't_L', 't_R': used to compute the vertical change or max secant deviation over episode.
        - 'start', 'stop': the half-open interval [start, stop) across which the pruning stage writes a sign.
    """
    # Step 1: Confirm that there are regions to analyse
    n_regions = s.shape[0]
    if n_regions == 0:
        return []

    # Step 2: Get the indices of the sign changes
    # Compare neighboring elements to detect where signs flip
    sign_change_flags = s[1:] != s[:-1] # (M-1,)
    # Get the index positions in s (i.e. the region index) where the sign changes happen
    indices = np.nonzero(sign_change_flags)[0] + 1  
    
    # Step 3: Get a list of tuples indicating the start and ending index positions in s/transitions
    starts = np.concatenate([[0], indices]) # The first episode starts at index position zero.
    stops = np.concatenate([indices, [n_regions]]) # The final episode ends at index position M-1 (nr of regions).
    
    # Step 4: Create the list of episodes, each of which is a dict
    episodes = []
    for start, stop in zip(starts, stops):
        episode =  {
            'sign': int(s[start]), 
            't_L': float(transitions[start]), 
            't_R': float(transitions[stop]), 
            'start': int(start),
            'stop': int(stop),
        }
        episodes.append(episode)

    return episodes


def compute_episode_score(mode, episode, coeffs, breakpoints):
    """Return the significance score of a slope or curvature episode.

    We distinguish between two modes: 
        - slope episodes are mode = 1,
        - curvature episodes are mode = 2. 

    For slope episodes the score of the epsiode is defined as the total
    vertical change of y over the epsiode. Since y is monotone for a slope
    episode this change is given as the difference between the two 
    endpoints. 

    For curvature episodes, the score is the maximum absolute deviation
    of y from the secant through the episode endpoints. The deviation y - l is
    piecewise cubic, obtained by subtracting the secant's linear part from the
    c0 and c1 coefficients of each overlapping piece. Its maximum is found
    by evaluating at the roots of its derivative that fall inside each
    piece's overlap with the episode, together with the overlap endpoints.

    Args:
        mode: int, 1 for slope pruning or 2 for curvature pruning.
        episode: dict including keys 't_L' and 't_R' for the episode endpoints.
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficients.
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus
            boundaries.

    Returns:
        float: The episode score, in the units of y.
    """
    if mode not in (1, 2):
        raise ValueError(f"Mode must be 1 or 2.")

    # Compute the spline value at all epsiode boundaries
    t_episode_L = episode['t_L']
    t_episode_R = episode['t_R']
    if t_episode_R <= t_episode_L:
        return 0.0
    y_episode_L, y_episode_R = evaluate_spline(coeffs, breakpoints, np.array([t_episode_L, t_episode_R]))

    # -------------------------------------------------------------------------
    # Mode 1: Slope significance
    # -------------------------------------------------------------------------
    if mode == 1:
        return float(np.abs(y_episode_R - y_episode_L))

    # -------------------------------------------------------------------------
    # Mode 2: Curvature significance
    # The curvature significance is the maximum vertical distance between
    # the spline curve y(t) and the secant line l(t) = m*t + b passing 
    # through the episode's start and end points.
    #
    # We determine this maximum deviation as follows:
    # 1. For each knot interval/ spline piece that the episode overlaps:
    #    a. We construct the deviation polynomial d(t) = y(t) - l(t):
    #          d(t) = (c0 - b) + (c1 - m)*t + c2*t^2 + c3*t^3
    #    b. We find the absolute maximum of this cubic polynomial over the
    #       sub-interval where the knot piece and the episode intersect.
    #       Candidates for the maximum include:
    #         - The boundaries of the sub-interval.
    #         - Any local extrema (i.e. where d'(t) = 0) that fall
    #           strictly within the sub-interval.
    # 2. Across all evaluated overlapping intervals, we identify the global 
    #    maximum deviation. 
    # -------------------------------------------------------------------------
    # Step 1: Build the secant l(t) = intercept slope*t that passes through
    # the episode endpoints
    secant_slope = (y_episode_R - y_episode_L) / (t_episode_R - t_episode_L)
    secant_intercept = y_episode_L - secant_slope * t_episode_L

    # Step 2: Determine the indices of the knot segements of the spline that
    # the episode overlaps with 
    unique_knots = np.unique(breakpoints)
    t_knot_start = unique_knots[:-1]
    t_knot_end = unique_knots[1:]
    active_knot_segement_indices = np.flatnonzero((t_knot_end > t_episode_L) & (t_knot_start < t_episode_R))

    max_deviation = 0.0
    for k in active_knot_segement_indices:
        # Step 3.1: Construct the coefficents of the deviation polynomial for piece k
        d0 = coeffs[k, 0] - secant_intercept
        d1 = coeffs[k, 1] - secant_slope
        d2 = coeffs[k, 2]
        d3 = coeffs[k, 3]

        # Step 3.2: Determine relevant boundaries of the monomial
        window_start = max(t_knot_start[k], t_episode_L)
        window_end = min(t_knot_end[k], t_episode_R)

        # Step 3.3: Obtain all candidate maximisers
        candidate_maximisers = []

        # 1/3 Quadratic derivative case.
        # Determine the points where d'(t) = 0  
        #                                0 = 3*d3*t^2 + 2*d2*t + d1
        #                                0 = t^2 + p*t + q -> two roots.
        # and which fall STRICTLY inside the intersection window.
        if d3 != 0:
            # I use the p-q formula.
            p = (2*d2) / (3*d3)
            q = d1 / (3*d3)
            discriminant = (p/2)**2 - q
            if discriminant >= 0:
                sqrt_discriminant = np.sqrt(discriminant)
                for root in (-(p/2) + sqrt_discriminant, 
                             -(p/2) - sqrt_discriminant):
                    if window_start < root < window_end:
                        candidate_maximisers.append(root)

        # 2/3 Linear derivative case
        elif d2 != 0:
            root = - d1 / (2*d2)
            if window_start < root < window_end:
                candidate_maximisers.append(root)

        # 3/3 Add the window endpoints
        candidate_maximisers.extend([window_start, window_end])

        # Step 4: Determine the maximum distance of over the current interval
        # and compare it to the currently recorded max_deviation
        candidate_maximisers = np.asarray(candidate_maximisers, dtype=float)
        deviations = np.abs(d0 + candidate_maximisers * (d1 + candidate_maximisers * (d2 + candidate_maximisers * d3)))
        max_deviation = max(max_deviation, float(np.max(deviations)))

    return max_deviation


def _select_lowest_scoring_episode(mode, episodes, coeffs, breakpoints, upsilon_prune):
    """Select the lowest-scoring signed episode below the pruning threshold.

    Iterate over the list of episodes and select the lowest scoring episode 
    (leftmost if tied). The score is computed according to compute_epsiode_score
    depending on the selected mode. Flat episodes are not considered.

    Args:
        mode: int, 1 for slope pruning or 2 for curvature pruning.
        episodes: list of episode dicts, as returned by episodes_from_signs.
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficients.
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus
                boundaries.
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
        score = compute_episode_score(mode, episode, coeffs, breakpoints)

        # Note that the strict inequality on the score keeps the leftmost episode 
        # if there is a score tie
        if score < upsilon_prune and score < best_score:
            selected = i
            best_score = score

    # Return the inex of the selected episode
    return selected


def _absorb_lowest_scoring_episode(mode, episodes, selected, s1, s2):
    """Absorb the selected episode into its neighbours, in place.

    Given a selected lowest-scoring epsiode with a score below upsilon_prune, 
    we apply the following absorbtion strategy (see prune)        
    We then re-form episodes and re-score. 

    Note:
    When mode is 1, when writing s1 == 0 this is set to imply s2 == 0.

    Args:
        mode: int, 1 for slope pruning or 2 for curvature pruning.
        episodes: list of episode dicts, as returned by episodes_from_signs.
        selected: int, index of the episode to absorb.
        s1: np.ndarray of shape (M - 1,) of first-derivative signs, modified in
            place.
        s2: np.ndarray of shape (M - 1,) of second-derivative signs, modified in
            place.

    Returns:
        None. (sign arrays are modified in place)
    """
    # Step 1: Take pruning target epsiode and check whether it has neighbours
    episode = episodes[selected]
    has_left = selected > 0
    has_right = selected < len(episodes) - 1

    # Step 2: Get the sign of the neighbours
    left_sign = episodes[selected - 1]['sign'] if has_left else None
    right_sign = episodes[selected + 1]['sign'] if has_right else None

    # Step 3: Determine what to do depending on the case

    # Case 1: Episode has two neighbours
    if has_left and has_right:
        # A. If either neighbour flat overwrite as flat
        if left_sign == 0 or right_sign == 0:
            value = 0
        # B. If both neighbours signed, assign their sign (both share a sign) 
        else:
            value = left_sign

    # Case 2: Episode has one neighbour only
    elif has_left or has_right:
        # If onlt one neighbour, assign the neigbours sign
        value = left_sign if has_left else right_sign

    # Case 3: Episode has no neighbours
    else:
        # If no neighbour overwrite slope to flat
        value = 0

    # Step 4: Implement the sign change (in place) inclduing propagation to s2 if applicable
    _overwrite_region_sign_values(mode, episode, value, s1, s2)


def _overwrite_region_sign_values(mode, episode, value, s1, s2):
    """Write the sign "value" across every region of an episode (in place).

    Args:
        mode: int, 1 for slope pruning or 2 for curvature pruning.
        episode: dict with keys 'start' and 'stop', giving the half-open region
            span of the episode.            
        value: int, the sign to write, in {-1, 0, 1}.
        s1: np.ndarray of shape (M - 1,) of first-derivative signs, modified in
                place.
        s2: np.ndarray of shape (M - 1,) of second-derivative signs, modified in
                place.

    Returns:
        None. (sign modified in place)
    """
    # The episode covers the regions given by the half-open interval [start, stop)
    start = episode['start']
    stop = episode['stop']

    if mode == 1:
        s1[start:stop] = value
        # In mode 1, a value of zero for s1 also writes s2 to zero. 
        if value == 0:
            s2[start:stop] = 0
    else:
        s2[start:stop] = value


def prune(mode, s1, s2, transitions, coeffs, breakpoints, upsilon_prune):
    """Iteratively absorb insignificant signed episodes into their neighbours.

    We distinguish two modes: mode = 1 prunes slope episodes, mode = 2 prunes
    curvature episodes. We then describe the spline a sequence of sign/curvature
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
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficients.
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus
                boundaries.
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
        selected = _select_lowest_scoring_episode(
                            mode, 
                            episodes, 
                            coeffs, 
                            breakpoints, 
                            upsilon_prune,
            )

        # End the loop if no epsiode scored insignificant remains
        if selected is None:
            break

        # Absorb the lowest scoring episode into its neighbour
        _absorb_lowest_scoring_episode(mode, episodes, selected, s1, s2)

    # Return the pruned derviative sign arrays 
    return s1, s2


def classify_signs(sign_d1, sign_d2):
    """Return the shape state from the sign of the first two derivatives. 

    Given inputs for the sign of the first and second derative, 
    return the associated shape state. The input signs are computed
    using the v-rule that etermined sign significance already.
    The function operates on integers. 

    Args:
        sign_d1: Sign of the first derivative; integer {-1, 0, 1}.
        sign_d2: Sign of the second derviative; integer {-1, 0, 1}.

    Returns:
        shape_state: An element of seven possiple shape states.
    """
    valid = {-1, 0, 1}
    if sign_d1 not in valid or sign_d2 not in valid:
        raise ValueError(f"Signs must be in {{-1, 0, 1}}; got ({sign_d1}, {sign_d2}).")

    state = SHAPE_STATES.get((sign_d1, sign_d2))
    if state is None:
        raise ValueError(
            f"Sign pair ({sign_d1}, {sign_d2}) has zero slope with non-zero "
            f"curvature, which cannot occur on an interval (only at isolated "
            f"points). Indicates a segmentation or threshold-coherence bug upstream."
        )

    return state


def merge_adjacent(summary):
    """ Compress a sequence shape states by merging consecutive duplicates.

    Args:
        summary (list of tuples (str, float)): A sequential list of shape states 
            and their associated start times, spanned by consecutive candidate 
            transition points.

    Returns:
        list of tuples (str, float): A summary of shape states with consecutive shape
            states of the same type removed. 
    """
    if not summary:
        return []

    merged_summary = [summary[0]]

    for current_item in summary[1:]:
        current_state, current_time = current_item
        last_added_state, last_added_time = merged_summary[-1]
        
        if current_state != last_added_state:
            merged_summary.append(current_item)

    return merged_summary


def extract_shape_summary(w, C, breakpoints, zeta_rel=0.0, upsilon_rel_1=0.0, upsilon_rel_2=0.0, upsilon_rel_prune=0.0, do_prune=False):
    """Extract a shape summary from a cubic spline. 

    Given a coefficient vector w of a cubic spline, return a shape 
    summary as a list of chronological (shape_state, start_time) tuples. 

    Possible toggleable settings:
        -  transition point de-duplication is disabled by zeta_rel = 0,
        - flatness detection is disabled by upsilon_rel_1 = upsilon_rel_2 = 0,
        - significance pruning is disabled by do_prune = False. 
    With all three disabled the procedure reduces to the timeview algorithm.

    Args:
        w: np.ndarray of shape (B, 1) of spline coefficients.
        C: np.ndarray of shape (K + 1, 4, B), where C[k] is the
            conversion matrix for the k-th knot interval, ordered left
            to right over the K + 1 intervals of [0, T].
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus boundaries.
        zeta_rel: float, minimum transition point distance as a fraction of the
            horizon T. Zero disables de-duplication.
        upsilon_rel_1 (float): scalar relative slope significance threshold.
        upsilon_rel_2 (float): scalar relative curvature significance threshold.
        upsilon_rel_prune: float, relative pruning significance threshold, as a
            fraction of the amplitude A. Ignored when do_prune is False.
        do_prune: bool, whether to apply slope and then curvature pruning.

    Returns:
        list of tuples (str, float): A summary of unique shape states with their 
            corresponding start times, sorted chronologically.
    """
    # Force unique and sorted knots to prevent duplicate calculation
    unique_knots = np.unique(breakpoints)
    T = unique_knots[-1]

    # Step 1: Compute the monomials for the spline
    coeffs = piece_coefficients(w, C)

    # Step 2: Compute the candidate transitions with retention class tags 
    gamma, classes = candidate_transitions(coeffs, unique_knots)

    # Step 3: Compute the spline amplitude
    A = compute_amplitude(coeffs, unique_knots, gamma)

    # Step 4: Compute absolute significance thresholds
    upsilon_1, upsilon_2 = absolute_thresholds(upsilon_rel_1, upsilon_rel_2, A, T)
    upsilon_prune = upsilon_rel_prune * A
    zeta = zeta_rel * T

    # Step 5: Perfrom transition point deduplication (identity with zeta ==0)
    gamma_star = deduplicate(gamma, classes, zeta)

    # Step 6: Construct regions
    regions = construct_regions(gamma_star)

    # Step 7: Obtain slope_flat and curvature_flat flags for each region
    #           This is disabled with upsilon_1 == 0 and upsilon_2 == 0
    slope_flat, curvature_flat = generate_region_flags(regions, coeffs, unique_knots, upsilon_1, upsilon_2)

    # Step 8: Assign a sign to each region
    s1, s2 = compute_region_signs(regions, coeffs, unique_knots, slope_flat, curvature_flat)

    # Step 9: Perform slope pruning and then significance pruning.
    if do_prune:
        s1, s2 = prune(1, s1, s2, gamma_star, coeffs, unique_knots, upsilon_prune)
        s1, s2 = prune(2, s1, s2, gamma_star, coeffs, unique_knots, upsilon_prune)


    # Step 10: Assign shape states to each candidate interval in the format (shape_state, start_time)
    candidate_shape_states = [
        (classify_signs(s1[i], s2[i]), gamma_star[i])  for i in range(0, len(s1))
    ]
    
    # Step 11: Merge adjacent identical shape states to obtain the shape summary
    shape_states = merge_adjacent(candidate_shape_states)

    return shape_states


def transition_points(shape_summary):
    """ Return a list of transition points and their timings. 

    A transition point is a boundary between two adjacent shape states. Its type
    is determined by the (state_a, state_b) pair. Boundaries whose pair is not in
    the table (e.g. compound boundaries where both slope and curvature sign change
    at once) are skipped.

    Note: the timing of movement_onset, plateau_onset, straightening, and
    bending_onset is threshold-dependent (sensitive to upsilon), since these
    events involve a derivative entering/leaving zero rather than changing sign.
    Maxima, minima, and inflections are threshold-robust.

    Args:
        shape_summary: list of (state, start_time) tuples, sorted chronologically
            (output of extract_shape_summary, already merged).

    Returns:
        list of (transition_type, time) tuples, where time is the start time of
            the second state in the boundary pair. Empty if fewer than two states.
    """
    transition_types = {
        # --- turning points (derivative sign flips; threshold-robust) ---
        ("concave_increasing", "concave_decreasing"): "maximum",
        ("convex_decreasing",  "convex_increasing"):  "minimum",

        ("convex_increasing",  "concave_increasing"): "inflection_cvx_ccv",
        ("convex_decreasing",  "concave_decreasing"): "inflection_cvx_ccv",
        ("concave_increasing", "convex_increasing"):  "inflection_ccv_cvx",
        ("concave_decreasing", "convex_decreasing"):  "inflection_ccv_cvx",

        # --- movement onset: constant -> moving (slope leaves zero) ---
        ("constant", "linear_increasing"):  "movement_onset_increasing",
        ("constant", "linear_decreasing"):  "movement_onset_decreasing",
        ("constant", "convex_increasing"):  "movement_onset_increasing",
        ("constant", "concave_increasing"): "movement_onset_increasing",
        ("constant", "convex_decreasing"):  "movement_onset_decreasing",
        ("constant", "concave_decreasing"): "movement_onset_decreasing",

        # --- plateau onset: moving -> constant (slope enters zero) ---
        ("linear_increasing",  "constant"): "plateau_onset",
        ("linear_decreasing",  "constant"): "plateau_onset",
        ("convex_increasing",  "constant"): "plateau_onset",
        ("concave_increasing", "constant"): "plateau_onset",
        ("convex_decreasing",  "constant"): "plateau_onset",
        ("concave_decreasing", "constant"): "plateau_onset",

        # --- straightening: curved -> linear (curvature enters zero, slope sign kept) ---
        ("convex_increasing",  "linear_increasing"): "straightening",
        ("concave_increasing", "linear_increasing"): "straightening",
        ("convex_decreasing",  "linear_decreasing"): "straightening",
        ("concave_decreasing", "linear_decreasing"): "straightening",

        # --- bending onset: linear -> curved (curvature leaves zero) ---
        ("linear_increasing", "convex_increasing"):  "bending_onset",
        ("linear_increasing", "concave_increasing"): "bending_onset",
        ("linear_decreasing", "convex_decreasing"):  "bending_onset",
        ("linear_decreasing", "concave_decreasing"): "bending_onset",
    }
    
    turning_summary = []
    for (state_a, _), (state_b, time_b) in zip(shape_summary[:-1], shape_summary[1:]):
        transition_type = transition_types.get((state_a, state_b))
        if transition_type is not None:
            turning_summary.append((transition_type, time_b))

    return turning_summary