import numpy as np

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


def amplitude(coeffs, breakpoints):
    """Return the peak-to-peak amplitude of the spline. 

    Given the matrix of monomial coefficents and array of
    breakpoints, i.e. [0, \xi_1, ..., \xi_K, T], return the 
    peak-to-peak amplitude of the spline. This is needed to
    set the significance thresholds upsilon_1 and upsilon_2 used
    for the shape state assignments.

    The approach evaluates roots strictly within open knot intervals
    and evaluates knots separately.

    Args:
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficents.
        breakpoints: np.ndarray of shape (K + 2,) of interior knots
            plus boundaries.
    Returns: 
        A: scalar amplitude of the spline
    """
    unique_knots = np.unique(breakpoints)
    t_start = unique_knots[:-1]  # Shape: (K + 1,): Left global knot \xi_k
    t_end = unique_knots[1:]     # Shape: (K + 1,):  Right global knot \xi_{k+1}

    c0 = coeffs[:, 0] 
    c1 = coeffs[:, 1]
    c2 = coeffs[:, 2]
    c3 = coeffs[:, 3]

    # Helper function to evaluate the pieces at a given array of t (matching shape K+1)
    def eval_poly(t):
        return c0 + c1 * t + c2 * (t**2) + c3 * (t**3)

    # --------------------------------------------------------------------------
    # Step 1: Evaluate all breakpoints / knots
    # --------------------------------------------------------------------------
    # Evaluate at the left global edge of all pieces
    left_edge_vals = eval_poly(t_start)
    # Evaluate at the very final right edge of the entire horizon
    final_edge_val = eval_poly(t_end)[-1:] 

    # --------------------------------------------------------------------------
    # Step 2: Vectorised ABC formula for interior knots
    # --------------------------------------------------------------------------
    # We set the derivaitive of the piecewise cubic on an iterval equal to zero
    # 3*c3*t^2 + 2*c2*t + c1 = 0. The abc formula solves this as:
    #   a t^2 + b t + c = 0
    #   Solution 1: t_1 = [-b + \\sqrt(b^2 - 4ac)] / 2a 
    #   Solution 2: t_2 = [-b - \\sqrt(b^2 - 4ac)] / 2a 
    a = 3*c3
    b = 2*c2
    c = c1
    discriminant = b**2 - 4*a*c

    # Prevent issues from sqrt of a negative or dividing by zero
    safe_a = np.where(a==0, 1.0, a) # Replace zero entries for a by 1. 
    safe_discriminant = np.where(discriminant<0, 0.0, discriminant)

    # Compute roots safely 
    sqrt_discriminant = np.sqrt(safe_discriminant)
    roots_1 = (-b + sqrt_discriminant) / (2 * safe_a)
    roots_2 = (-b - sqrt_discriminant) / (2 * safe_a)

    # Handle the case where we had a == 0 (must have b != 0)
    safe_b = np.where(b == 0, 1.0, b)
    linear_root_candidates = -c / safe_b
    roots_linear = np.where((a==0) &  (b!=0), linear_root_candidates, -1.0)

    # Build boolean masks to only keep valid, STRICTLY interior roots
    mask_1 = (discriminant >= 0) & (a != 0) & (roots_1 > t_start) & (roots_1 < t_end)
    mask_2 = (discriminant >= 0) & (a != 0) & (roots_2 > t_start) & (roots_2 < t_end)
    mask_lin = (a == 0) & (b != 0) & (roots_linear > t_start) & (roots_linear < t_end)
    
    # Evaluate and extract only the valid interior candidates
    valid_vals_1 = eval_poly(roots_1)[mask_1]
    valid_vals_2 = eval_poly(roots_2)[mask_2]
    valid_vals_lin = eval_poly(roots_linear)[mask_lin]

    # --------------------------------------------------------------------------
    # Step 3: Combine all evaluated values and find range
    # --------------------------------------------------------------------------
    all_candidate_values = np.concatenate([
        left_edge_vals, 
        final_edge_val, 
        valid_vals_1, 
        valid_vals_2, 
        valid_vals_lin
    ])

    spline_amplitude = np.max(all_candidate_values) - np.min(all_candidate_values)

    return spline_amplitude


def absolute_thresholds(upsilon_rel_1, upsilon_rel_2, A, T):
    """ Compute the shape significance thresholds upsilon_1 and upsilon_2. 

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
    upsilon_2 = upsilon_rel_2*A/(T**2)
    return upsilon_1, upsilon_2


def piece_flags(coeffs, breakpoints, upsilon_1, upsilon_2):
    """Return boolean flags for slope_flat and curvature_flat for each knot interval.

    Args:
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficents.
            Each row is [c0, c1, c2, c3]
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus boundaries.
        upsilon_1: scalar for slope significance (absolute tolerance for first derivative)
        upsilon_2: scalar for curvature significance (absolute tolerance for second derivative)
    
     Returns: 
        slope_flat: np.ndarray of shape (K + 1, ) of boolean flags
        curvature_flat: np.ndarray of shape (K + 1, ) of boolean flags
    """
    unique_knots = np.unique(breakpoints)
    t_start = unique_knots[:-1]  # Shape: (K + 1,): Left global knot \xi_k
    t_end = unique_knots[1:]     # Shape: (K + 1,):  Right global knot \xi_{k+1}

    c0 = coeffs[:, 0] 
    c1 = coeffs[:, 1]
    c2 = coeffs[:, 2]
    c3 = coeffs[:, 3]

    # --------------------------------------------------------------------------
    # Step 1: Determine whether a piece is curvature flat
    # --------------------------------------------------------------------------
    # A piece is curvature flat over a knot interval when the maximum curvature,
    # if it were sustained over the entire interval, moves by less than upsilon_2.  
    # We determine the maximum curvature as the (absolute) maximum of the 
    # second-order derivative. Since the second order derivative of a cubic is 
    # linear, we know that this maximum is achieved at a boundary. 
    
    # The second-order derivative is 6*c3*t + 2*c2 
    derivative2_start = 6*c3*t_start + 2*c2 
    derivative2_end = 6*c3*t_end + 2*c2 

    # Obtain a (K+1,) np.array with the max absolute curvature for each interval
    max_derivative2 =np.maximum(np.abs(derivative2_start), np.abs(derivative2_end)) 

    # Obtain boolean flag for curvature flag for each knot interval
    curvature_flat  = max_derivative2 < upsilon_2

    # --------------------------------------------------------------------------
    # Step 2: Determine whether a piece is slope flat
    # --------------------------------------------------------------------------
    # A piece is slope flat over a knot interval when the maximum slope, if it 
    # were sustained over the entire interval, moves by less than upsilon_1. 
    # We determine the maximum slope as the (absolute) maximum of the first-order 
    # derivative. Since the spline is cubic the first order derivative is a square
    # so its maximum is obtained at either of the endpoints or at the extremum
    # of the quadratic, if this falls within the interval. 

    # The first-order derivative is 3*c3*t^2 + 2*c2*t + c1
    derivative1_start = 3*c3*(t_start**2) + 2*c2*t_start + c1
    derivative1_end = 3*c3*(t_end**2) + 2*c2*t_end + c1

    # Obtain a (K+1,) np.array with the max absolute slope for each interval (at boundaries)
    max_derivative1 = np.maximum(np.abs(derivative1_start), np.abs(derivative1_end))

    # Consider the slope at the extremum where it falls within the knot interval.
    
    # Determine the extremum of the quadratic as the root of the second-order 
    # derivative. 
    safe_c3 =  np.where(c3 == 0, 1.0, c3)
    t_extremum = -1/3* c2/safe_c3

    # Write a mask that checks whether the extremum lies within the knot interval
    extremum_mask = (t_start < t_extremum) & (t_extremum < t_end) & (c3 != 0)

    # Evaluate the slope at the extremum
    derivative1_extremum = 3*c3*(t_extremum**2) + 2*c2*t_extremum + c1

    # Potentially update max_derivative1 for intervals where the extremum is in knot interval.
    max_derivative1 = np.where(
        extremum_mask, 
        np.maximum(max_derivative1, np.abs(derivative1_extremum)), 
        max_derivative1
    )

    slope_flat = max_derivative1 < upsilon_1

    return slope_flat, curvature_flat


def candidate_roots(coeffs, breakpoints, slope_flat, curvature_flat, knot_tol=1e-3):
    """Find the roots of the first two derivtives of the cubic spline.

    For each piecewise cubic, respectively, we find the roots of the first- and 
    second-order derivatives. We retain only those roots strictly within the 
    open interval (\xi_k, \xi_{k+1}). We further discard roots of the first
    derivative where the piece flags mark an interval as linear and roots of the 
    second derivative where the piece flags mark an interval as constant. 
    We also discard double roots of the first derivative since those concide
    with roots of the second derivative. Additionally we add all breakpoints to
    obtain the list of candidate roots.

    Args:
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficents.
            Each row is [c0, c1, c2, c3]
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus boundaries.
        slope_flat: np.ndarray of shape (K + 1, ) of boolean flags
        curvature_flat: np.ndarray of shape (K + 1, ) of boolean flags
        knot_tol (float): A tolerance value relative to the forecasting horizon 
            at which a knot and a derivative root are treated as the same. 

    Returns:
        roots: sorted np.array of interior candidate roots. 
    """
    unique_knots = np.unique(breakpoints)
    t_start = unique_knots[:-1]
    t_end = unique_knots[1:]

    c1 = coeffs[:, 1]
    c2 = coeffs[:, 2]
    c3 = coeffs[:, 3]
    
    valid_roots = []

    # -------------------------------------------------------------------------
    # 1. Roots of the second derivative 
    # -------------------------------------------------------------------------
    # The second derivative is: p''(t) =  6*c3*t + 2*c2
    #                                0 =  3*c3*t + c2
    # Find roots while preventing zero division error
    safe_c3 = np.where(c3 == 0, 1.0, c3)
    roots2 = -c2/(3*safe_c3)

    # We find the valid roots of the second derivative of a piece under the 
    # condition that c3 != 0, that the interval is not curvature flat and that 
    # the roots that are strictly inside the open interval of a piece.
    derivative2_mask = (~curvature_flat) & (c3 != 0) & (t_start < roots2) & (roots2 < t_end)

    valid_roots.extend(roots2[derivative2_mask])

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

    # Filter out roots where (1) the piece is slope flat 
    # (2) where we would be dividing by zero in the ABC formula, (3) where 
    # the discriminant would be negative (or zero, leading to a double root),
    # and where (4) the roots are not strictly inside the open interval of a piece. 
    deriv1_mask_r1_plus = (~slope_flat) & (a != 0) & (discriminant > 0) & (t_start < root1_plus) & (root1_plus < t_end)
    deriv1_mask_r1_minus = (~slope_flat) & (a != 0) & (discriminant > 0) & (t_start < roots1_minus) & (roots1_minus < t_end)

    valid_roots.extend(root1_plus[deriv1_mask_r1_plus])
    valid_roots.extend(roots1_minus[deriv1_mask_r1_minus])

    # Handle the case where we had a == 0 (must have b != 0)
    safe_b = np.where(b == 0, 1.0, b)
    linear_root_candidates = -c / safe_b
    deriv_mask_lin = (a == 0) & (b != 0) & (~slope_flat) & (t_start < linear_root_candidates) & (linear_root_candidates < t_end)
    valid_roots.extend(linear_root_candidates[deriv_mask_lin])

    # --------------------------------------------------------------------------
    # 3. Add roots from the knots and remove dublicates
    # --------------------------------------------------------------------------
    # Convert existing derivative roots into a clean numpy array
    derivative_roots = np.array(valid_roots)

    # Add knots, dropping any knot that nearly coincides with a transition root
    forecasting_horizon = unique_knots[-1] - unique_knots[0]
    knot_tol_adjusted = knot_tol * forecasting_horizon

    if len(derivative_roots) > 0:
        # Check absolute distance between every derivative root and every knot
        # np.abs(unique_knots[:, None] - deriv_roots) uses broadcasting to make an 
        # (num_knots, num_deriv_roots) matrix of distances.
        # .any(axis=1) returns True if a knot is close to *any* derivative root.
        is_duplicate_knot = (np.abs(unique_knots[:, None] - derivative_roots) < knot_tol_adjusted).any(axis=1)

        # Keep only the knots that are NOT duplicates of an existing root
        unique_knots_to_add = unique_knots[~is_duplicate_knot]

    else:
        # If no derivative roots were found, keep all unique knots
        unique_knots_to_add = unique_knots

    # Merge the arrays and sort them
    final_roots = np.concatenate([derivative_roots, unique_knots_to_add])

    return np.sort(np.unique(final_roots))


def signs_at(root_candidates, coeffs, breakpoints, slope_flat, curvature_flat):
    """Return the sign of the first- and second-order derivative over each root piece. 

    Args:
        root_candidates: np.ndarray of shape (M,), sorted unique candidate roots 
            (including all breakpoints and derivative roots).
        coeffs: np.ndarray of shape (K + 1, 4) of monomial coefficents.
            Each row is [c0, c1, c2, c3]
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus boundaries.
        upsilon_1: scalar for slope significance (absolute tolerance for first derivative)
        upsilon_2: scalar for curvature significance (absolute tolerance for second derivative)
        slope_flat: np.ndarray of shape (K + 1, ) of boolean flags
        curvature_flat: np.ndarray of shape (K + 1, ) of boolean flags

    Returns:
        (s1, s2): A tuple of two np.ndarray of shape (M - 1,), containing the signs 
            (-1, 0, or 1) of the first- and second-order derivatives over each 
            interval spanned by consecutive candidate roots.
    """
    # Calculate the midpoints between consecutive candidate roots
    # If there are M roots, there will be M - 1 midpoints
    midpoints = (root_candidates[:-1] + root_candidates[1:]) / 2.0

    unique_knots = np.unique(breakpoints)

    # Map each midpoint to its corresponding structural polynomial interval index [0, K]
    indices = np.searchsorted(unique_knots, midpoints, side='right') - 1
    indices = np.clip(indices, 0, len(coeffs) - 1) # Ensures that intervals are in [0, K]
    
    # Extract the monomial coefficients for each midpoint
    c1 = coeffs[indices, 1]
    c2 = coeffs[indices, 2]
    c3 = coeffs[indices, 3]
    
    # Extract flat flags matching the active intervals for each midpoint
    t_slope_flat = slope_flat[indices]
    t_curvature_flat = curvature_flat[indices]
    
    # --------------------------------------------------------------------------
    # 1. Compute first derivative (slope) signs at midpoints
    # --------------------------------------------------------------------------
    deriv1 = 3 * c3 * (midpoints**2) + 2 * c2 * midpoints + c1
    
    # Assign the sign of the first derivative of the relevant monomial to each midpoint. 
    s1 = np.sign(deriv1)              # raw sign, NO threshold here
    # If the interval is globally flagged as slope_flat, overwrite the sign to 0.
    s1 = np.where(t_slope_flat, 0, s1)

    # --------------------------------------------------------------------------
    # 2. Compute second derivative (curvature) signs at midpoints
    # --------------------------------------------------------------------------
    deriv2 = 6 * c3 * midpoints + 2 * c2
    
    # Assign the sign of the second derivative of the relevant monomial to each midpoint. 
    s2 = np.sign(deriv2)
    # If the interval is globally flagged as curvature_flat, overwrite the sign to 0.
    s2 = np.where(t_curvature_flat, 0, s2)
    # If the interval is flagged as slope_flat is is enforced to be curvature flat. 
    # This enforces that if f' == 0 then f'' == 0).
    s2 = np.where(t_slope_flat, 0, s2)
    
    return s1.astype(int), s2.astype(int)


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
    shape_states = { 
        (+1, +1): "convex_increasing",
        (-1, +1): "convex_decreasing",
        (+1, -1): "concave_increasing",
        (-1, -1): "concave_decreasing",
        (+1,  0): "linear_increasing",
        (-1,  0): "linear_decreasing", 
        ( 0,  0): "constant"
    }
    
    valid = {-1, 0, 1}
    if sign_d1 not in valid or sign_d2 not in valid:
        raise ValueError(f"Signs must be in {{-1, 0, 1}}; got ({sign_d1}, {sign_d2}).")

    state = shape_states.get((sign_d1, sign_d2))
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


def extract_shape_summary(w, C, breakpoints, upsilon_rel_1, upsilon_rel_2):
    """Extract a shape summary from a cubic spline. 

    Given a coefficient vector w of a cubic spline, return a shape 
    summary as a list of (shape_state, start_time) tuples. This function
    orchestrates all other shape-primitive helpers.

    Args:
        w: np.ndarray of shape (B, 1) of spline coefficients.
        C: np.ndarray of shape (K + 1, 4, B), where C[k] is the
            conversion matrix for the k-th knot interval, ordered left
            to right over the K + 1 intervals of [0, T].
        breakpoints: np.ndarray of shape (K + 2,) of interior knots plus boundaries.
        upsilon_rel_1 (float): scalar relative slope significance threshold.
        upsilon_rel_2 (float): scalar relative curvature significance threshold.

    Returns:
        list of tuples (str, float): A summary of unique shape states with their 
            corresponding start times, sorted chronologically.
    """
    # Force unique and sorted knots to prevent duplicate calculation
    unique_knots = np.unique(breakpoints)

    # 1. Obtain absolute slope/curvature significance thresholds
    T = unique_knots[-1]
    coeffs = piece_coefficients(w, C)
    A = amplitude(coeffs, breakpoints)
    upsilon_1, upsilon_2 = absolute_thresholds(upsilon_rel_1, upsilon_rel_2, A, T)

    # 2. Obtain slope_flat and curvature_flat flags for each knot piece
    slope_flat, curvature_flat = piece_flags(coeffs, unique_knots, upsilon_1, upsilon_2)

    # 3. Obtain candidate transition points (including the boundary points 0 and T)
    candidate_transitions = candidate_roots(coeffs, unique_knots, slope_flat, curvature_flat)

    # 4. Assign a sign to each candidate shape interval
    s1, s2 = signs_at(candidate_transitions, coeffs, unique_knots, slope_flat, curvature_flat)

    # 5. Assign shape states to each candidate interval in the format (shape_state, start_time)
    candidate_regions = [
        (classify_signs(s1[i], s2[i]), candidate_transitions[i]) 
        for i in range(0, len(s1))
    ]
    
    # 6. Merge adjacent identical shape states to obtain the shape summary
    regions = merge_adjacent(candidate_regions)

    return regions


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