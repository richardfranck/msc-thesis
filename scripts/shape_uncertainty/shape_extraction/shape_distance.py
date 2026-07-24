import numpy as np

from scripts.shape_uncertainty.shape_extraction.shape_summary import (
    SHAPE_STATES,
    construct_regions,
)

STATE_SIGNS = {state: signs for signs, state in SHAPE_STATES.items()}

def exact_shape_sequence_match(summary_a, summary_b):
    """Check whether two shape summaries have identical state sequences.

    This function compares whether the ordered sequence of shape states of 
    two shape summaries matches exactly. Transition times are ignored.

    Args:
        summary_a: list of (state, start_time) tuples.
        summary_b: list of (state, start_time) tuples.

    Returns:
        bool; True when the two state sequences are identical.
    """
    summary_a_states = [state for state, _ in summary_a]
    summary_b_states = [state for state, _ in summary_b]
    return summary_a_states == summary_b_states


def mean_shape_sequence_match(predicted, true):
    """Share of individuals whose predicted state sequence matches the truth.

    This function only compares the ordered sequence of shape states. 
    It ignroes transition times.

    Args:
        predicted_summaries: list of length D; predicted shape summaries
        [(state, start_time), ...].
        true_summaries: list of length D; reference (analytic) shape summaries.

    Returns:
        float in [0, 1]; the fraction of individuals with an exact state-sequence
            match (0.0 if the lists are empty).
    """
    D = len(predicted)
    if D == 0:
        return 0.0

    matches = 0
    for predicted_summary, true_summary in zip(predicted, true):
        if exact_shape_sequence_match(predicted_summary, true_summary):
            matches += 1

    return matches / D


def _merge_transition_points(summary_a, summary_b, T):
    """Assemble the ordered set of transition points of both two summaries.

    This function builds G defined as the union of the transition points of both 
    shape summaries and the domain boundaries 0 and T. 

    Args:
        summary_a: list of (state, start_time) tuples over [0, T].
        summary_b: list of (state, start_time) tuples over [0, T].
        T: float, the right endpoint of the forecasting horizon.

    Returns:
        np.ndarray of shape (n + 1,); the sorted grid G, with first entry 0.0
            and last entry T.
    """
    # Get the transition points and domain boundaries 
    start_times_a = [time for _, time in summary_a] 
    start_times_b = [time for _, time in summary_b]
    domain_boundaries = [0.0, T]

    # Obtain a sorted array
    combined = start_times_a + start_times_b + domain_boundaries
    return np.unique(np.asarray(combined, dtype=float)) 

def _get_states_at(shape_summary, query_times):
    """Return the shape state a summary assigns at each of the given times.

    Args:
        shape_summary: list of (state, start_time) tuples, sorted chronologically.
        query_times: np.ndarray of shape (n,) of query times in [0, T].

    Returns:
        list of length n of shape state strings.
    """
    # Step 1: Locate the last entry starting at or before each query time
    # The shape summary records each state alongside its start time.
    # The state at a time t is the state of the last entry whose 
    # start time does not exceed t.
    shape_states= [state for state, _ in shape_summary]
    start_times = [time for _, time in shape_summary]
    indices = np.searchsorted(start_times, query_times, side="right") - 1

    # Return a list of shape states active at the provided query times
    return [shape_states[index] for index in indices]

def _get_state_signs(state_label):
    """Return the slope and curvature signs of a shape state label.

    This function maps a shape state label to its slope and curvature signs.    
    It is therefore the inverse of classify_signs in shape_summary.py, which
    maps a sign pair to a shape state label. 

    Args:
        state_label: str, one of the seven shape states produced by classify_signs.

    Returns:
        tuple (s1, s2) of ints in {-1, 0, 1}; the signs of the first and second
            derivative associated with the state.
    """
    # Step 1: Invert the dict from (s1, s2): state_label into state_label: (s1, s2)
    state_signs = STATE_SIGNS
    # Step 2: Obtain the derivative signs associated with the given state label
    derivative_signs = state_signs.get(state_label)
    return derivative_signs

def _compute_pointwise_distance(state_a, state_b, alpha=1/3, beta=1/6):
    """Return the weighted l1 distance between two shape states.

    We compute the pointwise distance between two shape states as the weighted
    Manhatten distance of the absolute disagreement in slope and in curvature:

        d(s_a, s_b) = alpha * |s_a1 - s_b1| + beta * |s_a2 - s_b2|

    We enfore 2(alpha+beta)=1 to ensure d in [0, 1]. Further we set 
    alpha > beta to make slope error more costly than a curvature error. 
    As default we make slope errors twice as costly vs curvature errors.

    Args:
        state_a: str, shape state of the first summary.
        state_b: str, shape state of the second summary.
        alpha: float, slope disagreement weight.
        beta: float, curvature disagreement weight.

    Returns:
        float in [0, 1]; the pointwise distance, attaining 1 only where the two
            states are diametrically opposed in both slope and curvature.
    """
    # Ensure 2(alpha+beta)=1 so d in [0, 1] and that alpha > beta.
    if (beta>alpha) or (2*(alpha+beta) != 1):
        raise ValueError

    # Get the slope and curvature signs of both states.
    s_a1, s_a2 = _get_state_signs(state_a)
    s_b1, s_b2 = _get_state_signs(state_b)

    # Compute the pointwise distance as the weighted Manhatten distance.
    return alpha * abs(s_a1 - s_b1) + beta * abs(s_a2 - s_b2)


def shape_summary_distance(summary_a, summary_b, T, alpha=1/3, beta=1/6):
    """Return the distance between two shape summaries over [0, T].

    Shape summaries can disagree in their shape states and in the persistence
    of a given state, i.e. the transition point timing. To account for both,
    we proceed as follows:
        - merge the transition points of both summaries into a grid of
          regions within which neither summary's assignment changes; 
        - assign each region a score based on the "pointwise distance"
          between its two states
        - get a shape summary distance as a weighted sum of pointwise
          socres where each score is weighted by region duration,
          and normalised by T.

    Args:
        summary_a: list of (state, start_time) tuples over [0, T].
        summary_b: list of (state, start_time) tuples over [0, T].
        T: float, the right endpoint of the forecasting horizon.
        alpha: float, slope disagreement weight.
        beta: float, curvature disagreement weight.

    Returns:
        float in [0, 1]; the share of the horizon over which the two summaries
            disagree, scaled by the severity of each disagreement. Zero for
            identical summaries.
    """
    # Step 1: Merge the transition points into a sorted grid, including boundaries
    gamma = _merge_transition_points(summary_a, summary_b, T)

    # Step 2: Obtain the regions spanned by gamma 
    regions = construct_regions(gamma)

    # Step 3: Compute the length and midpoint of each region
    lengths = regions[:, 1] - regions[:, 0]
    midpoints = (regions[:, 0] + regions[:, 1]) / 2.0

    # Step 4: Obtain the shape state label for reach region under summary_a and summary_b
    states_a = _get_states_at(summary_a, midpoints)
    states_b = _get_states_at(summary_b, midpoints)

    # Step 5: Compute total duration weighted pointwise distance  
    D = 0.0
    for length, state_a, state_b in zip(lengths, states_a, states_b):
        D += length * _compute_pointwise_distance(state_a, state_b, alpha, beta)

    # Step 5: Normalise the  total duration weighted pointwise distance so it is in [0, 1] 
    return float(D / T)


def mean_shape_summary_distance(predicted, true, T, alpha=1/3, beta=1/6):
    """Compute the mean shape summary distance across a dataset.

    Args:
        predicted: list of length D of predicted shape summaries.
        true: list of length D of reference (analytic) shape summaries.
        T: float, the right endpoint of the forecasting horizon.
        alpha: float, slope disagreement weight.
        beta: float, curvature disagreement weight.

    Returns:
        float in [0, 1].
    """
    distances = [shape_summary_distance(predicted_summary, true_summary, T, alpha, beta)
            for predicted_summary, true_summary in zip(predicted, true)]
    return np.mean(distances, dtype=float)
