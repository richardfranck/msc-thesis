import numpy as np
from functools import lru_cache

from scripts.shape_uncertainty.shape_extraction.shape_summary import (
    SHAPE_STATES,
    construct_regions,
)

from scripts.shape_uncertainty.shape_extraction.shape_distance import (
    _compute_pointwise_distance,
    _get_states_at,
    _merge_transition_points,
)

# Extract shape states and construct a label-to-index mapping
STATE_LABELS = list(SHAPE_STATES.values())
STATE_INDEX = {label: index for index, label in enumerate(STATE_LABELS)}

def _get_shape_indices_by_region(summaries, midpoints):
    """Return the vocabulary index each ensemble member assigns to each region.

    We have an ensemble of size M and are looking to determine the shape state
    of each ensemble member at the midpoint of n regions. For this we construct
     an M-by-n matrix such that element [i, j] is the shape state, identified by
    a shape state index, that ensemble member i occupies in region j. 

    Args:
        summaries: sequence of M shape summaries over [0, T].
        midpoints: np.ndarray of shape (n,) of region midpoints.

    Returns:
        np.ndarray of shape (M, n) and dtype int.
    """
    ensemble_size = len(summaries)
    region_count = midpoints.shape[0]

    state_indices = np.empty((ensemble_size, region_count), dtype=int)
    for i, summary in enumerate(summaries):
        # Get the shape state string labels of the summary at the midpoints
        states = _get_states_at(summary, midpoints)
        # Convert the shape state strings into indices as per STATE_INDEX 
        state_indices[i] = [STATE_INDEX[state] for state in states]

    return state_indices

def _build_shape_state_histograms(state_indices):
    """Construct state histograms for all regions of the transition grid.

    For each region k, this function computes the histogram over the 
    vocabulary states. That is, it counts how many ensemble members 
    occupy each state in a given region k. This is done for all regions,
    respectively.
    We return a matrix of counts with one row per region of the transition
    grid and one column for each shape state. That is, counts[i, j] is the
    frequency count in region i for the jth shape state.  

    Args:
        state_indices: np.ndarray of shape (M, n) of vocabulary indices.
            state_indices[m, r] = state assigned to region r in summary m

    Returns:
        np.ndarray of shape (n, V); dtype int.
    """
    ensemble_size, region_count = state_indices.shape
    vocab_size = len(STATE_LABELS)

    # One histogram per region, one column per possible state
    state_histograms = np.zeros((region_count, vocab_size), dtype=int)

    # Count occurrences
    #for summary in range(ensemble_size):
    #    for region in range(region_count):
    #        state = state_indices[summary, region]
    #        state_histograms[region, state] += 1
    region_indices = np.broadcast_to(np.arange(region_count), (ensemble_size, region_count))
    np.add.at(state_histograms, (region_indices, state_indices), 1)

    return state_histograms

@lru_cache(maxsize=None)
def _build_distance_matrix(alpha=1/2, beta=1/4):
    """Compute the pointwise distance matrix over the shape state vocabulary.

    Entry (i, j) is the pointwise distance between STATE_LABELS[i] and
    STATE_LABELS[j], as defined in _compute_pointwise_distance, computed
    for the given (alpha, beta)-weight tuple. 

    We use lru_cache to cache the results, since the same weights are
    reused across every region, every individual, and every ensemble.

    Args:
        alpha: float, slope disagreement weight.
        beta: float, curvature disagreement weight.

    Returns:
        D (np.ndarray of shape (V, V) and dtype float), the distance matrix
            over all possible state combinations in the vocabulary.
    """
    # Initialize a zero-filled matrix of size V x V
    V = len(STATE_LABELS)
    distance_matrix = np.zeros((V, V), dtype=float)

    # Populate the matrix with all pairwise distances
    for i, state_a in enumerate(STATE_LABELS):
        for j, state_b in enumerate(STATE_LABELS):
            distance_matrix[i, j] = _compute_pointwise_distance(state_a, state_b, alpha, beta)

    return distance_matrix

def _compute_region_uncertainties(state_histograms, distance_matrix, ensemble_size):
    """Return the shape uncertainty Q_k of every region.

    Compute regional uncertainties

        Q_k = 1 / (M (M - 1)) * sum_sigma sum_sigma'
                  n_k(sigma) n_k(sigma') d(sigma, sigma'),

    for every region k; M = ensemble_size

    Args:
        state_histograms: np.ndarray of shape (region_count, V) of state histograms n_k.
        distance_matrix: np.ndarray of shape (V, V), the pointwise distance matrix.
        ensemble_size: int, the ensemble size.

    Returns:
        np.ndarray of shape (region_count,) and dtype float; the Q_k, all in [0, 1].
    """
    # Compute the total pairwise distances for each region
    region_count = state_histograms.shape[0]
    total_pairwise_distance = np.zeros(region_count, dtype=float)
    for region in range(region_count):
        total_pairwise_distance[region] = state_histograms[region] @ distance_matrix @ state_histograms[region]

    # Get the region uncertainties
    M = ensemble_size
    region_uncertainties =  total_pairwise_distance / (M * (M - 1))

    return region_uncertainties

def compute_shape_uncertainty(summaries, T, alpha=1/2, beta=1/4):
    """Compute the shape uncertainty of an ensemble of shape summaries over [0, T].

    The shape uncertainty is the average duration-weighted pairwise shape
    distance across an ensemble of M equally plausible shape summaries. We
    compute it in three steps:

    Step 1: Form regions from the sorted union of the transition points of all M
            summaries together with the domain boundaries 0 and T.
    Step 2: Within each region, build the histogram n_k of member states and
            compute a regional uncertainty score Q_k
    Step 3: Compute the shape uncertainty as the average regional uncertainty score Q_k,
            weighted by region duration and normalised by T.

    Args:
        summaries: sequence of M shape summaries, each a list of
            (state, start_time) tuples over [0, T].
        T: float, the right endpoint of the forecasting horizon.
        alpha: float, slope disagreement weight.
        beta: float, curvature disagreement weight.

    Returns:
        tuple (U, profile) where 
            - U is a float in [0, 1], zero for a unanimous ensemble 
            - profile is a dict with 
            profile = {
                'regions':            np.ndarray (n, 2) -- region boundaries R_k
                'lengths':            np.ndarray (n,)   -- region durations L_k
                'region_uncertainty': np.ndarray (n,)   -- regional uncertainty Q_k
                'state_histograms':   np.ndarray (n, V) -- state histograms n_k
                'state_indices': state_indices
            }
    """
    if len(summaries) < 2:
        raise ValueError(f"At least two summaries are required. Only {len(summaries)} provided")

    # Step 1: Merge the transition point grid and construct regions
    transition_points = _merge_transition_points(summaries, T)
    regions = construct_regions(transition_points)

    # Step 2: Compute region lengths and midpoints
    lengths = regions[:, 1] - regions[:, 0]
    midpoints = (regions[:, 0] + regions[:, 1]) / 2.0

    # Step 3.1: Build the shape state histograms by regions 
    state_indices = _get_shape_indices_by_region(summaries, midpoints)
    state_histograms = _build_shape_state_histograms(state_indices)

    # Step 3.2: Compute pointwise distances for all shape-state pairs
    distance_matrix = _build_distance_matrix(alpha, beta)
    
    # Step 3.3: Compute the shape uncertainties by region Q_k 
    region_uncertainty = _compute_region_uncertainties(state_histograms, distance_matrix, len(summaries))

    # Step 4: Compute ensemble uncertainty
    U = (lengths @ region_uncertainty) / T

    # Step 5: Construct uncertainty profile
    profile = {
        'regions': regions,
        'lengths': lengths,
        'region_uncertainty': region_uncertainty,
        'state_histograms': state_histograms,
        'state_indices': state_indices
    }

    return U, profile

def _compute_region_state_distances(state_histograms, distance_matrix, ensemble_size):
    """Compute the mean distance from each shape state to the ensemble, by region.

    We compute a matrix whose columns are the reference states we might have on
    a given region, holding the distance from that reference state to the
    ensemble. We do this for all regions to obtain a lookup table.
    The formula is,
        delta_k(sigma) = 1 / (M - 1) * sum_sigma' n_k(sigma') d(sigma, sigma'),

    which is the mean pointwise distance from shape state sigma to the states that
    the ensemble holds on region k. 

    Args:
        state_histograms: np.ndarray (region_count, V) of state histograms n_k.
        distance_matrix: np.ndarray of shape (V, V), the pointwise distance matrix.
        ensemble_size: int, the ensemble size M.

    Returns:
        np.ndarray of shape (region_count, V). [k, j] is the mean pointwise distance of
        for the jth state sigma to the states of the ensemble on region k
    """
    # (n, V) x (V, V) -> (n, V)
    region_state_distances = (state_histograms @ distance_matrix) / (ensemble_size - 1)
    return region_state_distances

def _compute_member_distances(state_indices, region_state_distances, lengths, T):
    """Compute the normalised duration-weighted mean distance from each member to the ensemble.

    We compute the distances delta_u for each ensemble member using the
    region_state_distances lookup table from above. 
    Here, delta_u is the duration-weighted mean distance  from each member
    to the ensemble, normalised by the forecasting horizon T, 
        delta_u = 1 / T * sum_k L_k delta_k(sigma_k^(u)).

    Args:
        state_indices: np.ndarray of shape (M, region_count) of vocabulary
            indices; element [u, k] is the state member u holds on region k.
        region_state_distances: np.ndarray of shape (region_count, V) of the
            regional state distances delta_k.
        lengths: np.ndarray of shape (region_count,) of region durations L_k.
        T: float, the right endpoint of the forecasting horizon.

    Returns:
        np.ndarray of shape (M,) and dtype float; the delta_u, all in [0, 1].
            Their mean over the ensemble is the shape uncertainty U, which makes
            them a cheap consistency check across the two algorithms.
    """
    member_count, region_count = state_indices.shape

    #scores = np.zeros(member_count, dtype=float)

    #for member in range(member_count):
        #total = 0.0

        # Accumulate the distance of the state this member holds on each region, weighted by the region duration
        #for region in range(region_count):
            #state = state_indices[member, region]
            #distance = region_state_distances[region, state]
            #total += distance * lengths[region]

        # Normalise by the horizon
        #scores[member] = total / T

    #return scores
    region_indices = np.arange(region_count)
    return (region_state_distances[region_indices, state_indices] @ lengths) / T

def _determine_medoid(member_distances):
    """Return the index of the ensemble member closest to the cloud.

    Return the index of the medoid, i.e. the summary minimising delta_u, 
    its duration-weighted mean pointwise distance to every other ensemble member.

    Ties are broken in favour of the smallest index.

    Args:
        member_distances: np.ndarray of shape (M,) of the delta_u.

    Returns:
        int; the index u* of the medoid summary.
    """
    # Note on tie breaker: np.argmin gives directly by returning the first occurrence of the minimum.
    return int(np.argmin(member_distances))

def medoid_summary(summaries, T, profile, alpha=1/2, beta=1/4):
    """Return the ensemble member that best represents the ensemble.

    The medoid is the summary minimising delta_u, its duration-weighted mean
    pointwise distance to every other member. 

    Step 1: Build a regional distance table delta_k(sigma).
    Step 2: Determine the distance delta_u for each summary. 
    Step 3: Identify the minimiser

    Args:
        summaries: sequence of M shape summaries, each a list of
            (state, start_time) tuples over [0, T].
        T: float, the right endpoint of the forecasting horizon.
        profile: dict as returned by shape_uncertainty.
        alpha: float, slope disagreement weight.
        beta: float, curvature disagreement weight.

    Returns:
        tuple (medoid, medoid_index) 
            - the selected shape summary, 
            - its position in summaries (to recover the trajectory)
    """
    if len(summaries) < 2:
            raise ValueError(f"At least two summaries are required. Only {len(summaries)} provided")

    # Step 1: Build the region distance matrix delta_k
    distance_matrix = _build_distance_matrix(alpha, beta)
    region_state_distances = _compute_region_state_distances(
        profile['state_histograms'], 
        distance_matrix, 
        len(summaries),
    )

    # Step 2: Compute the distances of every ensemble member
    member_distances = _compute_member_distances(
        profile['state_indices'], 
        region_state_distances, 
        profile['lengths'], 
        T,
    )
    # Step 3: Identify the minimiser the closest member, breaking ties towards parsimony
    medoid_index = _determine_medoid(member_distances)

    return summaries[medoid_index], medoid_index
