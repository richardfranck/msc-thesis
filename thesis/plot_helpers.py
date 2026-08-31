import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle

import plot_style

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Constants
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Shape states abbreviated for display inside a band
STATE_LABELS_SHORT = {
    "convex_increasing":  "cvx\ninc.",
    "convex_decreasing":  "cvx\ndec.",
    "concave_increasing": "ccv\ninc.",
    "concave_decreasing": "ccv\ndec.",
    "linear_increasing":  "lin\ninc.",
    "linear_decreasing":  "lin\ndec.",
    "constant":           "const.",
}

# Hatch per shape state
STATE_HATCH = {
    "concave_increasing": "///",
    "concave_decreasing": "\\\\\\",
    "constant":           "..",
}

# Vertical white space between two rows of bands, as a fraction of a row.
ROW_GAP = 0.10

# We use a white rule between neighbouring bands, and the the hatch to mark curvature.
BAND_EDGE_WIDTH = 1.2
HATCH_WIDTH = 0.6

# The ensemble cloud
CLOUD_WIDTH = 0.4 # line width for each cloud member
CLOUD_ALPHA = 0.15 # opacity of each cloud member line


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Trajectory layers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def add_true_curve(axis, times, curve):
    """Draw an individual's noise-free generating curve onto an axis.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        curve: np.ndarray of shape (M,), the generating curve at those times,
            as returned by SimulatedDataset.true_curves_at.

    Returns:
        None
    """
    axis.plot(times, curve, lw=0.9, ls="--", color=plot_style.COLORS["truth"],
              label="true curve", zorder=3)

def add_noisy_observations(axis, times, values, label="noisy obs.",
                           marker_size=3.0, marker_edge_width=0.7, zorder=4):
    """Draw the noisy observations an individual's fit was given onto an axis.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (N,), that individual's own observation
            times. 
        values: np.ndarray of shape (N,), the observed outcomes.
        label: str; the legend entry for the observations.
        marker_size: float; the marker diameter in points.
        marker_edge_width: float; the marker-outline width in points.
        zorder: float; the drawing order of the observations.

    Returns:
        None
    """
    axis.plot(times, values, ls="none", marker="o", ms=marker_size,
              mfc="none", mew=marker_edge_width,
              color=plot_style.COLORS["observations"], label=label,
              zorder=zorder)

def add_estimated_curve(axis, times, curve, label="model fit"):
    """Draw a model's predicted mean trajectory onto an axis.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        curve: np.ndarray of shape (M,), the predicted trajectory at those
            times, in original y units.
        label: str, the legend entry. 

    Returns:
        None
    """
    axis.plot(times, curve, lw=1.4, color=plot_style.COLORS["mean"],
              label=label, zorder=5)

def add_trajectory_cloud(axis, times, curves, skip=None, label=None,
                         alpha=None):
    """Draw a cloud of trajectories as one faint density behind the fit.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        curves: np.ndarray of shape (K, M), the cloud, in original y units. The
            cloud may be an ensemble, a set of aleatoric draws, or the two
            nestes.
        skip: int or None, index of a member to omit, where that member is drawn
            separately on top as the consensus.
        label: str or None, one legend entry standing for the whole cloud.
        alpha: float or None, the opacity of one member, or None for the shared
            CLOUD_ALPHA. A denser cloud reads better at a lower opacity.

    Returns:
        None
    """
    opacity = CLOUD_ALPHA if alpha is None else alpha

    # Step 1: Draw each member unlabelled
    for index, curve in enumerate(curves):
        if index == skip:
            continue
        axis.plot(times, curve, lw=CLOUD_WIDTH,
                  color=plot_style.COLORS["epistemic"], alpha=opacity,
                  zorder=2)

    # Step 2: Legend entry for the cloud
    if label is not None:
        axis.plot([], [], lw=1.0, color=plot_style.COLORS["epistemic"],
                  alpha=0.5, label=label)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Derivative layers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def add_true_derivative(axis, times, values):
    """Draw a derivative of the generating curve onto a derivative panel.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        values: np.ndarray of shape (M,), the derivative at those times.

    Returns:
        None
    """
    axis.plot(times, values, lw=0.9, ls="--", color=plot_style.COLORS["truth"],
              zorder=2)

def add_estimated_derivative(axis, times, values):
    """Draw a derivative of a fitted spline onto a derivative panel.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        values: np.ndarray of shape (M,), the derivative at those times, in
            original y units.

    Returns:
        None
    """
    axis.plot(times, values, lw=1.1, color=plot_style.COLORS["mean"], zorder=3)

def add_root_markers(axis, times, values):
    """Mark the points at which a drawn derivative crosses zero.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (R,), the crossing times.
        values: np.ndarray of shape (R,), the derivative there, which is zero up
            to the tolerance the caller selected them with.

    Returns:
        None
    """
    axis.plot(times, values, ls="none", marker="o", ms=4.0,
              color=plot_style.CYCLE[2], zorder=4, clip_on=False)

def add_candidate_rows(axis, rows):
    """Draw tagged point sets as one row each, along the time axis.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        rows: list of tuples (points, name, marker), ordered top row first.
            points is an np.ndarray of times, name is the row's tick label, and
            marker is a matplotlib marker code.

    Returns:
        None
    """
    # Step 1: One row per class
    for row, (points, _, marker) in enumerate(reversed(rows)):
        if len(points):
            axis.plot(points, np.full(len(points), float(row)), ls="none",
                      marker=marker, ms=4.5, color=plot_style.GRAY_DARK,
                      zorder=3, clip_on=False)

    # Step 2: Name the rows
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([name for _, name, _ in reversed(rows)],
                         fontsize=plot_style.FONT_SIZE_SMALL)
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.tick_params(axis="y", length=0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. Reference rules and annotation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def add_knot_rules(axis, breakpoints, interior_only=False):
    """Draw the knots as faint vertical rules behind everything else.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        breakpoints: np.ndarray of shape (K + 2,), interior knots plus
            boundaries.
        interior_only: bool, whether to skip the two domain boundaries, which
            the axis frame already draws.

    Returns:
        None
    """
    knots = np.unique(breakpoints)
    if interior_only:
        knots = knots[1:-1]
    for xi in knots:
        axis.axvline(xi, color=plot_style.GRAY_LIGHT, lw=0.5, ls=(0, (1, 2)),
                     zorder=0)

def add_knot_labels(axis, breakpoints, labels):
    """Name the interior knots just above the axis, at their own positions.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        breakpoints: np.ndarray of shape (K + 2,), interior knots plus
            boundaries.
        labels: sequence of str, one name per interior knot, in order.

    Returns:
        None
    """
    interior = np.unique(breakpoints)[1:-1]
    for xi, label in zip(interior, labels):
        axis.text(xi, 1.02, label, transform=axis.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=plot_style.FONT_SIZE_SMALL,
                  color=plot_style.GRAY_DARK)

def add_transition_rules(axis, transitions):
    """Draw guide lines down a panel at a summary's transition times.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        transitions: sequence of float, the times to rule.

    Returns:
        None
    """
    for t_star in transitions:
        axis.axvline(t_star, color=plot_style.GRAY_DARK, lw=0.5, alpha=0.5,
                     zorder=1)

def add_zero_rule(axis):
    """Draw the horizontal zero line a derivative panel is read against.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.

    Returns:
        None
    """
    axis.axhline(0.0, color=plot_style.GRAY_DARK, lw=0.7, zorder=2)

def add_panel_note(axis, text):
    """Note what a panel shows, right-aligned just above its own frame.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        text: str, the note, which may carry maths.

    Returns:
        None
    """
    axis.text(1.0, 1.03, text, transform=axis.transAxes, ha="right",
              va="bottom", fontsize=plot_style.FONT_SIZE_SMALL,
              color=plot_style.GRAY_DARK)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5. Shape-state bands
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def state_face(state):
    """Return the fill colour and hatch encoding one shape state.

    Args:
        state: str, one of the keys of STATE_LABELS_SHORT.

    Returns:
        tuple (facecolor, hatch): the fill colour, and a hatch pattern or None
            where the state is carried by its fill alone.
    """
    colour = (plot_style.CYCLE[0] if state.endswith("increasing")
              else plot_style.CYCLE[1] if state.endswith("decreasing")
              else plot_style.GRAY_LIGHT)
    return colour, STATE_HATCH.get(state)

def add_shape_bands(axis, summary, T, row, show_labels=True):
    """Draw a shape summary as one row of a strip of bands along the time axis.

    Each state occupies the interval from its own start time to the next state's
    start time, with the final state running to T. Rows share one axis so that
    the row names can be y-tick labels, leaving the y-label free for the panel.

    Args:
        axis: plt.Axes, the subplot holding the strip.
        summary: list of tuples (str, float), the shape summary as (state,
            start_time) pairs, sorted chronologically.
        T: float, the right endpoint of the forecasting horizon.
        row: int, which row to fill, counting upwards from 0.
        show_labels: bool, whether to name each state inside its own band. Set
            False where the bands are too narrow to hold a name, leaving the
            legend to carry them.

    Returns:
        list of Patch, the legend handles, one per distinct state in the row.
    """
    # Step 1: Each state runs until the next one starts; the last runs to T
    start_times = [time for _, time in summary]
    end_times = start_times[1:] + [T]

    # Step 2: Shade one band per state and name it in place
    handles, seen = [], []
    for (state, start), end in zip(summary, end_times):
        colour, hatch = state_face(state)
        band = Rectangle(
            (start, row + ROW_GAP / 2.0), end - start, 1.0 - ROW_GAP,
            facecolor=to_rgba(colour, plot_style.BAND), hatch=hatch,
            edgecolor="white", linewidth=BAND_EDGE_WIDTH, zorder=2)
        band.set_hatch_linewidth(HATCH_WIDTH)
        axis.add_patch(band)
        if show_labels:
            axis.text((start + end) / 2.0, row + 0.5, STATE_LABELS_SHORT[state],
                      ha="center", va="center", multialignment="center",
                      linespacing=1.1, fontsize=plot_style.FONT_SIZE_SMALL,
                      zorder=4, clip_on=True)
        if state not in seen:
            seen.append(state)
            key = Patch(facecolor=to_rgba(colour, plot_style.BAND), hatch=hatch,
                        edgecolor="white", lw=BAND_EDGE_WIDTH,
                        label=state.replace("_", " "))
            key.set_hatch_linewidth(HATCH_WIDTH)
            handles.append(key)

    return handles

def style_band_strip(axis, row_labels):
    """Strip a band axis of its frame and name its rows.

    Args:
        axis: plt.Axes, the subplot holding the strip.
        row_labels: sequence of str, one per row, ordered BOTTOM row first, to
            match the row indices add_shape_bands counts upwards from 0.

    Returns:
        None
    """
    axis.set_ylim(0.0, float(len(row_labels)))
    axis.set_yticks([row + 0.5 for row in range(len(row_labels))])
    axis.set_yticklabels(list(row_labels),
                         fontsize=plot_style.FONT_SIZE_SMALL)
    axis.tick_params(axis="y", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def merge_handles(upper, lower):
    """Merge two rows' legend handles, keeping the first of each label.

    Args:
        upper: list of Patch, the handles returned for the upper row.
        lower: list of Patch, the handles returned for the lower row.

    Returns:
        list of Patch, de-duplicated by label and in first-seen order.
    """
    merged = list(upper)
    labels = {handle.get_label() for handle in merged}
    for handle in lower:
        if handle.get_label() not in labels:
            labels.add(handle.get_label())
            merged.append(handle)
    return merged

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6. Uncertainty profiles
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def add_uncertainty_stairs(axis, regions, values, color=None, label=None):
    """Draw a piecewise-constant uncertainty profile as a filled step.

    The shape uncertainty Q_k is constant on each region of the transition grid
    and changes only where some member's summary transitions. 

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        regions: np.ndarray of shape (n, 2), the region boundaries R_k.
        values: np.ndarray of shape (n,), one value per region.
        color: the fill colour, or None for the epistemic role colour.
        label: str or None, the legend entry.

    Returns:
        None
    """
    edges = np.concatenate([regions[:, 0], [regions[-1, 1]]])
    axis.stairs(values, edges, fill=True,
                color=color or plot_style.COLORS["epistemic"],
                alpha=plot_style.BAND, linewidth=0.0, label=label, zorder=2)


def add_uncertainty_profile(axis, times, values, color=None, label=None):
    """Draw a continuous uncertainty profile as a filled area.

    The value-space uncertainty V(t) is defined at every instant rather than on
    regions, so it is filled smoothly.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        values: np.ndarray of shape (M,), the profile.
        color: the fill colour, or None for the epistemic role colour.
        label: str or None, the legend entry.

    Returns:
        None
    """
    plot_style.band(axis, times, np.zeros_like(values), values,
                    color=color or plot_style.COLORS["epistemic"], label=label)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7. Basis functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def add_basis_curves(axis, times, matrix, labels, styles, color=None):
    """Draw every member of a spline basis, each told apart by its dash pattern.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        matrix: np.ndarray of shape (M, B), the basis evaluated on that grid,
            one column per basis function.
        labels: sequence of str of length B, the legend entry per member.
        styles: sequence of length B of matplotlib linestyle specifications.
        color: the line colour, or None for the truth role colour.

    Returns:
        None
    """
    for b in range(matrix.shape[1]):
        axis.plot(times, matrix[:, b], lw=1.1,
                  color=color or plot_style.COLORS["truth"],
                  linestyle=styles[b], zorder=2, label=labels[b])
