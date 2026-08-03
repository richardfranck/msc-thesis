import matplotlib.pyplot as plt
import numpy as np

from scripts.shape_uncertainty.model.model_evaluation import ModelEvaluator


# Plot layout and resolution configurations
N_DENSE = 200
N_COLS = 3

# Shape states abbreviated for display inside the band strip.
STATE_LABELS_SHORT = {
    "convex_increasing":  "cvx\ninc.",
    "convex_decreasing":  "cvx\ndec.",
    "concave_increasing": "ccv\ninc.",
    "concave_decreasing": "ccv\ndec.",
    "linear_increasing":  "lin\ninc.",
    "linear_decreasing":  "lin\ndec.",
    "constant":           "const.",
}

# Colours 
TRUE_CURVE_COLOUR   = "#2a78d6"   # blue
OBSERVATION_COLOUR  = "#eb6834"   # orange
MODEL_FIT_COLOUR    = "#1baf7a"   # aqua

# Ensemble marks.
CLOUD_COLOUR        = "#c3c2b7"   # pale warm grey
CONSENSUS_COLOUR    = "#0b0b0b"   # near-black

# Panel fills and chrome
BAND_FILL           = "#f0efec"   # off-white
UNCERTAINTY_FILL    = "#52514e"   # dark warm grey (same as SPREAD_FILL: the two
                                  # disagreement panels are read as a pair)


def _add_true_curve(axis, dataset, i, times):
    """Draw noise-free, ground truth data-generating curve onto an axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis to receive the plot layer.
        dataset (SimulatedDataset): A simulated dataset instance.
        i (int): Index of a sample in the dataset.
        times: A sorted np.ndarray of the observation times for individual i

    Returns:
        None
    """
    y_true = dataset.true_curves_at(times, indices=[i])[0]
    axis.plot(times, y_true, lw=2, color=TRUE_CURVE_COLOUR, label="True Curve")


def _add_noisy_observations(axis, dataset, i):
    """Plot the observed noisy outcome data points onto an axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis to receive the plot layer.
        dataset (SimulatedDataset): A SimulatedDataset instance.
        i (int): Index of a sample in the dataset.

    Returns:
        None
    """
    axis.scatter(
            dataset.times[i],
            dataset.Y_noisy[i],
            s=24, # Size
            alpha=0.7, # Transparency
            color=OBSERVATION_COLOUR,
            label="Noisy Obs",
            zorder=3, # Pulled to the front.
        )


def _add_estimated_curve(axis, inference_engine, dataset, i, times, 
                         color=MODEL_FIT_COLOUR, lw=2, linestyle="--",
                         alpha=1.0, label="Model Fit", zorder=2):
    """Plot the model's predicted continuous mean trajectory onto an axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis to receive the plot layer.
        inference_engine (InferenceEngine): A populated inference tracking model engine.
        dataset (SimulatedDataset): The source evaluation dataset container.
        i (int): Index of a sample in the dataset.
        times (np.ndarray): 1-D sequence of time points in [0, T] at which 
            to evaluate the mean trajectory. This is an evaluation grid used for plotting.
        color, lw, linestyle, alpha, zorder: line styling. The defaults are the
            single-model appearance;
        label (str or None): legend entry.

    Returns:
        None
    """
    # Step 1: Build a one-row column dict for this individual
    X = {name: np.asarray([dataset.X[name][i]], dtype=float) for name in dataset.X}

    # Step 2: Predict; predict_trajectory returns numpy, already in original units
    y_pred = inference_engine.predict_trajectory(X, times)[0]

    # Step 3: Draw the line
    axis.plot(times, y_pred, lw=lw, linestyle=linestyle, color=color,
        alpha=alpha, label=label, zorder=zorder)

def plot_individual_diagnostic(
    dataset,
    indices,
    inference_engine,
    show_true=True,
    show_noisy=True,
    show_estimated=True,
):
    """Orchestrate atomic plot components into a grid layout for individual analysis.

    Args:
        dataset (SimulatedDataset): The simulated source dataset.
        indices (iterable): Integer indices of specific subjects to plot.
        inference_engine (InferenceEngine): A trained model engine.
        show_true (bool, optional): Toggles drawing the generator line.
        show_noisy (bool, optional): Toggles scattering observed dataset points.
        show_estimated (bool, optional): Toggles drawing the model's prediction line..

    Returns:
        plt.Figure: The complete multi-panel diagnostic figure.
    """
    # Step 1: Materialize indices and establish target grid dimensions
    indices = list(indices)
    k = len(indices)
    ncols = min(N_COLS, k)
    nrows = int(np.ceil(k / ncols))

    # Step 2: Initialize canvas and continuous tracking timeline
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    times = np.linspace(0.0, inference_engine.T, N_DENSE)

    # Step 3: Iterate over subjects and systematically layer active visual components
    for ax, i in zip(axes.flat, indices):

        if show_true:
            _add_true_curve(ax, dataset, i, times)

        if show_noisy:
            _add_noisy_observations(ax, dataset, i)

        if show_estimated and inference_engine is not None:
            _add_estimated_curve(ax, inference_engine, dataset, i, times)

        # Step 4: Apply clean labels and individual identification headers
        ax.set_title(f"Individual i={i}", fontsize=9)
        ax.set_xlabel("t")
        ax.set_ylabel("y")

    # Step 5: Shut off unneeded canvas regions and anchor uniform formatting
    for ax in axes.flat[k:]:
        ax.axis("off")

    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()

    return fig


def _add_ensemble_cloud(axis, engines, dataset, i, times, skip=None):
    """Draw every ensemble member's trajectory as one cloud.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis.
        engines (sequence): The ensemble's InferenceEngines.
        dataset (SimulatedDataset): The source evaluation dataset container.
        i (int): Index of a sample in the dataset.
        times (np.ndarray): 1-D sequence of time points at which to evaluate.
        skip (int or None): Index of a member to omit.

    Returns:
        None
    """
    # Step 1: Draw each member, unlabelled
    for member, engine in enumerate(engines):
        if member == skip:
            continue
        _add_estimated_curve(axis, engine, dataset, i, times,
                             color=CLOUD_COLOUR, lw=0.6, linestyle="-",
                             alpha=0.55, label=None, zorder=1)

    # Step 2: Add one legend entry for the cloud
    axis.plot([], [], lw=0.6, color=CLOUD_COLOUR,
              label=f"Ensemble (M={len(engines)})")


def _add_aleatoric_cloud(axis, times, curves, skip=None):
    """Draw one individual's cloud of aleatoric trajectory draws onto an axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis.
        times (np.ndarray): 1-D sequence of time points, shared by all curves.
        curves (np.ndarray): Shape (n_curves, len(times)); the trajectories.
        skip (int or None): Index of a member to omit, for one drawn
            separately on top as the consensus.

    Returns:
        plt.Axes
    """
    # Step 1: Draw each trajectory, unlabelled, skipping the highlighted member
    for index, trajectory in enumerate(curves):
        if index == skip:
            continue
        axis.plot(times, trajectory, color=CLOUD_COLOUR, lw=0.6, linestyle="-",
                  alpha=0.55, label=None, zorder=1)

    # Step 2: Add one legend entry for the cloud
    axis.plot([], [], lw=0.6, color=CLOUD_COLOUR, label=f"Aleatoric draws (n={len(curves)})")

    return axis

def _plot_shape_bands(axis, summary, T):
    """Draw a shape summary as a labelled strip of bands along the time axis.

    Each state occupies the interval from its own start time to the next state's
    start time, with the final state running to T, and is named inside its own
    band. The bands are one neutral fill separated by white rules.

    Args:
        axis (plt.Axes): A short subplot used solely for the strip.
        summary (list): The shape summary, as (state, start_time) tuples.
        T (float): The right endpoint of the forecasting horizon.

    Returns:
        None
    """
    # Step 1: Each state runs until the next one starts; the last runs to T
    start_times = [time for _, time in summary]
    end_times = start_times[1:] + [T]

    # Step 2: Shade one band per state and name it in place
    for (state, start), end in zip(summary, end_times):
        axis.axvspan(start, end, facecolor=BAND_FILL,
                     edgecolor="white", linewidth=1.2)

        axis.text((start + end) / 2.0, 0.5, STATE_LABELS_SHORT[state],
                  ha="center", va="center", multialignment="center",
                  linespacing=1.1, fontsize=7.5, clip_on=True)

    # Step 3: Strip the axis of everything but the bands themselves
    axis.set_xlim(0.0, T)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

def _add_uncertainty_panel(axis, profile, T):
    """Draw the regional epistemic uncertainty profile Q_k onto a subplot axis.

    Args:
        axis (plt.Axes): The target matplotlib subplot axis.
        profile (dict): Dictionary containing the uncertainty profile, expected
            to include keys 'regions' (an array of time interval edges) and
            'region_uncertainty' (the computed Q_k scores for each region).
        T (float): The total time horizon, used to set the x-axis limits.

    Returns:
        None
    """
    regions = profile["regions"]
    edges = np.concatenate([regions[:, 0], [regions[-1, 1]]])
    axis.stairs(profile["region_uncertainty"], edges, fill=True,
                color=UNCERTAINTY_FILL, alpha=0.35, linewidth=0.0)
    axis.set_ylabel(r"$Q_k$")
    axis.set_xlim(0.0, T)
    axis.set_ylim(0.0, max(1e-3, float(profile["region_uncertainty"].max()) * 1.15))



def plot_epistemic_uncertainty_diagnostic(dataset, i, engines, result,
                                show_true=True, show_noisy=True, show_cloud=True):
    """Show one individual's consensus trajectory, shape summary and epistemic uncertainty profile. 

    Three panels on a shared time axis:
        Top:    the consensus trajectory, true curve and the observations 
                + all ensemble members
        Middle: the consensus shape summary as a labelled strip of shape-state bands.
        Bottom: the regional uncertainty Q_k, which localises WHERE on the
                horizon the ensemble disagrees.

    Args:
        dataset (SimulatedDataset): The source evaluation dataset.
        i (int): Index of the individual to plot, as in the batch passed to
            predict_with_uncertainty.
        engines (sequence): The ensemble's InferenceEngines.
        result (dict): As returned by UncertaintyEngine.predict_with_uncertainty.
        show_true (bool): Toggles drawing the generator line. Must be False for
            real data, which has no generating process to evaluate.
        show_noisy (bool): Toggles scattering observed dataset points.
        show_cloud (bool): Toggles drawing every member behind the consensus.

    Returns:
        plt.Figure
    """
    # Step 1: Recover this individual's consensus, profile and score
    T = engines[0].T
    consensus = result["consensus"][i]
    member_index = int(result["selected_indices"][i])
    profile = result["profiles"][i]
    U = result["U"][i]

    # Step 2: Build the canvas and this individual's rendering grid
    fig, (ax_traj, ax_band, ax_unc) = plt.subplots(
        3, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.70, 1.1], "hspace": 0.15},
    )
    times = np.linspace(0.0, T, N_DENSE) 

    # Step 3: Layer the trajectory panel, cloud first so it sits behind
    if show_cloud:
        _add_ensemble_cloud(ax_traj, engines, dataset, i, times, skip=member_index)

    if show_true:
        _add_true_curve(ax_traj, dataset, i, times)

    if show_noisy:
        _add_noisy_observations(ax_traj, dataset, i)

    _add_estimated_curve(ax_traj, engines[member_index], dataset, i, times,
                         color=CONSENSUS_COLOUR, lw=2.2, linestyle="--",
                         label=f"Consensus (member {member_index})", zorder=5)

    ax_traj.set_ylabel("y")
    ax_traj.set_title(f"Individual i={i}    shape uncertainty U = {U:.3f}",
                      fontsize=10)
    ax_traj.legend(fontsize=8, framealpha=0.9)

    # Step 4: Draw the consensus summary as a labelled band strip
    _plot_shape_bands(ax_band, consensus, T)
    ax_band.set_ylabel("shape", fontsize=8, rotation=0, ha="right", va="center")

    # Step 5: Draw the regional uncertainty profile, aligned under the bands
    _add_uncertainty_panel(ax_unc, profile, T)
    ax_unc.set_xlabel("t")

    fig.tight_layout()

    return fig

def plot_aleatoric_uncertainty_diagnostic(dataset, i, engine, result,
                              true_summary=None, show_true=True, show_noisy=True,
                              show_cloud=True):
    """Show one individual's consensus trajectory, shape summary and aleatoric
    uncertainty profile.

     Three panels on a shared time axis:
        Top:    the consensus draw, the true curve and the observations, with the
                rest of the aleatoric cloud behind them.
        Middle: the consensus shape summary as a labelled strip of shape-state bands.
        Bottom: the regional uncertainty Q_k, which localises WHERE on the
                horizon the aleatoric draws disagree.

    Args:
        dataset (SimulatedDataset): The source evaluation dataset.
        i (int): Index of the individual to plot, as in the batch passed to
            predict_with_aleatoric_uncertainty.
        engine (InferenceEngine): The engine whose Sigma produced `result`, i.e.
            the `member` that was requested. Used only to evaluate the stored
            draws, which are expressed in that member's own normalised units.
        result (dict): As returned by
            UncertaintyEngine.predict_with_aleatoric_uncertainty. Must carry
            "coefficients", so the curves shown are the draws that were actually
            summarised rather than a fresh sample.
        show_true (bool): Toggles drawing the generator line. Must be False for
            real data, which has no generating process to evaluate.
        show_noisy (bool): Toggles scattering observed dataset points.
        show_cloud (bool): Toggles drawing every draw behind the consensus.

    Returns:
        plt.Figure
    """
    # Step 1: Recover this individual's consensus, profile, score and draws
    T = engine.T
    times = np.linspace(0.0, T, N_DENSE)

    consensus = result["consensus"][i]
    member_index = int(result["selected_indices"][i])
    profile = result["profiles"][i]
    U = float(result["U"][i])

    # Step 2: Obtain the consensus curve
    consensus_coefficents = result["coefficients"][i]
    curves = engine.predict_aleatoric_trajectories(consensus_coefficents, times)
    
    # Step 3: Build the canvas and this individual's rendering grid
    fig, (ax_traj, ax_band, ax_unc) = plt.subplots(
        3, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.70, 1.1], "hspace": 0.15},
    )

    # Step 4: Layer the trajectory panel, cloud first so it sits behind
    if show_cloud:
        _add_aleatoric_cloud(ax_traj, times, curves, skip=member_index)

    if show_true:
        _add_true_curve(ax_traj, dataset, i, times)

    if show_noisy:
        _add_noisy_observations(ax_traj, dataset, i)

    ax_traj.plot(times, curves[member_index], color=CONSENSUS_COLOUR,
                 lw=2.2, linestyle="--", zorder=5,
                 label=f"Consensus (draw {member_index})")

    ax_traj.set_ylabel("y")
    ax_traj.set_title(
        f"Individual i={i} "
        rf"aleatoric shape uncertainty U = {U:.3f}",
        fontsize=10)
    ax_traj.legend(fontsize=8, framealpha=0.9)

    # Step 5: Draw the consensus summary as a labelled band strip
    _plot_shape_bands(ax_band, consensus, T)
    ax_band.set_ylabel("shape", fontsize=8, rotation=0, ha="right", va="center")

    # Step 6: Draw the regional uncertainty profile, aligned under the bands
    _add_uncertainty_panel(ax_unc, profile, T)
    ax_unc.set_xlabel("t")

    fig.tight_layout()

    return fig