"""Thesis Figure 4: one individual's forecast, read as a shape summary.

This figure illustrates a static visualisation of the panels needed
for tri-level transparency:
    (a) the cloud of trajectories and the consensus draw
    (b) the consensus shape summary
    (c) the pointwise shape uncertainty Q(t) over the pooled cloud

    python thesis/thesis_figure_4.py
"""

from pathlib import Path
import sys

import numpy as np
import torch

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))
sys.path.insert(0, str(THESIS_DIR))

import config
import data_loader
import model_loader
import plot_helpers
import plot_style

from scripts.shape_uncertainty.model.model import GaussianModel
from scripts.shape_uncertainty.shape_extraction.shape_uncertainty import (
    compute_shape_uncertainty, medoid_summary)
from scripts.shape_uncertainty.shape_extraction.value_uncertainty import (
    compute_value_uncertainty)

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# The individual this figure draws. Chosen for a well-fitted forecast that turns
# once, close to the middle of the horizon.
INDEX = 109

# Plot layout and resolution configurations. The band strip carries one row
# rather than two, so it is given less height than in Figure 5.
N_DENSE = config.N_DENSE
ASPECT = 0.75
HEIGHT_RATIOS = [2.0, 0.7, 1.4]

# The pooled cloud holds NR_MEMBERS * NR_DRAWS trajectories. Panel (a) draws
# every CLOUD_STRIDE-th only; every measurement uses all of them. The cloud is
# drawn fainter than the shared default, so the consensus reads over it.
CLOUD_STRIDE = 10
CLOUD_ALPHA = 0.05

# The data regime, matching the ensemble this figure reads.
REGIME = config.REGIME_HETEROGENEOUS

# The training run this figure reads its ensemble and its draws from.
MODEL_NAME = f"gaussian_tumour_{REGIME.name}"
OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_4.pdf"


def load_fitted_ensemble(verbose=True):
    """Restore the ensemble its training run fitted.

    Args:
        verbose: bool; whether to report how many members were restored.

    Returns:
        BootstrapEnsemble; with its members restored.
    """
    ensemble = model_loader.load_bootstrap_ensemble(MODEL_NAME, GaussianModel)

    if verbose:
        print(f"Restored {len(ensemble.engines)} members of {MODEL_NAME}")

    return ensemble


def load_saved_draws():
    """Read the aleatoric coefficient draws the training run saved.

    Redrawing them here would give a different cloud, so the saved sample is
    reused and this figure and figure 5 report the same object.

    Returns:
        np.ndarray of shape (E, D, n, B); the draws, in member order.
    """
    return np.stack(model_loader.load_test_scores(MODEL_NAME)["coefficients"])


def collect_figure_inputs(ensemble, test, coefficients, index):
    """Rebuild one individual's cloud and score it in both spaces.

    Args:
        ensemble: BootstrapEnsemble; with its members restored.
        test: SimulatedDataset; the test split.
        coefficients: np.ndarray (E, D, n, B); the saved draws.
        index: int; the individual to draw, within the test split.

    Returns:
        dict carrying "times", "curves", "member", "consensus",
            "shape_profile", "U", "V", "amplitude" and "T". The value-space
            score is reported but not drawn.
    """
    # Step 1: The grid every curve and profile is evaluated on
    T = test.T
    times = np.linspace(0.0, T, N_DENSE)

    # Step 2: Pool this individual's block from every member
    summaries, curves = [], []
    for engine, block in zip(ensemble.engines, coefficients):
        individual = block[[index]]                              # (1, n, B)
        summaries.extend(engine.predict_aleatoric_summaries(individual)[0])
        curves.append(engine.predict_aleatoric_trajectories(individual,
                                                            times)[0])
    curves = np.concatenate(curves, axis=0)                      # (E * n, N)

    # Step 3: Score the cloud in shape space and take the draw representing it
    U, shape_profile = compute_shape_uncertainty(summaries, T)
    consensus, member = medoid_summary(summaries, T, shape_profile)

    # Step 4: Score the same cloud in value space, against that draw's amplitude
    V, value_profile = compute_value_uncertainty(curves, times, member)

    return {
        "times": times,
        "curves": curves,
        "member": int(member),
        "consensus": consensus,
        "shape_profile": shape_profile,
        "U": float(U),
        "V": float(V),
        "amplitude": value_profile["amplitude"],
        "T": T,
    }


def add_forecast_cloud(axis, times, curves, member, stride=CLOUD_STRIDE):
    """Draw panel (a): the cloud, with the consensus draw on top of it.

    Args:
        axis: plt.Axes; the target subplot.
        times: np.ndarray (N,); the evaluation grid.
        curves: np.ndarray (M, N); the pooled cloud.
        member: int; which draw supplied the consensus.
        stride: int; draw every stride-th member of the cloud.

    Returns:
        None
    """
    # The consensus is drawn separately, so it is skipped inside the cloud
    skip = member // stride if member % stride == 0 else None

    plot_helpers.add_trajectory_cloud(
        axis, times, curves[::stride], skip=skip, label="cloud",
        alpha=CLOUD_ALPHA)
    plot_helpers.add_estimated_curve(
        axis, times, curves[member], label="consensus")

    axis.set_ylabel(r"$\hat{y}(t)$")
    axis.legend(loc="upper left", ncol=1)


def add_consensus_summary(axis, summary, T):
    """Draw panel (b): the consensus summary as a single strip of bands.

    Only one summary is shown, so the strip needs no row name and the frame is
    stripped away entirely.

    Args:
        axis: plt.Axes; the target subplot.
        summary: list of (state, start_time); the consensus shape summary.
        T: float; the right endpoint of the horizon.

    Returns:
        list of Patch; the legend handles, one per distinct state.
    """
    handles = plot_helpers.add_shape_bands(axis, summary, T, row=0)

    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.set_ylabel(r"$\widehat{\bar{\Gamma}}_{\mathcal{V}}$")

    return handles


def add_shape_uncertainty(axis, regions, values):
    """Draw panel (c): the pointwise shape uncertainty.

    Args:
        axis: plt.Axes; the target subplot.
        regions: np.ndarray (n, 2); the region boundaries.
        values: np.ndarray (n,); the shape uncertainty on each region.

    Returns:
        None
    """
    plot_helpers.add_uncertainty_stairs(axis, regions, values)
    axis.set_ylabel(r"$\widehat{Q}(t \mid \mathbf{x})$")
    axis.set_ylim(0.0, config.Q_MAX)


def plot_consensus_forecast(reference, aspect=ASPECT):
    """Orchestrate the three panels onto one shared time axis.

    Args:
        reference: dict; as collect_figure_inputs returns.
        aspect: float; figure height divided by width.

    Returns:
        plt.Figure; the complete three-panel figure.
    """
    T = reference["T"]
    times = reference["times"]
    consensus = reference["consensus"]

    # Step 1: Build the canvas, three rows sharing one time axis
    fig, axes = plot_style.subplots(
        1.0, aspect=aspect, nrows=3, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.12})
    ax_traj, ax_band, ax_shape = axes

    # Step 2: Layer each panel
    add_forecast_cloud(ax_traj, times, reference["curves"],
                       reference["member"])
    handles = add_consensus_summary(ax_band, consensus, T)
    add_shape_uncertainty(ax_shape, reference["shape_profile"]["regions"],
                          reference["shape_profile"]["region_uncertainty"])

    # Step 3: Report the score on the panel it belongs to
    plot_helpers.add_panel_note(
        ax_shape, rf"$\widehat{{U}}(\mathbf{{x}}) = {reference['U']:.3f}$")

    # Step 4: Name the shape states once, beneath the figure
    ax_shape.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.35), ncol=min(4, len(handles)))
    ax_shape.set_xlabel("time $t$")
    ax_shape.set_xlim(0.0, T)

    return fig


def report_reference_individual(reference, index):
    """Print the figure's individual and its scores.

    Args:
        reference: dict; as collect_figure_inputs returns.
        index: int; the individual's position in the test split.

    Returns:
        None
    """
    states = [state for state, _ in reference["consensus"]]
    transitions = [f"{t_star:.3f}" for _, t_star in reference["consensus"][1:]]

    print(f"individual i={index}, consensus draw {reference['member']}")
    print(f"  consensus summary:    {states}")
    print(f"  transitions at:       {transitions}")
    print(f"  shape uncertainty U:  {reference['U']:.4f}")
    print(f"  value uncertainty V:  {reference['V']:.4f} "
          f"of amplitude {reference['amplitude']:.4f}")


if __name__ == "__main__":
    plot_style.use()

    test, _, _ = data_loader.draw_simulated_split(REGIME)
    ensemble = load_fitted_ensemble()
    coefficients = load_saved_draws()

    reference = collect_figure_inputs(ensemble, test, coefficients, INDEX)
    report_reference_individual(reference, INDEX)

    fig = plot_consensus_forecast(reference)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(fig, str(OUTPUT_PATH))
    print(f"\nWrote {OUTPUT_PATH}")
