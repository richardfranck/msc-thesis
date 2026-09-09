"""Figure 2: Least squares on one individual, in the basis the model uses.

    python thesis/thesis_figure_2.py
"""

from pathlib import Path
import sys

import numpy as np

# The simulation and extraction packages live at the repository root, one level up.
THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))

import config
import plot_helpers
import plot_style

from scripts.shape_uncertainty.simulated_data.data_generator import (
    WilkersonDGP, generate_simulated_dataset)
from scripts.shape_uncertainty.spline_basis.bspline_basis import (
    basis_matrix, build_knot_dictionary)
from scripts.shape_uncertainty.shape_extraction.shape_summary import (
    extract_shape_summary)
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import (
    get_analytic_shape_summary_single)

# Plot layout and resolution configurations
N_DENSE = config.N_DENSE
ASPECT = 0.70
HEIGHT_RATIOS = [3.0, 1.4]

# Output path
OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_2.pdf"

# Index of the individual shown, the one figure 3 works through
INDEX = 411

#  The homogeneous regime, i.e. noisy without unobserved heterogeneity
REGIME = config.REGIME_HOMOGENEOUS


def build_fit():
    """Generate the data and fit one individual by least squares.

    The coefficients solve Phi w = y over this individual's observations alone.
    With config.NR_BASIS coefficients for config.NR_OBS observations the system
    is overdetermined, so the fit smooths rather than interpolates.

    Returns:
        tuple (test, knot_objects, w): the test SimulatedDataset, the spline
            objects this knot configuration implies, and the np.ndarray of
            shape (B,) of fitted coefficients.
    """
    # Step 1: Fix every source of randomness
    np.random.seed(config.SEED)

    # Step 2: Draw the dataset from the regime's DGP and split it
    dgp = WilkersonDGP(T=config.T, hyperparams=REGIME.hyperparams)
    dataset = generate_simulated_dataset(dgp, D=config.D_TOTAL, **REGIME.design)
    test, _, _ = dataset.split_test_val_train(
        D_train=config.D_TRAIN, D_val=config.D_VAL, D_test=config.D_TEST)

    # Step 3: Build the spline basis this knot configuration implies
    knot_objects = build_knot_dictionary(
        nr_interior_knots=config.NR_INTERIOR_KNOTS, T=config.T)

    # Step 4: Solve for the coefficients on this individual's observations
    Phi = basis_matrix(test.times[INDEX], knot_objects["basis_functions"])
    w, *_ = np.linalg.lstsq(Phi, test.Y_noisy[INDEX], rcond=None)

    return test, knot_objects, w


def spline_shape_summary(w, knot_objects):
    """Extract the shape summary of the fitted spline.

    The fit is a cubic spline, so it goes through the repository's extractor
    unchanged. De-duplication, flatness detection and pruning are all off,
    matching config.SHAPE_CONFIG_NAIVE.

    Args:
        w: np.ndarray of shape (B,), the fitted spline coefficients.
        knot_objects: dict, the spline objects from build_knot_dictionary.

    Returns:
        list of tuples (str, float): the shape summary, as (state, start_time)
            pairs sorted chronologically.
    """
    return extract_shape_summary(w.reshape(-1, 1), knot_objects["C"],
                                 knot_objects["breakpoints"],
                                 **config.SHAPE_CONFIG_NAIVE)


def true_shape_summary(dataset, i):
    """Return the generating curve's shape summary, from analytic derivatives.

    Args:
        dataset: SimulatedDataset, the source evaluation dataset container.
        i: int, index of a sample in the dataset.

    Returns:
        list of tuples (str, float): the shape summary, as (state, start_time)
            pairs sorted chronologically.
    """
    naive = config.SHAPE_CONFIG_NAIVE
    return get_analytic_shape_summary_single(
        dataset.X["size"][i], dataset.params["g"][i], dataset.params["d"][i],
        dataset.params["rho"][i], dataset.T,
        upsilon_rel_1=naive["upsilon_rel_1"],
        upsilon_rel_2=naive["upsilon_rel_2"],
        upsilon_rel_prune=naive["upsilon_rel_prune"],
        do_prune=naive["do_prune"])


def _add_trajectory(axis, times, y_true, observation_times, observations, y_fit):
    """Draw panel (a): the fit, the observations and the generating curve.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        y_true: np.ndarray of shape (M,), the generating curve on that grid.
        observation_times: np.ndarray of shape (N,), this individual's own
            observation times, which are not the evaluation grid.
        observations: np.ndarray of shape (N,), the noisy outcomes the
            coefficients were solved against.
        y_fit: np.ndarray of shape (M,), the least-squares fit on the grid.

    Returns:
        None
    """
    # Step 1: The noise-free generating curve and the observations it was fitted to
    plot_helpers.add_true_curve(axis, times, y_true)
    plot_helpers.add_noisy_observations(axis, observation_times, observations)

    # Step 2: The fitted spline, which is what the shape summary describes
    plot_helpers.add_estimated_curve(axis, times, y_fit,
                                     label="least-squares fit")

    axis.set_ylabel("$y(t)$")
    # Stacked in the top-right corner, which the descending trajectory leaves
    # empty, so the legend costs no vertical space of its own.
    axis.legend(loc="upper right", ncol=1)

def plot_least_squares_fit(dataset, i, knot_objects, w):
    """Orchestrate the two panels onto one shared time axis.

    Args:
        dataset: SimulatedDataset, the source evaluation dataset container.
        i: int, index of the individual to plot.
        knot_objects: dict, the spline objects from build_knot_dictionary.
        w: np.ndarray of shape (B,), the fitted spline coefficients.

    Returns:
        plt.Figure, the complete two-panel figure.
    """
    # Step 1: Recover everything to be drawn, before any drawing starts: both
    # shape summaries, and both curves on the shared grid
    T = dataset.T
    times = np.linspace(0.0, T, N_DENSE)
    summary_true = true_shape_summary(dataset, i)
    summary_fit = spline_shape_summary(w, knot_objects)

    y_true = dataset.true_curves_at(times, indices=[i])[0]
    y_fit = basis_matrix(times, knot_objects["basis_functions"]) @ w

    # Step 2: Build the canvas at the full text width, two rows sharing one axis
    fig, axes = plot_style.subplots(
        1.0, aspect=ASPECT, nrows=2, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.06})
    ax_traj, ax_band = axes

    # Step 3: Layer each panel. The generating curve's summary goes on top, so
    # the figure reads downwards from a curve to the shape it actually has. The
    # fit's states go unnamed in place: most of its bands are narrower than a
    # label, and a label wider than its band reads as naming the neighbouring one.
    plot_helpers.add_knot_rules(ax_traj, knot_objects["breakpoints"],
                                interior_only=True)
    _add_trajectory(ax_traj, times, y_true, dataset.times[i],
                    dataset.Y_noisy[i], y_fit)

    handles = plot_helpers.merge_handles(
        plot_helpers.add_shape_bands(ax_band, summary_true, T, row=1),
        plot_helpers.add_shape_bands(ax_band, summary_fit, T, row=0,
                                     show_labels=False))
    plot_helpers.add_panel_note(
        ax_band,
        f"{len(summary_true)} true, {len(summary_fit)} fitted shape states")

    # Step 4: Name the two rows and strip the strip of everything else
    plot_helpers.style_band_strip(ax_band, ["FIT", "TRUTH"])
    ax_band.set_ylabel(r"$\Gamma_{\mathcal{V}}(y(t))$")

    ax_band.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.42), ncol=len(handles))
    ax_band.set_xlabel("time $t$")
    ax_band.set_xlim(0.0, T)

    return fig


if __name__ == "__main__":
    plot_style.use()
    dataset, knot_objects, w = build_fit()
    fig = plot_least_squares_fit(dataset, INDEX, knot_objects, w)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(fig, str(OUTPUT_PATH))
