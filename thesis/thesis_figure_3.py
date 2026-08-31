"""Figure 3: Worked example of the shape extraction algorithm.

This file trains a MeanOnlyModel on a frozen architecture and then visualises the
workings of the shape extraction algorithm for an example individual in a panel
with four elements:
    (a) the fitted trajectory, its observations and the ground truth
    (b) y' and y'', whose zero crossings generate the candidate transitions
    (c) the candidate set Gamma, tagged END / KNOT / ROOT
    (d) the resulting shape-state bands, before and after pruning

The architecture is not searched here. It is read from the run that already
searched it, so this worked example and the model the thesis reports at share
one encoder:
    python thesis/training/train_meanonly_tumour_homogeneous.py

    python thesis/thesis_figure_3.py
"""

from pathlib import Path
import sys

import numpy as np
import torch

# The model and extraction packages live at the repository root, one level up.
THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))
sys.path.insert(0, str(THESIS_DIR))

import config
import data_loader
import model_loader
import plot_helpers
import plot_style

from scripts.shape_uncertainty.model.model import MeanOnlyModel
from scripts.shape_uncertainty.model.model_training import Tuner
from scripts.shape_uncertainty.model.model_inference import build_inference_engine
from scripts.shape_uncertainty.shape_extraction.shape_summary import (
    CLASS_END, CLASS_KNOT, CLASS_ROOT, candidate_transitions, compute_amplitude,
    piece_coefficients)
# The generating curve's derivatives are analytic. They are module-private in
# shape_ground_truth because nothing else needed them outside it; panel (b) does.
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import (
    _evaluate_wilkerson_double_prime, _evaluate_wilkerson_prime)

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# Plot layout and resolution configurations
N_DENSE = config.N_DENSE
ASPECT = 1.1
HEIGHT_RATIOS = [3.0, 1.4, 1.4, 0.85, 1.7]

# Output path
OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_3.pdf"

# Index of the individual shown
INDEX = 411

#  The homogeneous regime, i.e. noisy without unobserved heterogeneity
REGIME = config.REGIME_HOMOGENEOUS

# The run whose search selected the architecture this file trains at.
ARCHITECTURE_NAME = "meanonly_tumour_homogeneous"

# Candidate transition classes, drawn on their own rows in panel (c).
CLASS_ROWS = [(CLASS_END, "END", "s"), (CLASS_KNOT, "KNOT", "^"),
              (CLASS_ROOT, "ROOT", "o")]


def build_engine(nr_epochs, verbose=True):
    """Draw the data, train on the frozen architecture, and wrap the model.

    Args:
        nr_epochs: int, training epochs, in place of config.NR_EPOCHS.
        verbose: bool, whether to report progress on stdout.

    Returns:
        tuple (engine, test): a populated InferenceEngine, and the test
            SimulatedDataset.
    """
    # Step 1: Draw the dataset from the regime's DGP and split it. This fixes
    #         every source of randomness, the encoder's initialisation included.
    test, val, train = data_loader.draw_simulated_split(REGIME)

    # Step 2: Build the spline basis this knot configuration implies
    knot_objects = data_loader.build_knot_objects()

    # Step 3: Train at the frozen architecture, with no Optuna search
    if verbose:
        print(f"Training MeanOnlyModel on the {REGIME.name} regime "
              f"for up to {nr_epochs} epochs")
    penalty = config.PENALTIES["MeanOnly"]
    tuner = Tuner(
        train_dataset=train,
        val_dataset=val,
        basis_functions=knot_objects["basis_functions"],
        Omega=knot_objects["Omega"],
        nr_basis=knot_objects["nr_basis"],
        lambda_mean=penalty["lambda_mean"],
        lambda_re=penalty["lambda_re"],
        nr_epochs=nr_epochs,
        model_cls=MeanOnlyModel,
        model_kwargs={},
        patience=config.PATIENCE,
    )
    architecture = model_loader.load_frozen_architecture(ARCHITECTURE_NAME)
    tuned = tuner.train_fixed_configuration(
        architecture, rng=np.random.default_rng(config.SEED))
    if verbose:
        print(f"Validation objective: {tuned['best_value']:.4f}")

    # Step 4: Wrap the trained model, carrying config's extraction thresholds
    engine = build_inference_engine(
        tuned["model"], tuned["normaliser"], knot_objects, config.SHAPE_CONFIG)

    return engine, test


def _individual_pieces(engine, dataset, i):
    """Return the monomial pieces of one individual's fitted spline.

    Args:
        engine: InferenceEngine, a populated inference engine.
        dataset: SimulatedDataset, the source evaluation dataset container.
        i: int, index of a sample in the dataset.

    Returns:
        tuple (coeffs, scale): the (K + 1, 4) monomial coefficients, and the
            factor converting values in those units back to original units.
    """
    X = dataset.select_individuals([i]).X
    W = engine._get_mean_coefficients(X).detach().cpu().numpy()
    coeffs = piece_coefficients(W[0].reshape(-1, 1), engine.knot_objects["C"])
    return coeffs, engine.normaliser.y_std


def _add_trajectory(axis, times, y_true, observation_times, observations, y_pred):
    """Draw panel (a): the fit, the observations and the generating curve.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        y_true: np.ndarray of shape (M,), the generating curve on that grid.
        observation_times: np.ndarray of shape (N,), this individual's own
            observation times, which are not the evaluation grid.
        observations: np.ndarray of shape (N,), the noisy outcomes the model saw.
        y_pred: np.ndarray of shape (M,), the fitted spline on the grid, which
            is the extraction algorithm's input.

    Returns:
        None
    """
    plot_helpers.add_true_curve(axis, times, y_true)
    plot_helpers.add_noisy_observations(axis, observation_times, observations)
    plot_helpers.add_estimated_curve(axis, times, y_pred)

    axis.set_ylabel("$y(t)$")
    # Stacked in the top-right corner, which the descending trajectory leaves
    # empty, so the legend costs no vertical space of its own.
    axis.legend(loc="upper right", ncol=1)


def _true_derivative(dataset, i, times, order):
    """Evaluate the generating curve's derivative on the grid.

    Args:
        dataset: SimulatedDataset, the source evaluation dataset container.
        i: int, index of a sample in the dataset.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        order: int, the derivative order, 1 or 2.

    Returns:
        np.ndarray of shape (M,), the derivative of the generating curve.
    """
    evaluate = (_evaluate_wilkerson_prime if order == 1
                else _evaluate_wilkerson_double_prime)
    return evaluate(times, dataset.X["size"][i], dataset.params["g"][i],
                    dataset.params["d"][i], dataset.params["rho"][i])



def _evaluate_derivative(coeffs, breakpoints, t, order):
    """Evaluate a derivative of the piecewise cubic at arbitrary times.

    Args:
        coeffs: np.ndarray of shape (K + 1, 4), the monomial coefficients,
            each row [c0, c1, c2, c3].
        breakpoints: np.ndarray of shape (K + 2,), interior knots plus
            boundaries.
        t: np.ndarray of shape (M,), evaluation times in [0, T].
        order: int, the derivative order, 1 or 2.

    Returns:
        np.ndarray of shape (M,), the derivative values at t.
    """
    unique_knots = np.unique(breakpoints)
    indices = np.searchsorted(unique_knots, t, side="right") - 1
    indices = np.clip(indices, 0, coeffs.shape[0] - 1)
    c1, c2, c3 = coeffs[indices, 1], coeffs[indices, 2], coeffs[indices, 3]

    if order == 1:
        return c1 + t * (2 * c2 + 3 * c3 * t)
    return 2 * c2 + 6 * c3 * t

def _derivative_and_roots(coeffs, breakpoints, times, scale, order, roots):
    """Evaluate a derivative of the fitted spline and pick out its own roots.

    Args:
        coeffs: np.ndarray of shape (K + 1, 4), the monomial coefficients.
        breakpoints: np.ndarray of shape (K + 2,), interior knots plus
            boundaries.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        scale: float, the factor converting to original units.
        order: int, the derivative order, 1 or 2.
        roots: np.ndarray, the ROOT-class candidate transitions, of either
            derivative.

    Returns:
        tuple (values, root_times, root_values): the derivative on the grid, and
            the times and values of the roots belonging to this derivative.
    """
    values = _evaluate_derivative(coeffs, breakpoints, times, order) * scale
    if not roots.size:
        return values, roots, roots

    at_roots = _evaluate_derivative(coeffs, breakpoints, roots, order) * scale
    tolerance = 1e-8 * max(float(np.max(np.abs(values))), 1.0)
    hit = np.abs(at_roots) <= tolerance
    return values, roots[hit], at_roots[hit]


def _add_derivative_panel(axis, times, true_values, values, root_times,
                          root_values, order):
    """Draw one half of panel (b): a derivative and its marked zero crossings.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        true_values: np.ndarray of shape (M,), the generating curve's derivative.
        values: np.ndarray of shape (M,), the fitted spline's derivative.
        root_times: np.ndarray of shape (R,), the crossings belonging here.
        root_values: np.ndarray of shape (R,), the derivative at those times.
        order: int, the derivative order, 1 or 2.

    Returns:
        None
    """
    plot_helpers.add_true_derivative(axis, times, true_values)
    plot_helpers.add_zero_rule(axis)
    plot_helpers.add_estimated_derivative(axis, times, values)
    plot_helpers.add_root_markers(axis, root_times, root_values)

    axis.set_ylabel(r"$y'(t)$" if order == 1 else r"$y''(t)$")
    axis.margins(y=0.25)



def plot_shape_extraction(engine, dataset, i, aspect=ASPECT):
    """Orchestrate the four panels of the worked example onto one time axis.

    Args:
        engine: InferenceEngine, a populated inference engine.
        dataset: SimulatedDataset, the source evaluation dataset container.
        i: int, index of the individual to plot.
        aspect: float, figure height divided by width.

    Returns:
        plt.Figure, the complete four-panel figure.
    """
        # Step 1: Recover this individual's spline pieces and candidate transitions
    T = engine.T
    times = np.linspace(0.0, T, N_DENSE)
    breakpoints = engine.knot_objects["breakpoints"]
    coeffs, scale = _individual_pieces(engine, dataset, i)
    gamma, classes = candidate_transitions(coeffs, breakpoints)
    roots = gamma[classes == CLASS_ROOT]

    # Step 2: Recover both summaries and the threshold that separates them.
    X = dataset.select_individuals([i]).X
    summary = engine.predict_mean_shape_summary(X)[0]
    unpruned_engine = build_inference_engine(
        engine.model, engine.normaliser, engine.knot_objects,
        {**config.SHAPE_CONFIG, "do_prune": False})
    unpruned_summary = unpruned_engine.predict_mean_shape_summary(X)[0]

    A = compute_amplitude(coeffs, breakpoints, gamma)
    upsilon_prune = config.SHAPE_CONFIG["upsilon_rel_prune"] * A

    # Step 3: Evaluate every curve the figure draws, before any canvas exists
    y_true = dataset.true_curves_at(times, indices=[i])[0]
    y_pred = engine.predict_trajectory(X, times)[0]

    derivative_panels = []
    for order in (1, 2):
        true_values = _true_derivative(dataset, i, times, order)
        values, root_times, root_values = _derivative_and_roots(
            coeffs, breakpoints, times, scale, order, roots)
        derivative_panels.append(
            (order, true_values, values, root_times, root_values))

    candidate_rows = [(gamma[classes == tag], name, marker)
                      for tag, name, marker in CLASS_ROWS]

    # Step 4: Build the canvas, five rows sharing one time axis
    fig, axes = plot_style.subplots(
        1.0, aspect=aspect, nrows=5, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.12})
    ax_traj, ax_d1, ax_d2, ax_gamma, ax_band = axes

    # Step 5: Layer each panel
    for axis in (ax_traj, ax_d1, ax_d2, ax_gamma):
        plot_helpers.add_knot_rules(axis, breakpoints)

    _add_trajectory(ax_traj, times, y_true, dataset.times[i],
                    dataset.Y_noisy[i], y_pred)

    for axis, panel in zip((ax_d1, ax_d2), derivative_panels):
        order, true_values, values, root_times, root_values = panel
        _add_derivative_panel(axis, times, true_values, values, root_times,
                              root_values, order)

    plot_helpers.add_candidate_rows(ax_gamma, candidate_rows)
    ax_gamma.set_ylabel(r"$G$")

    handles = plot_helpers.merge_handles(
        plot_helpers.add_shape_bands(ax_band, unpruned_summary, T, row=1),
        plot_helpers.add_shape_bands(ax_band, summary, T, row=0))

    plot_helpers.style_band_strip(ax_band, ["PRUNED", "NOT\nPRUNED"])
    ax_band.set_ylabel(r"$\Gamma_{\mathcal{V}}(y(t))$")
    
    for axis, note in (
        (ax_gamma, rf"Candidate transition count $|G| = {len(gamma)}$"),
        (ax_band, rf"{len(unpruned_summary)} shape states $\to$ {len(summary)} "
                  rf"at $\upsilon_{{\mathrm{{prune}}}} = "
                  rf"{config.SHAPE_CONFIG['upsilon_rel_prune']}\hat{{A}} = "
                  rf"{upsilon_prune * scale:.3f}$"),
    ):
        plot_helpers.add_panel_note(axis, note)

    # Step 7: Guide lines at the surviving transitions, and the state legend
    transitions = [t_star for _, t_star in summary[1:]]
    for axis in axes:
        plot_helpers.add_transition_rules(axis, transitions)

    ax_band.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.35), ncol=min(4, len(handles)))
    ax_band.set_xlabel("time $t$")
    ax_band.set_xlim(0.0, T)

    return fig

if __name__ == "__main__":
    plot_style.use()
    engine, test = build_engine(nr_epochs=config.NR_EPOCHS)
    fig = plot_shape_extraction(engine, test, INDEX)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(fig, str(OUTPUT_PATH))
    print(f"Wrote {OUTPUT_PATH}")
