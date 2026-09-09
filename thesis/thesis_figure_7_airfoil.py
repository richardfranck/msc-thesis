"""Figure 7 (airfoil): one individual, read by both model configurations.

Requires: meanonly_airfoil, gaussian_airfoil

    python thesis/thesis_figure_7_airfoil.py
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

from scripts.shape_uncertainty.model.model import GaussianModel, MeanOnlyModel
from scripts.shape_uncertainty.model.model_inference import UncertaintyEngine

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

DATASET = "airfoil"

# The individual both columns describe: the least covered of the split,
# which is where extrapolation is most visible.
INDEX = 5 

OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_7_airfoil.pdf"

# Aleatoric draws per member, matching the simulation's random-effects run.
NR_DRAWS = 20

# The seed the aleatoric draws are taken under, so the figure is reproducible.
SEED = config.SEED_EVAL

N_DENSE = config.N_DENSE

# Plot layout configuration. The spacing is set on the layout engine rather
# than through gridspec, which constrained layout overrides, as in figure 6.
ASPECT = 0.86
HEIGHT_RATIOS = [3.0, 0.8, 1.4, 1.4]
LAYOUT_PADS = {"hspace": 0.03, "wspace": 0.06, "h_pad": 0.02, "w_pad": 0.02}

# How many members of each cloud to draw. The two clouds differ in size by a
# factor of twenty, so a shared stride would flood one panel and thin the
# other; each strides to this count instead, and the panels stay comparable.
CLOUD_CURVES = 50
CLOUD_ALPHA = 0.1

# What the horizon variable means here. The machinery is defined on a generic
# horizon [0, T], so the symbol stays t; only its reading changes.
HORIZON_LABEL = r"normalised log frequency $t$"

# What the outcome is, in its own units rather than standardised ones, since
# the panel carries the recorded observations as well as the prediction.
OUTCOME_LABEL = "sound pressure level (dB)"

# How each column is labelled beneath the figure, so that the text can refer
# to panel (a) and panel (b) rather than to the configurations by name.
PANEL_LETTERS = ("a", "b")

# One column per configuration, in figure order. Each carries
#     "label":     str; the column caption.
#     "run_name":  str; the training run its ensemble was saved by.
#     "model_cls": the RandomEffectsModel subclass to rebuild.
#     "combined":  bool; whether the cloud draws random effects within each
#                  member as well as varying the member, which only the
#                  random-effects model can do.
COLUMNS = [
    {"label": r"Mean-only, epistemic cloud",
     "run_name": f"meanonly_{DATASET}",
     "model_cls": MeanOnlyModel,
     "combined": False},
    {"label": r"Random effects, combined cloud",
     "run_name": f"gaussian_{DATASET}",
     "model_cls": GaussianModel,
     "combined": True},
]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Restoring what each configuration contributes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def load_split():
    """Read the held-out split both columns describe an individual from.

    Returns:
        RealDataset; the test split.
    """
    _, split = data_loader.load_real_split(DATASET)

    return split.test


def load_column_ensemble(column, verbose=True):
    """Restore the bootstrap ensemble one column reads its cloud from.

    Args:
        column: dict; one entry of COLUMNS.
        verbose: bool; whether to report how many members were restored.

    Returns:
        BootstrapEnsemble; with its members restored.
    """
    ensemble = model_loader.load_bootstrap_ensemble(column["run_name"],
                                                    column["model_cls"])
    if verbose:
        print(f"  restored {len(ensemble.engines)} members of "
              f"{column['run_name']}", flush=True)

    return ensemble


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Scoring the individual under one cloud or the other
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_reference(test, index, member, consensus, shape_profile,
                    value_profile, uncertainty, value_uncertainty, times):
    """Collect what the four panels read, however the cloud was produced.

    Args:
        test: RealDataset; the split the individual belongs to.
        index: int; the individual drawn.
        member: int; which cloud position supplied the consensus.
        consensus: list of (state, start_time); the consensus summary.
        shape_profile: dict; the regional shape uncertainty profile.
        value_profile: dict; the value-space profile, carrying its curves.
        uncertainty: float; the shape uncertainty U.
        value_uncertainty: float; the value uncertainty V.
        times: np.ndarray; the evaluation grid.

    Returns:
        dict; the curves, summaries and profiles the panels read.
    """
    return {
        "times": times,
        "curves": value_profile["curves"],
        "member": member,
        "observation_times": test.times[index],
        "observations": test.Y_noisy[index],
        "consensus": consensus,
        "shape_profile": shape_profile,
        "value_profile": value_profile,
        "U": uncertainty,
        "V": value_uncertainty,
        "amplitude": value_profile["amplitude"],
        "coverage": float(test.compute_horizon_coverage()[index]),
        "T": test.T,
    }


def score_epistemic_reference(ensemble, test, index):
    """Collect the four panels from a cloud of member means.

    Args:
        ensemble: BootstrapEnsemble; with its members restored.
        test: RealDataset; the split the individual belongs to.
        index: int; the individual to draw.

    Returns:
        dict; as build_reference returns.
    """
    engine = UncertaintyEngine(ensemble.engines)
    result = engine.predict_with_epistemic_uncertainty(test.X)

    member = int(result["selected_indices"][index])
    times = np.linspace(0.0, test.T, N_DENSE)
    value_result = engine.predict_with_epistemic_value_uncertainty(
        test.select_individuals([index]).X, times, references=[member],
        keep_curves=True)

    return build_reference(
        test, index, member, result["consensus"][index],
        result["profiles"][index], value_result["profiles"][0],
        float(result["U"][index]), float(value_result["V"][0]), times)


def score_combined_reference(ensemble, test, index, rng):
    """Collect the four panels from a cloud carrying both sources at once.

    Every member draws NR_DRAWS random effects from its own covariance, so the
    cloud holds one summary per member and draw. Only this individual is
    scored, since the cost is one extraction per cloud position.

    Args:
        ensemble: BootstrapEnsemble; with its members restored.
        test: RealDataset; the split the individual belongs to.
        index: int; the individual to draw.
        rng: np.random.Generator; driving the aleatoric draws.

    Returns:
        dict; as build_reference returns.
    """
    engine = UncertaintyEngine(ensemble.engines)
    result = engine.predict_with_combined_uncertainty(
        test.select_individuals([index]).X, n_samples=NR_DRAWS, rng=rng,
        keep_coefficients=True)

    member = int(result["selected_indices"][0])
    times = np.linspace(0.0, test.T, N_DENSE)
    value_result = engine.predict_with_combined_value_uncertainty(
        result, times, references=[member], keep_curves=True)

    return build_reference(
        test, index, member, result["consensus"][0], result["profiles"][0],
        value_result["profiles"][0], float(result["U"][0]),
        float(value_result["V"][0]), times)


def collect_figure_inputs(index=INDEX, verbose=True):
    """Score the individual under both configurations, before drawing.

    Args:
        index: int; the individual both columns describe.
        verbose: bool; whether to report progress on stdout.

    Returns:
        list of dict; one per column, in the order of COLUMNS.
    """
    test = load_split()
    rng = np.random.default_rng(SEED)

    references = []
    for column in COLUMNS:
        ensemble = load_column_ensemble(column, verbose=verbose)
        references.append(
            score_combined_reference(ensemble, test, index, rng)
            if column["combined"]
            else score_epistemic_reference(ensemble, test, index))

    return references


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Drawing one column's four panels
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def choose_cloud_stride(nr_curves):
    """Return the stride that draws about CLOUD_CURVES members of a cloud.

    Args:
        nr_curves: int; how many trajectories the cloud holds.

    Returns:
        int; the stride to draw at, never below one.
    """
    return max(1, nr_curves // CLOUD_CURVES)


def add_trajectory(axis, reference):
    """Draw the trajectory cloud, observations and consensus curve.

    The cloud is strided to a fixed number of members, so both columns show
    equally many however large their clouds are.

    Args:
        axis: plt.Axes; the trajectory subplot.
        reference: dict; as build_reference returns.

    Returns:
        None
    """
    curves, member = reference["curves"], reference["member"]
    stride = choose_cloud_stride(len(curves))
    skip = member // stride if member % stride == 0 else None

    plot_helpers.add_trajectory_cloud(axis, reference["times"],
                                      curves[::stride], skip=skip,
                                      label="cloud", alpha=CLOUD_ALPHA)
    plot_helpers.add_estimated_curve(axis, reference["times"], curves[member],
                                     label="consensus")
    plot_helpers.add_noisy_observations(
        axis, reference["observation_times"], reference["observations"],
        label="observed", marker_size=4.5, marker_edge_width=1.0, zorder=6)


def add_shape_bands(axis, consensus, T):
    """Draw the consensus shape-summary band.

    Args:
        axis: plt.Axes; the shape-summary subplot.
        consensus: list of (state, start_time); the consensus summary.
        T: float; the right endpoint of the horizon.

    Returns:
        list of Patch; the legend handles for the states shown.
    """
    handles = plot_helpers.add_shape_bands(axis, consensus, T, row=0,
                                           show_labels=False)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    return handles


def add_shape_uncertainty(axis, shape_profile):
    """Draw the regional shape-uncertainty profile.

    Args:
        axis: plt.Axes; the shape-uncertainty subplot.
        shape_profile: dict; carrying "regions" and "region_uncertainty".

    Returns:
        None
    """
    plot_helpers.add_uncertainty_stairs(axis, shape_profile["regions"],
                                        shape_profile["region_uncertainty"])
    axis.set_ylim(0.0, config.Q_MAX)


def add_value_uncertainty(axis, times, value_profile):
    """Draw the value-space uncertainty profile.

    Args:
        axis: plt.Axes; the value-uncertainty subplot.
        times: np.ndarray; the evaluation grid.
        value_profile: dict; carrying "value_uncertainty".

    Returns:
        None
    """
    plot_helpers.add_uncertainty_profile(axis, times,
                                         value_profile["value_uncertainty"])
    axis.set_ylim(0.0, config.V_MAX)


def add_observed_window(axes, observation_times):
    """Shade the stretch of the horizon the observations cover.

    Args:
        axes: sequence of plt.Axes; the panels to carry the shading.
        observation_times: np.ndarray; the individual's measurement times.

    Returns:
        None
    """
    first, last = float(np.min(observation_times)), \
        float(np.max(observation_times))

    for axis in axes:
        axis.axvspan(first, last, color=plot_style.GRAY_LIGHT,
                     alpha=plot_style.BAND_OUTER, lw=0, zorder=0)


def add_column(axes, reference, caption):
    """Draw one configuration's four panels, and return its state handles.

    The caption rides on the axis label of the lowest panel so that the layout
    engine reserves room for it; a figure-level text would be placed before the
    panels are sized.

    Args:
        axes: sequence of four plt.Axes; that column, top to bottom.
        reference: dict; as build_reference returns.
        caption: str; the column caption, already labelled (a) or (b).

    Returns:
        list of Patch; the legend handles for the states shown.
    """
    trajectory_axis, band_axis, shape_axis, value_axis = axes

    add_observed_window((trajectory_axis, shape_axis, value_axis),
                        reference["observation_times"])
    add_trajectory(trajectory_axis, reference)

    handles = add_shape_bands(band_axis, reference["consensus"],
                              reference["T"])
    add_shape_uncertainty(shape_axis, reference["shape_profile"])
    add_value_uncertainty(value_axis, reference["times"],
                          reference["value_profile"])

    plot_helpers.add_panel_note(
        shape_axis, rf"$\widehat{{U}}(\mathbf{{x}}) = {reference['U']:.3f}$")
    plot_helpers.add_panel_note(
        value_axis, rf"$\widehat{{V}}(\mathbf{{x}}) = {reference['V']:.3f}$")

    value_axis.set_xlabel(f"{HORIZON_LABEL}\n\n{caption}")
    value_axis.set_xlim(0.0, reference["T"])

    return handles


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. Drawing the figure
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_deployment_uncertainty(references, aspect=ASPECT):
    """Orchestrate the four panels of each configuration side by side.

    The rows share a vertical scale, so only the left column is named and the
    right carries no duplicate tick labels to crowd the gap between them.

    Args:
        references: list of dict; as collect_figure_inputs returns.
        aspect: float; figure height divided by width.

    Returns:
        plt.Figure; the complete figure.
    """
    figure, axes = plot_style.subplots(
        1.0, aspect=aspect, nrows=4, ncols=len(references), sharex="col",
        sharey="row", gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    figure.get_layout_engine().set(**LAYOUT_PADS)

    handles = []
    for position, (column, reference) in enumerate(zip(COLUMNS, references)):
        caption = f"({PANEL_LETTERS[position]}) {column['label']}"
        handles = plot_helpers.merge_handles(
            handles, add_column(axes[:, position], reference, caption))

    # The cloud, consensus and observations are drawn alike in both columns,
    # so one key serves the figure.
    axes[0][0].legend(loc="lower left", ncol=1)

    for axis, name in zip(axes[:, 0],
                          (OUTCOME_LABEL,
                           r"$\widehat{\bar{\Gamma}}_{\mathcal{V}}$",
                           r"$\widehat{Q}(t \mid \mathbf{x})$",
                           r"$\widehat{V}(t \mid \mathbf{x})$")):
        axis.set_ylabel(name)

    figure.legend(handles=handles, loc="outside lower center",
                  ncol=min(4, len(handles)), frameon=False)

    return figure


def report_reference(reference, column, index=INDEX):
    """Print what one column reports for the individual.

    Args:
        reference: dict; as build_reference returns.
        column: dict; one entry of COLUMNS.
        index: int; the individual being drawn.

    Returns:
        None
    """
    states = [state for state, _ in reference["consensus"]]
    rising = any(state.endswith("increasing") for state in states)

    print(f"\n{column['label']}, individual {index}")
    print(f"  shape uncertainty U:  {reference['U']:.4f}")
    print(f"  value uncertainty V:  {reference['V']:.4f}")
    print(f"  horizon coverage:     {reference['coverage']:.1%}")
    print(f"  consensus summary:    {states}")
    print(f"  reports an increase:  {rising}")


if __name__ == "__main__":
    plot_style.use()

    print(f"Figure 7 ({DATASET}): individual {INDEX}, both configurations")
    references = collect_figure_inputs()
    for column, reference in zip(COLUMNS, references):
        report_reference(reference, column)

    figure = plot_deployment_uncertainty(references)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(figure, str(OUTPUT_PATH))
    print(f"\nWrote {OUTPUT_PATH}")
