"""Figure 6: What the cloud is unsure about, at a rebound it missed and at one
it caught.

Requires: meanonly_tumour_homogeneous, gaussian_tumour_heterogeneous

    python thesis/thesis_figure_6.py
"""

from pathlib import Path
import sys
import textwrap

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

from scripts.shape_uncertainty.evaluation.clinical_error_analysis import (
    ClinicalErrorEvaluator)
from scripts.shape_uncertainty.evaluation.rebound_localisation import (
    align_profile_on_offsets, resample_profile_on_grid)
from scripts.shape_uncertainty.model.model import GaussianModel, MeanOnlyModel
from scripts.shape_uncertainty.model.model_inference import UncertaintyEngine

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# Plot layout configuration. The spacing is set on the layout engine rather
# than through gridspec, which constrained layout overrides, as in figure 7.
ASPECT = 0.62
HEIGHT_RATIOS = [3.0, 1.0]
LAYOUT_PADS = {"hspace": 0.03, "wspace": 0.06, "h_pad": 0.02, "w_pad": 0.02}

OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_6.pdf"

# Aleatoric draws per ensemble member, as the random-effects run scored them.
NR_DRAWS = 20

# The settings the saved scores must have been produced under. Checking these
# also refuses a cache that does not describe this ensemble and this split.
SCORE_SETTINGS = {"nr_draws": NR_DRAWS, "nr_members": config.NR_MEMBERS,
                  "seed_eval": config.SEED_EVAL, "n_dense": config.N_DENSE,
                  "nr_test": config.D_TEST}

# The aligned axis, in time units either side of t*. Generous, since the drawn
# range is cut back to what the individuals actually cover.
OFFSETS = np.linspace(-0.90, 0.30, 241)

# How far before the turn to draw. Near enough that the missed rebounds all
# still reach it, so their left flank is one group throughout.
LEFT_LIMIT = -0.50

# The grid each profile is read on before it is aligned.
N_DENSE = 400

# The captions beneath each panel, which name the group it pools.
MISSED_TITLE = "Rebound reported as a monotone decline"
DETECTED_TITLE = "Rebound reported correctly"

# Characters per line of a panel caption, wide enough to hold either title and
# its label on one line, so the two panels sit on the same baseline.
CAPTION_WIDTH = 46

# One curve per configuration, in legend order. Each carries
#     "heading":            str; the legend entry.
#     "regime":             config.Regime; the regime its members were fitted
#                           under, which fixes the split every index refers to.
#     "run_name":           str; the training run its artefacts were saved by.
#     "model_cls":          the RandomEffectsModel subclass to rebuild.
#     "reads_saved_scores": bool; whether the cloud is read back from the run's
#                           saved draws rather than scored here. The
#                           random-effects cloud nests aleatoric draws inside
#                           ensemble members, so it is too large to hold and its
#                           training run saves it instead.
COLUMNS = [
    {"heading": r"Mean-only (\textsc{tumour} w/o UH)",
     "regime": config.REGIME_HOMOGENEOUS,
     "run_name": "meanonly_tumour_homogeneous",
     "model_cls": MeanOnlyModel,
     "reads_saved_scores": False},
    {"heading": r"Random effects (\textsc{tumour} w/ UH)",
     "regime": config.REGIME_HETEROGENEOUS,
     "run_name": "gaussian_tumour_heterogeneous",
     "model_cls": GaussianModel,
     "reads_saved_scores": True},
]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Restoring what each configuration contributes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def draw_test_split(column):
    """Redraw the split one column's members were fitted on.

    Args:
        column: dict; one entry of COLUMNS.

    Returns:
        SimulatedDataset; the test split every index refers to.
    """
    test, _, _ = data_loader.draw_simulated_split(column["regime"])

    return test


def load_ensemble(column):
    """Restore the bootstrap ensemble one column's training run saved.

    Args:
        column: dict; one entry of COLUMNS.

    Returns:
        BootstrapEnsemble; with its members restored.
    """
    return model_loader.load_bootstrap_ensemble(column["run_name"],
                                                column["model_cls"])


def score_test_set(ensemble, test):
    """Score every individual's epistemic cloud across the ensemble.

    Args:
        ensemble: BootstrapEnsemble; with its members restored.
        test: SimulatedDataset; the test split.

    Returns:
        dict; as predict_with_epistemic_uncertainty returns, carrying "U",
            "consensus", "selected_indices", "profiles" and "shapes".
    """
    uncertainty_engine = UncertaintyEngine(ensemble.engines)

    return uncertainty_engine.predict_with_epistemic_uncertainty(test.X)


def read_saved_scores(column):
    """Read the scores and coefficient draws one column's training run saved.

    Scoring a nested cloud takes about an hour, so the training run does it
    once. The draws are read back rather than redrawn, so that this figure and
    the tables reading this run see the same cloud.

    Args:
        column: dict; one entry of COLUMNS.

    Returns:
        dict; as load_test_scores returns, carrying "U", "V", "references",
            "amplitudes" and "coefficients".
    """
    return model_loader.load_test_scores(column["run_name"],
                                         expected=SCORE_SETTINGS)


def collect_uncertainty_result(column, ensemble, test):
    """Obtain the cloud one column is evaluated against, however it is held.

    Args:
        column: dict; one entry of COLUMNS.
        ensemble: BootstrapEnsemble; with its members restored.
        test: SimulatedDataset; the test split.

    Returns:
        dict; the result the evaluator reads its clouds from, either scored
            here or read back from the training run.
    """
    if column["reads_saved_scores"]:
        return read_saved_scores(column)

    return score_test_set(ensemble, test)


def build_error_evaluator(column):
    """Assemble the clinical error evaluator for one column.

    Args:
        column: dict; one entry of COLUMNS.

    Returns:
        tuple (evaluator, test); the ClinicalErrorEvaluator and the test split
            every index refers to, which the alignment reads its horizon from.
    """
    # Step 1: The split and the members fitted on it
    test = draw_test_split(column)
    ensemble = load_ensemble(column)

    # Step 2: The cloud those members report
    result = collect_uncertainty_result(column, ensemble, test)

    # Step 3: The single trained model a practitioner would report alone
    point_forecast_engine = model_loader.load_inference_engine(
        column["run_name"], column["model_cls"])

    # Step 4: Hand all three to the evaluator
    evaluator = ClinicalErrorEvaluator(
        test=test,
        uncertainty_engine=UncertaintyEngine(ensemble.engines),
        result=result,
        point_forecast_engine=point_forecast_engine,
        shape_config=config.SHAPE_CONFIG,
    )

    return evaluator, test


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Selecting the individuals each panel pools over
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def select_missed_rebounds(evaluator):
    """Return the rebounds reported as a monotone decline.

    Args:
        evaluator: ClinicalErrorEvaluator; over one column's test split.

    Returns:
        np.ndarray of int; indices into the test split.
    """
    return evaluator.get_missed_rebound_individuals()


def select_detected_rebounds(evaluator):
    """Return the rebounds the point forecast reports as such.

    Args:
        evaluator: ClinicalErrorEvaluator; over one column's test split.

    Returns:
        np.ndarray of int; indices into the test split.
    """
    return evaluator.get_detected_rebound_individuals()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Measuring the localisation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def align_profiles_on_turns(evaluator, indices, critical_points, T):
    """Re-express each individual's profile as time since its own turn.

    Args:
        evaluator: ClinicalErrorEvaluator; over one column's test split.
        indices: np.ndarray of int; the individuals to align.
        critical_points: np.ndarray of float; their turning points.
        T: float; the right endpoint of the horizon.

    Returns:
        np.ndarray of shape (len(indices), len(OFFSETS)); np.nan wherever an
            offset falls outside the horizon for that individual.
    """
    times = np.linspace(0.0, T, N_DENSE)
    aligned = np.full((len(indices), len(OFFSETS)), np.nan)

    for position, (index, t_star) in enumerate(zip(indices, critical_points)):
        values = resample_profile_on_grid(evaluator.get_shape_profile(index),
                                          times)
        aligned[position] = align_profile_on_offsets(values, times, t_star,
                                                     OFFSETS, T)

    return aligned


def summarise_localisation(evaluator, test, indices, heading):
    """Pool one group's aligned profiles, and count who stands behind each.

    Args:
        evaluator: ClinicalErrorEvaluator; over one column's test split.
        test: SimulatedDataset; the test split.
        indices: np.ndarray of int; the group to pool.
        heading: str; the configuration's legend entry.

    Returns:
        dict; carrying "heading", "mean_profile" and "counts" aligned with
            OFFSETS, and "nr_pooled" as int. The profile is np.nan wherever no
            individual reaches that offset.
    """
    # Every individual here rebounds, so its turn is real and inside the
    # horizon; the alignment needs no further filtering.
    critical_points = evaluator.compute_critical_points(indices)
    aligned = align_profiles_on_turns(evaluator, indices, critical_points,
                                      test.T)

    counts = np.sum(np.isfinite(aligned), axis=0)
    mean_profile = np.full(len(OFFSETS), np.nan)
    covered = counts > 0
    mean_profile[covered] = np.nanmean(aligned[:, covered], axis=0)

    return {"heading": heading, "mean_profile": mean_profile, "counts": counts,
            "nr_pooled": len(indices)}


def collect_figure_inputs():
    """Summarise both groups under both configurations, before drawing.

    Each configuration is restored once and asked for both groups, since
    rebuilding a nested cloud is the expensive step and the two groups do not
    overlap.

    Returns:
        list of dict; one per panel, carrying "label", "title" and "curves",
            the last being one summarise_localisation entry per configuration.
    """
    panels = [
        {"label": "(a)", "title": MISSED_TITLE,
         "select": select_missed_rebounds, "curves": []},
        {"label": "(b)", "title": DETECTED_TITLE,
         "select": select_detected_rebounds, "curves": []},
    ]

    for column in COLUMNS:
        evaluator, test = build_error_evaluator(column)
        print(f"  {column['heading']}", flush=True)

        for panel in panels:
            summary = summarise_localisation(
                evaluator, test, panel["select"](evaluator), column["heading"])
            panel["curves"].append(summary)
            print(f"    {panel['label']} pooled {summary['nr_pooled']}",
                  flush=True)

    return panels


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. The offsets the curves may be read over
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def find_covered_domain(curves, everyone=False):
    """Return the offsets every curve still has individuals at.

    Args:
        curves: list of dict; as summarise_localisation returns.
        everyone: bool; whether to require every individual rather than at
            least one, which gives the stretch the curves mean one thing over.

    Returns:
        tuple (lower, upper); the offsets bounding that stretch.
    """
    lower, upper = OFFSETS[0], OFFSETS[-1]

    for curve in curves:
        enough = curve["nr_pooled"] if everyone else 1
        covered = OFFSETS[curve["counts"] >= enough]
        lower = max(lower, covered.min())
        upper = min(upper, covered.max())

    return lower, upper


def find_drawn_domain(panels):
    """Return the offsets the whole figure is drawn over.

    Both panels share one axis, so a reader compares them over the same
    stretch. The left edge is fixed and the right is where the first curve runs
    out of individuals.

    Args:
        panels: list of dict; as collect_figure_inputs returns.

    Returns:
        tuple (lower, upper); the offsets to draw between.
    """
    curves = [curve for panel in panels for curve in panel["curves"]]

    return LEFT_LIMIT, find_covered_domain(curves)[1]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5. Drawing
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def add_localisation_profiles(axis, curves):
    """Draw one pooled profile per configuration on the aligned axis.

    Args:
        axis: plt.Axes; the target subplot axis to receive the plot layer.
        curves: list of dict; as summarise_localisation returns.

    Returns:
        None
    """
    # The group sizes differ between the panels and are read off the counts
    # beneath, so the legend names the configuration alone.
    for curve, colour in zip(curves, plot_style.CYCLE):
        axis.plot(OFFSETS, curve["mean_profile"], lw=1.4, color=colour,
                  zorder=3, label=curve["heading"])

    plot_helpers.add_transition_rules(axis, [0.0])
    axis.set_ylim(0.0, config.Q_MAX)


def add_support_counts(axis, curves):
    """Draw how many individuals stand behind each offset.

    Args:
        axis: plt.Axes; the target subplot axis to receive the plot layer.
        curves: list of dict; as summarise_localisation returns.

    Returns:
        None
    """
    for curve, colour in zip(curves, plot_style.CYCLE):
        axis.step(OFFSETS, curve["counts"], where="mid", lw=1.2, color=colour,
                  zorder=2)

    plot_helpers.add_transition_rules(axis, [0.0])
    axis.set_ylim(bottom=0.0)


def shade_partial_support(axis, drawn, full):
    """Shade the stretches where a curve no longer averages over everyone.

    Either side may be empty, where the drawn domain is narrower than the
    individuals cover.

    Args:
        axis: plt.Axes; the target subplot axis to receive the plot layer.
        drawn: tuple (lower, upper); the domain being drawn.
        full: tuple (lower, upper); the domain of full support.

    Returns:
        None
    """
    for start, end in ((drawn[0], full[0]), (full[1], drawn[1])):
        if start < end:
            axis.axvspan(start, end, color=plot_style.GRAY_LIGHT, alpha=0.18,
                         lw=0.0, zorder=0)


def label_panel_axis(axis, caption):
    """Name the shared time axis, and caption the panel under it.

    The caption rides on the axis label so that the layout engine reserves room
    for it; a figure-level text would be placed before the panels are sized.

    Args:
        axis: plt.Axes; the lower subplot of the column to label.
        caption: str; the caption, already labelled (a) or (b).

    Returns:
        None
    """
    axis.set_xlabel(rf"$t - t^{{*}}$"
                    "\n\n"
                    f"{textwrap.fill(caption, CAPTION_WIDTH)}")


def plot_localisation(panels, aspect=ASPECT):
    """Orchestrate the two panels, each over its profile and its support.

    Args:
        panels: list of dict; as collect_figure_inputs returns.
        aspect: float; figure height divided by width.

    Returns:
        plt.Figure; the complete figure.
    """
    fig, axes = plot_style.subplots(
        1.0, aspect=aspect, nrows=2, ncols=len(panels), sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.get_layout_engine().set(**LAYOUT_PADS)

    drawn = find_drawn_domain(panels)

    for position, panel in enumerate(panels):
        ax_profile, ax_count = axes[0][position], axes[1][position]

        add_localisation_profiles(ax_profile, panel["curves"])
        add_support_counts(ax_count, panel["curves"])

        # Mark the stretches the curves stop averaging over everyone, on both
        # panels, so the profile and its support are read against one region.
        full = find_covered_domain(panel["curves"], everyone=True)
        for axis in (ax_profile, ax_count):
            shade_partial_support(axis, drawn, full)

        label_panel_axis(ax_count, f"{panel['label']} {panel['title']}")

    # The axes carry one shared quantity each, so only the left column is named
    axes[0][0].set_ylabel(r"mean $\widehat{Q}(t \mid \mathbf{x})$")
    axes[1][0].set_ylabel("individuals")
    axes[0][0].legend(loc="upper left")
    axes[1][0].set_xlim(*drawn)

    return fig


def report_localisation(panels):
    """Print what each panel pools and where its curves may be read.

    Args:
        panels: list of dict; as collect_figure_inputs returns.

    Returns:
        None
    """
    lower, upper = find_drawn_domain(panels)
    print(f"\nDrawn over [{lower:+.3f}, {upper:+.3f}]")

    for panel in panels:
        full = find_covered_domain(panel["curves"], everyone=True)
        print(f"\n{panel['label']} {panel['title']}")
        print(f"  every individual over [{full[0]:+.3f}, {full[1]:+.3f}]")

        for curve in panel["curves"]:
            at_turn = curve["mean_profile"][np.argmin(np.abs(OFFSETS))]
            after = curve["mean_profile"][np.argmin(np.abs(OFFSETS - 0.10))]
            print(f"    {curve['heading']:<24} n = {curve['nr_pooled']:>3}   "
                  f"Q at the turn {at_turn:.3f}   Q at +0.10 {after:.3f}")


if __name__ == "__main__":
    plot_style.use()

    panels = collect_figure_inputs()
    report_localisation(panels)
    fig = plot_localisation(panels)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(fig, str(OUTPUT_PATH))
    print(f"\nWrote {OUTPUT_PATH}")
