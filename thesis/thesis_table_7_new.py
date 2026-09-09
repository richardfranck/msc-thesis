"""Table 7: What the shape uncertainty reflects, and what it can detect.

Requires: meanonly_tumour_homogeneous, gaussian_tumour_heterogeneous

    python thesis/thesis_table_7_new.py
"""

from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))
sys.path.insert(0, str(THESIS_DIR))

import config
import data_loader
import model_loader

from scripts.shape_uncertainty.evaluation.clinical_error_analysis import (
    ClinicalErrorEvaluator)
from scripts.shape_uncertainty.model.model import GaussianModel, MeanOnlyModel
from scripts.shape_uncertainty.model.model_inference import UncertaintyEngine
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import (
    get_analytic_shape_summary)

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_7_new.tex"

# Aleatoric draws per ensemble member, as the random-effects run scored them.
NR_DRAWS = 20

# The settings the saved scores must have been produced under.
SCORE_SETTINGS = {"nr_draws": NR_DRAWS, "nr_members": config.NR_MEMBERS,
                  "seed_eval": config.SEED_EVAL, "n_dense": config.N_DENSE,
                  "nr_test": config.D_TEST}

# How many individuals a warning would be issued to, when the top of the
# ranking is read as a shortlist.
SHORTLIST = 20

# One column per configuration, in table order. The heading is carried as two
# lines, which the renderer stacks, so that a column is no wider than the
# figures beneath it. Each entry carries
#     "heading":            str; the first line of the column heading, naming
#                           the model configuration.
#     "subheading":         str; the second line, naming the split it was
#                           fitted on.
#     "regime":             config.Regime; the regime its members were fitted
#                           under, which fixes the split every index refers to.
#     "run_name":           str; the training run its artefacts were saved by.
#     "model_cls":          the RandomEffectsModel subclass to rebuild.
#     "reads_saved_scores": bool; whether the cloud is read back from the run's
#                           saved draws rather than scored here.
COLUMNS = [
    {"heading": r"Mean-only",
     "subheading": r"(\textsc{tumour} w/o UH)",
     "regime": config.REGIME_HOMOGENEOUS,
     "run_name": "meanonly_tumour_homogeneous",
     "model_cls": MeanOnlyModel,
     "reads_saved_scores": False},
    {"heading": r"Random effects",
     "subheading": r"(\textsc{tumour} w/ UH)",
     "regime": config.REGIME_HETEROGENEOUS,
     "run_name": "gaussian_tumour_heterogeneous",
     "model_cls": GaussianModel,
     "reads_saved_scores": True},
]

# The table's rows, grouped so that a space separates the two panels.
#
# Every quantity named in the docstring is still measured and still reported on
# stdout. The table carries only what the section argues from, which is the two
# medians that show the confound, the ranking measure, and the one operating
# point that shows how little of that ranking survives a decision. Average
# precision and the shortlist precision say what the full-recall share already
# says, and the structure-matched area under the curve is a diagnostic rather
# than a setting.
ROW_GROUPS = [
    ("Shape uncertainty by reference-state count", [
        ("median_u_monotone", r"Median $U$ (one reference state)", "{:.3f}"),
        ("median_u_rebounding", r"Median $U$ (two reference states)",
         "{:.3f}"),
    ]),
    ("Missed-rebound detection", [
        ("population_size", r"Monotone declining predictions", "{:d}"),
        ("population_missed", r"\quad of which the curve rebounds", "{:d}"),
        ("area_under_curve", r"AUROC of $U$", "{:.3f}"),
        ("full_recall_share",
         r"Fraction flagged at 100\% missed-rebound recall", "{:.1%}"),
    ]),
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


def load_ensemble(column, verbose=True):
    """Restore the bootstrap ensemble one column's training run saved.

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
        return model_loader.load_test_scores(column["run_name"],
                                             expected=SCORE_SETTINGS)

    return UncertaintyEngine(
        ensemble.engines).predict_with_epistemic_uncertainty(test.X)


def build_error_evaluator(column, verbose=True):
    """Assemble the clinical error evaluator for one column.

    Args:
        column: dict; one entry of COLUMNS.
        verbose: bool; whether to report progress on stdout.

    Returns:
        tuple (evaluator, test); the ClinicalErrorEvaluator and the split every
            index refers to.
    """
    test = draw_test_split(column)
    ensemble = load_ensemble(column, verbose=verbose)
    result = collect_uncertainty_result(column, ensemble, test)

    point_forecast_engine = model_loader.load_inference_engine(
        column["run_name"], column["model_cls"])

    evaluator = ClinicalErrorEvaluator(
        test=test,
        uncertainty_engine=UncertaintyEngine(ensemble.engines),
        result=result,
        point_forecast_engine=point_forecast_engine,
        shape_config=config.SHAPE_CONFIG,
    )

    return evaluator, test


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. What the shape uncertainty reflects
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def count_reference_states(test):
    """Count the states each individual's generating curve actually has.

    The reference summary is the analytic one, pruned as everywhere else, so
    the count is a property of the curve rather than of any forecast.

    Args:
        test: SimulatedDataset; the test split.

    Returns:
        np.ndarray of shape (D,) and dtype int.
    """
    summaries = get_analytic_shape_summary(
        test, config.SHAPE_CONFIG["upsilon_rel_1"],
        config.SHAPE_CONFIG["upsilon_rel_2"],
        config.SHAPE_CONFIG["upsilon_rel_prune"],
        config.SHAPE_CONFIG["do_prune"])

    return np.array([len(summary) for summary in summaries], dtype=int)


def summarise_uncertainty_by_structure(uncertainty, states):
    """Report how the shape uncertainty grows with a curve's own structure.

    On this data the reference summary of a monotone curve carries one state
    and that of a rebounding curve two, so the two groups are also the two
    state counts, and the correlation is over a binary predictor.

    Args:
        uncertainty: np.ndarray of shape (D,); the shape uncertainty U.
        states: np.ndarray of shape (D,) and dtype int; the reference states.

    Returns:
        dict; the median U of each group, and the rank correlation between U
            and the state count.
    """
    return {
        "median_u_monotone": float(np.median(uncertainty[states == 1])),
        "median_u_rebounding": float(np.median(uncertainty[states >= 2])),
        "state_correlation": float(spearmanr(uncertainty, states).statistic),
    }


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. The populations U is asked to separate
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def flag_missed_rebounds(evaluator):
    """Flag the rebounds the point forecast reports as a monotone decline.

    This is the clinically consequential error and the one positive label used
    throughout, so that the panel and the diagnostic beneath it count the same
    individuals. A rebound reported as a monotone rise is wrong too, but it
    reassures nobody and is left out of both.

    Args:
        evaluator: ClinicalErrorEvaluator; for one configuration.

    Returns:
        np.ndarray of shape (D,) and dtype bool.
    """
    return (evaluator.get_true_rebound_flags()
            & evaluator.get_point_forecast_monotone_flags("decreasing"))


def select_declining_population(evaluator):
    """Select the predictions a missed rebound could hide behind.

    This is the population a warning would be issued to: every individual whose
    point forecast declines throughout, whether or not the curve turns.

    Args:
        evaluator: ClinicalErrorEvaluator; for one configuration.

    Returns:
        tuple (indices, labels); ascending indices into the test split, and a
            bool array marking the missed rebounds among them.
    """
    indices = np.flatnonzero(
        evaluator.get_point_forecast_monotone_flags("decreasing"))

    return indices, flag_missed_rebounds(evaluator)[indices]


def select_rebounding_population(evaluator):
    """Select the rebounding curves, whose reference structure is constant.

    Every rebounding curve carries two reference states and every monotone one
    carries a single state, so the declining population cannot be split at a
    matched state count. Restricting to the curves that do rebound holds the
    structure fixed instead, and asks whether U still separates the ones
    reported as a decline from the ones reported correctly. Rebounds reported
    as a rise are excluded, so that both sides of the comparison are
    unambiguous.

    Args:
        evaluator: ClinicalErrorEvaluator; for one configuration.

    Returns:
        tuple (indices, labels); as select_declining_population returns.
    """
    missed = flag_missed_rebounds(evaluator)
    detected = (evaluator.get_true_rebound_flags()
                & evaluator.get_point_forecast_rebound_flags())
    indices = np.flatnonzero(missed | detected)

    return indices, missed[indices]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. What the shape uncertainty detects
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def rank_labels_by_score(scores, labels):
    """Sort a population's labels from most to least suspect.

    Args:
        scores: np.ndarray; higher means more suspect.
        labels: np.ndarray of bool; the individuals worth catching.

    Returns:
        np.ndarray of bool; the labels in descending score order.
    """
    order = np.argsort(-np.asarray(scores), kind="stable")

    return np.asarray(labels, dtype=bool)[order]


def locate_weakest_positive(ranked):
    """Find how far down a ranking its last positive sits.

    This is how far a practitioner reading from the top would have to go to
    miss nobody, and so the cheapest threshold reaching full recall.

    Args:
        ranked: np.ndarray of bool; as rank_labels_by_score returns.

    Returns:
        int; that position, counting from one.
    """
    return int(np.flatnonzero(ranked).max() + 1)


def score_detection_performance(scores, labels):
    """Score how far U concentrates the missed rebounds in one population.

    Both a threshold-free ranking measure and an operating point are reported,
    since the first says there is signal and the second says how little of it
    survives a decision. The operating point is given as a count as well as a
    share, since narrowing the population moves the two in opposite directions.

    Args:
        scores: np.ndarray; the shape uncertainty of each individual in the
            population.
        labels: np.ndarray of bool; True where the point forecast missed a
            rebound.

    Returns:
        dict; the population size and positives, the base rate, the area under
            the ROC curve, the average precision, the precision in the top
            SHORTLIST, and the count and share needed for full recall.
    """
    ranked = rank_labels_by_score(scores, labels)
    weakest = locate_weakest_positive(ranked)

    return {
        "population_size": int(len(labels)),
        "population_missed": int(np.sum(labels)),
        "base_rate": float(np.mean(labels)),
        "area_under_curve": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "shortlist_precision": float(np.mean(ranked[:SHORTLIST])),
        "full_recall_count": weakest,
        "full_recall_share": weakest / len(labels),
    }


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5. The same question with the curve's structure held fixed
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def score_structure_matched_diagnostic(scores, labels):
    """Score U over the rebounding curves, which all carry two states.

    Whatever separation survives here is the error rather than the state count,
    which is what the panel above cannot distinguish. It is a diagnostic and
    not a setting, since selecting this population needs the ground truth.

    Args:
        scores: np.ndarray; the shape uncertainty of each individual in the
            population.
        labels: np.ndarray of bool; True where the rebound was reported as a
            decline rather than caught.

    Returns:
        dict; the group's size and how many of it were missed, the median U of
            each side, and the area under the ROC curve within the group.
    """
    separable = 0 < int(np.sum(labels)) < len(labels)

    return {
        "rebounding_size": int(len(labels)),
        "rebounding_missed": int(np.sum(labels)),
        "rebounding_median_missed": float(np.median(scores[labels])),
        "rebounding_median_detected": float(np.median(scores[~labels])),
        "rebounding_auc": (float(roc_auc_score(labels, scores))
                           if separable else float("nan")),
    }


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6. Scoring every configuration
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def summarise_column(column, verbose=True):
    """Score both panels and the diagnostic under one configuration.

    Args:
        column: dict; one entry of COLUMNS.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict; carrying the two heading lines and one entry per key named in
            ROW_GROUPS, alongside the quantities reported only on stdout.
    """
    evaluator, test = build_error_evaluator(column, verbose=verbose)
    uncertainty = np.asarray(evaluator.result["U"], dtype=float)

    declining, missed = select_declining_population(evaluator)
    rebounding, reported_as_decline = select_rebounding_population(evaluator)

    return {
        "heading": column["heading"],
        "subheading": column["subheading"],
        **summarise_uncertainty_by_structure(uncertainty,
                                             count_reference_states(test)),
        **score_detection_performance(uncertainty[declining], missed),
        **score_structure_matched_diagnostic(uncertainty[rebounding],
                                             reported_as_decline),
    }


def tabulate_columns(verbose=True):
    """Score every configuration.

    Args:
        verbose: bool; whether to report each column as it completes.

    Returns:
        list of dict; one per column in the order of COLUMNS.
    """
    summaries = []
    for column in COLUMNS:
        if verbose:
            print(f"\n{column['heading']} {column['subheading']}", flush=True)

        summary = summarise_column(column, verbose=verbose)
        summaries.append(summary)

        if verbose:
            report_column(summary)

    return summaries


def report_column(summary):
    """Print one column's figures as it completes.

    Args:
        summary: dict; one entry of tabulate_columns.

    Returns:
        None
    """
    print(f"  median U: monotone {summary['median_u_monotone']:.3f}, "
          f"rebounding {summary['median_u_rebounding']:.3f}, "
          f"rho {summary['state_correlation']:.2f}")
    print(f"  declining predictions: {summary['population_missed']} missed of "
          f"{summary['population_size']} at a "
          f"{summary['base_rate']:.1%} base rate")
    print(f"  detection: AUC {summary['area_under_curve']:.3f}, "
          f"AP {summary['average_precision']:.3f}, "
          f"top-{SHORTLIST} precision "
          f"{summary['shortlist_precision']:.1%}, "
          f"{summary['full_recall_count']} flagged "
          f"({summary['full_recall_share']:.1%}) for full recall")
    print(f"  structure matched: {summary['rebounding_missed']} missed of "
          f"{summary['rebounding_size']} rebounding, median U "
          f"{summary['rebounding_median_missed']:.3f} against "
          f"{summary['rebounding_median_detected']:.3f}, "
          f"AUC {summary['rebounding_auc']:.3f}")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7. Rendering
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def format_metric(value, spec):
    """Format one figure to the precision its row is reported at.

    Args:
        value: int or float; the figure to render.
        spec: str; the row's format string.

    Returns:
        str; the formatted figure, safe to place in a latex cell.
    """
    if isinstance(value, float) and np.isnan(value):
        return "--"

    return spec.format(value).replace("%", r"\%")


def render_column_heading(summary):
    """Stack one column's two heading lines over the figures beneath them.

    Args:
        summary: dict; one entry of tabulate_columns.

    Returns:
        str; the heading cell, centred over its column.
    """
    return (rf"\multicolumn{{1}}{{c}}{{\shortstack{{{summary['heading']}"
            rf"\\{summary['subheading']}}}}}")


def render_latex_table(summaries):
    """Assemble the columns into a booktabs tabular, one panel per question.

    Args:
        summaries: list of dict; as tabulate_columns returns.

    Returns:
        str; the latex table.
    """
    lines = [r"\begin{tabular}{l" + "r" * len(summaries) + "}", r"\toprule"]

    # Step 1: The stacked headings, one to a line so the source stays readable
    headings = [render_column_heading(summary) for summary in summaries]
    lines.extend(f"& {heading}" for heading in headings[:-1])
    lines.append(f"& {headings[-1]}" + r" \\")
    lines.append(r"\midrule")

    # Step 2: One panel per question, ruled off from the next
    for position, (group_label, rows) in enumerate(ROW_GROUPS):
        lines.append(rf"\multicolumn{{{1 + len(summaries)}}}{{l}}"
                     rf"{{\textit{{{group_label}}}}} \\")

        for key, label, spec in rows:
            cells = [format_metric(summary[key], spec)
                     for summary in summaries]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        if position != len(ROW_GROUPS) - 1:
            lines.extend([r"\addlinespace", r"\midrule"])

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


def write_latex_table(table, output_path=OUTPUT_PATH):
    """Write the rendered table alongside the other thesis tables.

    Args:
        table: str; the latex tabular returned by render_latex_table.
        output_path: pathlib.Path; destination of the generated ``.tex`` file.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n")


if __name__ == "__main__":
    print("Table 7: what the shape uncertainty reflects, and what it detects")
    summaries = tabulate_columns()

    write_latex_table(render_latex_table(summaries))
    print(f"\nWrote {OUTPUT_PATH}")
