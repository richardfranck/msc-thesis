"""Table 6: The rebounds the point forecast misses, and who they belong to.

Requires: meanonly_tumour_homogeneous, gaussian_tumour_heterogeneous

    python thesis/thesis_table_6.py
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

from scripts.shape_uncertainty.evaluation.clinical_error_analysis import (
    ClinicalErrorEvaluator)
from scripts.shape_uncertainty.evaluation.model_evaluation import ModelEvaluator
from scripts.shape_uncertainty.model.model import GaussianModel, MeanOnlyModel
from scripts.shape_uncertainty.model.model_inference import UncertaintyEngine

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_6.tex"

# Aleatoric draws per ensemble member, as the random-effects run scored them.
NR_DRAWS = 20

# The settings the saved scores must have been produced under. Checking these
# also refuses a cache that does not describe this ensemble and this split.
SCORE_SETTINGS = {"nr_draws": NR_DRAWS, "nr_members": config.NR_MEMBERS,
                  "seed_eval": config.SEED_EVAL, "n_dense": config.N_DENSE,
                  "nr_test": config.D_TEST}


# One column per configuration, in table order. Each carries
#     "heading":            str; the column heading.
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
    {"heading": "MeanOnly (w/o UH)",
     "regime": config.REGIME_HOMOGENEOUS,
     "run_name": "meanonly_tumour_homogeneous",
     "model_cls": MeanOnlyModel,
     "reads_saved_scores": False},
    {"heading": "RandomEffects (w/ UH)",
     "regime": config.REGIME_HETEROGENEOUS,
     "run_name": "gaussian_tumour_heterogeneous",
     "model_cls": GaussianModel,
     "reads_saved_scores": True},
]

# The table's rows, grouped so that a space separates the three blocks.
ROW_GROUPS = [
    ("Clinical error", [
        ("nr_rebounding", r"Rebounding individuals", "{:d}"),
        ("nr_missed_decline", r"\quad reported as monotone decline", "{:d}"),
        ("nr_missed_increase", r"\quad reported as monotone increase", "{:d}"),
        ("miss_rate", r"\quad monotone decline miss rate", "{:.1%}"),
    ]),
    (r"Point-forecast accuracy (median $R^2$)", [
        ("median_r2_missed", r"Missed rebounds", "{:.3f}"),
        ("median_r2_detected", r"Detected rebounds", "{:.3f}"),
        ("median_r2_all", r"All individuals", "{:.3f}"),
    ]),
    (r"Location of the turn (median $t^{*}$)", [
        ("median_t_star_missed", r"Missed rebounds", "{:.2f}"),
        ("median_t_star_detected", r"Detected rebounds", "{:.2f}"),
        ("median_t_star_rebounding", r"All rebounds", "{:.2f}"),
    ]),
]


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
              f"{column['run_name']}")

    return ensemble


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
    once. The draws are read back rather than redrawn, so that this table and
    the figures reading this run see the same cloud.

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


def build_error_evaluator(column, verbose=True):
    """Assemble the clinical error evaluator for one column.

    Args:
        column: dict; one entry of COLUMNS.
        verbose: bool; whether to report progress on stdout.

    Returns:
        tuple (evaluator, point_forecast_engine, test); the
            ClinicalErrorEvaluator, the single trained model standing in for a
            bare point forecast, and the test split every index refers to.
    """
    # Step 1: The split and the members fitted on it
    test = draw_test_split(column)
    ensemble = load_ensemble(column, verbose=verbose)

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

    return evaluator, point_forecast_engine, test


def score_observed_r_squared(point_forecast_engine, test):
    """Score how well the point forecast fits each individual's observations.

    Args:
        point_forecast_engine: InferenceEngine; the single trained model.
        test: SimulatedDataset; the test split.

    Returns:
        np.ndarray of shape (D,) and dtype float; the per-individual R^2
            against the observations, np.nan where it is undefined.
    """
    evaluator = ModelEvaluator(point_forecast_engine, test)
    metrics = evaluator.compute_value_space_metrics(target="observed")

    return np.asarray(metrics["individual_r2"], dtype=float)


def summarise_missed_rebounds(column, verbose=True):
    """Count one column's missed rebounds and describe who they belong to.

    Args:
        column: dict; one entry of COLUMNS.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict; carrying "heading" and one entry per key named in ROW_GROUPS.
    """
    evaluator, point_forecast_engine, test = build_error_evaluator(
        column, verbose=verbose)

    # Step 1: The individuals whose curve turns, by what the forecast reported.
    #         The three groups partition the rebounding individuals.
    rebounding = evaluator.get_rebounding_individuals()
    missed = evaluator.get_missed_rebound_individuals()
    detected = evaluator.get_detected_rebound_individuals()
    misreported_increase = evaluator.get_misreported_increase_individuals()

    # Step 2: How well the point forecast fitted them in value space. The
    #         missed rebounds are read against the detected ones, since a
    #         rebound is harder to fit than a monotone curve and the whole test
    #         set would charge that difficulty to the error.
    r_squared = score_observed_r_squared(point_forecast_engine, test)

    # Step 3: Where on the horizon the missed turns fall, read against the
    #         detected ones and against every rebound, as the accuracy is
    t_star_missed = evaluator.compute_critical_points(missed)
    t_star_detected = evaluator.compute_critical_points(detected)
    t_star_rebounding = evaluator.compute_critical_points(rebounding)

    # Step 4: Collect the figures the rows read
    miss_rate = len(missed) / len(rebounding) if len(rebounding) else np.nan

    return {
        "heading": column["heading"],
        "nr_rebounding": len(rebounding),
        "nr_missed_decline": len(missed),
        "nr_missed_increase": len(misreported_increase),
        "miss_rate": miss_rate,
        "median_r2_missed": float(np.nanmedian(r_squared[missed])),
        "median_r2_detected": float(np.nanmedian(r_squared[detected])),
        "median_r2_all": float(np.nanmedian(r_squared)),
        "median_t_star_missed": float(np.nanmedian(t_star_missed)),
        "median_t_star_detected": float(np.nanmedian(t_star_detected)),
        "median_t_star_rebounding": float(np.nanmedian(t_star_rebounding)),
    }


def report_column(summary):
    """Print one column's figures as it completes.

    Args:
        summary: dict; one entry of tabulate_missed_rebounds.

    Returns:
        None
    """
    print(f"  {summary['heading']}: {summary['nr_missed_decline']} of "
          f"{summary['nr_rebounding']} rebounds reported as a decline, "
          f"{summary['nr_missed_increase']} as an increase", flush=True)


def tabulate_missed_rebounds(verbose=True):
    """Summarise the missed rebounds of every column.

    Args:
        verbose: bool; whether to report each column as it completes.

    Returns:
        list of dict; one per column in the order of COLUMNS, as
            summarise_missed_rebounds returns.
    """
    summaries = []
    for column in COLUMNS:
        if verbose:
            print(f"\n{column['heading']}")

        summary = summarise_missed_rebounds(column, verbose=verbose)
        summaries.append(summary)

        if verbose:
            report_column(summary)

    return summaries


def format_metric(value, spec):
    """Format one figure to the precision its row is reported at.

    Args:
        value: int or float; the figure to render.
        spec: str; the row's format string.

    Returns:
        str; the formatted figure, safe to place in a latex cell.
    """
    return spec.format(value).replace("%", r"\%")


def render_latex_table(summaries):
    """Assemble the columns into a booktabs tabular.

    Args:
        summaries: list of dict; as tabulate_missed_rebounds returns.

    Returns:
        str; the latex table.
    """
    # Step 1: Open the tabular, the label column left, one column per setting
    lines = [r"\begin{tabular}{l" + "r" * len(summaries) + "}", r"\toprule"]

    # Step 2: Head each column with its configuration
    headings = " & ".join(summary["heading"] for summary in summaries)
    lines.append(" & " + headings + r" \\")
    lines.append(r"\midrule")

    # Step 3: One panel per group, its rows underneath it. The row labels carry
    #         their own indentation, so nothing is added here.
    for position, (group_label, rows) in enumerate(ROW_GROUPS):
        lines.append(rf"\multicolumn{{{1 + len(summaries)}}}{{l}}"
                     rf"{{\textit{{{group_label}}}}} \\")

        for key, label, spec in rows:
            cells = [format_metric(summary[key], spec) for summary in summaries]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        if position != len(ROW_GROUPS) - 1:
            lines.append(r"\addlinespace")

    # Step 4: Close the tabular
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
    print("Table 6: the rebounds the point forecast misses")
    summaries = tabulate_missed_rebounds()
    table = render_latex_table(summaries)
    write_latex_table(table)
    print(f"\nWrote {OUTPUT_PATH}")
