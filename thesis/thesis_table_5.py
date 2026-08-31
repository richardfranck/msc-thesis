"""Table 5: shape-state pruning, faithfulness and observed individual R-squared.

Note that all shape scores assessed against pruned analytic truth.

    python thesis/thesis_table_5_pruned_truth.py
"""

from itertools import groupby
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

from scripts.shape_uncertainty.evaluation.model_evaluation import ModelEvaluator
from scripts.shape_uncertainty.model.model import MeanOnlyModel
from scripts.shape_uncertainty.model.model_inference import (
    build_inference_engine, UncertaintyEngine)
from scripts.shape_uncertainty.shape_extraction.shape_distance import (
    mean_shape_sequence_match)
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import (
    get_analytic_shape_summary)


# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# The regime: simulated data without unobserved heterogeneity.
REGIME = config.REGIME_HOMOGENEOUS

OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_5_pruned_truth.tex"
PRUNED_TRUTH_CONFIG = config.SHAPE_CONFIG


# The two configurations, one column group each.
CONFIGURATIONS = [
    ("TimeView", f"timeview_tumour_{REGIME.name}",
     config.LAMBDA_MEAN_UNPENALISED),
    ("Mean-only", f"meanonly_tumour_{REGIME.name}",
     config.LAMBDA_MEAN_MEANONLY),
]

# The two extraction settings, one column each within a group.
SETTINGS = [
    ("No pruning", config.SHAPE_CONFIG_NAIVE),
    ("Pruning", config.SHAPE_CONFIG),
]

# The table's rows, grouped so that a rule separates the three blocks.
ROW_GROUPS = [
    [("estimated_states", "Estimated states per summary", "{:.2f}"),
     ("true_states", "True states per summary", "{:.2f}"),
     ("sequence_match_rate", "Sequence match rate", "{:.1%}")],
    [("mean_uncertainty", r"Mean $\overline{U}$", "{:.3f}"),
     ("median_uncertainty", "Median $U$", "{:.3f}"),
     ("share_certain", "Share with $U = 0$", "{:.1%}")],
    [("median_observed_r2", r"Median $R^2$ (observed)", "{:.3f}"),
     ("observed_r2_interquartile_range", "Interquartile range", "text")],
]


def load_single_model(name, verbose=True):
    """Reload the reported model of one configuration.

    Args:
        name: str; the training run to read.
        verbose: bool; report progress on stdout.

    Returns:
        dict; as load_run returns, carrying "model", "normaliser",
            "knot_objects" and "best_params".
    """
    if verbose:
        print(f"  reading the single model of {name}")

    return model_loader.load_single_model(name, MeanOnlyModel)


def load_ensemble(name, verbose=True):
    """Restore the bootstrap ensemble of one configuration.

    The stored shape_config only sets the ensemble's own engines. Both columns
    rebuild their engines from the weights, so it does not bind here.

    Args:
        name: str; the training run to read.
        verbose: bool; report progress on stdout.

    Returns:
        BootstrapEnsemble; with its members restored.
    """
    if verbose:
        print(f"  reading the ensemble of {name}")

    return model_loader.load_bootstrap_ensemble(name, MeanOnlyModel)


def load_configuration_models(configuration, verbose=True):
    """Read both fitted objects one configuration contributes.

    Args:
        configuration: tuple; the heading, run name, and roughness penalty.
        verbose: bool; report progress on stdout.

    Returns:
        tuple (run, ensemble); the single model's run and the ensemble.
    """
    heading, name, lambda_mean = configuration
    if verbose:
        print(f"\n{heading} (lambda_mean = {lambda_mean:g})")

    return (load_single_model(name, verbose=verbose),
            load_ensemble(name, verbose=verbose))


def build_engine_at_pruning(run, shape_config):
    """Wrap the single model's weights at one pruning threshold.

    Args:
        run: dict; as load_single_model returns.
        shape_config: dict; the extraction thresholds for this column.

    Returns:
        InferenceEngine; ready to predict shape summaries.
    """
    return build_inference_engine(
        model=run["model"],
        normaliser=run["normaliser"],
        knot_objects=run["knot_objects"],
        shape_config=shape_config,
    )


def rebuild_member_engines(ensemble, shape_config):
    """Rewrap every ensemble member at one pruning threshold.

    The saved ensemble carries one extraction setting. Rebuilding the engines
    from the stored weights lets both columns read the same fitted members.

    Args:
        ensemble: BootstrapEnsemble; fitted.
        shape_config: dict; the extraction thresholds for this column.

    Returns:
        list of InferenceEngine, in fit order.
    """
    return [
        build_inference_engine(
            model=member["model"],
            normaliser=member["normaliser"],
            knot_objects=ensemble.knot_objects,
            shape_config=shape_config,
        )
        for member in ensemble.members
    ]


def extract_true_summaries(test, shape_config):
    """Read the analytic shape summaries at one pruning threshold.

    Args:
        test: SimulatedDataset; the test split.
        shape_config: dict; the extraction thresholds for this column.

    Returns:
        list of length D_test; one analytic shape summary per individual.
    """
    return get_analytic_shape_summary(
        test,
        shape_config["upsilon_rel_1"],
        shape_config["upsilon_rel_2"],
        shape_config["upsilon_rel_prune"],
        shape_config["do_prune"],
    )


def count_mean_states(summaries):
    """Average the number of shape states over a set of summaries.

    Args:
        summaries: list of shape summaries.

    Returns:
        float; the mean state count.
    """
    return float(np.mean([len(summary) for summary in summaries]))


def score_summary_faithfulness(engine, test, true_summaries):
    """Score the single model's summaries against the analytic truth.

    Args:
        engine: InferenceEngine; the single model at this column's threshold.
        test: SimulatedDataset; the test split.
        true_summaries: list; the analytic truth used for both columns.

    Returns:
        dict with keys "estimated_states", "true_states" and
            "sequence_match_rate".
    """
    predicted = engine.predict_mean_shape_summary(test.X)

    return {
        "estimated_states": count_mean_states(predicted),
        "true_states": count_mean_states(true_summaries),
        "sequence_match_rate": mean_shape_sequence_match(
            predicted, true_summaries),
    }


def score_shape_uncertainty(engines, test):
    """Score how far the ensemble's summaries disagree.

    Args:
        engines: list of InferenceEngine; the members at this column's
            threshold.
        test: SimulatedDataset; the test split.

    Returns:
        dict with keys "mean_uncertainty", "median_uncertainty" and
            "share_certain", the last being the share of individuals every
            member agrees on.
    """
    result = UncertaintyEngine(engines).predict_with_epistemic_uncertainty(
        test.X)
    U = result["U"]

    return {
        "mean_uncertainty": float(np.mean(U)),
        "median_uncertainty": float(np.median(U)),
        "share_certain": float(np.mean(U == 0.0)),
    }


def summarise_observed_individual_r_squared(engine, test):
    """Summarise observed individual R-squared for one fitted model.

    Args:
        engine: InferenceEngine; the fitted model to evaluate.
        test: SimulatedDataset; the held-out observations.

    Returns:
        dict: The median and interquartile range of observed individual R2.
    """
    metrics = ModelEvaluator(engine, test).compute_value_space_metrics(
        target="observed")
    individual_r_squared = np.asarray(metrics["individual_r2"], dtype=float)
    lower, upper = np.nanpercentile(individual_r_squared, [25, 75])

    return {
        "median_observed_r2": float(np.nanmedian(individual_r_squared)),
        "observed_r2_interquartile_range": f"{lower:.3f}--{upper:.3f}",
    }


def score_pruning_setting(run, ensemble, test, shape_config):
    """Score one configuration against the pruned analytic truth.

    Args:
        run: dict; as load_single_model returns.
        ensemble: BootstrapEnsemble; fitted.
        test: SimulatedDataset; the test split.
        shape_config: dict; the extraction thresholds for this column.

    Returns:
        dict; the keys the table's rows read.
    """
    # Step 1: The single model, read against the fixed pruned truth
    engine = build_engine_at_pruning(run, shape_config)
    true_summaries = extract_true_summaries(test, PRUNED_TRUTH_CONFIG)
    faithfulness = score_summary_faithfulness(engine, test, true_summaries)
    observed_r_squared = summarise_observed_individual_r_squared(engine, test)

    # Step 2: The ensemble, read for how far its members disagree
    engines = rebuild_member_engines(ensemble, shape_config)
    agreement = score_shape_uncertainty(engines, test)

    return {**faithfulness, **agreement, **observed_r_squared}


def report_column(column):
    """Print one column's figures as it completes.

    Args:
        column: dict; one entry of tabulate_configurations.

    Returns:
        None
    """
    print(f"  {column['group']} / {column['heading']}: "
          f"states {column['estimated_states']:.2f} "
          f"vs true {column['true_states']:.2f}   "
          f"match {column['sequence_match_rate']:.1%}   "
          f"mean U {column['mean_uncertainty']:.3f}   "
          f"share U=0 {column['share_certain']:.1%}", flush=True)


def tabulate_configurations(test, verbose=True):
    """Score every configuration at every pruning threshold.

    Args:
        test: SimulatedDataset; the test split.
        verbose: bool; report each column as it completes.

    Returns:
        list of dicts, one per column in table order, each carrying "group",
            "heading" and the keys the table's rows read.
    """
    columns = []
    for configuration in CONFIGURATIONS:
        group_heading, _, _ = configuration
        run, ensemble = load_configuration_models(
            configuration, verbose=verbose)

        for setting_heading, shape_config in SETTINGS:
            scores = score_pruning_setting(run, ensemble, test, shape_config)
            column = {"group": group_heading,
                      "heading": setting_heading,
                      **scores}
            columns.append(column)

            if verbose:
                report_column(column)

    return columns


def format_metric(value, spec):
    """Format one figure to the precision its row is reported at.

    Args:
        value: float or str; the figure to render.
        spec: str; the row's format string, or ``text`` for a prepared value.

    Returns:
        str; the formatted figure, safe to place in a latex cell.
    """
    if spec == "text":
        return value

    return spec.format(value).replace("%", r"\%")


def format_group_headings(columns):
    """Build the two header rows that span and then name the columns.

    Args:
        columns: list of dicts, as tabulate_configurations returns, with the
            columns of one group adjacent.

    Returns:
        list of str; the spanning row, its rules, and the per-column row.
    """
    groups = [(name, len(list(members)))
              for name, members in groupby(columns, key=lambda c: c["group"])]

    # The label column is first, so the groups start at column two
    spans, rules, first = [], [], 2
    for name, width in groups:
        spans.append(rf"\multicolumn{{{width}}}{{c}}{{{name}}}")
        rules.append(rf"\cmidrule(lr){{{first}-{first + width - 1}}}")
        first += width

    headings = " & ".join(column["heading"] for column in columns)

    return [" & " + " & ".join(spans) + r" \\",
            " ".join(rules),
            " & " + headings + r" \\"]


def render_latex_table(columns):
    """Assemble the columns into a booktabs tabular.

    Args:
        columns: list of dicts, as tabulate_configurations returns.

    Returns:
        str; the latex table.
    """
    # Step 1: Open the tabular, the label column left, one column per setting
    lines = [r"\begin{tabular}{l" + "c" * len(columns) + "}", r"\toprule"]

    # Step 2: Head the columns, grouped by configuration
    lines.extend(format_group_headings(columns))
    lines.append(r"\midrule")

    # Step 3: One block of rows per panel, separated by a space
    for position, group in enumerate(ROW_GROUPS):
        for key, label, spec in group:
            cells = [format_metric(column[key], spec) for column in columns]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        if position != len(ROW_GROUPS) - 1:
            lines.append(r"\addlinespace")

    # Step 4: Close the tabular
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


def write_latex_table(table, output_path=OUTPUT_PATH):
    """Write the rendered table alongside the other thesis tables.

    Args:
        table: str; the LaTex tabular returned by render_latex_table.
        output_path: pathlib.Path; destination of the generated ``.tex`` file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n")


if __name__ == "__main__":
    print(f"Table 5 with pruned analytic truth on {REGIME.name}")
    test, _, _ = data_loader.draw_simulated_split(REGIME)
    columns = tabulate_configurations(test)
    table = render_latex_table(columns)
    write_latex_table(table)
    print(f"Wrote {OUTPUT_PATH}")
