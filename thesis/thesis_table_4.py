"""Table 4: Observed individual R-squared across reported configurations.

Requires: timeview_tumour_homogeneous, meanonly_tumour_homogeneous,
    gaussian_tumour_heterogeneous, meanonly_flchain, gaussian_flchain,
    meanonly_airfoil, gaussian_airfoil

    python thesis/thesis_table_4.py
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

from scripts.shape_uncertainty.evaluation.accuracy_ceilings import (
    AccuracyCeilings)
from scripts.shape_uncertainty.evaluation.model_evaluation import ModelEvaluator
from scripts.shape_uncertainty.model.model import GaussianModel, MeanOnlyModel


torch.set_default_dtype(torch.float64)

OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_4.tex"

# The reported dataset--model configurations, one table row each, grouped by
# dataset and running from the simplest model configuration to the richest.
# Each carries
#     "setting":   str; the data setting, keying both the test datasets and the
#                  accuracy ceilings.
#     "dataset":   str; how that dataset is named in the table.
#     "model":     str; how that model configuration is named in the table.
#     "prefix":    str; the run prefix its artefact was saved under.
#     "model_cls": type; the model class used when the artefact was saved.
ROW_SETTINGS = [
    {"setting": "tumour_homogeneous", "dataset": "Tumour (w/o UH)",
     "model": r"\textsc{TimeView}", "prefix": "timeview",
     "model_cls": MeanOnlyModel},
    {"setting": "tumour_homogeneous", "dataset": "Tumour (w/o UH)",
     "model": "Mean-only", "prefix": "meanonly",
     "model_cls": MeanOnlyModel},
    {"setting": "tumour_heterogeneous", "dataset": "Tumour (w/ UH)",
     "model": "Random effects", "prefix": "gaussian",
     "model_cls": GaussianModel},
    {"setting": "flchain", "dataset": "Flchain",
     "model": "Mean-only", "prefix": "meanonly",
     "model_cls": MeanOnlyModel},
    {"setting": "flchain", "dataset": "Flchain",
     "model": "Random effects", "prefix": "gaussian",
     "model_cls": GaussianModel},
    {"setting": "airfoil", "dataset": "Airfoil",
     "model": "Mean-only", "prefix": "meanonly",
     "model_cls": MeanOnlyModel},
    {"setting": "airfoil", "dataset": "Airfoil",
     "model": "Random effects", "prefix": "gaussian",
     "model_cls": GaussianModel},
]

# The settings whose ceiling is computed, which are the simulated ones: the
# conditional mean can be estimated by Monte Carlo only where the process the
# data was drawn from is available.
COMPUTED_CEILINGS = ("tumour_homogeneous", "tumour_heterogeneous")

# The settings whose ceiling is known without computing it. A flchain
# trajectory is an exact evaluation of a deterministic random survival forest
# at that subject's own covariates, so it carries neither observation noise nor
# variation between subjects sharing a covariate vector, and a predictor
# reading those covariates can in principle reproduce it exactly.
CONSTRUCTED_CEILINGS = {"flchain": 1.0}

# What is written where a dataset has no process to read a ceiling from.
UNAVAILABLE_CEILING = "---"

# The column headings, each broken over two lines so the table still fits the
# page now that it carries a fifth column.
COLUMN_HEADINGS = [
    "Dataset",
    r"\makecell[l]{Model\\configuration}",
    r"\makecell[r]{Median\\individual $R^2$}",
    r"\makecell[r]{Interquartile\\range}",
    r"\makecell[r]{Median\\individual $R^2_{\max}$}",
]


def load_test_datasets():
    """Load the held-out test dataset for every reported column.

    Returns:
        dict: Maps each dataset setting name to its test dataset.
    """
    tumour_homogeneous = data_loader.draw_simulated_split(
        config.REGIME_HOMOGENEOUS)
    tumour_heterogeneous = data_loader.draw_simulated_split(
        config.REGIME_HETEROGENEOUS)
    _, flchain = data_loader.load_real_split("flchain")
    _, airfoil = data_loader.load_real_split("airfoil")

    return {
        "tumour_homogeneous": tumour_homogeneous.test,
        "tumour_heterogeneous": tumour_heterogeneous.test,
        "flchain": flchain.test,
        "airfoil": airfoil.test,
    }


def load_model_engine(model_prefix, dataset_name, model_cls):
    """Reload the single model fitted to one dataset setting.

    Args:
        model_prefix: str; the run prefix, ``meanonly`` or ``gaussian``.
        dataset_name: str; the suffix identifying the data setting.
        model_cls: type; the model class used when the artefact was saved.

    Returns:
        InferenceEngine: The restored single model, ready to predict.
    """
    return model_loader.load_inference_engine(
        f"{model_prefix}_{dataset_name}", model_cls)


def calculate_observed_individual_r_squared(engine, test):
    """Calculate each test individual's R-squared against its observations.

    Args:
        engine: InferenceEngine; the restored single-model prediction engine.
        test: SimulatedDataset or RealDataset; the held-out observations.

    Returns:
        np.ndarray: Individual observed R-squared values, with NaN where
            R-squared is undefined.
    """
    metrics = ModelEvaluator(engine, test).compute_value_space_metrics(
        target="observed")
    return np.asarray(metrics["individual_r2"], dtype=float)


def summarise_individual_r_squared(r_squared):
    """Summarise individual R-squared values by their median and quartiles.

    Args:
        r_squared: np.ndarray; individual observed R-squared values.

    Returns:
        dict: Carries ``median``, ``lower_quartile`` and ``upper_quartile``.
    """
    lower, upper = np.nanpercentile(r_squared, [25, 75])
    return {
        "median": float(np.nanmedian(r_squared)),
        "lower_quartile": float(lower),
        "upper_quartile": float(upper),
    }


def compute_accuracy_ceiling(test):
    """Compute the median-individual accuracy ceiling of one simulated split.

    The ceiling is the R-squared attained by the conditional mean trajectory,
    which is the best any predictor reading only the observed covariates can
    do. It is scored against the recorded noisy observations, so it carries the
    cost of measurement error as well as of unobserved heterogeneity, matching
    the R-squared reported beside it.

    Args:
        test: SimulatedDataset; a held-out split whose generating process is
            known, so that the conditional mean can be estimated by Monte
            Carlo over the latent factors.

    Returns:
        float: The median of the ceiling over the individuals in the split.
    """
    ceilings = AccuracyCeilings(test, n_mc=config.N_MC, seed=config.SEED_EVAL)
    conditional_mean, _ = ceilings.compute_conditional_curve_moments()
    individual = ceilings.compute_individual_metrics(conditional_mean)

    return float(np.nanmedian(individual["r2_ceiling_observed"]))


def collect_accuracy_ceilings(test_datasets, verbose=True):
    """Collect the accuracy ceiling of every setting that has one.

    A setting is absent from the result where its data carries no process to
    read a ceiling from, which build_table_row reports as unavailable.

    Args:
        test_datasets: dict; maps setting names to held-out datasets.
        verbose: bool; whether to report each ceiling as it is computed.

    Returns:
        dict: Maps a setting name to its ceiling, as a float.
    """
    ceilings = dict(CONSTRUCTED_CEILINGS)

    for dataset_name in COMPUTED_CEILINGS:
        ceiling = compute_accuracy_ceiling(test_datasets[dataset_name])
        ceilings[dataset_name] = ceiling

        if verbose:
            print(f"  {dataset_name}: ceiling {ceiling:.3f}", flush=True)

    return ceilings


def build_table_row(row_setting, test_datasets, ceilings, verbose=True):
    """Score one reported dataset--model configuration.

    Args:
        row_setting: dict; one entry of ROW_SETTINGS.
        test_datasets: dict; maps setting names to held-out datasets.
        ceilings: dict; maps setting names to their accuracy ceiling, as
            collect_accuracy_ceilings returns.
        verbose: bool; whether to report the completed row.

    Returns:
        dict: The dataset, model configuration, R-squared summary and ceiling,
            the last being None where the dataset has no ceiling.
    """
    setting = row_setting["setting"]
    engine = load_model_engine(row_setting["prefix"], setting,
                               row_setting["model_cls"])
    r_squared = calculate_observed_individual_r_squared(
        engine, test_datasets[setting])
    summary = summarise_individual_r_squared(r_squared)

    if verbose:
        print(f"  {row_setting['dataset']}, {row_setting['model']}: "
              f"median R2 {summary['median']:.3f}", flush=True)

    return {"dataset": row_setting["dataset"], "model": row_setting["model"],
            "summary": summary, "ceiling": ceilings.get(setting)}


def build_table_rows(test_datasets, ceilings, verbose=True):
    """Build every reported observed individual R-squared row.

    Args:
        test_datasets: dict; maps setting names to held-out datasets.
        ceilings: dict; maps setting names to their accuracy ceiling.
        verbose: bool; whether to report progress on stdout.

    Returns:
        list of dict: One observed R-squared summary per table row.
    """
    return [
        build_table_row(setting, test_datasets, ceilings, verbose=verbose)
        for setting in ROW_SETTINGS
    ]


def format_interquartile_range(summary):
    """Format the interquartile range of one R-squared summary.

    Args:
        summary: dict; carries ``median``, ``lower_quartile`` and
            ``upper_quartile``.

    Returns:
        str: The rendered ``lower--upper`` table cell.
    """
    return (f"{summary['lower_quartile']:.3f}--"
            f"{summary['upper_quartile']:.3f}")


def format_accuracy_ceiling(ceiling):
    """Format one row's accuracy ceiling, or mark it unavailable.

    Args:
        ceiling: float or None; the ceiling, or None where the dataset has no
            generating process to read one from.

    Returns:
        str: The rendered table cell.
    """
    if ceiling is None:
        return UNAVAILABLE_CEILING

    return f"{ceiling:.3f}"


def render_latex_table(rows):
    """Render the observed individual R-squared summaries as a tabular.

    Args:
        rows: sequence of dict; the dataset--model summaries from
            build_table_rows.

    Returns:
        str: The complete LaTex tabular environment.
    """
    lines = [r"\begin{tabular}{llrrr}", r"\toprule"]
    lines.append(" & ".join(COLUMN_HEADINGS) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        cells = [
            row["dataset"],
            row["model"],
            f"{row['summary']['median']:.3f}",
            format_interquartile_range(row["summary"]),
            format_accuracy_ceiling(row["ceiling"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_latex_table(table, output_path=OUTPUT_PATH):
    """Write the rendered table to the thesis tables directory.

    Args:
        table: str; the LaTex tabular returned by render_latex_table.
        output_path: pathlib.Path; destination of the generated ``.tex`` file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n")


if __name__ == "__main__":
    test_datasets = load_test_datasets()
    ceilings = collect_accuracy_ceilings(test_datasets)
    rows = build_table_rows(test_datasets, ceilings)
    write_latex_table(render_latex_table(rows))
    print(f"Wrote {OUTPUT_PATH}")
