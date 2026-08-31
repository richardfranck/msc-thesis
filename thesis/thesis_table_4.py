"""Table 4: Observed individual R-squared of the random-effects model.

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

from scripts.shape_uncertainty.evaluation.model_evaluation import ModelEvaluator
from scripts.shape_uncertainty.model.model import GaussianModel


torch.set_default_dtype(torch.float64)

OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_4.tex"

# The reported datasets exclude the homogeneous Tumour calibration setting.
DATASET_SETTINGS = [
    ("tumour_heterogeneous", "Tumour, with unobserved heterogeneity"),
    ("flchain", "Flchain"),
    ("airfoil", "Airfoil"),
]

# The random-effects model is the only model reported in this table.
MODEL_SETTINGS = [
    (r"Median $R^2$ [$Q_1$, $Q_3$]", "gaussian", GaussianModel),
]


def load_test_datasets():
    """Load the held-out test dataset for every reported column.

    Returns:
        dict: Maps each dataset setting name to its test dataset.
    """
    tumour_heterogeneous = data_loader.draw_simulated_split(
        config.REGIME_HETEROGENEOUS)
    _, flchain = data_loader.load_real_split("flchain")
    _, airfoil = data_loader.load_real_split("airfoil")

    return {
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


def build_model_row(model_setting, test_datasets, verbose=True):
    """Score one model type across all reported held-out datasets.

    Args:
        model_setting: tuple; one entry of MODEL_SETTINGS.
        test_datasets: dict; maps setting names to held-out datasets.
        verbose: bool; whether to report each completed cell.

    Returns:
        dict: The row label and one R-squared summary per dataset setting.
    """
    label, model_prefix, model_cls = model_setting
    row = {"label": label}

    for dataset_name, _ in DATASET_SETTINGS:
        engine = load_model_engine(model_prefix, dataset_name, model_cls)
        r_squared = calculate_observed_individual_r_squared(
            engine, test_datasets[dataset_name])
        row[dataset_name] = summarise_individual_r_squared(r_squared)

        if verbose:
            median = row[dataset_name]["median"]
            print(f"  {label}, {dataset_name}: median R2 {median:.3f}",
                  flush=True)

    return row


def build_table_rows(test_datasets, verbose=True):
    """Build the random-effects row of observed individual R-squared summaries.

    Args:
        test_datasets: dict; maps setting names to held-out datasets.
        verbose: bool; whether to report progress on stdout.

    Returns:
        list of dict: The one random-effects row.
    """
    return [
        build_model_row(setting, test_datasets, verbose=verbose)
        for setting in MODEL_SETTINGS
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


def render_latex_table(rows):
    """Render the observed individual R-squared summaries as a tabular.

    Args:
        rows: sequence of dict; the random-effects summary from
            build_table_rows.

    Returns:
        str: The complete LaTex tabular environment.
    """
    summary = rows[0]
    lines = [r"\begin{tabular}{lrr}", r"\toprule"]
    lines.append(r"Dataset & Median $R^2$ & Interquartile range \\")
    lines.append(r"\midrule")

    for dataset_name, heading in DATASET_SETTINGS:
        median = f"{summary[dataset_name]['median']:.3f}"
        interquartile_range = format_interquartile_range(summary[dataset_name])
        lines.append(f"{heading} & {median} & {interquartile_range} \\")

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
    rows = build_table_rows(test_datasets)
    write_latex_table(render_latex_table(rows))
    print(f"Wrote {OUTPUT_PATH}")
