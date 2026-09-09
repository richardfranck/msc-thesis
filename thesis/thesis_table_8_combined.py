"""Table 8: What each configuration reports on the deployment datasets.

Requires: meanonly_flchain, gaussian_flchain, meanonly_airfoil,
    gaussian_airfoil

    python thesis/thesis_table_8_combined.py
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

from scripts.shape_uncertainty.model.model import GaussianModel, MeanOnlyModel
from scripts.shape_uncertainty.model.model_inference import UncertaintyEngine

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_8_combined.tex"

# Aleatoric draws per member, matching the simulation's random-effects run.
NR_DRAWS = 20

# The seed the aleatoric draws are taken under, so the table is reproducible.
SEED = config.SEED_EVAL

# One panel per deployment dataset, in table order. Each carries
#     "name":    str; the dataset, as data_loader and the run names know it.
#     "heading": str; the panel heading.
DATASETS = [
    {"name": "flchain", "heading": r"\textsc{Flchain}"},
    {"name": "airfoil", "heading": r"\textsc{Airfoil}"},
]

# One column per configuration, in table order. Each carries
#     "heading":   str; the column heading.
#     "prefix":    str; what its training runs are named for.
#     "model_cls": the RandomEffectsModel subclass to rebuild.
#     "combined":  bool; whether the cloud draws random effects within each
#                  member as well as varying the member.
CONFIGURATIONS = [
    {"heading": r"\makecell{Mean-only model\\(epistemic cloud)}",
     "prefix": "meanonly",
     "model_cls": MeanOnlyModel,
     "combined": False},
    {"heading": r"\makecell{Random effects model\\(combined cloud)}",
     "prefix": "gaussian",
     "model_cls": GaussianModel,
     "combined": True},
]

# The rows each panel carries, in order. The split size and the share of
# individuals at U = 0 are still measured and reported on stdout, but the table
# itself carries only what the section argues from.
ROWS = [
    ("median_states", r"Median consensus states", "{:.1f}"),
    ("shape_quartiles", r"Median $U$ $[Q_1, Q_3]$", "median-iqr"),
    ("share_increasing",
     r"\makecell[l]{Proportion of consensus sets\\"
     r"containing an increasing state}", "{:.1%}"),
]

# Rows that only mean something on some datasets. An increasing state is an
# error for a monotone outcome and expected otherwise, so the row that counts
# them is reported only where it counts a mistake.
ROW_DATASETS = {"share_increasing": ("flchain",)}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Restoring what each configuration contributes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def load_real_ensemble(dataset, configuration, verbose=True):
    """Restore one configuration's ensemble and the split it is scored on.

    Args:
        dataset: dict; one entry of DATASETS.
        configuration: dict; one entry of CONFIGURATIONS.
        verbose: bool; whether to report which run is being scored.

    Returns:
        tuple (ensemble, test); the BootstrapEnsemble with its members
            restored, and the held-out RealDataset.
    """
    _, split = data_loader.load_real_split(dataset["name"])
    run_name = f"{configuration['prefix']}_{dataset['name']}"
    ensemble = model_loader.load_bootstrap_ensemble(run_name,
                                                    configuration["model_cls"])
    if verbose:
        print(f"  scoring {run_name} over {len(ensemble.engines)} members",
              flush=True)

    return ensemble, split.test


def score_shape_uncertainty(ensemble, test, combined, rng):
    """Score every individual's cloud, however that cloud is built.

    Args:
        ensemble: BootstrapEnsemble; with its members restored.
        test: RealDataset; the held-out split.
        combined: bool; whether to draw random effects within each member.
        rng: np.random.Generator; driving the aleatoric draws.

    Returns:
        dict; as the uncertainty engine returns, carrying "U" and "consensus".
    """
    engine = UncertaintyEngine(ensemble.engines)
    if combined:
        return engine.predict_with_combined_uncertainty(
            test.X, n_samples=NR_DRAWS, rng=rng)

    return engine.predict_with_epistemic_uncertainty(test.X)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Reducing a scored split to the rows the table reports
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def summarise_quartiles(values):
    """Reduce a distribution to the median and the quartiles around it.

    Args:
        values: np.ndarray of shape (D,); the distribution to summarise.

    Returns:
        tuple (median, q1, q3) of float.
    """
    return (float(np.median(values)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)))


def count_consensus_states(consensus):
    """Count the shape states each consensus summary reports.

    Args:
        consensus: list of length D; one shape summary per individual.

    Returns:
        np.ndarray of shape (D,) and dtype int; the state count per individual.
    """
    return np.array([len(summary) for summary in consensus], dtype=int)


def flag_increasing_consensus(consensus):
    """Flag the consensus summaries that report an increasing state.

    Args:
        consensus: list of length D; one shape summary per individual.

    Returns:
        np.ndarray of shape (D,) and dtype bool; True where any state of that
            individual's summary increases.
    """
    return np.array(
        [any(state.endswith("increasing") for state, _ in summary)
         for summary in consensus],
        dtype=bool)


def summarise_split(dataset, configuration, rng, verbose=True):
    """Summarise one dataset's split under one configuration.

    Args:
        dataset: dict; one entry of DATASETS.
        configuration: dict; one entry of CONFIGURATIONS.
        rng: np.random.Generator; driving the aleatoric draws.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict; one entry per key named in ROWS.
    """
    ensemble, test = load_real_ensemble(dataset, configuration,
                                        verbose=verbose)
    result = score_shape_uncertainty(ensemble, test,
                                     configuration["combined"], rng)

    uncertainty = np.asarray(result["U"], dtype=float)
    states = count_consensus_states(result["consensus"])
    increasing = flag_increasing_consensus(result["consensus"])

    return {
        "nr_individuals": int(test.D),
        "median_states": float(np.median(states)),
        "shape_quartiles": summarise_quartiles(uncertainty),
        "share_certain": float(np.mean(uncertainty == 0.0)),
        "share_increasing": float(np.mean(increasing)),
    }


def tabulate_datasets(verbose=True):
    """Summarise every dataset under every configuration.

    Args:
        verbose: bool; whether to report progress on stdout.

    Returns:
        list of dict; one per dataset, carrying "name", "heading" and
            "summaries", the last one entry per configuration in order.
    """
    rng = np.random.default_rng(SEED)

    panels = []
    for dataset in DATASETS:
        if verbose:
            print(f"\n{dataset['name']}", flush=True)

        summaries = [summarise_split(dataset, configuration, rng,
                                     verbose=verbose)
                     for configuration in CONFIGURATIONS]
        panels.append({**dataset, "summaries": summaries})

    return panels


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Rendering
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def keep_row(key, dataset_name):
    """Say whether one row is reported for one dataset.

    Args:
        key: str; the row's key in a summary.
        dataset_name: str; the dataset the panel describes.

    Returns:
        bool; whether to render the row.
    """
    return dataset_name in ROW_DATASETS.get(key, (dataset_name,))


def format_metric(value, spec):
    """Format one figure to the precision its row is reported at.

    Args:
        value: the figure to render; a triple (median, Q1, Q3) where the spec
            names one, otherwise a single number.
        spec: str; the row's format string, or "median-iqr" for a median
            written with its quartiles beside it.

    Returns:
        str; the formatted figure, safe to place in a latex cell.
    """
    if spec == "median-iqr":
        median, q1, q3 = value

        return f"{median:.3f} $[{q1:.3f}, {q3:.3f}]$"

    return spec.format(value).replace("%", r"\%")


def render_latex_table(panels):
    """Assemble the configurations into a booktabs tabular, one panel each.

    Args:
        panels: list of dict; as tabulate_datasets returns.

    Returns:
        str; the latex table.
    """
    nr_columns = len(CONFIGURATIONS)
    lines = [r"\begin{tabular}{l" + "r" * nr_columns + "}", r"\toprule"]

    headings = " & ".join(configuration["heading"]
                          for configuration in CONFIGURATIONS)
    lines.append("Quantity & " + headings + r" \\")
    lines.append(r"\midrule")

    for position, panel in enumerate(panels):
        lines.append(rf"\multicolumn{{{1 + nr_columns}}}{{l}}"
                     rf"{{{panel['heading']}}} \\")

        for key, label, spec in ROWS:
            if not keep_row(key, panel["name"]):
                continue

            cells = [format_metric(summary[key], spec)
                     for summary in panel["summaries"]]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        if position != len(panels) - 1:
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


def report_panels(panels):
    """Print the figures each panel carries, once they are all scored.

    Args:
        panels: list of dict; as tabulate_datasets returns.

    Returns:
        None
    """
    for panel in panels:
        print(f"\n{panel['name']}")
        for configuration, summary in zip(CONFIGURATIONS, panel["summaries"]):
            median, q1, q3 = summary["shape_quartiles"]
            print(f"  {configuration['prefix']:<9} "
                  f"median states {summary['median_states']:.1f}   "
                  f"median U {median:.4f} [{q1:.4f}, {q3:.4f}]   "
                  f"U = 0 in {summary['share_certain']:.1%}   "
                  f"increasing in {summary['share_increasing']:.1%}")


if __name__ == "__main__":
    print("Table 8: what each configuration reports on the deployment data")
    panels = tabulate_datasets()
    report_panels(panels)

    write_latex_table(render_latex_table(panels))
    print(f"\nWrote {OUTPUT_PATH}")
