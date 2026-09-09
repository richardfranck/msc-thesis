"""Table 3: Summary of the datasets used throughout the thesis.

    python thesis/thesis_table_3.py
"""

from pathlib import Path
import sys

import numpy as np

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))
sys.path.insert(0, str(THESIS_DIR))

import config
import data_loader


OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_3.tex"

# The datasets appear in the order they are introduced in the thesis, one
# column each.
DATASET_LABELS = ("Tumour", "Flchain", "Airfoil")

# The table's rows. A "text" row carries a value that is already written out,
# so that the formatting of every other row stays in one place.
ROWS = [
    ("individuals", r"Individuals", "{:,}"),
    ("split_sizes", r"Train / validation / test", "text"),
    ("covariates", r"Covariates", "{:d}"),
    ("observations", r"Observation count range", "text"),
    ("median_coverage", r"Median domain coverage", "{:.1%}"),
]


def load_dataset_splits():
    """Load the three dataset splits used by the thesis analyses.

    Returns:
        dict: Maps each printed dataset label to a Split namedtuple with
            ``train``, ``val`` and ``test`` datasets.
    """
    _, flchain_split = data_loader.load_real_split("flchain")
    _, airfoil_split = data_loader.load_real_split("airfoil")

    return {
        "Tumour": data_loader.draw_simulated_split(
            config.REGIME_HOMOGENEOUS),
        "Flchain": flchain_split,
        "Airfoil": airfoil_split,
    }


def calculate_median_domain_coverage(datasets):
    """Calculate the median share of the observed time horizon.

    Each individual's coverage is ``(max_j t_ij - min_j t_ij) / T``.

    Args:
        datasets: sequence of SimulatedDataset or RealDataset; the partitions
            whose individuals are included in the table row.

    Returns:
        float: The median individual domain coverage.
    """
    coverages = []
    for dataset in datasets:
        if hasattr(dataset, "compute_horizon_coverage"):
            coverages.extend(dataset.compute_horizon_coverage())
            continue

        coverages.extend((times.max() - times.min()) / dataset.T
                         for times in dataset.times)

    return float(np.median(coverages))


def format_observation_counts(datasets):
    """Format the per-individual observation counts for one table cell.

    Args:
        datasets: sequence of SimulatedDataset or RealDataset; the partitions
            whose observation counts are reported.

    Returns:
        str: The common count, or ``minimum--maximum`` when counts differ
            between individuals.
    """
    counts = [count for dataset in datasets for count in dataset.N]
    lower, upper = min(counts), max(counts)

    if lower == upper:
        return f"{lower:,}"

    return f"{lower:,}--{upper:,}"


def count_raw_covariates(dataset):
    """Count the covariates recorded before categorical encoding.

    Args:
        dataset: SimulatedDataset or RealDataset; one partition of the dataset
            being summarised.

    Returns:
        int: The number of raw covariates in the source dataset.
    """
    if hasattr(dataset, "feature_ranges"):
        return len(dataset.feature_ranges)

    return len(dataset.X)


def summarise_dataset_split(label, split):
    """Summarise one full dataset from its train, validation and test splits.

    Args:
        label: str; the dataset name printed in the table.
        split: Split; a namedtuple with ``train``, ``val`` and ``test``
            dataset partitions.

    Returns:
        dict: The values required for one rendered table column.
    """
    datasets = (split.train, split.val, split.test)
    split_sizes = f"{split.train.D:,} / {split.val.D:,} / {split.test.D:,}"

    return {
        "label": label,
        "individuals": sum(dataset.D for dataset in datasets),
        "split_sizes": split_sizes,
        "covariates": count_raw_covariates(split.train),
        "observations": format_observation_counts(datasets),
        "median_coverage": calculate_median_domain_coverage(datasets),
    }


def build_dataset_summaries(dataset_splits):
    """Build the ordered columns of the dataset-summary table.

    Args:
        dataset_splits: dict mapping each printed dataset label to its Split,
            as returned by :func:`load_dataset_splits`.

    Returns:
        list of dict: One summary per dataset, in column order.
    """
    return [
        summarise_dataset_split(label, dataset_splits[label])
        for label in DATASET_LABELS
    ]


def format_metric(value, spec):
    """Format one figure to the precision its row is reported at.

    Args:
        value: int, float or str; the figure to render.
        spec: str; the row's format string, or ``text`` for a prepared value.

    Returns:
        str: The formatted figure, safe to place in a LaTex cell.
    """
    if spec == "text":
        return value

    return spec.format(value).replace("%", r"\%")


def render_latex_table(summaries):
    """Render the dataset summaries as a booktabs LaTex tabular.

    One column per dataset, one row per reported quantity.

    Args:
        summaries: sequence of dict; one summary per dataset, in column order.

    Returns:
        str: The complete LaTex tabular environment.
    """
    # Step 1: Open the tabular, the label column left, one column per dataset
    lines = [r"\begin{tabular}{l" + "r" * len(summaries) + "}", r"\toprule"]

    # Step 2: Head each column with its dataset
    headings = " & ".join(summary["label"] for summary in summaries)
    lines.append("Quantity & " + headings + r" \\")
    lines.append(r"\midrule")

    # Step 3: One row per reported quantity
    for key, label, spec in ROWS:
        cells = [format_metric(summary[key], spec) for summary in summaries]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    # Step 4: Close the tabular
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


def write_latex_table(table, output_path=OUTPUT_PATH):
    """Write the rendered table to the thesis tables directory.

    Args:
        table: str; the LaTex tabular returned by :func:`render_latex_table`.
        output_path: pathlib.Path; destination of the generated ``.tex`` file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n")


if __name__ == "__main__":
    dataset_splits = load_dataset_splits()
    summaries = build_dataset_summaries(dataset_splits)
    write_latex_table(render_latex_table(summaries))
    print(f"Wrote {OUTPUT_PATH}")
