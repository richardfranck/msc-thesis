"""Appendix Table D: Covariates used by the three thesis datasets.

    python thesis/thesis_table_appendix_D.py
"""

from pathlib import Path
import sys

import numpy as np

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))
sys.path.insert(0, str(THESIS_DIR))

import config

from scripts.shape_uncertainty.real_data.real_dataset import load_real_dataset


OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_appendix_D.tex"

# The labels and meanings used to make the source covariate names readable.
COVARIATE_DETAILS = {
    "Tumour": {
        "size": ("Initial tumour volume", "Baseline tumour size", ""),
        "age": ("Age", "Patient age", ""),
        "weight": ("Weight", "Patient weight", ""),
        "dosage": ("Dosage", "Normalised drug dosage", ""),
    },
    "Flchain": {
        "age": ("Age", "Patient age", ""),
        "sex": ("Sex", "Patient sex", ""),
        "creatinine": ("Serum creatinine", "Serum creatinine level", ""),
        "kappa": ("Kappa FLC", "Kappa free light chain", ""),
        "lambda": ("Lambda FLC", "Lambda free light chain", ""),
        "flc.grp": ("FLC group", "Free light chain group", ""),
        "mgus": ("MGUS", "MGUS status", ""),
    },
    "Airfoil": {
        "angle": ("Angle", "Angle of attack", r"$^\circ$"),
        "chord": ("Chord", "Chord length", " m"),
        "velocity": ("Velocity", "Free-stream velocity", " m/s"),
        "thickness": ("Thickness", "Suction-side displacement thickness", " m"),
    },
}


def load_real_datasets():
    """Load the two real datasets whose empirical ranges are reported.

    Returns:
        dict: Maps ``Flchain`` and ``Airfoil`` to their RealDataset objects.
    """
    return {
        "Flchain": load_real_dataset("flchain", seed=config.SEED),
        "Airfoil": load_real_dataset("airfoil", seed=config.SEED),
    }


def format_numeric_range(values, unit=""):
    """Format the lower and upper bounds of one continuous covariate.

    Args:
        values: np.ndarray; the raw covariate values.
        unit: str; text appended after the upper bound.

    Returns:
        str: The range written as ``minimum--maximum`` with its unit.
    """
    lower = float(np.min(values))
    upper = float(np.max(values))
    return f"{lower:g}--{upper:g}{unit}"


def collect_observed_levels(dataset, covariate, levels):
    """Collect the categorical levels that occur in a loaded dataset.

    Args:
        dataset: RealDataset; the encoded dataset holding the covariate.
        covariate: str; the original categorical covariate name.
        levels: list; the levels declared by the dataset specification.

    Returns:
        list: The declared levels that are present in the loaded data.
    """
    if len(levels) == 2:
        indicator = dataset.X[covariate]
        return [
            level for value, level in enumerate(levels)
            if np.any(indicator == value)
        ]

    return [
        level for level in levels
        if np.any(dataset.X[f"{covariate}_{level}"] == 1.0)
    ]


def format_observed_levels(levels):
    """Format categorical levels for the range-or-levels column.

    Args:
        levels: list; the observed categorical levels in display order.

    Returns:
        str: A compact range for consecutive integer levels, or a comma list.
    """
    if all(isinstance(level, int) for level in levels):
        expected = list(range(levels[0], levels[-1] + 1))
        if levels == expected:
            return f"{levels[0]}--{levels[-1]}"

    return ", ".join(str(level).capitalize() for level in levels)


def build_tumour_rows():
    """Build the rows from the Tumour data-generating design ranges.

    Returns:
        list of dict: The Tumour covariate rows in display order.
    """
    rows = []
    for covariate, design_range in config.COVARIATE_RANGES.items():
        label, meaning, unit = COVARIATE_DETAILS["Tumour"][covariate]
        rows.append({
            "label": label,
            "meaning": meaning,
            "range": format_numeric_range(np.asarray(design_range), unit),
        })

    return rows


def build_real_dataset_rows(label, dataset):
    """Build the rows from one real dataset's observed covariate values.

    Args:
        label: str; the dataset heading, ``Flchain`` or ``Airfoil``.
        dataset: RealDataset; the loaded dataset whose covariates are reported.

    Returns:
        list of dict: The covariate rows in source-dataset order.
    """
    rows = []
    for covariate, declared_range in dataset.feature_ranges.items():
        name, meaning, unit = COVARIATE_DETAILS[label][covariate]

        if isinstance(declared_range, tuple):
            range_or_levels = format_numeric_range(dataset.X[covariate], unit)
        else:
            levels = collect_observed_levels(
                dataset, covariate, declared_range)
            range_or_levels = format_observed_levels(levels)

        rows.append({
            "label": name,
            "meaning": meaning,
            "range": range_or_levels,
        })

    return rows


def build_table_sections(real_datasets):
    """Assemble the three labelled row groups of the appendix table.

    Args:
        real_datasets: dict mapping real-data labels to RealDataset objects,
            as returned by :func:`load_real_datasets`.

    Returns:
        list of tuple: Each ``(label, rows)`` table section, in display order.
    """
    return [
        ("Tumour", build_tumour_rows()),
        ("Flchain", build_real_dataset_rows("Flchain", real_datasets["Flchain"])),
        ("Airfoil", build_real_dataset_rows("Airfoil", real_datasets["Airfoil"])),
    ]


def render_latex_table(sections):
    """Render the covariate sections as a booktabs LaTex tabular.

    Args:
        sections: sequence of ``(label, rows)`` tuples from
            :func:`build_table_sections`.

    Returns:
        str: The complete LaTex tabular environment.
    """
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Covariate & Meaning & Range / levels " + r"\\",
        r"\midrule",
    ]

    for position, (label, rows) in enumerate(sections):
        if position > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{label}}}}} " + r"\\")

        for row in rows:
            lines.append(
                f"{row['label']} & {row['meaning']} & {row['range']} "
                + r"\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_latex_table(table, output_path=OUTPUT_PATH):
    """Write the rendered appendix table to the thesis tables directory.

    Args:
        table: str; the LaTex tabular returned by :func:`render_latex_table`.
        output_path: pathlib.Path; destination of the generated ``.tex`` file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n")


if __name__ == "__main__":
    real_datasets = load_real_datasets()
    sections = build_table_sections(real_datasets)
    write_latex_table(render_latex_table(sections))
    print(f"Wrote {OUTPUT_PATH}")
