"""Appendix Table D1.2: Reference shape summaries produced by each data regime.

    python thesis/thesis_table_D1_2.py
"""

from collections import Counter
from pathlib import Path
import sys

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))

import config

from scripts.shape_uncertainty.simulated_data.data_generator import (
    WilkersonDGP, generate_simulated_dataset)
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import (
    get_analytic_shape_summary)

# Output path
OUTPUT_PATH = THESIS_DIR / "tables" / "thesis_table_D1_2.tex"

# The rows, in the order they appear in the table.
REGIMES = [config.REGIME_HOMOGENEOUS, config.REGIME_HETEROGENEOUS]


def _build_regime_dataset_split(regime):
    """Draw and split the dataset for the given data regime.

    1. Rebuild the Wilkerson process at the regime's heterogeneity level,
    2. Draw config.D_TOTAL individuals under the shared observation design,
    3. SPlit the datainto (test, val train).

    Args:
        regime: the DGP regime, supplying the hyperparameters
            and the observation design.

    Returns:
        Split: the namedtuple (test, val, train) of SimulatedDatasets.
    """
    # Step 1: Build the Wilkerson process at this regime's heterogeneity level
    dgp = WilkersonDGP(T=config.T, hyperparams=regime.hyperparams)

    # Step 2: Draw the full population under the shared observation design.
    dataset = generate_simulated_dataset(dgp, D=config.D_TOTAL, **regime.design)

    # Step 3: Split the dataset.
    return dataset.split_test_val_train(
        D_train=config.D_TRAIN, D_val=config.D_VAL, D_test=config.D_TEST)


def _get_true_shape_summaries(dataset):
    """Read the analytic ground-truth shape summary of every individual.

    Args:
        dataset: SimulatedDataset; the individuals to summarise.

    Returns:
        list of length dataset.D; element i is individual i's shape summary as
            a chronological list of (state, start_time) tuples.
    """
    return get_analytic_shape_summary(
        dataset,
        config.SHAPE_CONFIG["upsilon_rel_1"],
        config.SHAPE_CONFIG["upsilon_rel_2"],
        config.SHAPE_CONFIG["upsilon_rel_prune"],
        config.SHAPE_CONFIG["do_prune"],
    )


def get_shape_class(summary):
    """Reduce a shape summary to the shape class it belongs to.

    We use a shape class as the ordered sequence of states alone, with the transition
    times dropped, so that two individuals whose curves turn at different times
    but pass through the same states count as one class.

    Args:
        summary: list of (state, start_time) tuples, sorted chronologically.

    Returns:
        tuple of str: the ordered state sequence, hashable so it can key a
            count.
    """
    return tuple(state for state, _ in summary)


def _count_shape_classes(summaries):
    """Count how many individuals fall in each shape class.

    Args:
        summaries: list of shape summaries, each a list of (state, start_time)
            tuples.

    Returns:
        collections.Counter: maps each shape class,
            to the number of individuals showing it.
    """
    return Counter(get_shape_class(summary) for summary in summaries)


def tabulate_shape_classes(regimes):
    """Count the shape classes each regime produces on its test split.

    We consider two data regimes that share the same seed and differ only
    in unobserved heterogeneity. We:
        1. Draw a dataset per regime,
        2. read the ground truth shape summary off the TEST set only,
        3. get the shape class counts.

    Args:
        regimes: sequence of config.Regime; one per table row, in the order the
            rows should appear.

    Returns:
        tuple (counts, classes):
            counts: dict mapping each regime's name to the Counter of shape
                class to number of individuals.
            classes: list of tuples of str; every shape class seen in any
                regime, ordered by total count descending.
    """
    # Step 1: Count each regime over its test split
    counts = {}
    for regime in regimes:
        split = _build_regime_dataset_split(regime)
        counts[regime.name] = _count_shape_classes(
            _get_true_shape_summaries(split.test))

    # Step 2: Pool the rows to fix one column order across the whole table.
    totals = Counter()
    for counter in counts.values():
        totals.update(counter)
    classes = sorted(totals, key=lambda shape_class: (-totals[shape_class],
                                                      shape_class))

    return counts, classes


def format_column_heading(name):
    """Render one regime as its column heading.

    Args:
        name: str; the regime's name, a key of config.REGIMES.

    Returns:
        str: the heading, abbreviating unobserved heterogeneity as UH.
    """
    label = "w/ UH" if config.REGIMES[name].has_heterogeneity else "w/o UH"

    return rf"\makecell[r]{{{label}}}"


def format_class_label(shape_class):
    """Render a shape class as a row label.

    Args:
        shape_class: tuple of str; a shape class, as returned by
            get_shape_class.

    Returns:
        str: the row label, the states joined by arrows and boxed in a makecell
            where there is more than one of them.
    """
    # Step 1: Name the states, the first capitalised and the rest running on
    states = [state.replace("_", " ") for state in shape_class]
    states[0] = states[0].capitalize()

    # Step 2: A class of one state is short enough to need no box
    if len(states) == 1:
        return states[0]

    # Step 3: Join the states by arrows, keeping the label on one line
    return r"\makecell[l]{" + r" $\rightarrow$ ".join(states) + "}"


def format_class_row(shape_class, cells):
    """Render one shape class and its counts as a table row.

    A single-state label is short enough to sit beside its counts. A boxed
    multi-state label is not, so its counts are carried to the next line.

    Args:
        shape_class: tuple of str; the shape class the row reports.
        cells: list of str; that class's count in each regime, in column order.

    Returns:
        str: the row, which may span two lines.
    """
    label = format_class_label(shape_class)
    counted = " & ".join(cells) + r" \\"

    if len(shape_class) == 1:
        return f"{label} & {counted}"

    return f"{label}\n    & {counted}"


def print_latex_table(counts, classes):
    """Assemble the counts into a booktabs tabular.

    Args:
        counts: dict mapping regime name to a Counter of shape class to number
            of test individuals, as returned by tabulate_shape_classes.
        classes: list of tuples of str; the shape classes in row order, as
            returned by tabulate_shape_classes.

    Returns:
        str: The latex table
    """
    # Step 1: We open the tabular, the label column left, every count right
    column_spec = "l" + "r" * len(counts)
    lines = [r"\begin{tabular}{" + column_spec + "}", r"\toprule"]

    # Step 2: We head each column with its regime, one heading per line
    names = list(counts)
    for position, name in enumerate(names):
        closing = r" \\" if position == len(names) - 1 else ""
        lines.append(f"& {format_column_heading(name)}{closing}")
    lines.append(r"\midrule")

    # Step 3: One row per shape class, the most common first
    for shape_class in classes:
        cells = [f"{counts[name].get(shape_class, 0)}" for name in names]
        lines.append(format_class_row(shape_class, cells))

    # Step 4: We close with the size of the split the counts were read off
    lines.append(r"\midrule")
    sizes = [f"{sum(counts[name].values())}" for name in names]
    lines.append(" & ".join(["Test dataset size"] + sizes) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


if __name__ == "__main__":
    counts, classes = tabulate_shape_classes(REGIMES)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(print_latex_table(counts, classes) + "\n")
    print(f"\nWrote {OUTPUT_PATH}")
