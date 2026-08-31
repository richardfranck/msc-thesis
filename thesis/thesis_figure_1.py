"""Figure 1: Two bases for the same cubic spline space, with two interior knots.

This file draws both bases a cubic spline with two interior knots can be written
in, in a panel with two elements:
    (a) the truncated power basis, global and badly scaled
    (b) the B-spline basis, a set of bumps with local support

    python thesis/thesis_figure_1.py
"""

from pathlib import Path
import sys

import numpy as np
from matplotlib.lines import Line2D

# The basis package lives at the repository root, one level up.
THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))

import config
import plot_helpers
import plot_style

from scripts.shape_uncertainty.spline_basis.bspline_basis import (
    basis_matrix, build_knot_dictionary)

# Plot layout and resolution configurations
N_DENSE = config.N_DENSE
ASPECT = 0.38

# Ouptut path
OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_1.pdf"

# Two interior knots. Cubic spline.
NR_INTERIOR_KNOTS = 2
DEGREE = 3

# Labels for interior knots.
KNOT_LABELS = [r"$\xi_1$", r"$\xi_2$"]

# The truncated power basis.
POWER_LABELS = ["$1$", "$t$", "$t^2$", "$t^3$",
                r"$(t-\xi_1)^3_+$", r"$(t-\xi_2)^3_+$"]

# Pattern per basis function line
BASIS_STYLES = [
    "solid",
    (0, (5, 1.5)),                    # dashed
    (0, (1, 1.5)),                    # dotted
    (0, (5, 1.5, 1, 1.5)),            # dash-dot
    (0, (9, 2)),                      # long dash
    (0, (5, 1.5, 1, 1.5, 1, 1.5)),    # dash-dot-dot
]


def build_basis():
    """Build the spline objects the knot configuration implies.

    Returns:
        dict, the spline objects from build_knot_dictionary, holding the basis
            functions, the augmented knot vector, the breakpoints and the
            coefficient dimension.
    """
    return build_knot_dictionary(nr_interior_knots=NR_INTERIOR_KNOTS, T=config.T)

def truncated_power_matrix(times, breakpoints):
    """Evaluate the truncated power basis of this configuration on a grid.

    For a cubic with interior knots xi_1, ..., xi_K the basis is the monomials
    up to the degree, followed by one truncated power per interior knot:

        1, t, t^2, t^3, (t - xi_1)^3_+, ..., (t - xi_K)^3_+,

    where x_+ is x for positive x and zero otherwise.

    Args:
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        breakpoints: np.ndarray of shape (K + 2,), interior knots plus
            boundaries.

    Returns:
        np.ndarray of shape (M, K + 4), the basis functions evaluated at times,
            ordered as above.
    """
    interior = np.unique(breakpoints)[1:-1]
    monomials = [times ** power for power in range(DEGREE + 1)]
    truncated = [np.clip(times - xi, 0.0, None) ** DEGREE for xi in interior]
    return np.column_stack(monomials + truncated)

def _stack_three_two_one(handles, labels):
    """Reorder legend entries so they lay out in rows of three, two and one.

    Args:
        handles: list of Artist, the legend handles in their drawn order.
        labels: list of str, the labels matching handles.

    Returns:
        tuple (handles, labels): the same entries, padded and reordered for a
            column-major fill that reads as rows of three, two and one.
    """
    blank = Line2D([], [], linestyle="none")
    order = [0, 3, 5, 1, 4, None, 2, None, None]
    padded_handles = [blank if k is None else handles[k] for k in order]
    padded_labels = ["" if k is None else labels[k] for k in order]
    return padded_handles, padded_labels


def _add_truncated_power(axis, times, Psi):
    """Draw panel (a): the truncated power basis, each member named.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        Psi: np.ndarray of shape (M, K + 4), the truncated power basis evaluated
            on that grid.

    Returns:
        None
    """
    # Step 1: Draw each member in one ink, told apart by its dash pattern
    plot_helpers.add_basis_curves(axis, times, Psi, POWER_LABELS, BASIS_STYLES)

    # Step 2: Scale the panel, then lay the legend out in rows of three, two and one
    axis.set_ylabel(r"$\phi_b(t)$")
    axis.set_ylim(0.0, 1.05)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    handles, labels = axis.get_legend_handles_labels()
    handles, labels = _stack_three_two_one(handles, labels)

    axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.95),
                ncol=3, columnspacing=0.8, handlelength=1.6, handletextpad=0.4,
                labelspacing=0.3, borderpad=0.2,
                fontsize=plot_style.FONT_SIZE_SMALL - 1)

def _add_basis_functions(axis, times, Phi):
    """Draw panel (b): the B-spline basis, each member named.

    Args:
        axis: plt.Axes, the target subplot axis to receive the plot layer.
        times: np.ndarray of shape (M,), the evaluation grid on [0, T].
        Phi: np.ndarray of shape (M, B), the B-spline basis evaluated on that
            grid.

    Returns:
        None
    """
    # Step 1: Draw each member in one ink, told apart by its dash pattern.
    # The members are numbered rather than written out: unlike the truncated
    # powers, no B-spline has a short closed form to name it by.
    labels = [rf"$\phi_{{{b + 1}}}$" for b in range(Phi.shape[1])]
    plot_helpers.add_basis_curves(axis, times, Phi, labels, BASIS_STYLES)

    # Step 2: Scale the panel and name the members
    axis.set_ylabel(r"$\phi_b(t)$")
    axis.set_ylim(0.0, 1.05)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.legend(loc="upper center", ncol=3, columnspacing=0.9,
                handlelength=1.9, handletextpad=0.4, labelspacing=0.3,
                borderpad=0.2, fontsize=plot_style.FONT_SIZE_SMALL - 1)

def plot_spline_basis(knot_objects):
    """Orchestrate the two basis panels side by side.

    Args:
        knot_objects: dict, the spline objects from build_knot_dictionary.

    Returns:
        plt.Figure, the complete two-panel figure.
    """
    # Step 1: Evaluate both bases on the shared grid, before any drawing starts
    T = config.T
    times = np.linspace(0.0, T, N_DENSE)
    breakpoints = knot_objects["breakpoints"]

    Psi = truncated_power_matrix(times, breakpoints)
    Phi = basis_matrix(times, knot_objects["basis_functions"])

    # Step 2: Build the canvas at the full text width, two panels side by side
    fig, axes = plot_style.subplots(1.0, aspect=ASPECT, ncols=2,
                                   gridspec_kw={"wspace": 0.18})
    ax_power, ax_bspline = axes

    # Step 3: Rule and name the knots on both panels
    for axis in axes:
        plot_helpers.add_knot_rules(axis, breakpoints, interior_only=True)
        plot_helpers.add_knot_labels(axis, breakpoints, KNOT_LABELS)
        axis.set_xlabel("time $t$")
        axis.set_xlim(0.0, T)

    # Step 4: Layer each panel
    _add_truncated_power(ax_power, times, Psi)
    _add_basis_functions(ax_bspline, times, Phi)

    return fig


if __name__ == "__main__":
    plot_style.use()
    knot_objects = build_basis()
    fig = plot_spline_basis(knot_objects)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(fig, str(OUTPUT_PATH))
