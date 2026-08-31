"""Figure style for thesis plots.

The sections are:
    1.  Geometry
    2.  Colour
    3.  Typography
    4.  Overrides on the science stylesheet
    5.  Entry points
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import scienceplots


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Geometry
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PAPER_WIDTH_CM = 21.0
MARGIN_CM = 2.5
CM_PER_IN = 2.54
TEXTWIDTH = (PAPER_WIDTH_CM - 2 * MARGIN_CM) / CM_PER_IN

# Width of one of a side-by-side pair
GUTTER = 0.20
HALFWIDTH = (TEXTWIDTH - GUTTER) / 2
HALF = HALFWIDTH / TEXTWIDTH

# Default height-to-width ratio
ASPECT = 0.62


def figsize(fraction=1.0, aspect=ASPECT):
    """Figure size in inches as a fraction of the document textwidth.

    Args:
        fraction: float; width as a fraction of TEXTWIDTH. Use 1.0 for a
            full-width figure and HALF for one of a side-by-side pair.
        aspect: float; height divided by width. Default ASPECT.

    Returns:
        tuple (width, height) in inches, for matplotlib's figsize argument.
    """
    w = TEXTWIDTH * fraction
    return (w, w * aspect)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Colour
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CYCLE = [
    "#0C5DA5",  # blue
    "#FF9500",  # orange
    "#FF2C00",  # red
    "#00B945",  # green   -- 4th series onward: vary linestyle or marker too
    "#845B97",  # violet
]

# Reserved, never used as a series colour.
GRAY_DARK = "#474747"   # ground truth
GRAY_LIGHT = "#9E9E9E"  # observation scatter

# Semantic roles for the uncertainty plots.
COLORS = {
    "mean": CYCLE[0],
    "epistemic": CYCLE[0],
    "aleatoric": CYCLE[1],
    "truth": GRAY_DARK,
    "observations": GRAY_LIGHT,
}

# Fill opacity for uncertainty bands. Two nested bands (e.g. 1 and 2 SD) should
# use BAND and BAND_OUTER so the inner still reads as darker.
BAND = 0.25
BAND_OUTER = 0.12


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Typography
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Point sizes are absolute, so they are only correct if the figure is not
# rescaled in LaTeX. 
FONT_SIZE = 11
FONT_SIZE_SMALL = 9


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. Overrides on the science stylesheet
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
RC = {
    # Figure
    "figure.figsize": figsize(),
    "figure.dpi": 120,
    "figure.constrained_layout.use": True,

    # Colour cycle
    "axes.prop_cycle": cycler("color", CYCLE),

    # Type
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE_SMALL,
    "ytick.labelsize": FONT_SIZE_SMALL,
    "legend.fontsize": FONT_SIZE_SMALL,

    # Legend, tightened a little for the wider figure
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.4,

    # Output.
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "savefig.dpi": 600,
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5. Entry points
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def use(latex=True):
    """Apply the thesis figure style to the global matplotlib state.

    Call once per notebook, before creating any figure.

    Args:
        latex: bool; whether to render text through a real LaTeX installation.
            True gives maths identical to the surrounding document and is what
            the thesis figures use. False selects the stylesheet's "no-latex"
            variant, which falls back to matplotlib's mathtext with Computer
            Modern; glyphs are very close but not identical, so do not mix the
            two within a chapter.
    """
    # Step 1: Start from the science stylesheet, which resets rcParams.
    plt.style.use(["science"] if latex else ["science", "no-latex"])

    # Step 2: Layer the thesis overrides on top.
    mpl.rcParams.update(RC)


def subplots(fraction=1.0, aspect=ASPECT, **kwargs):
    """Create a figure sized to a fraction of the textwidth.

    Args:
        fraction: float; width as a fraction of TEXTWIDTH.
        aspect: float; height divided by width. Default ASPECT.
        **kwargs: forwarded to plt.subplots, e.g. nrows, ncols, sharex.

    Returns:
        tuple (fig, ax) as returned by plt.subplots.
    """
    size = figsize(fraction, aspect)
    fig, ax = plt.subplots(figsize=size, **kwargs)
    fig._thesis_size = size
    return fig, ax


def save(fig, path):
    """Write a figure at exactly the size it was created with.

    Args:
        fig: the matplotlib Figure to write.
        path: destination path. The extension selects the format; prefer .pdf
            so the figure stays vector.
    """
    size = getattr(fig, "_thesis_size", None)
    if size is not None:
        fig.set_size_inches(size)
    fig.savefig(path, bbox_inches=None, pad_inches=0.0)


def band(ax, t, lower, upper, color, alpha=BAND, label=None):
    """Shade an uncertainty band with no edge line.

    Args:
        ax: the Axes to draw on.
        t: 1-D array of times, shape (M,).
        lower: 1-D array of lower band values, shape (M,).
        upper: 1-D array of upper band values, shape (M,).
        color: fill colour, e.g. one of COLORS.
        alpha: fill opacity. Default BAND; use BAND_OUTER for the wider of two
            nested bands.
        label: legend label, or None to keep the band out of the legend.

    Returns:
        The PolyCollection returned by fill_between.
    """
    return ax.fill_between(t, lower, upper, color=color, alpha=alpha,
                           linewidth=0, label=label)


