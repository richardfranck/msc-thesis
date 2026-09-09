"""Appendix Figure D1: How many airfoil configurations observe each band.

    python thesis/thesis_figure_D3.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))

import plot_style

from scripts.shape_uncertainty.real_data.real_dataset import DATA_DIR

OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_D3.pdf"

ASPECT = 0.46

# The raw columns, in the order the file gives them.
RAW_COLUMNS = ["t", "angle", "chord", "velocity", "thickness", "y"]

# The frequency the horizon is measured from, so that t = 0 is the lowest band.
BASE_FREQUENCY = 200.0

# The bands labelled on the lower axis, in Hz. The rest are left unlabelled so
# the ticks stay legible.
LABELLED_BANDS = [200, 500, 1000, 2000, 5000, 10000, 20000]

# Where the upper axis is ticked, on the horizon the model is fitted over.
HORIZON_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def count_configurations_per_band():
    """Count how many configurations report each one-third-octave band.

    The raw file carries no identifier, so the configurations are recovered as
    _read_airfoil recovers them: each is a sweep upwards through the bands, and
    a falling frequency marks the start of the next.

    Returns:
        tuple (frequencies, counts, nr_configurations):
            frequencies: np.ndarray of shape (21,); the band centres in Hz,
                ascending.
            counts: np.ndarray of shape (21,) and dtype int; how many
                configurations report each band.
            nr_configurations: int; the number of configurations in the file.
    """
    frame = pd.read_csv(DATA_DIR / "airfoil" / "airfoil_self_noise.dat",
                        sep="\t", header=None, names=RAW_COLUMNS)
    frame["id"] = (frame["t"].diff().fillna(-1.0) < 0).cumsum()

    per_band = frame.groupby("t")["id"].nunique().sort_index()

    return (per_band.index.to_numpy(dtype=float),
            per_band.to_numpy(dtype=int),
            int(frame["id"].nunique()))


def rescale_to_horizon(frequencies):
    """Put the band centres on the horizon the model is fitted over.

    Args:
        frequencies: np.ndarray; the band centres in Hz.

    Returns:
        np.ndarray of the same shape; the band centres as t in [0, 1].
    """
    horizon = np.log(frequencies / BASE_FREQUENCY)

    return horizon / horizon.max()


def add_horizon_axis(axis):
    """Label the upper axis with the horizon the model is fitted over.

    The bars are placed at their own position on that horizon, so the upper
    axis needs only to be ticked; it reads the same coordinate as the lower one.

    Args:
        axis: plt.Axes; the axes the bars are drawn on.

    Returns:
        plt.Axes; the twinned upper axis.
    """
    upper = axis.twiny()
    upper.set_xlim(axis.get_xlim())

    upper.set_xticks(HORIZON_TICKS)
    upper.set_xticklabels([f"{tick:.1f}" for tick in HORIZON_TICKS])
    upper.set_xlabel(r"normalised log frequency $t$")

    return upper


def plot_band_coverage(times, counts, nr_configurations, frequencies,
                       aspect=ASPECT):
    """Draw the number of configurations reporting each band.

    Args:
        times: np.ndarray; the band centres as t in [0, 1].
        counts: np.ndarray; how many configurations report each band.
        nr_configurations: int; the cohort size, drawn as the reference line.
        frequencies: np.ndarray; the band centres in Hz, for the upper axis.
        aspect: float; figure height divided by figure width.

    Returns:
        plt.Figure: the completed figure.
    """
    figure, axis = plot_style.subplots(1.0, aspect=aspect)

    # Step 1: The cohort size, so the shortfall at each band is readable
    axis.axhline(nr_configurations, color=plot_style.GRAY_DARK, lw=0.8,
                 ls="--", zorder=1)
    axis.text(0.012, nr_configurations - 3.5,
              f"all {nr_configurations} configurations",
              color=plot_style.GRAY_DARK, fontsize=plot_style.FONT_SIZE_SMALL,
              va="top")

    # Step 2: One bar per band, at the band's own place on the horizon
    width = 0.8 * np.min(np.diff(times))
    axis.bar(times, counts, width=width, facecolor="none",
             edgecolor="black", lw=1.0, zorder=2)

    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(0, nr_configurations * 1.12)
    axis.set_ylabel("configurations")

    # Step 3: The lower axis names the bands in their own units, the upper one
    #         the horizon the model is fitted over
    labelled = [f in LABELLED_BANDS for f in frequencies.astype(int)]
    axis.set_xticks(times[labelled])
    axis.set_xticklabels([f"{int(f):,}" for f in frequencies[labelled]])
    axis.set_xlabel("band centre frequency (Hz)")

    add_horizon_axis(axis)

    return figure


def report_band_coverage(frequencies, counts, nr_configurations):
    """Print the counts behind the figure, for the appendix text.

    Args:
        frequencies: np.ndarray; the band centres in Hz.
        counts: np.ndarray; how many configurations report each band.
        nr_configurations: int; the cohort size.
    """
    print(f"{len(frequencies)} bands, {nr_configurations} configurations")
    for frequency, count in zip(frequencies, counts):
        print(f"  {int(frequency):6d} Hz  {count:3d}  "
              f"{100 * count / nr_configurations:5.1f}%")


def save_figure(figure, output_path=OUTPUT_PATH):
    """Save the completed figure as a PDF.

    Args:
        figure: plt.Figure; the completed figure.
        output_path: pathlib.Path; destination of the PDF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(figure, str(output_path))


if __name__ == "__main__":
    plot_style.use()
    frequencies, counts, nr_configurations = count_configurations_per_band()
    times = rescale_to_horizon(frequencies)

    report_band_coverage(frequencies, counts, nr_configurations)
    figure = plot_band_coverage(times, counts, nr_configurations, frequencies)
    save_figure(figure)
    print(f"Wrote {OUTPUT_PATH}")
