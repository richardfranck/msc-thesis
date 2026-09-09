"""Figure 8: where the dispersion sits, on both deployment datasets.

Requires: meanonly_airfoil, gaussian_airfoil, meanonly_flchain,
    gaussian_flchain

    python thesis/thesis_figure_8_final.py
"""

from pathlib import Path
import sys
import textwrap

import numpy as np
import torch

# The model and extraction packages live at the repository root, one level up.
THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))
sys.path.insert(0, str(THESIS_DIR))

import config
import data_loader
import model_loader
import plot_helpers
import plot_style

from scripts.shape_uncertainty.evaluation.rebound_localisation import (
    align_profile_on_offsets, resample_profile_on_grid)
from scripts.shape_uncertainty.model.model import GaussianModel, MeanOnlyModel
from scripts.shape_uncertainty.model.model_inference import UncertaintyEngine

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_8_final.pdf"

# The dataset each panel is measured on.
LEFT_DATASET = "airfoil"
RIGHT_DATASET = "flchain"

# Aleatoric draws per member, matching the simulation's random-effects run.
NR_DRAWS = 20

# The seed the aleatoric draws are taken under, so the figure is reproducible.
SEED = config.SEED_EVAL

# Plot layout configuration. The spacing is set on the layout engine rather
# than through gridspec, which constrained layout overrides, as in figure 6.
ASPECT = 0.72
HEIGHT_RATIOS = [3.0, 1.0]
LAYOUT_PADS = {"hspace": 0.03, "wspace": 0.06, "h_pad": 0.02, "w_pad": 0.02}

# The aligned axis of each panel, in time units either side of its turning
# point. The peaks of panel (a) sit mid-horizon and are approached from both
# sides; the reported increases of panel (b) run to the horizon's end, so that
# axis reaches further back than forward.
LEFT_OFFSETS = np.linspace(-0.90, 0.90, 361)
RIGHT_OFFSETS = np.linspace(-0.90, 0.30, 241)

# The share of the largest support panel (a) must retain to be drawn.
SUPPORT_FRACTION = 0.5

# Whether panel (a) pools an individual only where it was actually measured.
# \textsc{airfoil} is observed over part of its horizon and forecast over all
# of it, so this decides whether the panel reads as disagreement between
# observations or also carries the extrapolated stretch.
RESTRICT_TO_OBSERVED = True

# How far before the onset panel (b) draws. Near enough that every wrong
# summary still reaches it, so its left flank is one group throughout.
RIGHT_LEFT_LIMIT = -0.50

# The grid every profile is read on before it is aligned.
N_DENSE = 400

# The offsets each panel is reported at on stdout.
LEFT_PROBES = (-0.05, 0.0, 0.05, 0.10, 0.20)
RIGHT_PROBES = (-0.10, -0.05, 0.0, 0.05, 0.10)

# How wide a caption may run before it wraps.
CAPTION_WIDTH = 34

# One configuration per curve, in legend order. Each carries
#     "label":     str; the legend entry.
#     "prefix":    str; the training run its ensemble was saved by, before the
#                  dataset name.
#     "model_cls": the RandomEffectsModel subclass to rebuild.
#     "combined":  bool; whether the cloud draws random effects within each
#                  member as well as varying the member.
CONFIGURATIONS = [
    {"label": r"Mean-only, epistemic",
     "prefix": "meanonly",
     "model_cls": MeanOnlyModel,
     "combined": False},
    {"label": r"Random effects, combined",
     "prefix": "gaussian",
     "model_cls": GaussianModel,
     "combined": True},
]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Restoring and scoring what each configuration contributes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def load_test_split(dataset):
    """Read the held-out split both configurations are scored on.

    Args:
        dataset: str; the deployment dataset, "airfoil" or "flchain".

    Returns:
        RealDataset; the test split.
    """
    _, split = data_loader.load_real_split(dataset)

    return split.test


def score_configuration_cloud(configuration, dataset, test, rng, verbose=True):
    """Restore one configuration's ensemble and score its cloud.

    The mean-only model has no random effects to draw, so its cloud is the
    members' mean trajectories; the random-effects model draws NR_DRAWS from
    each member's covariance as well, which is the slow path.

    Args:
        configuration: dict; one entry of CONFIGURATIONS.
        dataset: str; the deployment dataset being scored.
        test: RealDataset; the split to score.
        rng: np.random.Generator; driving the aleatoric draws.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict; as the uncertainty engine returns, carrying "consensus", "U" and
            "profiles".
    """
    run_name = f"{configuration['prefix']}_{dataset}"
    ensemble = model_loader.load_bootstrap_ensemble(run_name,
                                                    configuration["model_cls"])
    if verbose:
        print(f"  scoring {run_name} over {len(ensemble.engines)} members",
              flush=True)

    engine = UncertaintyEngine(ensemble.engines)
    if configuration["combined"]:
        return engine.predict_with_combined_uncertainty(test.X,
                                                        n_samples=NR_DRAWS,
                                                        rng=rng)

    return engine.predict_with_epistemic_uncertainty(test.X)


def build_shared_grid(horizon):
    """Return the grid every profile is read on before it is aligned.

    Args:
        horizon: float; the right endpoint of the horizon.

    Returns:
        np.ndarray of shape (N_DENSE,).
    """
    return np.linspace(0.0, horizon, N_DENSE)


def read_profiles_on_grid(result, times):
    """Read every individual's profile on one shared grid.

    The profile is piecewise constant on that individual's own transition grid,
    and those grids differ, so they are put on a common one before pooling.

    Args:
        result: dict; as score_configuration_cloud returns.
        times: np.ndarray of shape (N_DENSE,); the shared grid.

    Returns:
        np.ndarray of shape (D, N_DENSE); one profile per individual.
    """
    return np.array([resample_profile_on_grid(profile, times)
                     for profile in result["profiles"]])


def average_aligned_profiles(aligned, offsets):
    """Reduce aligned profiles to their mean and the group behind it.

    Both panels pool this way, over whichever turning point they aligned on.

    Args:
        aligned: np.ndarray of shape (n, len(offsets)).
        offsets: np.ndarray; the aligned axis.

    Returns:
        tuple (mean_profile, counts); both aligned with offsets, the profile
            np.nan wherever no individual reaches that offset.
    """
    counts = np.sum(np.isfinite(aligned), axis=0)

    mean_profile = np.full(len(offsets), np.nan)
    covered = counts > 0
    mean_profile[covered] = np.nanmean(aligned[:, covered], axis=0)

    return mean_profile, counts


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Panel (a): aligning \textsc{airfoil} on the reported peak
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def locate_first_peak(summary):
    """Return when a summary first turns from increasing to decreasing.

    Args:
        summary: list of (state, start_time); one consensus shape summary.

    Returns:
        float; the start time of that decreasing state, np.nan where the
            summary never falls after rising.
    """
    seen_increasing = False
    for state, start_time in summary:
        if state.endswith("increasing"):
            seen_increasing = True
        elif state.endswith("decreasing") and seen_increasing:
            return float(start_time)

    return np.nan


def measure_observation_windows(test):
    """Return the stretch of the horizon each individual is observed over.

    Args:
        test: RealDataset; the split being read.

    Returns:
        tuple (first, last); two arrays of shape (D,), the earliest and latest
            observation time of each individual.
    """
    first = np.array([np.min(times) for times in test.times], dtype=float)
    last = np.array([np.max(times) for times in test.times], dtype=float)

    return first, last


def align_profiles_on_peaks(values, times, peaks, windows):
    """Re-express each profile as time since its peak.

    Where RESTRICT_TO_OBSERVED holds, an individual contributes only over the
    stretch it was actually measured on, so the profile is disagreement between
    observations. Otherwise it contributes over the whole horizon, and the
    profile carries the extrapolated forecasts as well.

    Args:
        values: np.ndarray of shape (D, N_DENSE); the profiles on the grid.
        times: np.ndarray of shape (N_DENSE,); that grid.
        peaks: np.ndarray of float; each individual's reported peak.
        windows: tuple (first, last); their observed stretches.

    Returns:
        np.ndarray of shape (D, len(LEFT_OFFSETS)); np.nan wherever an offset
            falls outside the stretch that individual contributes over.
    """
    aligned = np.full((len(values), len(LEFT_OFFSETS)), np.nan)

    individuals = zip(values, peaks, windows[0], windows[1])
    for position, (row, peak, first, last) in enumerate(individuals):
        queried = peak + LEFT_OFFSETS
        if RESTRICT_TO_OBSERVED:
            inside = (queried >= first) & (queried <= last)
        else:
            inside = (queried >= times[0]) & (queried <= times[-1])

        aligned[position, inside] = np.interp(queried[inside], times, row)

    return aligned


def measure_peak_alignment(configuration, test, windows, rng, verbose=True):
    """Measure panel (a) for one configuration.

    Args:
        configuration: dict; one entry of CONFIGURATIONS.
        test: RealDataset; the \\textsc{airfoil} split to score.
        windows: tuple (first, last); each individual's observed stretch.
        rng: np.random.Generator; driving the aleatoric draws.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict; one curve, carrying "label", the aligned "profile" with its
            "counts", and a "detail" line for the report.
    """
    result = score_configuration_cloud(configuration, LEFT_DATASET, test, rng,
                                       verbose=verbose)

    peaks = np.array([locate_first_peak(summary)
                      for summary in result["consensus"]])
    times = build_shared_grid(test.T)
    values = read_profiles_on_grid(result, times)

    aligned = align_profiles_on_peaks(values, times, peaks, windows)
    profile, counts = average_aligned_profiles(aligned, LEFT_OFFSETS)

    reported = int(np.sum(np.isfinite(peaks)))

    return {"label": configuration["label"],
            "profile": profile, "counts": counts,
            "detail": f"{reported} of {test.D} individuals report a peak"}


def find_observed_offsets(counts):
    """Return the offsets enough individuals are still observed at.

    Args:
        counts: np.ndarray of int; individuals contributing at each offset.

    Returns:
        tuple (lower, upper); the offsets to draw between.
    """
    covered = LEFT_OFFSETS[counts >= SUPPORT_FRACTION * counts.max()]

    return float(covered.min()), float(covered.max())


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Panel (b): aligning \textsc{flchain} on the impossible increase
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def locate_increase_extent(summary, horizon):
    """Return when a summary's first increasing state begins and ends.

    The state ends where the next one starts, or at the horizon if it is the
    last one the summary reports.

    Args:
        summary: list of (state, start_time); one consensus shape summary.
        horizon: float; the right endpoint of the horizon.

    Returns:
        tuple (onset, end); both np.nan where the summary reports no increase.
    """
    for position, (state, start_time) in enumerate(summary):
        if state.endswith("increasing"):
            following = summary[position + 1:]

            return float(start_time), (float(following[0][1]) if following
                                       else horizon)

    return np.nan, np.nan


def align_profiles_on_onsets(values, times, onsets, horizon):
    """Re-express each profile as time since its own reported increase.

    Args:
        values: np.ndarray of shape (n, N_DENSE); the profiles on the grid.
        times: np.ndarray of shape (N_DENSE,); that grid.
        onsets: np.ndarray of float; when each summary reports its increase.
        horizon: float; the right endpoint of the horizon.

    Returns:
        np.ndarray of shape (n, len(RIGHT_OFFSETS)); np.nan wherever an offset
            falls outside the horizon for that individual.
    """
    return np.array([align_profile_on_offsets(row, times, onset,
                                              RIGHT_OFFSETS, horizon)
                     for row, onset in zip(values, onsets)])


def measure_increase_alignment(configuration, test, rng, verbose=True):
    """Measure panel (b) for one configuration.

    Where a configuration reports no impossible state there is nothing to pool,
    and the curve is returned empty rather than faked.

    Args:
        configuration: dict; one entry of CONFIGURATIONS.
        test: RealDataset; the \\textsc{flchain} split to score.
        rng: np.random.Generator; driving the aleatoric draws.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict; one curve as measure_peak_alignment returns, whose "profile" and
            "counts" are None where no increase is reported anywhere.
    """
    result = score_configuration_cloud(configuration, RIGHT_DATASET, test, rng,
                                       verbose=verbose)

    extents = np.array([locate_increase_extent(summary, test.T)
                        for summary in result["consensus"]])
    reported = np.isfinite(extents[:, 0])

    curve = {"label": configuration["label"], "profile": None, "counts": None,
             "detail": f"reports no impossible state, over all {test.D}"}
    if not reported.any():
        return curve

    times = build_shared_grid(test.T)
    values = read_profiles_on_grid(result, times)[reported]
    onsets = extents[reported, 0]

    aligned = align_profiles_on_onsets(values, times, onsets, test.T)
    profile, counts = average_aligned_profiles(aligned, RIGHT_OFFSETS)

    nr_wrong = int(reported.sum())
    durations = extents[reported, 1] - onsets

    return {**curve, "profile": profile, "counts": counts,
            "detail": f"{nr_wrong} of {test.D} report an impossible increase, "
                      f"running {durations.mean():.3f} on average, "
                      f"{durations.max():.3f} at the longest"}


def find_reached_offsets(counts):
    """Return the offsets panel (b) is drawn over.

    The left edge is fixed and the right is where the curve runs out of
    individuals.

    Args:
        counts: np.ndarray of int; individuals contributing at each offset.

    Returns:
        tuple (lower, upper); the offsets to draw between.
    """
    return RIGHT_LEFT_LIMIT, float(RIGHT_OFFSETS[counts > 0].max())


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. The stretch each panel may be read over
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def find_full_support_offsets(counts, offsets):
    """Return the offsets every pooled individual contributes to.

    Args:
        counts: np.ndarray of int; individuals contributing at each offset.
        offsets: np.ndarray; the aligned axis.

    Returns:
        tuple (lower, upper); the offsets bounding full support.
    """
    covered = offsets[counts == counts.max()]

    return float(covered.min()), float(covered.max())


def intersect_domains(domains):
    """Narrow several domains to the stretch they all cover.

    Where both configurations are drawn they lose individuals towards the
    edges, so a panel is read over what they share rather than over either
    alone.

    Args:
        domains: list of tuple (lower, upper); at least one.

    Returns:
        tuple (lower, upper); the shared stretch.
    """
    return (max(lower for lower, _ in domains),
            min(upper for _, upper in domains))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5. Assembling each panel before drawing
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_panel(title, axis_name, caption, offsets, curves, probes,
                find_drawn, count_limit=None):
    """Gather one panel's curves and the stretch they are drawn over.

    A configuration that reports nothing to align on still belongs to the
    panel, so that it keeps its colour and is named in the report; it simply
    contributes no domain.

    Args:
        title: str; the dataset the panel is measured on.
        axis_name: str; what the horizontal axis measures.
        caption: str; the caption, already labelled (a) or (b).
        offsets: np.ndarray; the aligned axis.
        curves: list of dict; every measured curve, drawn or empty.
        probes: tuple of float; the offsets to report at.
        find_drawn: callable; maps counts to the domain to draw over.
        count_limit: float; the top of the group-size subplot, or None to take
            it from the largest group drawn.

    Returns:
        dict; the panel, ready to draw.
    """
    filled = [curve for curve in curves if curve["profile"] is not None]
    largest = max(curve["counts"].max() for curve in filled)

    return {"title": title, "axis_name": axis_name, "caption": caption,
            "offsets": offsets, "curves": curves, "probes": probes,
            "count_limit": count_limit or largest * 1.15,
            "drawn": intersect_domains([find_drawn(curve["counts"])
                                        for curve in filled]),
            "full": intersect_domains([find_full_support_offsets(
                curve["counts"], offsets) for curve in filled])}


def collect_figure_inputs(verbose=True):
    """Measure both panels for both configurations, before drawing.

    Returns:
        tuple (left, right); one panel dict each, as build_panel returns.
    """
    left_test = load_test_split(LEFT_DATASET)
    right_test = load_test_split(RIGHT_DATASET)
    windows = measure_observation_windows(left_test)
    rng = np.random.default_rng(SEED)

    if verbose:
        print(f"{LEFT_DATASET}: {left_test.D} individuals")
    left_curves = [measure_peak_alignment(configuration, left_test, windows,
                                          rng, verbose=verbose)
                   for configuration in CONFIGURATIONS]

    if verbose:
        print(f"{RIGHT_DATASET}: {right_test.D} individuals")
    right_curves = [measure_increase_alignment(configuration, right_test, rng,
                                               verbose=verbose)
                    for configuration in CONFIGURATIONS]

    left = build_panel(
        title=r"\textsc{airfoil}",
        axis_name="time since the reported peak",
        caption="(a) Aligned on the reported peak, where observed",
        offsets=LEFT_OFFSETS, curves=left_curves, probes=LEFT_PROBES,
        find_drawn=find_observed_offsets, count_limit=float(left_test.D))

    right = build_panel(
        title=r"\textsc{flchain}",
        axis_name="time since the reported increase",
        caption="(b) Aligned on the onset of an impossible increase",
        offsets=RIGHT_OFFSETS, curves=right_curves, probes=RIGHT_PROBES,
        find_drawn=find_reached_offsets)

    return left, right


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6. Drawing
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def shade_partial_support(axis, drawn, full):
    """Shade the stretches where the mean no longer covers everyone.

    Args:
        axis: plt.Axes; the target subplot axis to receive the plot layer.
        drawn: tuple (lower, upper); the domain being drawn.
        full: tuple (lower, upper); the domain of full support.

    Returns:
        None
    """
    for start, end in ((drawn[0], full[0]), (full[1], drawn[1])):
        if start < end:
            axis.axvspan(start, end, color=plot_style.GRAY_LIGHT, alpha=0.18,
                         lw=0.0, zorder=0)


def label_panel(ax_profile, ax_count, panel):
    """Title a panel above, and name and caption its horizontal axis below.

    The caption rides on the axis label so that the layout engine reserves room
    for it; a figure-level text would be placed before the panels are sized.

    Args:
        ax_profile: plt.Axes; the upper subplot of the panel.
        ax_count: plt.Axes; the lower subplot of the panel.
        panel: dict; as build_panel returns.

    Returns:
        None
    """
    ax_profile.set_title(panel["title"])
    ax_count.set_xlabel(f"{panel['axis_name']}\n\n"
                        f"{textwrap.fill(panel['caption'], CAPTION_WIDTH)}")


def draw_aligned_panel(ax_profile, ax_count, panel):
    """Draw one panel's aligned profiles over the stretch they are read on.

    Both panels align on a turning point and pool the same way, so one routine
    draws either; they differ only in the offsets and domains it is handed.

    Args:
        ax_profile: plt.Axes; the upper subplot.
        ax_count: plt.Axes; the lower subplot.
        panel: dict; as build_panel returns.

    Returns:
        None
    """
    offsets = panel["offsets"]

    # The cycle is walked over every measured curve, not only those with
    # something to draw, so a configuration keeps its colour across both panels
    # even where it reports nothing.
    for curve, colour in zip(panel["curves"], plot_style.CYCLE):
        if curve["profile"] is None:
            continue

        ax_profile.plot(offsets, curve["profile"], lw=1.6, color=colour,
                        zorder=3, label=curve["label"])
        ax_count.step(offsets, curve["counts"], where="mid", lw=1.2,
                      color=colour, zorder=2)

    for axis in (ax_profile, ax_count):
        plot_helpers.add_transition_rules(axis, [0.0])
        shade_partial_support(axis, panel["drawn"], panel["full"])
        axis.set_xlim(*panel["drawn"])

    ax_profile.set_ylim(0.0, config.Q_MAX)
    ax_count.set_ylim(0.0, panel["count_limit"])


def assemble_figure(panels, aspect=ASPECT):
    """Orchestrate the two panels, each over its profiles and its group sizes.

    Args:
        panels: tuple (left, right); as collect_figure_inputs returns.
        aspect: float; figure height divided by width.

    Returns:
        plt.Figure; the complete figure.
    """
    fig, axes = plot_style.subplots(
        1.0, aspect=aspect, nrows=2, ncols=2,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.get_layout_engine().set(**LAYOUT_PADS)

    for column, panel in enumerate(panels):
        ax_profile, ax_count = axes[0][column], axes[1][column]
        draw_aligned_panel(ax_profile, ax_count, panel)
        label_panel(ax_profile, ax_count, panel)
        ax_profile.legend(loc="upper left")

    axes[0][0].set_ylabel(r"mean $\widehat{Q}(t \mid \mathbf{x})$")
    axes[1][0].set_ylabel("individuals")

    return fig


def report_measurements(panels):
    """Print what each configuration reports, so the two can be compared.

    Args:
        panels: tuple (left, right); as collect_figure_inputs returns.

    Returns:
        None
    """
    for panel in panels:
        for curve in panel["curves"]:
            print(f"\n{panel['title']}, {curve['label']}: {curve['detail']}")

            if curve["profile"] is None:
                continue

            for probe in panel["probes"]:
                position = int(np.argmin(np.abs(panel["offsets"] - probe)))
                print(f"    offset {probe:+.2f}  "
                      f"Q {curve['profile'][position]:.3f}  "
                      f"on {curve['counts'][position]:>3}")


if __name__ == "__main__":
    plot_style.use()

    print("Figure 8: both deployment datasets, both configurations")
    panels = collect_figure_inputs()
    report_measurements(panels)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(assemble_figure(panels), str(OUTPUT_PATH))
    print(f"\nWrote {OUTPUT_PATH}")
