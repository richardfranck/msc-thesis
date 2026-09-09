"""Figure 5: Combined shape uncertainty, under unobserved heterogeneity.

Requires: gaussian_tumour_heterogeneous

    python thesis/thesis_figure_5.py
"""

from pathlib import Path
import sys

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

from scripts.shape_uncertainty.simulated_data.shape_ground_truth import get_analytic_shape_summary_single
from scripts.shape_uncertainty.model.model import GaussianModel
from scripts.shape_uncertainty.shape_extraction.shape_uncertainty import (
    compute_shape_uncertainty, medoid_summary)
from scripts.shape_uncertainty.shape_extraction.value_uncertainty import (
    compute_value_uncertainty)
from scripts.shape_uncertainty.shape_extraction.shape_distance import (
    exact_shape_sequence_match, shape_summary_distance)

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# Plot layout and resolution configurations
N_DENSE = config.N_DENSE
ASPECT = 1.05
HEIGHT_RATIOS = [3.0, 1.0, 1.4, 1.4]

# The pooled cloud holds NR_MEMBERS * NR_DRAWS trajectories.
# Panel (a) draws about as many cloud members as Figure 7 alternative.
# The pooled cloud has NR_MEMBERS * NR_DRAWS = 2,000 trajectories.
CLOUD_STRIDE = 40
# Match the faint cloud and foreground observations of figures 4 and 7.
CLOUD_ALPHA = 0.1

# The observations are drawn over a dense cloud, so they are marked larger and
# heavier than the shared default, and in black rather than the observation role
# colour, which the cloud would otherwise swallow.
OBSERVATION_MARKER_SIZE = 4.0
OBSERVATION_MARKER_EDGE_WIDTH = 1.0
OBSERVATION_COLOR = "black"

# The cloud rises towards the right of panel (a), leaving the upper left free.
TRAJECTORY_LEGEND_LOCATION = "upper left"

# Output path
OUTPUT_PATH = THESIS_DIR / "figures" / "thesis_figure_5.pdf"

# Where on the U / V ranking the reference individual is taken from. 
# 0.00 is the individual with the most shape uncertainty per unit of value uncertainty, 1.00 the least.
PERCENTILE = 0.00 # 0.00

# Aleatoric draws per ensemble member.
NR_DRAWS = 20

# The data regime
REGIME = config.REGIME_HETEROGENEOUS

# The penalty pair the Gaussian model is trained under.
PENALTY = config.PENALTIES["Gaussian"]

# The training run this file reads its ensemble and its scores from.
MODEL_NAME = f"gaussian_tumour_{REGIME.name}"

# The settings the saved scores must have been produced under.
SCORE_SETTINGS = {"nr_draws": NR_DRAWS, "nr_members": config.NR_MEMBERS,
                  "seed_eval": config.SEED_EVAL, "n_dense": N_DENSE,
                  "nr_test": config.D_TEST}


def load_gaussian_ensemble(verbose=True):
    """Redraw the data and restore the Gaussian ensemble fitted on it.

    Args:
        verbose: bool; whether to report how many members were restored.

    Returns:
        tuple (ensemble, test); the BootstrapEnsemble of GaussianModels with its
            members restored, and the test SimulatedDataset.
    """
    # Step 1: Redraw the split the members were fitted on
    test, _, _ = data_loader.draw_simulated_split(REGIME)

    # Step 2: Restore the members its training run saved
    ensemble = model_loader.load_bootstrap_ensemble(MODEL_NAME, GaussianModel)
    if verbose:
        print(f"Restored {len(ensemble.engines)} members of {MODEL_NAME}")

    return ensemble, test


def load_uncertainty_scores(ensemble, test, verbose=True):
    """Read the test set's shape and value uncertainty, and the draws behind it.

    Scoring the test set takes about an hour, so the training run does it once
    and saves the result. Redrawing the sample here would give a different
    cloud, so the saved draws are reused and this figure and figure 4 report the
    same object. The ensemble and the split are taken so that the saved scores
    can be checked against the objects they are about to be read alongside.

    Args:
        ensemble: BootstrapEnsemble; the members the scores were produced from.
        test: SimulatedDataset; the test split they were produced on.
        verbose: bool; whether to report how many individuals were read.

    Returns:
        dict carrying "U" and "V", the two scores per individual, "references"
            and "amplitudes", the draw each V was scaled by and its own
            amplitude, and "coefficients", the E blocks of draws behind them.

    Raises:
        ValueError: where the saved scores do not describe this ensemble and
            this split.
    """
    scores = model_loader.load_test_scores(MODEL_NAME, expected=SCORE_SETTINGS)

    # The scores only mean anything for the members and the individuals they
    # were measured on, so refuse a pairing that does not match.
    if (len(scores["coefficients"]) != len(ensemble.engines)
            or len(scores["U"]) != test.D):
        raise ValueError(
            f"The saved scores describe {len(scores['coefficients'])} members "
            f"and {len(scores['U'])} individuals, but this ensemble has "
            f"{len(ensemble.engines)} members and this split {test.D}. "
            f"Retrain them:\n"
            f"    python thesis/training/train_{MODEL_NAME}.py")

    if verbose:
        print(f"Read scores for {len(scores['U'])} individuals", flush=True)

    return scores


def select_reference_individual(scores, percentile=PERCENTILE):
    """Return the index of the individual at a given point of the U / V ranking.

    Individuals are ordered by shape uncertainty per unit of value uncertainty,
    highest first, so percentile 0.00 selects the individual whose shape the
    cloud is least certain about relative to how far apart it lies in value, and
    1.00 the one where the two spaces agree most.

    Args:
        scores: dict; as load_uncertainty_scores returns.
        percentile: float in [0, 1]; the position on the ranking.

    Returns:
        int; the index of the selected individual within the test split.
    """
    # Step 1: Order by shape uncertainty per unit of value uncertainty
    ratio = scores["U"] / scores["V"]
    order = np.argsort(-ratio, kind="stable")

    # Step 2: Take the individual sitting at the requested percentile
    return int(order[int(percentile * (len(order) - 1))])


def collect_figure_inputs(ensemble, test, scores, i):
    """Rebuild one individual's cloud and score it in both spaces.

    The saved scores cover the whole test split but hold no profile, so the
    E * NR_DRAWS summaries behind the drawn individual are re-extracted from its
    own block of the saved draws and rescored here.

    Args:
        ensemble: BootstrapEnsemble; with its members restored.
        test: SimulatedDataset; the test split.
        scores: dict; as load_uncertainty_scores returns.
        i: int; the individual to draw, within the test split.

    Returns:
        dict carrying "times", "curves", "member", "y_true",
            "observation_times", "observations", "true_summary", "consensus",
            "shape_profile", "value_profile", "U", "V", "amplitude" and "T".
    """
    # Step 1: The grid every curve and profile is evaluated on
    T = test.T
    times = np.linspace(0.0, T, N_DENSE)

    # Step 2: Rebuild this individual's pooled cloud from the saved draws
    summaries, curves = [], []
    for engine, block in zip(ensemble.engines, scores["coefficients"]):
        individual = block[[i]]                                  # (1, n, B)
        summaries.extend(engine.predict_aleatoric_summaries(individual)[0])
        curves.append(engine.predict_aleatoric_trajectories(individual,
                                                            times)[0])
    curves = np.concatenate(curves, axis=0)                      # (E * n, N)

    # Step 3: Score the cloud in shape space and take the draw that represents it
    U, shape_profile = compute_shape_uncertainty(summaries, T)
    consensus, member = medoid_summary(summaries, T, shape_profile)

    # Step 4: Score the same cloud in value space, against that draw's amplitude
    V, value_profile = compute_value_uncertainty(curves, times, member)

    # Step 5: The generating curve and its exact shape summary
    y_true = test.true_curves_at(times, indices=[i])[0]
    true_summary = get_analytic_shape_summary_single(
        test.X["size"][i], test.params["g"][i], test.params["d"][i],
        test.params["rho"][i], T,
        upsilon_rel_1=config.SHAPE_CONFIG["upsilon_rel_1"],
        upsilon_rel_2=config.SHAPE_CONFIG["upsilon_rel_2"],
        upsilon_rel_prune=config.SHAPE_CONFIG["upsilon_rel_prune"],
        do_prune=config.SHAPE_CONFIG["do_prune"])

    return {
        "times": times,
        "curves": curves,
        "member": int(member),
        "y_true": y_true,
        "observation_times": test.times[i],
        "observations": test.Y_noisy[i],
        "true_summary": true_summary,
        "consensus": consensus,
        "shape_profile": shape_profile,
        "value_profile": value_profile,
        "U": float(U),
        "V": float(V),
        "amplitude": value_profile["amplitude"],
        "T": T,
    }

def report_reference_individual(reference, test, scores, i):
    """Print the figure's individual, its scores and how well it recovers.

    Args:
        reference: dict; as collect_figure_inputs returns.
        test: SimulatedDataset; the test split.
        scores: dict; as load_uncertainty_scores returns.
        i: int; the individual's position in the test split.

    Returns:
        None
    """
    # Step 1: Set this individual against the ranking it was drawn from
    ratio = scores["U"] / scores["V"]
    order = np.argsort(-ratio, kind="stable")
    rank = int(np.flatnonzero(order == i)[0])

    # Step 2: Score the consensus against the truth the bands show
    match = exact_shape_sequence_match(reference["consensus"],
                                       reference["true_summary"])
    distance = shape_summary_distance(reference["consensus"],
                                      reference["true_summary"], reference["T"])

    print(f"Individual i={i}, rank {rank + 1} of {test.D} by U / V")
    print(f"  shape uncertainty U:     {reference['U']:.4f}  "
          f"(test-set median {np.median(scores['U']):.4f})")
    print(f"  value uncertainty V:     {reference['V']:.4f}  "
          f"of estimated amplitude {reference['amplitude']:.4f}")
    print(f"  ratio U / V:             {ratio[i]:.4f}  "
          f"(test-set median {np.median(ratio):.4f})")
    print(f"  consensus draw:          {reference['member']}")
    print(f"  consensus summary:       "
          f"{[state for state, _ in reference['consensus']]}")
    print(f"  true summary:            "
          f"{[state for state, _ in reference['true_summary']]}")
    print(f"  exact match:             {match}")
    print(f"  shape summary distance:  {distance:.4f}")


def add_forecast_cloud(axis, times, y_true, observation_times, observations,
                       curves, member, stride=CLOUD_STRIDE):
    """Draw panel (a): the cloud, the observations, the truth and the consensus.

    Args:
        axis: plt.Axes; the target subplot.
        times: np.ndarray (N,); the evaluation grid on [0, T].
        y_true: np.ndarray (N,); the generating curve on that grid.
        observation_times: np.ndarray (K,); this individual's own observation
            times, which are not the evaluation grid.
        observations: np.ndarray (K,); the noisy outcomes.
        curves: np.ndarray (M, N); the pooled cloud.
        member: int; which draw supplied the consensus.
        stride: int; draw every stride-th member of the cloud.

    Returns:
        None
    """
    # Step 1: The consensus is drawn separately, so it is skipped in the cloud
    skip = member // stride if member % stride == 0 else None

    plot_helpers.add_trajectory_cloud(
        axis, times, curves[::stride], skip=skip, label="cloud",
        alpha=CLOUD_ALPHA)

    # Step 2: Add the true curve and the observations
    plot_helpers.add_true_curve(axis, times, y_true)
    plot_helpers.add_noisy_observations(
        axis, observation_times, observations,
        marker_size=OBSERVATION_MARKER_SIZE,
        marker_edge_width=OBSERVATION_MARKER_EDGE_WIDTH, zorder=6,
        color=OBSERVATION_COLOR)

    # Step 3: Add the draw the cloud agreed on
    plot_helpers.add_estimated_curve(axis, times, curves[member],
                                     label="consensus")

    axis.set_ylabel(r"$y(t)$")
    axis.legend(loc=TRAJECTORY_LEGEND_LOCATION, ncol=1)


def add_summary_strip(axis, consensus, true_summary, T):
    """Draw panel (b): the consensus summary, over the generating curve's own.

    The two summaries share one axis, so that their row names can be y-tick
    labels and the y-label is left free to name the quantity both rows carry.

    Args:
        axis: plt.Axes; the target subplot.
        consensus: list of (state, start_time); the summary the cloud agreed on.
        true_summary: list of (state, start_time); the generating curve's exact
            summary.
        T: float; the right endpoint of the horizon.

    Returns:
        list of Patch; the legend handles, one per distinct state across rows.
    """
    handles = plot_helpers.merge_handles(
        plot_helpers.add_shape_bands(axis, true_summary, T, row=1),
        plot_helpers.add_shape_bands(axis, consensus, T, row=0))

    plot_helpers.style_band_strip(axis, ["CONSENSUS", "TRUTH"])
    axis.set_ylabel(r"$\widehat{\bar{\Gamma}}_{\mathcal{V}}$")

    return handles


def add_shape_uncertainty(axis, regions, values, score):
    """Draw panel (c): the regional shape uncertainty, and report its score.

    Args:
        axis: plt.Axes; the target subplot.
        regions: np.ndarray (n, 2); the region boundaries R_k.
        values: np.ndarray (n,); the shape uncertainty on each region, all in
            [0, Q_MAX].
        score: float; the shape uncertainty the profile integrates to.

    Returns:
        None
    """
    plot_helpers.add_uncertainty_stairs(axis, regions, values)
    plot_helpers.add_panel_note(
        axis, rf"$\widehat{{U}}(\mathbf{{x}}) = {score:.3f}$")

    axis.set_ylabel(r"$\widehat{Q}(t \mid \mathbf{x})$")
    axis.set_ylim(0.0, config.Q_MAX)


def add_value_uncertainty(axis, times, values, score):
    """Draw panel (d): the value-space uncertainty, and report its score.

    Args:
        axis: plt.Axes; the target subplot.
        times: np.ndarray (N,); the evaluation grid on [0, T].
        values: np.ndarray (N,); the value-space uncertainty at those times, as
            a fraction of the trajectory amplitude.
        score: float; the value uncertainty the profile integrates to.

    Returns:
        None
    """
    plot_helpers.add_uncertainty_profile(axis, times, values)
    plot_helpers.add_panel_note(
        axis, rf"$\widehat{{V}}(\mathbf{{x}}) = {score:.3f}$")

    axis.set_ylabel(r"$\widehat{V}(t \mid \mathbf{x})$")
    axis.set_ylim(0.0, config.V_MAX)


def plot_uncertainty(reference, aspect=ASPECT):
    """Orchestrate the four panels onto one shared time axis.

    Args:
        reference: dict; as collect_figure_inputs returns.
        aspect: float; figure height divided by width.

    Returns:
        plt.Figure; the complete four-panel figure.
    """
    T = reference["T"]
    times = reference["times"]

    # Step 1: Build the canvas, four rows sharing one time axis
    fig, axes = plot_style.subplots(
        1.0, aspect=aspect, nrows=4, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.12})
    ax_traj, ax_band, ax_shape, ax_value = axes

    # Step 2: Layer each panel
    add_forecast_cloud(ax_traj, times, reference["y_true"],
                       reference["observation_times"],
                       reference["observations"], reference["curves"],
                       reference["member"])
    handles = add_summary_strip(ax_band, reference["consensus"],
                                reference["true_summary"], T)
    add_shape_uncertainty(ax_shape, reference["shape_profile"]["regions"],
                          reference["shape_profile"]["region_uncertainty"],
                          reference["U"])
    add_value_uncertainty(ax_value, times,
                          reference["value_profile"]["value_uncertainty"],
                          reference["V"])

    # Step 3: Name the shape states once, beneath the figure
    ax_value.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.35), ncol=min(4, len(handles)))
    ax_value.set_xlabel("time $t$")
    ax_value.set_xlim(0.0, T)

    return fig


if __name__ == "__main__":
    plot_style.use()
    ensemble, test = load_gaussian_ensemble()
    scores = load_uncertainty_scores(ensemble, test)
    i = select_reference_individual(scores)
    reference = collect_figure_inputs(ensemble, test, scores, i)
    report_reference_individual(reference, test, scores, i)
    fig = plot_uncertainty(reference)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(fig, str(OUTPUT_PATH))
    print(f"Wrote {OUTPUT_PATH}")
