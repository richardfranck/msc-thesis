"""Training run: Gaussian model, flchain deployment data.

Step 1: Search the encoder architecture and train a GaussianModel at the Gaussian
penalty pair, writing models/gaussian_flchain, then fit NR_MEMBERS bootstrap
members at that frozen architecture, writing models/bootstrap_gaussian_flchain.
Step 2: Draw NR_DRAWS random effects from every member and score the test split,
writing draws.npz and scores.npz into the ensemble's directory.

This is the deployment counterpart of train_gaussian_tumour_heterogeneous.py.
The architecture is searched with a plain Tuner rather than through the
covariance evaluator that run uses: the evaluator reconstructs the covariance
the DGP implies, and real data has no DGP to reconstruct it from.

    python thesis/training/train_gaussian_flchain.py
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

THESIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DIR))
sys.path.insert(0, str(THESIS_DIR.parent))

import config
import data_loader

from scripts.shape_uncertainty.model.model import GaussianModel
from scripts.shape_uncertainty.model.model_training import Tuner
from scripts.shape_uncertainty.model.model_persistence import save_run
from scripts.shape_uncertainty.model.bootstrap import BootstrapEnsemble
from scripts.shape_uncertainty.model.model_inference import UncertaintyEngine

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# The deployment dataset this run is fitted on.
DATASET = "flchain"

# The penalty pair this model is searched and trained under. The Gaussian model
# leaves the mean unpenalised and penalises the random effects.
PENALTY = config.PENALTIES["Gaussian"]

# Aleatoric draws per ensemble member, and how many individuals are scored per
# call, so a large test split need not hold every cloud in memory at once.
NR_DRAWS = 20
CHUNK = 20

# The grid the value-space profiles are evaluated on.
N_DENSE = config.N_DENSE

# Where the artefacts are written. The scores sit beside the members they were
# produced from, because they are only meaningful for that ensemble.
MODELS_DIR = THESIS_DIR / "models"
MODEL_PATH = MODELS_DIR / f"gaussian_{DATASET}"
ENSEMBLE_PATH = MODELS_DIR / f"bootstrap_gaussian_{DATASET}"
SCORES_PATH = ENSEMBLE_PATH / "scores.npz"
DRAWS_PATH = ENSEMBLE_PATH / "draws.npz"


def train_single_model(split, knot_objects, verbose=True):
    """Search the encoder architecture and train the reported model.

    Args:
        split: Split namedtuple; supplies the train and val datasets.
        knot_objects: dict from build_knot_objects, supplying the basis
            functions, the penalty matrix Omega and the coefficient dimension.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict; the Tuner.run result, with keys "best_params", "best_value",
            "model", "normaliser" and "study".
    """
    # Step 1: Fix the seeds the search and the fit are run under
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Step 2: Build the tuner at this run's penalty pair
    tuner = Tuner(
        train_dataset=split.train,
        val_dataset=split.val,
        basis_functions=knot_objects["basis_functions"],
        Omega=knot_objects["Omega"],
        nr_basis=knot_objects["nr_basis"],
        lambda_mean=PENALTY["lambda_mean"],
        lambda_re=PENALTY["lambda_re"],
        nr_epochs=config.NR_EPOCHS,
        model_cls=GaussianModel,
        model_kwargs={},
        patience=config.PATIENCE,
    )

    # Step 3: Search the architecture and retrain the best configuration
    started = time.time()
    if verbose:
        print(f"{DATASET}: searching {config.NR_TRIALS} architectures",
              flush=True)
    tuned = tuner.run(nr_trials=config.NR_TRIALS, seed=config.SEED)

    # Step 4: Report the architecture the search settled on
    if verbose:
        print(f"{DATASET}: fitted in {time.time() - started:.0f}s", flush=True)
        print("  Selected architecture:", flush=True)
        for name, value in tuned["best_params"].items():
            print(f"    {name:<13} {value}", flush=True)
        print(f"  Validation objective: {tuned['best_value']:.4f}", flush=True)

    return tuned


def save_single_model(tuned, dataset, verbose=True):
    """Persist the trained model, its normaliser and its provenance.

    Args:
        tuned: dict; the train_single_model result.
        dataset: RealDataset; the whole dataset, whose gen_config is recorded
            as the fit's data provenance.
        verbose: bool; whether to report the directory written to.

    Returns:
        pathlib.Path; the directory written to.
    """
    save_run(
        tuned,
        knot_config=data_loader.KNOT_CONFIG,
        generate_config=dataset.gen_config,
        lambda_mean=PENALTY["lambda_mean"],
        lambda_re=PENALTY["lambda_re"],
        path=str(MODEL_PATH),
    )

    if verbose:
        print(f"  Saved the fit to {MODEL_PATH}", flush=True)

    return MODEL_PATH


def load_frozen_architecture():
    """Read the architecture the single model's search selected.

    Every ensemble member shares this configuration, so members differ only in
    the data each of them saw, not in their capacity.

    Returns:
        dict; the hyperparameters, in the form set_architecture consumes.
    """
    return json.loads((MODEL_PATH / "run.json").read_text())["best_params"]


def fit_bootstrap_ensemble(split, knot_objects, architecture, verbose=True):
    """Fit the bootstrap members at the frozen architecture.

    Args:
        split: Split namedtuple; supplies the pool that is resampled and the
            validation set held fixed across members.
        knot_objects: dict from build_knot_objects.
        architecture: dict; the architecture every member is trained at.
        verbose: bool; whether to report progress on stdout.

    Returns:
        BootstrapEnsemble; fitted, carrying its members.
    """
    # Step 1: Fix the seeds. The members seed themselves from the ensemble's
    #         own seed, so this only fixes what happens outside their fits.
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Step 2: Build the ensemble at the same penalty pair as the single fit
    ensemble = BootstrapEnsemble(
        train_dataset=split.train,
        val_dataset=split.val,
        knot_objects=knot_objects,
        lambda_mean=PENALTY["lambda_mean"],
        lambda_re=PENALTY["lambda_re"],
        shape_config=config.SHAPE_CONFIG,
        model_cls=GaussianModel,
        model_kwargs={},
        nr_epochs=config.NR_EPOCHS,
        patience=config.PATIENCE,
        seed=config.SEED,
    )

    # Step 3: Freeze the architecture the search already paid for
    ensemble.set_architecture(architecture)

    # Step 4: Fit one member per bootstrap resample
    started = time.time()
    if verbose:
        print(f"{DATASET}: fitting {config.NR_MEMBERS} members", flush=True)
    ensemble.fit(nr_members=config.NR_MEMBERS)
    if verbose:
        print(f"{DATASET}: fitted in {time.time() - started:.0f}s", flush=True)

    return ensemble


def load_bootstrap_ensemble(verbose=True):
    """Restore the fitted members from disk.

    Args:
        verbose: bool; whether to report how many members were restored.

    Returns:
        BootstrapEnsemble; with its members restored.
    """
    ensemble = BootstrapEnsemble.load(str(ENSEMBLE_PATH), model_cls=GaussianModel)

    if verbose:
        print(f"  Restored {len(ensemble.engines)} members from "
              f"{ENSEMBLE_PATH.name}", flush=True)

    return ensemble


def save_bootstrap_ensemble(ensemble, verbose=True):
    """Persist every member's weights and the settings they were fitted under.

    Args:
        ensemble: BootstrapEnsemble; fitted.
        verbose: bool; whether to report the directory written to.

    Returns:
        pathlib.Path; the directory written to.
    """
    ensemble.save(str(ENSEMBLE_PATH))

    if verbose:
        print(f"  Saved {len(ensemble.members)} members to {ENSEMBLE_PATH}",
              flush=True)

    return ENSEMBLE_PATH


def score_shape_uncertainty(ensemble, test, verbose=True):
    """Compute every test individual's shape uncertainty, keeping the draws.

    Args:
        ensemble: BootstrapEnsemble; with its members restored or fitted.
        test: RealDataset; the test split.
        verbose: bool; whether to report progress on stdout.

    Returns:
        tuple (U, references, coefficients):
            U: np.ndarray of shape (D,), the pooled shape uncertainty.
            references: np.ndarray of shape (D,) of int, which draw of each
                pooled cloud supplied the consensus.
            coefficients: list of E arrays of shape (D, n, B), the draws.
    """
    uncertainty_engine = UncertaintyEngine(ensemble.engines)

    if verbose:
        print(f"Scoring shape uncertainty: {test.D} individuals, "
              f"{len(ensemble.engines)} members x {NR_DRAWS} draws", flush=True)

    result = uncertainty_engine.predict_with_combined_uncertainty(
        test.X, n_samples=NR_DRAWS, rng=config.SEED_EVAL,
        keep_coefficients=True)

    return result["U"], result["selected_indices"], result["coefficients"]


def score_value_uncertainty(ensemble, test, coefficients, references,
                            verbose=True):
    """Compute every test individual's value-space uncertainty.

    Args:
        ensemble: BootstrapEnsemble; with its members restored or fitted.
        test: RealDataset; the test split.
        coefficients: list of E arrays of shape (D, n, B), as score_shape_
            uncertainty returns, so that the trajectories scored here are the
            very draws whose summaries were scored there.
        references: np.ndarray of shape (D,) of int; which draw of each pooled
            cloud the uncertainty is reported for. V is a fraction of that
            draw's own amplitude.
        verbose: bool; whether to report progress on stdout.

    Returns:
        tuple (V, amplitudes), each an np.ndarray of shape (D,).
    """
    uncertainty_engine = UncertaintyEngine(ensemble.engines)
    times = np.linspace(0.0, test.T, N_DENSE)
    V = np.empty(test.D, dtype=float)
    amplitudes = np.empty(test.D, dtype=float)

    stored = {"coefficients": coefficients,
              "group_ids": np.repeat(np.arange(len(ensemble.engines)), NR_DRAWS)}

    for start in range(0, test.D, CHUNK):
        stop = min(start + CHUNK, test.D)
        result = uncertainty_engine.predict_with_combined_value_uncertainty(
            stored, times, references[start:stop],
            indices=list(range(start, stop)))
        V[start:stop] = result["V"]
        amplitudes[start:stop] = result["amplitudes"]
        if verbose:
            print(f"  scored {stop} / {test.D}", flush=True)

    return V, amplitudes


def score_test_uncertainty(ensemble, test, verbose=True):
    """Score the whole test split in shape and in value space.

    Both scores read the same cloud: the draws kept by the shape pass are the
    trajectories the value pass measures, so U and V describe one sample.

    Args:
        ensemble: BootstrapEnsemble; with its members restored or fitted.
        test: RealDataset; the test split.
        verbose: bool; whether to report progress on stdout.

    Returns:
        dict with keys "U", "V", "references" and "amplitudes", each an
            np.ndarray of shape (D,), and "coefficients", the draws behind both.
    """
    # Step 1: Score the test set in shape space, keeping the draws
    U, references, coefficients = score_shape_uncertainty(
        ensemble, test, verbose=verbose)

    # Step 2: Score those very draws in value space
    if verbose:
        print("Scoring value-space uncertainty ...", flush=True)
    V, amplitudes = score_value_uncertainty(
        ensemble, test, coefficients, references, verbose=verbose)

    return {"U": U, "V": V, "references": references,
            "amplitudes": amplitudes, "coefficients": coefficients}


def save_test_scores(scores, verbose=True):
    """Persist the scores and the coefficient draws behind them.

    The draws are kept rather than the trajectories: they reconstruct the
    trajectories exactly at a fraction of the space, and every file reading this
    run has to see the same cloud rather than draw one of its own. The settings
    are recorded alongside, so a later run can check what it is about to reuse
    against what it is asking for. The test split is sized by fraction rather
    than by a constant, so its size is read off the scores themselves.

    Args:
        scores: dict; the score_test_uncertainty result.
        verbose: bool; whether to report the files written.

    Returns:
        pathlib.Path; the directory written into.
    """
    ENSEMBLE_PATH.mkdir(parents=True, exist_ok=True)

    # Step 1: The scores, and the constants the readers validate them against
    np.savez(SCORES_PATH,
             U=scores["U"], V=scores["V"],
             references=scores["references"],
             amplitudes=scores["amplitudes"],
             nr_draws=NR_DRAWS,
             nr_members=config.NR_MEMBERS,
             seed_eval=config.SEED_EVAL,
             n_dense=N_DENSE,
             nr_test=len(scores["U"]))

    # Step 2: The coefficient draws every cloud is evaluated from
    np.savez(DRAWS_PATH, coefficients=np.stack(scores["coefficients"]))

    if verbose:
        print(f"  Saved scores for {len(scores['U'])} individuals to "
              f"{ENSEMBLE_PATH}", flush=True)

    return ENSEMBLE_PATH


def run_training(verbose=True):
    """Run all three stages, skipping each where its artefact is already on disk.

    Args:
        verbose: bool; whether to report progress on stdout.
    """
    # Step 1: The data and the basis every stage works on
    dataset, split = data_loader.load_real_split(DATASET)
    knot_objects = data_loader.build_knot_objects()

    # Step 2: Search the architecture and train the reported model
    if MODEL_PATH.exists():
        if verbose:
            print(f"{MODEL_PATH.name} is already on disk; skipping the search")
    else:
        tuned = train_single_model(split, knot_objects, verbose=verbose)
        save_single_model(tuned, dataset, verbose=verbose)

    # Step 3: Fit the ensemble at the architecture that search selected
    if ENSEMBLE_PATH.exists():
        if verbose:
            print(f"{ENSEMBLE_PATH.name} is already on disk; skipping the ensemble")
        ensemble = load_bootstrap_ensemble(verbose=verbose)
    else:
        ensemble = fit_bootstrap_ensemble(
            split, knot_objects, load_frozen_architecture(), verbose=verbose)
        save_bootstrap_ensemble(ensemble, verbose=verbose)

    # Step 4: Draw the clouds and score the test split
    if SCORES_PATH.exists() and DRAWS_PATH.exists():
        if verbose:
            print(f"{SCORES_PATH.name} is already on disk; skipping the scoring")
    else:
        scores = score_test_uncertainty(ensemble, split.test, verbose=verbose)
        save_test_scores(scores, verbose=verbose)


if __name__ == "__main__":
    print(f"Training the Gaussian model on the {DATASET} deployment data")
    run_training()
