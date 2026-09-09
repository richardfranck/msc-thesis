"""Training run: Gaussian model, tumour data without unobserved heterogeneity.

Search the encoder architecture and train a GaussianModel at the Gaussian
penalty pair, writing models/gaussian_tumour_homogeneous.

    python thesis/training/train_gaussian_tumour_homogeneous.py
"""

from pathlib import Path
import sys

import torch

THESIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DIR))
sys.path.insert(0, str(THESIS_DIR.parent))

import config
import data_loader

from scripts.shape_uncertainty.model.model import GaussianModel
from scripts.shape_uncertainty.model.model_persistence import save_run
from scripts.shape_uncertainty.evaluation.covariance_recovery import (
    GaussianModelEvaluator)

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# The regime: simulated data without unobserved heterogeneity. The evaluator's
# methods take the heterogeneity as a pair of scalars rather than as a regime.
REGIME = config.REGIME_HOMOGENEOUS
ALPHA_G, ALPHA_D = float(REGIME.alpha_g), float(REGIME.alpha_d)

# The penalty pair this model is searched and trained under. The Gaussian model
# leaves the mean unpenalised and penalises the random effects.
PENALTY = config.PENALTIES["Gaussian"]

# Where the artefact is written.
MODELS_DIR = THESIS_DIR / "models"
MODEL_PATH = MODELS_DIR / "gaussian_tumour_homogeneous"


def build_covariance_evaluator():
    """Build the evaluator that owns this run's data, fitting and scoring.

    The Gaussian model is searched and trained through the same evaluator table
    6 scores it with, so the fit reported there and the fit saved here are one
    and the same. The evaluator draws and caches its own split.

    Returns:
        GaussianModelEvaluator; configured in accordance with config.py.
    """
    # Step 1: The observation design and split sizes, shared across regimes
    data_config = {
        "D_train": config.D_TRAIN,
        "D_val": config.D_VAL,
        "D_test": config.D_TEST,
        "nr_obs": config.NR_OBS,
        "sigma": config.SIGMA,
        "regular": config.REGULAR,
        "include_endpoints": config.INCLUDE_ENDPOINTS,
    }

    # Step 2: The optimisation schedule, shared across regimes
    train_config = {
        "nr_epochs": config.NR_EPOCHS,
        "patience": config.PATIENCE,
        "model_cls": GaussianModel,
        "model_kwargs": {},
    }

    # Step 3: Build the evaluator on the shared spline basis
    return GaussianModelEvaluator(
        base_hyperparams=config.BASE_HYPERPARAMS,
        knot_objects=data_loader.build_knot_objects(),
        shape_config=config.SHAPE_CONFIG,
        data_config=data_config,
        train_config=train_config,
        T=config.T,
        n_ref=config.N_REF,
        n_mc=config.N_MC,
        seed=config.SEED,
    )


def search_encoder_architecture(evaluator, verbose=True):
    """Search the encoder architecture at this run's penalty pair.

    Args:
        evaluator: GaussianModelEvaluator; supplies the data and the tuner.
        verbose: bool; whether to report the selected architecture on stdout.

    Returns:
        dict; the architecture, in the form fit_model consumes it.
    """
    # Step 1: Search at the penalty pair the reported fit is trained with
    if verbose:
        print(f"Searching the architecture over {config.NR_TRIALS} trials",
              flush=True)
    architecture = evaluator.search_fixed_architecture(
        ALPHA_G, ALPHA_D, config.NR_TRIALS,
        lambda_mean=PENALTY["lambda_mean"],
        lambda_re=PENALTY["lambda_re"],
    )

    # Step 2: Report the architecture the search settled on
    if verbose:
        print("  Selected architecture:", flush=True)
        for name, value in architecture.items():
            print(f"    {name:<13} {value}", flush=True)

    return architecture


def train_single_model(evaluator, architecture, verbose=True):
    """Train the reported model at the selected architecture.

    Args:
        evaluator: GaussianModelEvaluator; supplies the data and the tuner.
        architecture: dict; the architecture the search selected.
        verbose: bool; whether to report the validation objective on stdout.

    Returns:
        dict with keys "engine", the InferenceEngine wrapping the trained
            model, and "tuned", the fitted dict save_run consumes.
    """
    if verbose:
        print("  Training at the selected architecture", flush=True)
    fitted = evaluator.fit_model(
        ALPHA_G, ALPHA_D, architecture,
        lambda_mean=PENALTY["lambda_mean"],
        lambda_re=PENALTY["lambda_re"],
        seed=config.SEED,
    )

    if verbose:
        print(f"  Validation objective: {fitted['tuned']['best_value']:.4f}",
              flush=True)

    return fitted


def save_single_model(evaluator, tuned, verbose=True):
    """Persist the trained model, its normaliser and its provenance.

    Args:
        evaluator: GaussianModelEvaluator; supplies the split this fit was
            trained on, whose gen_config is recorded as its data provenance.
        tuned: dict; the "tuned" entry of the train_single_model result.
        verbose: bool; whether to report the directory written to.

    Returns:
        pathlib.Path; the directory written to.
    """
    split = evaluator.get_dataset_split(ALPHA_G, ALPHA_D)

    save_run(
        tuned,
        knot_config=data_loader.KNOT_CONFIG,
        generate_config=split.train.gen_config,
        lambda_mean=PENALTY["lambda_mean"],
        lambda_re=PENALTY["lambda_re"],
        path=MODEL_PATH,
    )

    if verbose:
        print(f"  Saved the fit to {MODEL_PATH}", flush=True)

    return MODEL_PATH


def run_training(verbose=True):
    """Search, train and save, skipping where the artefact is already on disk.

    Args:
        verbose: bool; whether to report progress on stdout.
    """
    # Step 1: Stop where a previous run has already written the artefact
    if MODEL_PATH.exists():
        if verbose:
            print(f"{MODEL_PATH.name} is already on disk; skipping the search")
        return

    # Step 2: The evaluator that owns this run's data
    evaluator = build_covariance_evaluator()

    # Step 3: Search the architecture, train, and persist the fit
    architecture = search_encoder_architecture(evaluator, verbose=verbose)
    fitted = train_single_model(evaluator, architecture, verbose=verbose)
    save_single_model(evaluator, fitted["tuned"], verbose=verbose)


if __name__ == "__main__":
    print(f"Training the Gaussian model on the {REGIME.name} tumour regime")
    run_training()
