"""Training run: MeanOnly model, tumour data with unobserved heterogeneity.

Search the encoder architecture and trainsa MeanOnlyModel at the MeanOnly
roughness penalty, writing models/meanonly_tumour_heterogeneous.

    python thesis/training/train_meanonly_tumour_heterogeneous.py
"""

from pathlib import Path
import sys

import numpy as np
import torch

THESIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DIR))
sys.path.insert(0, str(THESIS_DIR.parent))

import config
import data_loader

from scripts.shape_uncertainty.model.model import MeanOnlyModel
from scripts.shape_uncertainty.model.model_training import Tuner
from scripts.shape_uncertainty.model.model_persistence import save_run

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# The regime: simulated data with unobserved heterogeneity.
REGIME = config.REGIME_HETEROGENEOUS

# The penalty pair this model is searched and trained under. MeanOnlyModel has
# no random effects, so lambda_re is zero structurally.
PENALTY = config.PENALTIES["MeanOnly"]

# Where the artefact is written.
MODELS_DIR = THESIS_DIR / "models"
MODEL_PATH = MODELS_DIR / "meanonly_tumour_heterogeneous"


def train_single_model(split, knot_objects, verbose=True):
    """Search the encoder architecture and train the reported model.

    Args:
        split: Split namedtuple; supplies the train and val datasets.
        knot_objects: dict from build_knot_objects, supplying the basis
            functions, the penalty matrix Omega and the coefficient dimension.
        verbose: bool; whether to report the selected architecture on stdout.

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
        model_cls=MeanOnlyModel,
        model_kwargs={},
        patience=config.PATIENCE,
    )

    # Step 3: Search the architecture and retrain the best configuration
    if verbose:
        print(f"Searching the architecture over {config.NR_TRIALS} trials",
              flush=True)
    tuned = tuner.run(nr_trials=config.NR_TRIALS, seed=config.SEED)

    # Step 4: Report the architecture the search settled on
    if verbose:
        print("  Selected architecture:", flush=True)
        for name, value in tuned["best_params"].items():
            print(f"    {name:<13} {value}", flush=True)
        print(f"  Validation objective: {tuned['best_value']:.4f}", flush=True)

    return tuned


def save_single_model(tuned, split, verbose=True):
    """Persist the trained model, its normaliser and its provenance.

    Args:
        tuned: dict; the train_single_model result.
        split: Split namedtuple; supplies the training dataset, whose
            gen_config is recorded as the fit's data provenance.
        verbose: bool; whether to report the directory written to.

    Returns:
        pathlib.Path; the directory written to.
    """
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

    # Step 2: The data and the basis the model is fitted on
    split = data_loader.draw_simulated_split(REGIME)
    knot_objects = data_loader.build_knot_objects()

    # Step 3: Search the architecture, train, and persist the fit
    tuned = train_single_model(split, knot_objects, verbose=verbose)
    save_single_model(tuned, split, verbose=verbose)


if __name__ == "__main__":
    print(f"Training the MeanOnly model on the {REGIME.name} tumour regime")
    run_training()
