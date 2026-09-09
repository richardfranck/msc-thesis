"""Training run: MeanOnly model, airfoil deployment data.

Step 1: Search the encoder architecture and train a MeanOnlyModel at the MeanOnly
roughness penalty, writing models/meanonly_airfoil.
Step 2: Fit NR_MEMBERS bootstrap members at that frozen architecture, and write
to models/bootstrap_meanonly_airfoil.

    python thesis/training/train_meanonly_airfoil.py
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

from scripts.shape_uncertainty.model.model import MeanOnlyModel
from scripts.shape_uncertainty.model.model_training import Tuner
from scripts.shape_uncertainty.model.model_persistence import save_run
from scripts.shape_uncertainty.model.bootstrap import BootstrapEnsemble

# The models are trained in double precision.
torch.set_default_dtype(torch.float64)

# The deployment dataset this run is fitted on.
DATASET = "airfoil"

# The penalty pair this model is searched and trained under. MeanOnlyModel has
# no random effects, so lambda_re is zero structurally. The model is
# misspecified against data with unexplained variation between individuals,
# which is deliberate: it is what a practitioner deploys.
PENALTY = config.PENALTIES["MeanOnly"]

# Where the two artefacts are written.
MODELS_DIR = THESIS_DIR / "models"
MODEL_PATH = MODELS_DIR / f"meanonly_{DATASET}"
ENSEMBLE_PATH = MODELS_DIR / f"bootstrap_meanonly_{DATASET}"


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
        model_cls=MeanOnlyModel,
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
        print(f"{DATASET}: fitted in {time.time() - started:.0f}s, "
              f"validation objective {tuned['best_value']:.4f}", flush=True)

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
        model_cls=MeanOnlyModel,
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


def run_training(verbose=True):
    """Run both stages, skipping either where its artefact is already on disk.

    Args:
        verbose: bool; whether to report progress on stdout.
    """
    # Step 1: The data and the basis both stages are fitted on
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
    else:
        ensemble = fit_bootstrap_ensemble(
            split, knot_objects, load_frozen_architecture(), verbose=verbose)
        save_bootstrap_ensemble(ensemble, verbose=verbose)


if __name__ == "__main__":
    print(f"Training the MeanOnly model on the {DATASET} deployment data")
    run_training()
