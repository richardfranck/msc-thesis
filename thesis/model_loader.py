import json
from pathlib import Path
import sys

import numpy as np

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))

import config

from scripts.shape_uncertainty.model.model_persistence import load_run
from scripts.shape_uncertainty.model.bootstrap import BootstrapEnsemble
from scripts.shape_uncertainty.model.model_inference import build_inference_engine


MODELS_DIR = THESIS_DIR / "models"


def model_path(name):
    """Return where the single model of one run is saved.

    Args:
        name: str; the run, e.g. "meanonly_tumour_homogeneous".

    Returns:
        pathlib.Path; the directory save_run wrote to.
    """
    return MODELS_DIR / name


def ensemble_path(name):
    """Return where the bootstrap ensemble of one run is saved.

    Args:
        name: str; the run, e.g. "meanonly_tumour_homogeneous".

    Returns:
        pathlib.Path; the directory BootstrapEnsemble.save wrote to.
    """
    return MODELS_DIR / f"bootstrap_{name}"


def require_artefact(path, name):
    """Fail with the script to run where an artefact has not been trained.

    Args:
        path: pathlib.Path; the artefact the caller wants.
        name: str; the run it belongs to, naming the training script.

    Returns:
        pathlib.Path; the path, where it exists.

    Raises:
        FileNotFoundError: where it does not, naming the script that writes it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No artefact at {path}. Train it first:\n"
            f"    python thesis/training/train_{name}.py")

    return path


def load_single_model(name, model_cls):
    """Reload the trained weights, the normaliser and the basis of one run.

    Args:
        name: str; the run whose single model to read.
        model_cls: the RandomEffectsModel subclass to rebuild.

    Returns:
        dict; as load_run returns, carrying "model", "normaliser",
            "knot_objects", "best_params", "generate_config", "lambda_mean"
            and "lambda_re".
    """
    path = require_artefact(model_path(name), name)

    return load_run(str(path), model_cls=model_cls)


def load_inference_engine(name, model_cls, shape_config=config.SHAPE_CONFIG):
    """Reload one run's single model, ready to predict.

    Args:
        name: str; the run whose single model to read.
        model_cls: the RandomEffectsModel subclass to rebuild.
        shape_config: dict; the extraction thresholds the engine reads shape
            summaries at. Defaults to the setting the thesis reports.

    Returns:
        InferenceEngine; populated and ready to predict.
    """
    saved = load_single_model(name, model_cls)

    return build_inference_engine(saved["model"], saved["normaliser"],
                                  saved["knot_objects"], shape_config)


def load_frozen_architecture(name):
    """Read the architecture one run's search selected.

    Args:
        name: str; the run whose search to read.

    Returns:
        dict; the hyperparameters, in the form set_architecture consumes.
    """
    path = require_artefact(model_path(name), name)

    with open(path / "run.json") as f:
        return json.load(f)["best_params"]


def load_bootstrap_ensemble(name, model_cls):
    """Restore the fitted members of one run's bootstrap ensemble.

    Args:
        name: str; the run whose ensemble to read.
        model_cls: the RandomEffectsModel subclass every member is built from.

    Returns:
        BootstrapEnsemble; with its members restored.
    """
    path = require_artefact(ensemble_path(name), name)

    return BootstrapEnsemble.load(str(path), model_cls=model_cls)


def load_test_scores(name, expected=None):
    """Read the uncertainty scores and coefficient draws of one run's ensemble.

    Scoring the test set is slow, so the training script does it once and saves
    the result. The draws are read back rather than redrawn, so that every
    figure reading this run sees the same cloud.

    Args:
        name: str; the run whose scores to read.
        expected: dict or None; settings the saved scores must have been
            produced under, checked against the constants recorded beside them.

    Returns:
        dict with keys "U", "V", "references" and "amplitudes", each an
            np.ndarray of shape (D,), and "coefficients", the E blocks of shape
            (D, n, B) that every cloud is evaluated from.

    Raises:
        ValueError: where the saved settings disagree with expected.
    """
    path = require_artefact(ensemble_path(name), name)
    saved = np.load(require_artefact(path / "scores.npz", name))

    # Step 1: Refuse a cache that was produced under different settings
    if expected is not None:
        mismatched = {key: (int(saved[key]), wanted)
                      for key, wanted in expected.items()
                      if int(saved[key]) != wanted}
        if mismatched:
            raise ValueError(
                f"The scores at {path} were produced under different settings "
                f"{mismatched} (saved, wanted). Retrain them:\n"
                f"    python thesis/training/train_{name}.py")

    # Step 2: The draws both scores were measured on
    coefficients = list(np.load(require_artefact(path / "draws.npz", name))
                        ["coefficients"])

    return {"U": saved["U"], "V": saved["V"],
            "references": saved["references"],
            "amplitudes": saved["amplitudes"],
            "coefficients": coefficients}
