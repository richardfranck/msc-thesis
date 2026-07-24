import os, json
import numpy as np
import torch

from scripts.shape_uncertainty.spline_basis.bspline_basis import build_knot_dictionary
from scripts.shape_uncertainty.model.model import Encoder, GaussianModel
from scripts.shape_uncertainty.model.data_preparation import Normaliser
from scripts.shape_uncertainty.model.model_inference import InferenceEngine


def save_run(tuned, knot_config, generate_config, lambda_mean, lambda_re=0.0,  path="."):
    """Persist a trained run: weights plus everything needed to reload it.

    Writes the model state_dict to model.pt and a run.json bundling the best
    hyperparameters (to rebuild the architecture), the normaliser statistics (to
    standardise new inputs), the knot and generation configs, and the evaluation
    metrics. Together these fully reconstitute a working InferenceEngine.

    Args:
        tuned: the dict returned by Tuner.run (model, normaliser, best_params, ...).
        knot_config: dict {"nr_interior_knots", "T"} used to build knot objects.
        generate_config: the data-generation config (provenance).
        lambda_mean: float, the mean-trajectory roughness penalty this run was
            trained with (saved so the reloaded run's regularisation is known).
        lambda_re: float, the random-effects roughness penalty this run was
            trained with (fixed at 0.0 unless swept separately).
        path: directory to write into (default: current working directory).
    """
    os.makedirs(path, exist_ok=True)
    torch.save(tuned["model"].state_dict(), os.path.join(path, "model.pt"))
    norm = tuned["normaliser"]
    meta = {
        "best_params": tuned["best_params"],
        "best_value": tuned["best_value"],
        "knot_config": knot_config,
        "generate_config": generate_config,
        "lambda_mean": lambda_mean,
        "lambda_re": lambda_re,
        "normaliser": {
            "names": list(norm.names),
            "mean": norm.mean.tolist(),
            "std": norm.std.tolist(),
            "y_mean": norm.y_mean,
            "y_std": norm.y_std,
            "epsilon": norm.epsilon,
        },
    }
    with open(os.path.join(path, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_run(path=".", model_cls=GaussianModel):
    """Reload a saved run and rebuild a ready-to-use InferenceEngine.

    Rebuilds the Encoder + model_cls from the saved architecture, loads the
    trained weights, restores the normaliser object and the knot objects, and
    returns them.

    Args:
        path: directory a previous save_run wrote to (default: current working directory).
        model_cls: the RandomEffectsModel subclass to reconstruct (e.g.
            GaussianModel or MeanOnlyModel).

    Returns:
        dict with "engine", "model", "normaliser", "knot_objects", "metrics",
            "best_params", "generate_config", "lambda_mean", "lambda_re"
    """
    with open(os.path.join(path, "run.json")) as f:
        meta = json.load(f)

    best = meta["best_params"]
    knot_objects = build_knot_dictionary(**meta["knot_config"])

    # Rebuild the exact architecture, then load the trained weights into it.
    encoder = Encoder(
        nr_covariates=len(meta["normaliser"]["names"]),
        nr_basis=knot_objects["nr_basis"],
        hidden_sizes=tuple(best["hidden_sizes"]),
        activation=best["activation"],
        dropout=best["dropout"],
    )
    model = model_cls(encoder, knot_objects["nr_basis"])
    model.load_state_dict(torch.load(os.path.join(path, "model.pt"), weights_only=True))
    model.eval()

    # Restore the normaliser as a usable object.
    ns = meta["normaliser"]
    norm = Normaliser(epsilon=ns["epsilon"])
    norm.names = tuple(ns["names"])
    norm.mean = np.array(ns["mean"])
    norm.std = np.array(ns["std"])
    norm.y_mean = ns["y_mean"]
    norm.y_std = ns["y_std"]


    return {
            "model": model, 
            "normaliser": norm,
            "knot_objects": knot_objects, 
            "best_params": best, 
            "generate_config": meta["generate_config"],
            "lambda_mean": meta.get("lambda_mean", 0.0),
            "lambda_re": meta.get("lambda_re", 0.0),
    }
