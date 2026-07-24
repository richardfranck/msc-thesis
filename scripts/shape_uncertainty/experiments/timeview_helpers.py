import json
import os

import numpy as np
import torch
import pytorch_lightning as pl
import shutil
import tempfile

from vendor.timeview.timeview.config import Config
from vendor.timeview.timeview.data import TTSDataset, create_dataloader
from vendor.timeview.timeview.lit_module import LitTTS

from scripts.shape_uncertainty.shape_extraction.shape_summary import extract_shape_summary

from scripts.shape_uncertainty.model.data_preparation import Normaliser


def fit_timeview_normaliser(train_data):
    """Fit covariate + target normalisers on the training SimulatedDataset.

    TimeView  standardises both the static covariates and the trajectory 
    targets before training (see TTSBenchmark.prepare_data in
    vendor/timeview/experiments/baselines.py).

    The normaliser is fit on the training data only. The same statistics
    transform the validation/test data and standardise covariates at inference
    time (i.e. no information leakage).

    Args:
        train_data: SimulatedDataset used for training

    Returns:
        Normaliser: a normaliser fitted for both covariates (X) and targets (y).
    """
    normaliser = Normaliser()
    normaliser.fit(train_data) # X per covariate
    normaliser.fit_y(train_data) # y global
    return normaliser


def convert_sim_dataset_to_timeview_tuple(sim_dataset, normaliser):
    """Convert a SimulatedDataset into the (X, ts, ys) tuple TimeView's TTSDataset expects.

    Our SimulatedDataset class stores model-visible data as a covariate dictionary
    (X), a per-individual list of observation times (times), and a
    per-individual list of noisy observations (Y_noisy). 

    Timeview's TTSDataset expects these ingredients as a tuple (X, ts, ys), where
        - X is a numpy matrix of static covariates of shape (D, M)
        - ts is a length-D list of one-dimensional arrays of shape (N_i,), and
        - ys is a length-D list of one-dimensional arrays of shape (N_i,)
    (see the TTSDataset constructor in vendor/timeview/timeview/data.py.

    Args:
        sim_dataset: SimulatedDataset containing the model-visible data:
        - X: dict mapping covariate names to arrays of shape (D,).
        - times: length-D list where times[i] has shape (N_i,).
        - Y_noisy: length-D list where Y_noisy[i] has shape (N_i,).
        normaliser: fitted Normaliser where X is standardised via
            transform and ys via transform_y.

    Returns:
        X: float32 array (D, M) of static covariates.
        ts: length-D list of float32 arrays (N_i,) of observation times.
        ys: length-D list of float32 arrays (N_i,) of noisy observations.
    """
    # Step 1: Convert a sim_dataset dict of arrays into a 2D numpy array of shape (D, M). 
    X = normaliser.transform(sim_dataset.X).numpy().astype(np.float32)   # (D, M) standardised

    # Step 2: Convert each individual's noisy trajectory observations to float32 arrays.
    ys = [normaliser.transform_y(y).astype(np.float32) for y in sim_dataset.Y_noisy]

    # Step 3: Convert each individual's observation times to float32 arrays.
    ts = [np.asarray(t, dtype=np.float32) for t in sim_dataset.times]

    return (X, ts, ys)


def sample_timeview_hyperparams(trial, search_space):
    """Sample one TimeView hyperparameter set for one Optuna trial.

        Args:
            trial: an optuna.Trial; the object Optuna passes in to record and
                propose hyperparameter values for this trial.
            search_space: A dictionary of the search_space.
                
        Returns:
            dict of the sampled hyperparameters.
    """
    hs = search_space["hidden_sizes"]
    av = search_space["activation"]
    dr = search_space["dropout"]
    lr = search_space["lr"]
    bz = search_space["batch_size"]
    wd = search_space["weight_decay"]

    hyperparameter_selection =  {
        "hidden_sizes": [trial.suggest_int(f"hidden_{i}", hs["low"], hs["high"])
                          for i in range(hs["nr_layers"])],
        "activation": trial.suggest_categorical("activation", av),
        "dropout": trial.suggest_float("dropout", dr["low"], dr["high"]),
        "lr": trial.suggest_float("lr", lr["low"], lr["high"], log=lr["log"]),
        "batch_size": trial.suggest_categorical("batch_size", bz),
        "weight_decay": trial.suggest_float("weight_decay", wd["low"], wd["high"], log=wd["log"]),
    }

    return hyperparameter_selection


def make_timeview_config(hyperparas, features, T, n_basis, seed=42, dataloader_type="iterative"):
    """Build a TimeView Config from a structured hyperparameter dict.

    Timeview stores model, data, and optimiser settings in a Config object.
    (see vendor/timeview/timeview/config.py).

    Args:
        hyperparas: Dict of sampled hyperparameters produced by sample_timeview_hyperparams.
        features: Ordered names of features used as Timeview inpus.
        T: Right endpoint T of the observation window.
        n_basis: Number of B-spline basis functions used by Timeview.
        seed: Random seed passed into the Timeview config.
        dataloader_type: Timeview dataloader mode. 
            - "Iterative" means each parentints trajectory is kepy at length N_i.

    Returns:
        Config: Timeview configuration object for one training run.
    """
    timeview_config = Config(
                    n_features=len(features),
                    n_basis=n_basis,
                    T=float(T),
                    seed=seed,
                    dataloader_type=dataloader_type,
                    encoder={
                        "hidden_sizes": hyperparas["hidden_sizes"],
                        "activation": hyperparas["activation"],
                        "dropout_p": hyperparas["dropout"],
                    },
                    training={
                        "optimizer": "adam",
                        "lr": hyperparas["lr"],
                        "batch_size": hyperparas["batch_size"],
                        "weight_decay": hyperparas["weight_decay"],
                    },
                    dataset_split={"train": 0.8, "val": 0.1, "test": 0.1},  # required by Config but unused as we pass our own train/val sets
                )

    return timeview_config


def fit_timeview_model(config, train_data, val_data, epochs, patience):
    """Train one LitTTS model on prepared train/val data under early stopping.

        1. Build TTSDataset objects and dataloaders for the given train/val
            SimulatedDatasets, 
        2. Train a fresh LitTTS(config) with a pl.Trainer configured 
            for checkpointing (best val_loss) and early stopping (with patience).
        3. Return the checkpoint callback (for Optuna trial scoring).

    Args:
        config: the TimeView configuration for this run
        train_data: SimulatedDataset used for training.
        val_data: SimulatedDataset used for validation and early stopping.
        epochs: maximum int number of training epochs.
        patience: int number of epochs with no val_loss improvement before
            early stopping triggers.

    Returns:
        turple of pl.callbacks.ModelCheckpoint and a fitted normaliser
           - pl.callbacks.ModelCheckpoint: the checkpoint callback used during
            training, exposing best_model_score (float validation loss) and
            best_model_path (path to the best-epoch weights on disk).
          - normaliser fitted on the training dataset
        
    """ 
    # Step 0: Fit normalisers on train_data
    normaliser = fit_timeview_normaliser(train_data)

    # Step 1: Create normalised timeview TTSDatasets for training and validation
    train_tts = TTSDataset(config, convert_sim_dataset_to_timeview_tuple(train_data, normaliser))
    val_tts = TTSDataset(config, convert_sim_dataset_to_timeview_tuple(val_data, normaliser))

    # Step 2: Build a timveview dataloader for training and validation batches
    train_loader = create_dataloader(config, train_tts, shuffle=True) # randomise order for better training
    val_loader = create_dataloader(config, val_tts, shuffle=False) # do not randomise order for comparable validation

    # Step 3: Create the timeview model using pytorch_lightning.
    model = LitTTS(config)

    # Step 4: Configure pytorch_lightning.Trainer class with checkpointing and early stopping
    checkpoint_dir = tempfile.mkdtemp(prefix="timeview_ckpt_")
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min", save_top_k=1)
    early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        deterministic=True,
        max_epochs=epochs,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[checkpoint, early_stop],
        log_every_n_steps=1,
    )

    # Step 5: Train the the timeview model on the training data and score on the validation data
    trainer.fit(model, train_loader, val_loader)

    # Step 6: Return the pytorch_lightning checkpoint and the fitted normaliser
    return checkpoint, normaliser


def run_timeview_optuna_trial(trial, train_data, val_data, nr_basis, search_space, epochs=200, patience=10):
    """Run one Optuna trial for TimeView using our provided training and validation data.

    Because TimeView's training() creates its own train/validation/test
    split, this function implements training on provded train/val SimulatedDatasets. 
    
    Args:
        trial: optuna.Trial for sampling and recording one hyperpara config.
        train_data: SimulatedDataset used for training.
        val_data: SimulatedDataset used for validation and early stopping.
        nr_basis (int): number of B-spline basis functions for the TimeView model.
        epochs (int): maximum number (int) of training epochs for this trial.
        patience (int): early-stopping patience (in epochs) for this trial.

    Returns:
        float: the best validation loss observed during this trial's training run.
    """
    # Step 1: Sample one Optuna hyperparameter configuration
    hyperparams = sample_timeview_hyperparams(trial, search_space)

    # Step 2: Build the TimeView Config for this trial
    features = list(train_data.X.keys())
    config = make_timeview_config(hyperparams, features, train_data.T, nr_basis, seed=42)

    # Step 3: Train a LitTTS model on train_data with early stopping on val_data.
    checkpoint, _ = fit_timeview_model(config, train_data, val_data, epochs, patience)
 
    # Step 4: Save best validation loss value for ranking the trial against others
    best_value = float(checkpoint.best_model_score)
    
    # Step 5: Delete this trial's throwaway  directory
    shutil.rmtree(os.path.dirname(checkpoint.best_model_path), ignore_errors=True) 

    # Step 6: Return the best val loss 
    return best_value


def best_timeview_hyperparams(study, search_space):
    """Return a dict of the best hyperparameters found by a completed Optuna study.

    Args:
        study (optuna.Study): a completed Optuna study produced by optimising
            run_timeview_optuna_trial.
        search_space: the dictionary of the hyperparameter search space

    Returns:
        dict: the best configuration, shaped like sample_timeview_hyperparams's
        return value, e.g.:
            {
                "hidden_sizes": [64, 97, 31],
                "activation": "elu",
                "dropout": 0.18,
                "lr": 0.0023,
                "batch_size": 64,
                "weight_decay": 0.0001,
            }
    """
    b = study.best_params
    hs = search_space["hidden_sizes"]
    best_hyperparas = {
        "hidden_sizes": [b[f"hidden_{i}"] for i in range(hs["nr_layers"])],
        "activation": b["activation"],
        "dropout": b["dropout"],
        "lr": b["lr"],
        "batch_size": b["batch_size"],
        "weight_decay": b["weight_decay"],
    }
    return best_hyperparas


def train_final_timeview_model(best_params, train_data, val_data, nr_basis, epochs=200, patience=10, seed=42):
    """Retrain the best Optuna configuration.

    Args:
        best_params: dict shaped like sample_timeview_hyperparams return dict
        train_data: SimulatedDataset used for training.
        val_data: SimulatedDataset used for validation/ early stopping.
        nr_basis (int): number of B-spline basis functions for the TimeView model.
        epochs (int): maximum number of training epochs.
        patience (int): early-stopping patience (in epochs).
        seed (int): random seed for the final Config (for encoder init and dataloader shuffling).

    Returns:
        dict with keys:
            "model": the retrained LitTTS in eval mode.
            "config": the Config used for this final run.
            "best_params": the input best_params 
            "best_value": the best validation loss achieved during final training.
    """
    # Step 1: Build a TimeView Config from the best hyperparameter configuration
    features = list(train_data.X.keys())
    config = make_timeview_config(best_params, features, train_data.T, nr_basis, seed=seed)

    # Step 2: Train a LitTTS model on train_data with early stopping on val_data
    checkpoint, normaliser = fit_timeview_model(config, train_data, val_data, epochs, patience)

    # Step 3: Load the best-checkpoint weights back into a LitTTS model.
    model = LitTTS.load_from_checkpoint(checkpoint.best_model_path, config=config)
    model.eval()
    
    # Step 4: Delete the checkpoint directory 
    shutil.rmtree(os.path.dirname(checkpoint.best_model_path), ignore_errors=True)

    return {
        "model": model,
        "config": config,
        "best_params": best_params,
        "best_value": float(checkpoint.best_model_score),
        "normaliser": normaliser,
    }


def save_timeview_run(tuned, generate_config, path="."):
    """Save a trained TimeView run.

    Write two files into path:
        - model.pt: the model
        - run.json: hyperparameters, validation loss, and other configs

    Args:
        tuned: dict returned by train_final_timeview_model.
        generate_config: the data-generation config.
        path: directory to write into with the cwd as the default.
    """
    # Step 1: Save the TTS module's weights to path/model.pt.
    os.makedirs(path, exist_ok=True)
    torch.save(tuned["model"].model.state_dict(), os.path.join(path, "model.pt"))

    # Step 2: Creare a metadata dictionary
    config = tuned["config"]
    meta = {
        "best_params": tuned["best_params"],
        "best_value": tuned["best_value"],
        "generate_config": generate_config,
        "config": {
            "n_features": config.n_features,
            "n_basis": config.n_basis,
            "T": config.T,
            "seed": config.seed,
            "encoder": vars(config.encoder),
            "training": vars(config.training), # Required by Config's constructor. Not used otherwise.
            "dataset_split": vars(config.dataset_split), # Required by Config's constructor. Not used otherwise.
            "dataloader_type": config.dataloader_type,
            "internal_knots": config.internal_knots, # Required by Config's constructor. Not used otherwise.
        },
        "normaliser": {
            "names": list(tuned["normaliser"].names),
            "mean": tuned["normaliser"].mean.tolist(),
            "std": tuned["normaliser"].std.tolist(),
            "y_mean": tuned["normaliser"].y_mean,
            "y_std": tuned["normaliser"].y_std,
            "epsilon": tuned["normaliser"].epsilon,
        },
    }

    # Step 3: Write the metadata dict to path/run.json.
    with open(os.path.join(path, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_timeview_run(path="."):
    """Reload a saved TimeView run.

    Read the two files written by save_timeview_run from path.

    Args:
        path: directory a previous save_timeview_run wrote to.

    Returns:
        dict with "model" (a LitTTS in eval mode), "config", "best_params",
        "best_value", and "generate_config".
    """
    # Step 1: Read path/run.json.
    with open(os.path.join(path, "run.json")) as f:
        meta = json.load(f)

    # Step 2: Rebuild TimeView Config from saved metadata
    config = Config(**meta["config"])

    # Step 3: Load the saved model weights into a model and set it into eval mode
    model = LitTTS(config)
    model.model.load_state_dict(torch.load(os.path.join(path, "model.pt"), weights_only=True))
    model.eval()

    # Step 4: Rebuild the normaliser
    ns = meta["normaliser"]
    normaliser = Normaliser(epsilon=ns["epsilon"])
    normaliser.names = tuple(ns["names"])
    normaliser.mean = np.array(ns["mean"])
    normaliser.std = np.array(ns["std"])
    normaliser.y_mean = ns["y_mean"]
    normaliser.y_std = ns["y_std"]

    # Step 5: Return a dict with the loaded model, config, best parameter values, 
    # best validation loss and the data generation configuration
    return {
        "model": model,
        "config": config,
        "best_params": meta["best_params"],
        "best_value": meta["best_value"],
        "generate_config": meta["generate_config"],
        "normaliser": normaliser
    }



class TimeViewInferenceEngine:
    """This brings the InferenceEngine structure to the TimeView model, 
        which is needed for e.g. plotting of results. 
    """

    def __init__(self, model, feature_names, C=None, breakpoints=None, 
        zeta_rel=0.0, upsilon_rel_1=None, upsilon_rel_2=None, 
        upsilon_rel_prune=0.0, do_prune=False, normaliser=None,):
        """Wrap a trained LitTTS model for use with the shared plotting interface.

        Args:
            model (LitTTS) a traind TimeView lightning module.
            feature_names: Ordered names of the static covariates.

            C: np.ndarray of shape (K + 1, 4, B), where C[k] is the
                conversion matrix for the k-th knot interval, ordered left
                to right over the K + 1 intervals of [0, T].
            breakpoints: breakpoints: np.ndarray of shape (K + 2,) of interior knots
                plus boundaries.
            upsilon_rel_1 (float) relative slope significance threshold.
            upsilon_rel_2 (float) relative curvature significance threshold.
        """
        self.model = model.model  # the wrapped TTS module
        self.feature_names = feature_names
        self.C = C
        self.breakpoints = breakpoints
        self.zeta_rel = zeta_rel
        self.upsilon_rel_1 = upsilon_rel_1
        self.upsilon_rel_2 = upsilon_rel_2
        self.upsilon_rel_prune = upsilon_rel_prune
        self.do_prune = do_prune
        self.normaliser = normaliser

    def _get_mean_coefficients(self, x):
        """Return the encoder's predicted B-spline coefficients h_theta(x) for one profile.

        Analogue of InferenceEngine._get_mean_coefficients for the custom
        model, useful for inspecting what the encoder produces before it's
        multiplied through the B-spline basis.

        Args:
            x: dict mapping each covariate name to a scalar value for one
                individual.

        Returns:
            coeff_vector: A torch.Tensor of shape (B,) of the mean coefficients.
        """
        # Step 1: Normalise feature values
        feature_values = self.normaliser.transform_one(x)
        feature_values = np.array(feature_values, dtype=np.float32) # shape (M,)
        feature_values = feature_values.reshape(1, -1) # shape (1, M)

        # Step 2: Get the B-spline coefficient predictions
        coeff_vector = self.model.predict_latent_variables(feature_values)
        coeff_vector = torch.from_numpy(coeff_vector).view(-1)
        return coeff_vector

    def predict_mean_shape_summary(self, x):
        """Predict the shape summary for the mean trajectory for one covariate dict x.

        Args:
            x: dict mapping each covariate name to a scalar value for one
                individual.

        Returns:
            a shape summary [(state, start_time), ...].
        """
        w = self._get_mean_coefficients(x).reshape(-1, 1)  # (B, 1)
        w = w.detach().cpu().numpy() #  extract_shape_summary needs numpy not torch.Tensor
        shape_summary = extract_shape_summary(
            w,
            self.C,
            self.breakpoints,
            zeta_rel=self.zeta_rel,
            upsilon_rel_1=self.upsilon_rel_1,
            upsilon_rel_2=self.upsilon_rel_2,
            upsilon_rel_prune=self.upsilon_rel_prune,
            do_prune=self.do_prune,
        )
        return shape_summary


    def predict_trajectory_values(self, x, times):
        """Predict the mean trajectory at times for one covariate covariate dict x.

        Args:
            Args:
            x: dict mapping each covariate name to a scalar value for one
                individual.
            times: 1-D sequence of time points (floats) in [0, T] at which to 
                evaluate the mean trajectory. This is used for plotting.

        Returns:
            torch.Tensor of shape (len(times),);.
        """
        # Step 1: Normalise the covariates (as at training time)
        feature_values = self.normaliser.transform_one(x)
        feature_values = np.array(feature_values, dtype=np.float32) # shape (M,)

        # Step 2: Get the observation times as an np.array 
        times = np.asarray(times, dtype=np.float32)

        # Step 3: Get the forecast values using the timeview forecast_trajectory() function
        y_pred = self.model.forecast_trajectory(feature_values, times)

        # Step 4: Inverse-transform predictions back to real units
        y_pred = self.normaliser.inverse_transform_y(y_pred)  # (N,) np
        y_pred = torch.from_numpy(y_pred)

        return y_pred