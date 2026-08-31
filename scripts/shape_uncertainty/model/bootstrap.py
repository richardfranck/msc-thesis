import os
import json

import numpy as np
import torch

from scripts.shape_uncertainty.model.model import MeanOnlyModel
from scripts.shape_uncertainty.model.model_training import Tuner

from scripts.shape_uncertainty.spline_basis.bspline_basis import build_knot_dictionary
from scripts.shape_uncertainty.model.model_persistence import save_run, load_run

from scripts.shape_uncertainty.model.model_inference import (
    InferenceEngine, build_inference_engine)


class BootstrapEnsemble:
    """Train an ensemble of models on bootstrap resamples at one frozen architecture.

    This class produces the ensemble from which we measure epistemic shape
    uncertainty. That is, variation in the predicted shape summary that is due to 
    which data a model happened to be trained on. 

    The class operates in two stages:
        Stage A: Select an architecture once, via an Optuna search on the entire
                 training data, and freeze it. Every ensemble member shares this
                 configuration, so members differ only in the data they saw (and
                 possibly in their random initialisation), not in their capacity.
        Stage B: Draw nr_members bootstrap resamples of the training set (size
                 D_train, sampled WITH replacement) and train one model per
                 resample at the frozen configuration. Each member fits its own
                 covariate normaliser on its own resample and is wrapped in its
                 own InferenceEngine.
    The validation set is held fixed across members and is not resampled. 
    The class returns a list of InferenceEngines. These are consumed by the
    UncertaintyEngine (in model_inference.py) to build a per-individual cloud 
    of shape summaries.

    Args:
        train_dataset: SimulatedDataset; the data pool that is bootstrap-resampled 
            to produce each bootstrap member's training data.
        val_dataset: SimulatedDataset; held fixed across members and used for
            early stopping and for scoring the Optuna search.
        knot_objects: dict from build_knot_dictionary, supplying
            "basis_functions", "C", "breakpoints", "Omega" and "nr_basis".
        lambda_mean: float; the mean-trajectory roughness weight, fixed across
            the whole ensemble 
        lambda_re: float; the random-effects roughness weight, fixed across the
            whole ensemble. 
        shape_config: dict; the shape-extraction thresholds
            ("zeta_rel", "upsilon_rel_1", "upsilon_rel_2", "upsilon_rel_prune",
            "do_prune") every member engine is populated with.
        model_cls: the RandomEffectsModel subclass every member is built from.
        model_kwargs: dict or None; extra keyword arguments forwarded to
            model_cls beyond (encoder, nr_basis).
        nr_epochs: int; maximum training epochs per member and per search trial.
        patience: int; early-stopping patience in epochs.
        seed: int; the master seed for the generator that bootstrap index draws 
            come from. 

    Attributes:
        hyperparas: dict or None; the frozen architecture. None until
            select_architecture or set_architecture has been called.
        study: the completed optuna.Study from the architecture search, or None
            if the architecture was handed rather than searched.
        members: list of the per-member fit records produced in Stage B.
    """

    def __init__(self, train_dataset, val_dataset, knot_objects,    
                 lambda_mean, lambda_re, 
                 shape_config, # zeta_rel, upsilon_rel_1/2, prune -> passed to each engine
                 model_cls, model_kwargs=None,
                 nr_epochs=200, patience=10,    
                 seed=0):
        # Data
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Spline constants, fixed
        self.knot_objects = knot_objects

        # Roughness penalties, fixed
        self.lambda_mean = lambda_mean
        self.lambda_re = lambda_re

        # Model family, fixed
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}

        # Optimisation constants, fixed
        self.nr_epochs = nr_epochs
        self.patience = patience

        # Shape-extraction thresholds, fixed
        self.shape_config = dict(shape_config)

        # Master seed
        self.seed = seed

        # Populated by architecture selection stage (Stage A)
        self.hyperparas = None # Frozen architecture
        self.study = None # Optuna study

        # Populated by bootstrapping stage (Stage B)
        self.members = [] # List of {"engine", "model", "normaliser", "indices", "best_value"}


    def _make_tuner(self, train):
        """Build a Tuner over the given train data.

        Args:
            train: SimulatedDataset; the training data this Tuner fits on. 
                - In stage A, train = full training pool 
                - In stage B, train = a bootstrap resample
        Returns:
            A configured Tuner instance
        """
        return Tuner(
            train_dataset=train,
            val_dataset=self.val_dataset,
            basis_functions=self.knot_objects["basis_functions"],
            Omega=self.knot_objects["Omega"],
            nr_basis=self.knot_objects["nr_basis"],
            lambda_mean=self.lambda_mean,
            lambda_re=self.lambda_re,
            nr_epochs=self.nr_epochs,
            patience=self.patience,
            model_cls=self.model_cls,
            model_kwargs=self.model_kwargs,
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Architecture Selection -- Stage A
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def select_architecture(self, nr_trials=100):
        """Search for an architecture once and freeze it for the whole ensemble.

        Run an Optuna search over the encoder architecture.
        
        Note:
        The search runs on the full training pool, which is the same pool the 
        bootstrap resamples are later drawn from. 

        Args:
            nr_trials: int; number of Optuna trials in the search.
        
        Returns:
            dict; the frozen hyperparameter configuration, shaped as
                {
                    "hidden_sizes": [64, 97, 31],
                    "activation": "elu",
                    "dropout": 0.18,
                    "lr": 0.0023,
                    "batch_size": 64,
                    "weight_decay": 0.0001,
                }
            The same dict is stored as self.hyperparas.
        """
        # Step 1: Set a seed
        torch.manual_seed(self.seed)

        # Step 2: Build a Tuner 
        tuner = self._make_tuner(self.train_dataset)

        # Step 3: Run the hyperparameter search 
        search_result = tuner.run(nr_trials=nr_trials, seed=self.seed)

        # Step 4: Save the winning configuration
        self.hyperparas = search_result["best_params"]
        self.study = search_result["study"]

        # Step 5: Report the frozen architecture
        print(f"Architecture frozen: val={search_result['best_value']:.5f}")
        print(self.hyperparas)

        return self.hyperparas

    def set_architecture(self, hyperparas):
        """Set an architecture to be used for the bootstrap models without a search. 

        Args:
            hyperparas: dict shaped like select_architecture's return value,
                carrying "hidden_sizes", "activation", "dropout", "lr",
                "batch_size" and "weight_decay".

        Returns:
            dict; the frozen configuration, also stored as self.hyperparas.
        """
        # Step 1: Check validity
        required = {"hidden_sizes", "activation", "dropout", "lr", "batch_size", "weight_decay"}
        missing = [key for key in required if key not in hyperparas.keys()]
        if missing:
            raise ValueError(f"Missing hyperparameters {missing}")

        # Step 2: Freeze the configuration
        self.study = None
        self.hyperparas = dict(hyperparas)

        return self.hyperparas

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Bootstrapping -- Stage B
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def _draw_bootstrap_indices(self, rng):
        """Draw the data indices of one bootstrap resample.

        Sample D_train indices uniformly WITH replacement from the training
        pool. The resample is the same size as the pool but contains some
        individuals more than once and omits others. 

        Args:
            rng: np.random.Generator; advanced in place, so successive calls
                yield different resamples.

        Returns:
            np.ndarray of shape (D_train,) and dtype int; the sampled indices,
                with repeats.
        """
        D = self.train_dataset.D
        return rng.integers(0, D, size=D)

    def _draw_bootstrap_dataset(self, rng):
        """Draw one bootstrap resample to obtain a SimulatedDataset

        Args:
            rng: np.random.Generator; advanced in place.

        Returns:
            tuple (bootstrap_dataset, indices):
                bootstrap_dataset: SimulatedDataset of D_train individuals drawn
                    with replacement from the training pool.
                indices: np.ndarray of shape (D_train,); the drawn indices,
                    returned alongside so the caller can record which
                    individuals this member actually saw.
        """
        # Step 1: Draw the resample's individual indices
        indices = self._draw_bootstrap_indices(rng)

        # Step 2: Obtain the bootstrap dataset
        bootstrap_dataset = self.train_dataset.select_individuals(indices)

        return bootstrap_dataset, indices

    def _fit_member(self, member_train_dataset, member_seed, rng=None):
        """Train one ensemble member at the frozen architecture.

        Train a model on its own bootstrap resample for a fixed frozen 
        architechture (early stopping using a shared validation set). 

        Args:
            member_train_dataset: SimulatedDataset; this member's bootstrap
                resample.
            member_seed: int; seeds torch's global generator so this member's
                weight initialisation is reproducible, and differs from the
                other members'.
            rng: np.random.Generator or None; this member's mini-batch shuffling
                order.

        Returns:
            dict from Tuner.train_fixed_configuration, with keys "best_params",
                "best_value", "model" and "normaliser".
        """
        # Step 1: Set weight initialisation seed
        torch.manual_seed(member_seed)

        # Step 2: Build a Tuner
        tuner = self._make_tuner(member_train_dataset)

        # Step 3: Train a model
        return tuner.train_fixed_configuration(self.hyperparas, rng=rng)


    def _build_engine(self, model, normaliser):
        """Wrap one trained ensemble member in an InferenceEngine.

        Note:
        Every ensemble member engine is populated with the same shape-extraction
        thresholds. Otherwise the shape summaries are not comparable. 

        Args:
            model: the trained RandomEffectsModel.
            normaliser: the covariate normaliser fitted on this member's
                resample.

        Returns:
            InferenceEngine; populated and ready to predict.
        """
        return build_inference_engine(
            model=model,
            normaliser=normaliser,
            knot_objects=self.knot_objects,
            shape_config=self.shape_config,
        )


    def fit(self, nr_members=100):
        """Train an ensemble of size nr_members of bootstrap resamples.

        For each member: 
            - draw a bootstrap resample, 
            - train a model of given  frozen architecture on the resample,
            -  wrap the result in an InferenceEngine. 

        Args:
            nr_members: int; the ensemble size M. 
                This is the number of summaries each individual's shape cloud contains

        Returns:
            list of nr_members InferenceEngines, in fit order. The full records
                are kept on self.members.
        """
        if self.hyperparas is None:
            raise RuntimeError("Set training architecture before fitting members.")


        # Step 1: Random number generator
        rng = np.random.default_rng(self.seed)

        # Step 2: Clear member list
        self.members = []

        # Step 3: Fit the members one at a time
        for m in range(nr_members):
            # Step 3.1: Draw this member's bootstrap resample
            member_train_dataset, indices = self._draw_bootstrap_dataset(rng)

            # Step 3.2: Give the member its own initialisation and shuffling order
            member_seed = self.seed + m
            member_rng = np.random.default_rng(member_seed)

            # Step 3.3: Train at the frozen architecture
            fit_result = self._fit_member(member_train_dataset, member_seed, rng=member_rng)

            # Step 3.4: Wrap the trained model in its own inference engine
            engine = self._build_engine(fit_result["model"], fit_result["normaliser"])

            # Step 3.5: Record the member
            self.members.append({
                "engine": engine,
                "model": fit_result["model"],
                "normaliser": fit_result["normaliser"],
                "indices": indices,
                "best_value": fit_result["best_value"],
            })

            # Step 3.6: Report progress
            print(f"member {m + 1}/{nr_members}  val={fit_result['best_value']:.5f}", flush=True)

        return self.engines

    @property
    def engines(self):
        """The fitted members' InferenceEngines, in fit order; consumed by UncertaintyEngine.

        Returns:
            list of InferenceEngine; empty until fit has been called.
        """
        return [member["engine"] for member in self.members]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Persistence
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def _knot_config(self):
        """Recover the knot configuration that produced self.knot_objects.

        Returns:
            dict with keys "nr_interior_knots" (int) and "T" (float), shaped for
                build_knot_dictionary(**knot_config).
        """
        return {
            "nr_interior_knots": int(self.knot_objects["nr_basis"]) - 4,
            "T": float(self.knot_objects["breakpoints"][-1]),
        }

    def save(self, path):
        """Save the whole ensemble

        Write each member through save_run into its own subdirectory, so a
        single member can be reloaded on its own with load_run if needed. 
        The following is written once to ensemble.json:
            - the frozen architecture, 
            - the shape-extraction thresholds, 
            - the roughness weights, 
            - the seed

        Args:
            path: str; the directory to write into.
        """
        if not self.members:
            raise RuntimeError("Nothing to save.")

        # Step 1: Create the target directory
        os.makedirs(path, exist_ok=True)

        # Step 2: Recover the knot configuration
        knot_config = self._knot_config()
        generate_config = (self.train_dataset.gen_config
                           if self.train_dataset is not None else None)


        # Step 3: Write each member through the single-run saver
        for m, member in enumerate(self.members):
            tuned = {
                "model": member["model"],
                "normaliser": member["normaliser"],
                "best_params": self.hyperparas,
                "best_value": member["best_value"],
            }
            save_run(
                tuned, knot_config, generate_config,
                lambda_mean=self.lambda_mean, lambda_re=self.lambda_re,
                path=os.path.join(path, f"member_{m:03d}"),
            )

        # Step 4: Write the ensemble-level record
        meta = {
            "model_cls": self.model_cls.__name__,
            "model_kwargs": self.model_kwargs,
            "hyperparas": self.hyperparas,
            "shape_config": self.shape_config,
            "lambda_mean": self.lambda_mean,
            "lambda_re": self.lambda_re,
            "nr_epochs": self.nr_epochs,
            "patience": self.patience,
            "seed": self.seed,
            "nr_members": len(self.members),
            "knot_config": knot_config,
            "generate_config": generate_config,
            "best_values": [member["best_value"] for member in self.members],
        }
        with open(os.path.join(path, "ensemble.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path, model_cls):
        """Reload a saved ensemble, ready for UncertaintyEngine.

        Rebuilds the knot objects ONCE from the saved config, then restores each
        member's weights and normaliser and rewraps it in an InferenceEngine
        populated with the ensemble's shared shape-extraction thresholds.

        Args:
            path: str; a directory a previous save() wrote to.
            model_cls: the RandomEffectsModel subclass the members were trained
                as. Checked against the saved record.

        Returns:
            BootstrapEnsemble; with hyperparas frozen and members restored, so
                .engines is immediately usable.
        """
        # Step 1: Read the ensemble-level record
        with open(os.path.join(path, "ensemble.json")) as f:
            meta = json.load(f)

        # Step 2: Check the model family matches what was saved
        if model_cls.__name__ != meta["model_cls"]:
            raise ValueError(
                f"Ensemble was saved as {meta['model_cls']}, "
                f"but {model_cls.__name__} was requested."
            )

        # Step 3: Rebuild the spline objects once, shared by every member
        knot_objects = build_knot_dictionary(**meta["knot_config"])

        # Step 4: Reconstruct the ensemble and freeze its architecture
        ensemble = cls(
            train_dataset=None,
            val_dataset=None,
            knot_objects=knot_objects,
            lambda_mean=meta["lambda_mean"],
            lambda_re=meta["lambda_re"],
            shape_config=meta["shape_config"],
            model_cls=model_cls,
            model_kwargs=meta.get("model_kwargs"),
            nr_epochs=meta["nr_epochs"],
            patience=meta["patience"],
            seed=meta["seed"],
        )
        ensemble.set_architecture(meta["hyperparas"])

        # Step 5: Restore each member and rewrap it in an engine
        for m in range(meta["nr_members"]):
            loaded = load_run(
                path=os.path.join(path, f"member_{m:03d}"), model_cls=model_cls,
            )
            engine = ensemble._build_engine(loaded["model"], loaded["normaliser"])
            ensemble.members.append({
                "engine": engine,
                "model": loaded["model"],
                "normaliser": loaded["normaliser"],
                "indices": None,
                "best_value": meta["best_values"][m],
            })

        return ensemble