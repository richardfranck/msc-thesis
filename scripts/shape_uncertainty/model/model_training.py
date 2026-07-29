import torch
import numpy as np
import optuna

from scripts.shape_uncertainty.model.optimisation_objective import objective
from scripts.shape_uncertainty.model.model import Encoder
from scripts.shape_uncertainty.model.data_preparation import Normaliser, prepare_training_data

def _make_optimiser(model, lr, weight_decay):
    """Adam with L2 weight decay on the ENCODER parameters only.

    Initialise the Adam optimiser with L2 regularisation on the
    NN parameters only and not on any other parameters
    (e.g random effects parameters).

    Args:
        model: a RandomEffects model.
        lr: float, learning rate.
        weight_decay: float, L2 weight decay applied to the encoder only.

    Returns:
        torch.optim.Adam configured with the two parameter groups.
    """
    # Separate encoder paramerers from all other parameters
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    other_paras = [
        p for p in model.parameters()
        if id(p) not in encoder_param_ids and p.requires_grad
    ]

    # Initialise the Adam optimiser with L2 regularisaton only on the NN parameters.
    optimiser = torch.optim.Adam([
        {"params": model.encoder.parameters(), "weight_decay": weight_decay},
        {"params": other_paras, "weight_decay": 0.0},
    ], lr=lr)

    return optimiser


def evaluate_objective(model, batch, Omega, lambda_mean, lambda_re):
    """Compute the objective value on a batch with no gradient step (for logging).
    
    Args:
        model: a RandomEffectsModel.
        batch: a Batch to evaluate on.
        Omega, lambda_mean, lambda_re: penalty matrix and weights.

    Returns:
        float; the objective on the batch.
    """
    with torch.no_grad():
        return objective(model, batch, Omega, lambda_mean, lambda_re).item()


def train(model, train_batch, Omega, lambda_mean, lambda_re,
          lr, weight_decay, nr_epochs, batch_size=None,
          val_batch=None, patience=None, rng=None):
    """Fit a random-effects model by minimising NLL + wiggle penalty.

    Run gradient descent (Adam) on
        L = NLL(train_batch) + penalty(Omega, X, lambda_mean, lambda_re)

        - If batch_size is set and smaller than D, use a random mini-batch of
            individuals for each optimisation step (Batch.subbatch). 
        - If a validation batch is given, track validation loss for early 
            stopping / model selection.

    Note:
        - The lambda_* weights are fixed inputs (not tuned).  
        - The weight_decay applies  L2 regularisation to the encoder parameters.

    Args:
        model: a RandomEffectsModel (e.g. GaussianModel).
        train_batch: a Batch of prepared training data.
        Omega: torch.Tensor (B, B); the spline penalty matrix.
        lambda_mean: float, weight on the mean-trajectory roughness.
        lambda_re: float, weight on the random-effects roughness.
        lr: float, optimiser learning rate.
        weight_decay: float, L2 weight decay on the network parameters.
        nr_epochs: int, number of training epochs.
        batch_size: int or None; mini-batch size over individuals (None = full batch).
        val_batch: a Batch or None; validation data for early stopping/selection.
        patience: int or None; early-stopping patience in epochs (None = no early stop).
        rng: np.random.Generator or None; for mini-batch shuffling.

    Returns:
        dict with the trained model and training history, e.g.
            {"model": model, "train_losses": [...], "val_losses": [...],
             "best_epoch": int}.
    """
    # Step 1: If no random number generator was provided, create one for random shuffling
    if rng is None:
        rng = np.random.default_rng(0)

    # Step 2: Create an optimiser with weight decay only on encoder parameters (not sigma^2 and Sigma)
    optimiser = _make_optimiser(model, lr, weight_decay)

    # Step 3: Create variables to track losses and early stopping based on validation set
    history = {"train_losses": [], "val_losses": []}
    best = {"val": float("inf"), "epoch": -1, "state": None}
    no_improve = 0

    # Step 4: Run training loop for the specified nr of epochs
    D = train_batch.D
    for epoch in range(nr_epochs):
        model.train()

        # Step 4.1: Use full batch if batch_size is None or too large; else create a mini-batch
        if batch_size is None or batch_size >= D:
            batches = [np.arange(D)] # Create one batch of individuals [0, 1, 2, ..., D-1]
        else:
            perm = rng.permutation(D) # Create a random permutation of [0, 1, 2, ..., D-1] and split into batches of size batch_size.
            batches = [perm[k:k + batch_size] for k in range(0, D, batch_size)]

        # Step 4.2: Take one step per mini-batch on the objective.
        for ids in batches:
            # Create a mini batch of selected individuals
            mini_batch = train_batch.subbatch(ids)
            # Clear gradients from the previous step
            optimiser.zero_grad()
            # Compute loss on the mini-batch
            loss = objective(model, mini_batch, Omega, lambda_mean, lambda_re)
            # Computes gradient of the loss w.r.t. all trainable parameters
            loss.backward()
            # Updates the parameters using Adam
            optimiser.step()

        # Step 4.3: record the full-batch training loss for this epoch.
        # (full training objective loss after the epoch’s updates)
        model.eval()
        history["train_losses"].append(
            evaluate_objective(model, train_batch, Omega, lambda_mean, lambda_re))
        
        # Step 4.4: If using validation set record val loss, track the best epoch, and early-stop on patience
        if val_batch is not None:
            v = evaluate_objective(model, val_batch, Omega, lambda_mean, lambda_re)
            history["val_losses"].append(v)
            if v < best["val"]:
                best = {"val": v, "epoch": epoch,
                        "state": {k: t.detach().clone() for k, t in model.state_dict().items()}}
                no_improve = 0
            else:
                no_improve += 1
                if patience is not None and no_improve >= patience:
                    break

    # Step 5: Restore the best-validation parameters if any were recorded
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # Step 6: Return the trained model and the loss history
    return {
        "model": model, 
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "best_epoch": best["epoch"],
    }


class Tuner:
    """Optuna hyperparameter search for a RandomEffectsModel.

    Run a hyperparameter search for
        - Architectural hyperparameters (hidden sizes, activation, dropout; and
        - Optimisation hyperparameters (learning rate, batch size, weight decay).
    The smoothness weights lambda_mean and lambda_re are not searched but taken
    as given. 

    For each Optuna trial:
        1. sample hyperparameters
        2. prepare train/validation batches
        3. build Encoder + RandomEffectsModel
        4. train the model
        5. return validation objective
    After all trials:
        6. recover best hyperparameters
        7. retrain best model
        8. return model, normaliser, study, and best params

    Args:
        train_dataset, val_dataset: SimulatedDatasets for fitting and scoring.
        basis: the B-spline basis used to build design matrices.
        Omega: torch.Tensor (B, B); the spline penalty matrix.
        nr_basis: int, B = K + 4.
        lambda_mean, lambda_re: floats; fixed smoothness weights.
        nr_epochs: int; training epochs per trial.
            prepare_fn: callable(dataset, basis, normaliser) -> Batch
                (the data-preparation function).
            normaliser_cls: class constructing a fresh covariate normaliser.
            train_fn: the training function train(model, train_batch, ...).
            evaluate_fn: evaluate_objective(model, batch, Omega, lm, lr).
        patience: int or None; early-stopping patience passed to train_fn.

        model_cls: the RandomEffectsModel subclass to build and tune
            (GaussianModel or MixtureModel).
        model_kwargs: dict or None; extra keyword arguments forwarded to
            model_cls beyond (encoder, nr_basis); e.g. a mixture model's
            number of components.

        search_space: dict of candidate ranges
    """
    def __init__(self, train_dataset, val_dataset, basis_functions, Omega, nr_basis,
                 lambda_mean, lambda_re, nr_epochs, 
                model_cls, model_kwargs=None,
                patience=10):
        # Fixed across all trials
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.basis_functions = basis_functions
        self.Omega = Omega
        self.nr_basis = nr_basis
        self.lambda_mean = lambda_mean
        self.lambda_re = lambda_re
        self.nr_epochs = nr_epochs
        self.patience = patience

        # Specification of the type of model (Gaussian or GMM) 
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}

        # Tunable parameters:
        self.search_space = {
            "hidden_sizes": {"nr_layers": 3, "low": 16, "high": 128},
            "activation": ["relu", "sigmoid", "tanh", "leaky_relu", "elu", "selu"],
            "dropout": {"low": 0.0, "high": 0.5},
            "lr": {"low": 1e-4, "high": 1e-1, "log": True},
            "batch_size": [64, 128],
            "weight_decay": {"low": 1e-6, "high": 1e-1, "log": True},
        }


    def _get_hyperparameter_selection(self, trial):
        """Sample one set of hyperparameters for one Optuna trial.

        We draw the following
            - one integer width per hidden layer, 
            - a categorical activation, 
            - a uniform dropout probability, 
            - a log-uniform learning rate, 
            - a categorical batch size, and 
            - a log-uniform weight decay. 
        The number of hidden layers is fixed and taken from 
        search_space["hidden_sizes"]["nr_layers"].

        Args:
            trial: an optuna.Trial; the object Optuna passes in to record and
                propose hyperparameter values for this trial.

        Returns:
            dict of the sampled hyperparameters. For example:
                {
                    "hidden_sizes": [64, 97, 31],
                    "activation": "elu",
                    "dropout": 0.18,
                    "lr": 0.0023,
                    "batch_size": 64,
                    "weight_decay": 0.0001,
                }
        """
        hs = self.search_space["hidden_sizes"]
        av = self.search_space["activation"]
        dr = self.search_space["dropout"]
        lr = self.search_space["lr"]
        bz = self.search_space["batch_size"]
        wd = self.search_space["weight_decay"]

        # Make optuna select values for all hyperparametrs. 
        hyperparameter_selection = {
            "hidden_sizes": [trial.suggest_int(f"hidden_{i}", hs["low"], hs["high"])
                                for i in range(hs["nr_layers"])],
            "activation": trial.suggest_categorical("activation", av),
            "dropout": trial.suggest_float("dropout", dr["low"], dr["high"]),
            "lr": trial.suggest_float("lr", lr["low"], lr["high"], log=lr["log"]),
            "batch_size": trial.suggest_categorical("batch_size", bz),
            "weight_decay": trial.suggest_float("weight_decay", wd["low"], wd["high"], log=wd["log"]),
        }

        return hyperparameter_selection


    def _prepare_batches(self):
        """Prepare train and validation Batches sharing one normaliser fitted on training.

        Step 1: Construct a new covariate normaliser, 
        Step 2.1: Fit the covariate normaliser on the training dataset
        Step 2.2: Prepare the training Batch
        Step 3: Prepare the validation Batch with that already fitted normaliser
                    (i.e. reusing  training statistics on the validation set.

        Returns:
            (train_batch, val_batch, normaliser): the two prepared Batches and
                the fitted normaliser (returned so it can travel with a
                retrained model for use at prediction time).
        """
        # Step 1: Construct a new covariate normaliser
        normaliser = Normaliser()

        # Step 2: Prepare the training batch (and fit the covariate normaliser on the training dataset)
        train_batch = prepare_training_data(self.train_dataset, self.basis_functions, normaliser)

        # Step 3: Prepare the validation batch with that already fitted) normaliser
        val_batch = prepare_training_data(self.val_dataset, self.basis_functions, normaliser)

        return train_batch, val_batch, normaliser


    def _build_re_model(self, hyperparas, nr_covariates):
        """Construct an Encoder and wrap it in a RandomEffectsModel.

        Args:
            hyperparas: dict of sampled hyperparameters.
            nr_covariates: int, M; the encoder's input dimension

        Returns:
            RandomEffectsMpde;; an untrained model ready to be optimised.
        """
        # Step 1: Build the encoder from the architectural hyperparameters,
        encoder = Encoder(
                    nr_covariates, 
                    self.nr_basis,
                    hidden_sizes=tuple(hyperparas["hidden_sizes"]),
                    activation=hyperparas["activation"],
                    dropout=hyperparas["dropout"],
        )

        # Step 2: Wrap the encoder in self.model_cls (a RandomEffectsModel) with any
        #           model-specific keyword arguments
        random_effects_model = self.model_cls(encoder, self.nr_basis, **self.model_kwargs)

        return random_effects_model


    def _train_re_model(self, hyperparas, train_batch, val_batch, rng=None):
        """Build and train a model for one hyperparameter configuration.

        Step 1: Construct the model from hyperparas and the batch's covariate dimension, 
        Step 2: Run the training loop with 
            - this configs's optimisation hyperparas (learning rate, weight decay, batch size); and 
            - the fixed context (Omega, the lambda weights, epochs, patience).
        Step 3: Return the best model
        Args:
            hyperparas: dict of sampled hyperparameters
            train_batch: the prepared training Batch
            val_batch: the prepared validation Batch
            rng: np.random.Generator or None; drives mini-batch shuffling. When
                None, train() falls back to its own fixed default, so every call
                walks the data in the same order.


        Returns:
            RandomEffectModel; the trained model (best-validation state restored).
        """
        # Step 1: Construct the model 
        model = self._build_re_model(hyperparas, train_batch.X.shape[1])

        # Step 2: Run the training loop
        train_result = train(model, train_batch, self.Omega, self.lambda_mean, self.lambda_re,
                        lr=hyperparas["lr"], weight_decay=hyperparas["weight_decay"], nr_epochs=self.nr_epochs,
                        batch_size=hyperparas["batch_size"], val_batch=val_batch, patience=self.patience,
                        rng=rng)
                    
        # Step 3: Return the best model 
        return train_result["model"]


    def _execute_optuna_trial_iteration(self, trial, train_batch, val_batch):
        """Execute one Optuna search trial on pre-prepared batches.

        Args:
            trial: an optuna.Trial.
            train_batch: the prepared training Batch (shared across all trials).
            val_batch: the prepared validation Batch (shared across all trials).

        Returns:
            float; the validation objective (NLL + penalty) of the trained model.
        """
        # Step 1: Draw hyperparameters
        hyperparas = self._get_hyperparameter_selection(trial)

        # Step 2: Train a model with the drawn configuration (batches passed in)
        model = self._train_re_model(hyperparas, train_batch, val_batch)

        # Step 3: Score the trained model on the validation batch
        score =  evaluate_objective(model, val_batch, self.Omega,
                                  self.lambda_mean, self.lambda_re)

        return score

    def _best_hyperparas(self, study):
        """Return a dict of the best hyperparameter found after a full search a finished study.

        Rebuilt the best trial's parameters from Optuna format into the 
        following format:
                {
                    "hidden_sizes": [64, 97, 31],
                    "activation": "elu",
                    "dropout": 0.18,
                    "lr": 0.0023,
                    "batch_size": 64,
                    "weight_decay": 0.0001,
                }

        Args:
            study: a completed optuna.Study.

        Returns:
            dict of the best configuration
        """
        b = study.best_params
        hs = self.search_space["hidden_sizes"]
        return {
            "hidden_sizes": [b[f"hidden_{i}"] for i in range(hs["nr_layers"])],
            "activation": b["activation"], 
            "dropout": b["dropout"],
            "lr": b["lr"], 
            "batch_size": b["batch_size"], 
            "weight_decay": b["weight_decay"],
        }

    def run(self, nr_trials=100, seed=None):
        """Run the search and retrain the best configuration.
        
        Step 1: Optimise the validation objective over nr_trials, 
        Step 2: Train the best config training data with validation-based early stopping
        
        Note that we return the fitted normaliser later predictionsmust standardise 
        features with the same training statistics.

        Args:
            nr_trials: int; number of Optuna trials.
            seed: int 

        Returns:
            dict with keys:
                "best_params": the best structured hyperparameter dict,
                "best_value": the best validation objective achieved,
                "model": the retrained best RandomEffectsModel,
                "normaliser": the fitted covariate normaliser,
                "study": the completed optuna.Study.
        """
        # Step 1: Prepare train/validation batches (shared by every trial).
        train_batch, val_batch, normaliser = self._prepare_batches()

        # Step 2: Optimise the validation objective; the lambda injects the shared
        #         batches into Optuna's single-argument objective callable.
        study = optuna.create_study(
                direction="minimize", 
                sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(
            lambda trial: self._execute_optuna_trial_iteration(trial, train_batch, val_batch),
            n_trials=nr_trials,
        )

        # Step 3: Retrain the best configuration on the same batches (data unchanged).
        hyperparas = self._best_hyperparas(study)
        model = self._train_re_model(hyperparas, train_batch, val_batch)

        return {"best_params": hyperparas, "best_value": study.best_value,
                "model": model, "normaliser": normaliser, "study": study}


    def train_fixed_configuration(self, hyperparas, rng=None):
        """Train a model for a GIVEN hyperparameter configuration (no Optuna search).

        Train a model with the given hyperparam config on the training data 
        with validation-based early stopping.

        Args:
            hyperparas: dict shaped like run()'s "best_params" (hidden_sizes,
                activation, dropout, lr, batch_size, weight_decay).
            rng: np.random.Generator or None; mini-batch shuffling order.

        Returns:
            dict with keys:
                "best_params": the best structured hyperparameter dict,
                "best_value": the best validation objective achieved,
                "model": the retrained best RandomEffectsModel,
                "normaliser": the fitted covariate normaliser
        """
        # Step 1: Prepare train/validation batches  
        train_batch, val_batch, normaliser = self._prepare_batches()

        # Step 2: Train at the given hyperparameters, no search
        model = self._train_re_model(hyperparas, train_batch, val_batch, rng=rng)

        # Step 3: Score on validation for logging/provenance
        best_value = evaluate_objective(model, val_batch, self.Omega,
                                        self.lambda_mean, self.lambda_re)

        return {"best_params": hyperparas, "best_value": best_value,
                "model": model, "normaliser": normaliser}
