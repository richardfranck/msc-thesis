import numpy as np
import pandas as pd
import torch

from scripts.shape_uncertainty.model.model_training import Tuner
from scripts.shape_uncertainty.model.model_inference import InferenceEngine
from scripts.shape_uncertainty.model.model_evaluation import ModelEvaluator, ExtractionEvaluator
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import get_analytic_shape_summary


class RegularisationSweep:
    """Train + evaluate one model_cls across a grid of lambda_mean values.

    This class implements a sweep over lambda_mean wiggle regularisation
    parameters, while keeping lambda_re fixed at 0.0 by defualt (i.e. no
    smoothness regularisation for random effects). The class works for
    any random effects model (e.g. MeanOnlyModel or GaussianModel)

    Args:
        train_dataset: SimulatedDataset used for training.
        val_dataset: SimulatedDataset used for validation and early stopping.
        test_dataset: SimulatedDataset used for shape, value-space, and
            roughness evaluation of each trained model.
        knot_objects: dict from build_knot_dictionary (basis_functions, C,
            breakpoints, Omega, nr_basis).
        model_cls: the RandomEffectsModel subclass to train and evaluate at
            every lambda_mean (e.g. MeanOnlyModel or GaussianModel).
        model_kwargs: dict or None; extra keyword arguments forwarded to
            model_cls beyond (encoder, nr_basis).
        upsilon_rel_1: float, relative slope significance threshold used for
            both the ground-truth and predicted shape summaries.
        upsilon_rel_2: float, relative curvature significance threshold used
            for both the ground-truth and predicted shape summaries.
        evaluation_times: 1-D array of times for value-space accuracy, or
            None to default to 200 points evenly spaced over
            [0, test_dataset.T].
        lambda_re: float, fixed random-effects penalty weight applied at
            every lambda_mean in the sweep (not itself swept).
        nr_trials: int, Optuna trials per lambda_mean (passed to Tuner).
        nr_epochs: int, maximum training epochs per trial.
        patience: int, early-stopping patience in epochs.
    """
    def __init__(self, train_dataset, val_dataset, test_dataset, knot_objects,
                 model_cls, model_kwargs=None,
                 upsilon_rel_1=0.02, upsilon_rel_2=0.02,
                 evaluation_times=None,
                 lambda_re=0.0, nr_trials=25, nr_epochs=150, patience=10):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.knot_objects = knot_objects
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.upsilon_rel_1 = upsilon_rel_1
        self.upsilon_rel_2 = upsilon_rel_2
        self.evaluation_times = (evaluation_times if evaluation_times is not None
                                  else np.linspace(0.0, test_dataset.T, 200))
        self.lambda_re = lambda_re
        self.nr_trials = nr_trials
        self.nr_epochs = nr_epochs
        self.patience = patience

        # Ground truth shapes (independent of lambda_mean and model_cls)
        self.true_summaries = get_analytic_shape_summary(
            test_dataset, upsilon_rel_1, upsilon_rel_2)

    def _run_one(self, lambda_mean):
        """Tune and train model_cls at one lambda_mean value.

        Args:
            lambda_mean (float): mean-trajectory roughness weight.

        Returns:
            dict returned by Tuner.run: "best_params", "best_value", "model",
                "normaliser", "study".
        """
        # Initialise a Tuner instance for a Optuna training/tuning trials
        tuner = Tuner(
            train_dataset=self.train_dataset, 
            val_dataset=self.val_dataset,
            basis_functions=self.knot_objects["basis_functions"],
            Omega=self.knot_objects["Omega"], 
            nr_basis=self.knot_objects["nr_basis"],
            lambda_mean=lambda_mean, 
            lambda_re=self.lambda_re,
            nr_epochs=self.nr_epochs, 
            patience=self.patience,
            model_cls=self.model_cls, 
            model_kwargs=self.model_kwargs,
        )
        # Run Optuna trials
        return tuner.run(nr_trials=self.nr_trials)

    def _evaluate_one(self, tuned):
        """Score one trained model on the test set.

        Compute evaluation metrics for a trained RandomEffectsModel (trained 
        with a given lambda_mean) according to its
            - shape summary accuracy,
            - value fit (via RMSE and R_squared)
            - mean wiggle

        Args:
            tuned: dict returned by _run_one.

        Returns:
            tuple (metrics, engine):
                metrics: dict with keys "shape_accuracy", "value_r2",
                    "value_rmse", "mean_wiggle"
                engine: the populated InferenceEngine wrapping the trained
                    model, returned so callers can reuse it for further
                    inspection (e.g. qualitative plotting) without rebuilding it.
        """
        model = tuned["model"]
        normaliser = tuned["normaliser"]
        # Step 0: For evaluation instantiate an InferenceEngine and ModelEvaluator
        engine = InferenceEngine()
        engine.populate_attributes(
            model=model, 
            normaliser=normaliser,
            basis_functions=self.knot_objects["basis_functions"],
            C=self.knot_objects["C"], 
            breakpoints=self.knot_objects["breakpoints"],
            upsilon_rel_1=self.upsilon_rel_1, 
            upsilon_rel_2=self.upsilon_rel_2,
        )
        evaluator = ModelEvaluator(engine, self.test_dataset)

        # Step 1: Evaluate shape extraction performance using the ExtractionEvaluator
        extraction_eval = ExtractionEvaluator(self.test_dataset)
        predicted_summaries = evaluator.get_shape_summary_estimates()
        shape_accuracy = extraction_eval.exact_sequence_match(
            predicted_summaries, 
            self.true_summaries
        )

        # Step 2: Evaluate value-space fit (RMSE, R^2) against the true trajectories.
        value_metrics = evaluator.compute_value_space_accuracy(self.evaluation_times)  

        # Step 3: Compute all trajectories mean_wiggle
        X_test_normalised = normaliser.transform(self.test_dataset.X)
        model.eval()
        with torch.no_grad():
            mean_wiggle = model.mean_wiggle(
                X_test_normalised,
                self.knot_objects["Omega"],
            )
            mean_wiggle = mean_wiggle.detach().cpu().numpy()
  
        # Step 4: Assemble a dictionary of evaluation metrics
        evaluation_metrics = {
            "shape_accuracy": shape_accuracy,
            "value_r2": value_metrics["r2"],
            "value_rmse": value_metrics["rmse"],
            "mean_wiggle": mean_wiggle,
            "best_params": tuned["best_params"],
            "best_value": tuned["best_value"],
        }

        return evaluation_metrics, engine


    def sweep_lambda_mean_grid(self, lambda_mean_grid):
        """Run a hyperparameter sweep over a grid of lambda_mean values.

        Args:
            lambda_mean_grid (list): Regularisation penalty values 
                (lambda_mean) to evaluate during the grid search.

        Returns:
            results_df (pd.DataFrame): A DataFrame with metrics (lambda_mean, 
                shape_accuracy, value_r2, value_rmse, mean_wiggle) for each tested value
    
            engines_by_lambda (dict): A dictionary with lambda_mean values as keys and
                values as the trained InferenceEngine instances.
        """
        rows, engines_by_lambda = [], {}
        for lambda_mean in lambda_mean_grid:
            # Step 1: Tune a model
            tuned = self._run_one(lambda_mean)

            # Step 2: Get performance metrics and the used engine. Append lambda_mean to the metrics. 
            evaluation_metrics, engine = self._evaluate_one(tuned)
            evaluation_metrics["lambda_mean"] = lambda_mean

            # Step 3: Save results
            rows.append(evaluation_metrics)
            engines_by_lambda[lambda_mean] = engine

        # Step 4: Built the results into a  DataFrame
        df = pd.DataFrame(rows)
        results_df = df[[
        "lambda_mean", "shape_accuracy", "value_r2", "value_rmse",
        "mean_wiggle"
        ]]
        
        return results_df, engines_by_lambda