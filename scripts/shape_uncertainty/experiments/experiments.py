import numpy as np
import pandas as pd
import torch

# B-Spline utilities
from scripts.shape_uncertainty.spline_basis.bspline_basis import (
    basis_matrix,
    build_knot_dictionary,
)

# Shape extraction utilities
from scripts.shape_uncertainty.shape_extraction.shape_summary import extract_shape_summary
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import get_analytic_shape_summary
from scripts.shape_uncertainty.shape_extraction.shape_distance import (
    mean_shape_sequence_match,
    mean_shape_summary_distance,
)


# Model training and inference utilities
from scripts.shape_uncertainty.model.model_training import Tuner
from scripts.shape_uncertainty.model.model_inference import InferenceEngine
from scripts.shape_uncertainty.model.model_evaluation import ModelEvaluator



class ExtractionEvaluator:
    """Assess the faithfulness of shape extraction, independent of any model."""
    def __init__(self, dataset):
        self.dataset = dataset

    def _fit_ols_splines(self, nr_obs, nr_interior_knots):
        """Fit the best-possible spline to each individual's noise-free curve.

        Solve for the least-squares problem for the spline coefficients.
        Snice there is no model for mapping covariates X to spline coefficents
        and no observation noise here, the result is the best trajectory that
        can be achieved for a given number of observation points and spline
        dimension.

        Args:
            nr_obs: int; number of evenly spaced observation times on [0, T].
            nr_interior_knots: int; number of interior knots for the basis.

        Returns:
            dict with keys:
                "coefficients": np.ndarray of shape (D, B); one spline
                    coefficient vector per individual.
                "C": the monomial conversion tensor for the basis.
                "breakpoints": np.ndarray of the interior plus boundary knots.
        """
        # Step 1: Build the shared observation grid
        T = self.dataset.T
        observation_times = np.linspace(0.0, T, nr_obs)

        # Step 2: Build the spline basis 
        knot_objects = build_knot_dictionary(nr_interior_knots, T)
        Phi = basis_matrix(observation_times, knot_objects["basis_functions"])

        # Step 3: Evaluate every individual's noise-free curve on that grid
        Y_true = np.array(self.dataset.true_curves_at(observation_times))

        # Step 4: Solve for all individuals at once (Phi is shared)
        coefficients, _, _, _ = np.linalg.lstsq(Phi, Y_true.T, rcond=None)

        return {
            "coefficients": coefficients.T,
            "C": knot_objects["C"],
            "breakpoints": knot_objects["breakpoints"],
        }

    def evaluate_extraction(self, nr_obs, nr_interior_knots, zeta_rel=0.0,
                            u1_rel=0.0, u2_rel=0.0, upsilon_rel_prune=0.0,
                            do_prune=False):
        """Score shape extraction from best-possible splines against the truth.

        1. Fit a the splines at the given resolution of noise-free y using OLS,
        2. Extract a shape summary from each fitted spline, 
        2. Get the ground truth shape summary for each trajectory, 
        3. Grade the extracted shape summary against the ground truth shape summary. 
            - Note that both estimated and true summary use the same thresholds 
              so they describe shapes in one vocabulary
        4. Return the shape extraction accuracy and the generated summaries. 

        Args:
            nr_obs: int; number of evenly spaced observation times on [0, T].
            nr_interior_knots: int; number of interior knots for the basis.
            zeta_rel: float; relative transition de-duplication tolerance
                (applies to spline extraction only).
            u1_rel: float; relative slope significance threshold.
            u2_rel: float; relative curvature significance threshold.
            upsilon_rel_prune: float; relative pruning significance threshold.
            do_prune: bool; whether to apply slope then curvature pruning.

        Returns:
            dict with keys:
                "accuracy": float in [0, 1]; the exact-sequence-match share.
                "distance": float in [0, 1]; the mean shape-summary distance (0 = identical).
                "predicted": list of length D; summaries from the fitted splines.
                "truth": list of length D; analytic ground-truth summaries.
        """
        # Step 1: Fit the best-possible spline for every individual
        fit = self._fit_ols_splines(nr_obs, nr_interior_knots)

        # Step 2: Extract a shape summary from each fitted spline
        predicted = []
        for w in fit["coefficients"]:
            summary = extract_shape_summary(
                w.reshape(-1, 1), fit["C"], fit["breakpoints"],
                zeta_rel=zeta_rel,
                upsilon_rel_1=u1_rel,
                upsilon_rel_2=u2_rel,
                upsilon_rel_prune=upsilon_rel_prune,
                do_prune=do_prune,
            )
            predicted.append(summary)

        # Step 3: Get the analytic ground truth at the same thresholds
        truth = get_analytic_shape_summary(self.dataset, u1_rel, u2_rel,
                               upsilon_rel_prune, do_prune)

        # Step 4: Score the predicted state sequences against the truth
        accuracy = mean_shape_sequence_match(predicted, truth)
        distance = mean_shape_summary_distance(predicted, truth, self.dataset.T)

        return {"accuracy": accuracy, "distance": distance,
                "predicted": predicted, "truth": truth}

    # -------------------------------------------------------------------- #
    # --------------------- Shape Extraction Analysis -------------------- #
    # -------------------------------------------------------------------- #

    def sweep_observations(self, variable_nr_obs, fixed_nr_knots, zeta_rel, fixed_u1_rel, fixed_u2_rel, upsilon_rel_prune, do_prune):
        """Evaluate shape-extraction accuracy across varying observation densities.

        Evaluate how extraction accuracy varies as we vary the number of observations
        nr_obs. The other inputs (nr_knots, u1_rel, u2_rel) remain fixed.
        
        Note: We compare the analytic shape summart and the shape summary our extractor
        produces based on the data generating process. No model is involved. 

        Args:
            variable_nr_obs (list): The observation number counts over which to sweep.
            fixed_nr_knots (list): The interior knot counts (fixed). 
            zeta_rel (float): Relative transition de-duplication tolerance (fixed).
            fixed_u1_rel (float): Relative slope threshold (fixed).
            fixed_u2_rel (float): Relative curvature threshold (fixed).
            upsilon_rel_prune (float): Relative pruning threshold (fixed).
            do_prune (bool): Whether to apply slope then curvature pruning (fixed).

        Returns:
            list of (nr_obs, accuracy) pairs.
        """
        rows = []
        for n in variable_nr_obs:
            result = self.evaluate_extraction(
                n, fixed_nr_knots, zeta_rel=zeta_rel,
                u1_rel=fixed_u1_rel, u2_rel=fixed_u2_rel,
                upsilon_rel_prune=upsilon_rel_prune, do_prune=do_prune,
            )
            rows.append({
                "nr_obs": n,
                "accuracy": result["accuracy"],
                "distance": result["distance"],
            })
        return rows


    def sweep_knots(self, fixed_nr_obs, variable_nr_knots, zeta_rel, fixed_u1_rel, fixed_u2_rel, upsilon_rel_prune, do_prune):
        """Evaluate shape-extraction accuracy across varying spline complexities.

        Args:
            fixed_nr_obs (int): The observation count per individual (fixed).
            variable_nr_knots (list): The interior knot counts over which to sweep.
            zeta_rel (float): Relative transition de-duplication tolerance (fixed).
            fixed_u1_rel (float): Relative slope threshold (fixed).
            fixed_u2_rel (float): Relative curvature threshold (fixed).
            upsilon_rel_prune (float): Relative pruning threshold (fixed).
            do_prune (bool): Whether to apply slope then curvature pruning (fixed).


        Returns:
            list of dicts, one per swept value, each with keys
                "nr_interior_knots" (int), "accuracy" (float in [0, 1], higher
                is better) and "distance" (float in [0, 1], lower is better).
        """
        rows = []
        for k in variable_nr_knots:
            result = self.evaluate_extraction(
                fixed_nr_obs, k,
                zeta_rel=zeta_rel,
                u1_rel=fixed_u1_rel,
                u2_rel=fixed_u2_rel,
                upsilon_rel_prune=upsilon_rel_prune,
                do_prune=do_prune,
            )
            rows.append({
                "nr_interior_knots": k,
                "accuracy": result["accuracy"],
                "distance": result["distance"],
            })
        return rows


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
                 zeta_rel=0.0, upsilon_rel_1=0.02, upsilon_rel_2=0.02,
                 upsilon_rel_prune=0.0, do_prune=False, evaluation_times=None,
                 lambda_re=0.0, nr_trials=25, nr_epochs=150, patience=10):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.knot_objects = knot_objects
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.zeta_rel = zeta_rel
        self.upsilon_rel_1 = upsilon_rel_1
        self.upsilon_rel_2 = upsilon_rel_2
        self.upsilon_rel_prune = upsilon_rel_prune
        self.do_prune = do_prune
        self.evaluation_times = (evaluation_times if evaluation_times is not None
                                  else np.linspace(0.0, test_dataset.T, 200))
        self.lambda_re = lambda_re
        self.nr_trials = nr_trials
        self.nr_epochs = nr_epochs
        self.patience = patience

        # Ground truth shapes (independent of lambda_mean and model_cls)
        self.true_summaries = get_analytic_shape_summary(
            test_dataset, upsilon_rel_1, upsilon_rel_2,
            upsilon_rel_prune, do_prune,
        )

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
            zeta_rel=self.zeta_rel,
            upsilon_rel_1=self.upsilon_rel_1, 
            upsilon_rel_2=self.upsilon_rel_2,
            upsilon_rel_prune=self.upsilon_rel_prune,
            do_prune=self.do_prune,
        )
        evaluator = ModelEvaluator(engine, self.test_dataset)

        # Step 1: Evaluate shape extraction performance
        predicted_summaries = evaluator.get_shape_summary_estimates()
        shape_accuracy = mean_shape_sequence_match(predicted_summaries, self.true_summaries)

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

            tuned_by_lambda (dict): A dictionary with lambda_mean values as keys and
                values as the full tuned dict returned by Tuner.run (model, normaliser,
                best_params, best_value, study), retained so each trained model can be
                persisted with save_run.
        """
        rows, engines_by_lambda, tuned_by_lambda = [], {}, {}
        for lambda_mean in lambda_mean_grid:
            # Step 1: Tune a model
            tuned = self._run_one(lambda_mean)

            # Step 2: Get performance metrics and the used engine. Append lambda_mean to the metrics. 
            evaluation_metrics, engine = self._evaluate_one(tuned)
            evaluation_metrics["lambda_mean"] = lambda_mean

            # Step 3: Save results
            rows.append(evaluation_metrics)
            engines_by_lambda[lambda_mean] = engine
            tuned_by_lambda[lambda_mean] = tuned

        # Step 4: Built the results into a  DataFrame
        df = pd.DataFrame(rows)
        results_df = df[[
        "lambda_mean", "shape_accuracy", "value_r2", "value_rmse",
        "mean_wiggle"
        ]]
        
        return results_df, engines_by_lambda, tuned_by_lambda