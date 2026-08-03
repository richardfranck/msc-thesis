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

