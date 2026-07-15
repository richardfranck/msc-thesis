import torch
import numpy as np

# Shape extraction utilities
from scripts.shape_uncertainty.shape_extraction.shape_summary import extract_shape_summary
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import get_analytic_shape_summary

# B-Spline mathematical utilities
from scripts.shape_uncertainty.spline_basis.bspline_basis import (
    make_knots,
    basis_functions,
    basis_matrix,
    monomial_coefficients,
)

# Trajectory simulation
from scripts.shape_uncertainty.simulated_data.data_generator import compute_wilkerson


class ExtractionEvaluator:
    """Assess the faithfulness of shape extraction, independent of any model."""
    def __init__(self, dataset):
        self.dataset = dataset

    # -------------------------------------------------------------------- #
    # --------------------- Shape Extraction Machinery ------------------- #
    # -------------------------------------------------------------------- #

    def shape_summaries_at(self, nr_obs, nr_interior_knots,zeta_rel=0.0, 
        u1_rel=0.0, u2_rel=0.0, upsilon_rel_prune=0.0, do_prune=False):
        """Extract shape summaries for the dataset at a GIVEN resolution.

        This function:
          - Computes the best-possible spline coefficients for one all trajectory
            using OLS based on noise-free observations. 
          - Extracts the shape summaries associated with all predicted trajectorie. 
        The observation time points and the spline resolution are fixed across all
        individuals. 

        Args:
            nr_obs: Number of evenly spaced observation times over [0, T].
            nr_interior_knots: Number of interior knots for the spline basis.
            u1_rel: float, relative slope threshold for shape extraction.
            u2_rel: float, relative curvature threshold for shape extraction.

        Returns:
            list: A shape summary for each individual in the dataset.
        """
        # Generate nr_obs evenly spaced observation times over [0, T] 
        T = self.dataset.T
        observation_times = np.linspace(0.0, T, nr_obs)

        # Create ingredients needed for OLS coefficent estimation and shape extractons
        augmented_knots = make_knots(nr_interior_knots, T)
        basis_func = basis_functions(augmented_knots, degree=3)
        C = monomial_coefficients(basis_func, augmented_knots)
        breakpoints = np.unique(augmented_knots)
        Phi = basis_matrix(observation_times, basis_func)

        summaries = []
        for i in range(self.dataset.D):
            # Create noise-free Wilkerson outcomes at observation times from self.dataset
            y_i = compute_wilkerson(
                observation_times,
                self.dataset.X["size"][i],
                self.dataset.params["g"][i],
                self.dataset.params["d"][i],
                self.dataset.params["rho"][i]
            )
            # Compute the OLS coefficients
            w, _, _, _ = np.linalg.lstsq(Phi, y_i, rcond=None)

            # Extract the shape summary
            shape_summary = extract_shape_summary(
                        w.reshape(-1, 1), C, breakpoints,
                        zeta_rel=zeta_rel, 
                        upsilon_rel_1=u1_rel, 
                        upsilon_rel_2=u2_rel,
                        upsilon_rel_prune=upsilon_rel_prune, 
                        do_prune=do_prune,
            )
            summaries.append(shape_summary)

        return summaries

    def get_true_shape_summary(self, upsilon_rel_1, upsilon_rel_2):
        """Return analytic ground-truth shape summaries using the DGP.

        Returns:
            A D-length list of shape summaires
        """
        return get_analytic_shape_summary(self.dataset, upsilon_rel_1, upsilon_rel_2)


    # -------------------------------------------------------------------- #
    # --------------------- Extraction Scoring Machinery ----------------- #
    # -------------------------------------------------------------------- #

    def exact_sequence_match(self, predicted_summaries, true_summaries):
        """Compute the share of samples whose predicted STATE SEQUENCE (no timing) 
            exactly matches the ground truth.

        This function only compares the ordered sequence of shape states. 
        It ignroes transition times.

        Args:
            predicted_summaries: list of length D; predicted shape summaries
                [(state, start_time), ...].
            true_summaries: list of length D; reference (analytic) shape summaries.

        Returns:
            float in [0, 1]; the fraction of individuals with an exact state-sequence
                match (0.0 if the lists are empty).
        """
        D = len(predicted_summaries)
        if D == 0:
            return 0.0
        matches = 0
        for pred, true in zip(predicted_summaries, true_summaries):
            pred_states = [state for state, _ in pred]
            true_states = [state for state, _ in true]
            if pred_states == true_states:
                matches += 1
        return matches / D


    def extraction_accuracy(self, nr_obs, nr_interior_knots, u1_rel, u2_rel):
        """Exact-sequence-match accuracy for one extraction configuration.

        Extracts shapes at the given (nr_obs, nr_interior_knots, u1_rel, u2_rel) and
        grades their state sequences against the analytic ground truth (evaluated at
        the SAME thresholds, so both speak one vocabulary).

        Args:
            evaluator: an ExtractionEvaluator (holds the dataset).
            nr_obs: number of observation times for the OLS fit.
            nr_interior_knots: number of interior knots of the spline basis.
            u1_rel: relative slope threshold.
            u2_rel: relative curvature threshold.

        Returns:
            float in [0, 1]; the exact-sequence-match share.
        """
        predicted = self.shape_summaries_at(nr_obs, nr_interior_knots, u1_rel, u2_rel)
        truth = self.get_true_shape_summary(u1_rel, u2_rel)
        return self.exact_sequence_match(predicted, truth)

    # -------------------------------------------------------------------- #
    # --------------------- Shape Extraction Analysis -------------------- #
    # -------------------------------------------------------------------- #

    def sweep_observations(self, variable_nr_obs, fixed_nr_knots, fixed_u1_rel, fixed_u2_rel):
        """Evaluate shape-extraction accuracy across varying observation densities.

        Evaluate how extraction accuracy varies as we vary the number of observations
        nr_obs. The other inputs (nr_knots, u1_rel, u2_rel) remain fixed.
        
        Note: We compare the analytic shape summart and the shape summary our extractor
        produces based on the data generating process. No model is involved. 

        Args:
            variable_nr_obs (list): The observation number counts over which to sweep.
            fixed_nr_knots (int): The number of interior knots (fixed).
            fixed_u1_rel (float): Relative slope threshold (fixed).
            fixed_u2_rel (float): Relative curvature threshold (fixed).

        Returns:
            list of (nr_obs, accuracy) pairs.
        """
        return [
            (n, self.extraction_accuracy(n, fixed_nr_knots, fixed_u1_rel, fixed_u2_rel))
            for n in variable_nr_obs
        ]


    def sweep_knots(self, fixed_nr_obs, variable_nr_knots, fixed_u1_rel, fixed_u2_rel):
        """Evaluate shape-extraction accuracy across varying spline complexities.

        Args:
            fixed_nr_obs (int): The observation number count (fixed). 
            variable_nr_knots (list): The interior knot counts over which to sweep.
            fixed_u1_rel (float): Relative slope threshold (fixed).
            fixed_u2_rel (float): Relative curvature threshold (fixed).

        Returns:
            list of (nr_interior_knots, accuracy) pairs.
        """
        return [
            (k, self.extraction_accuracy(fixed_nr_obs, k, fixed_u1_rel, fixed_u2_rel))
            for k in variable_nr_knots
        ]

    # -------------------------------------------------------------------- #
    # --------------------- Mismatch Exploration ------------------------- #
    # -------------------------------------------------------------------- #

    def find_mismatches(self, nr_obs, nr_interior_knots, u1_rel, u2_rel):
        """Collect the individuals whose predicted shape sequence misses the truth.

        Run extraction at the given configuration and return a record for every
        individual whose predicted STATE SEQUENCE differs from the analytic truth.

        Args:
            nr_obs: number of observation times for the OLS fit.
            nr_interior_knots: number of interior knots of the spline basis.
            u1_rel: relative slope threshold.
            u2_rel: relative curvature threshold.

        Returns:
            list of dicts, one per mismatch, each with keys:
                "i": individual index into self.dataset,
                "true": the analytic ground-truth shape summary,
                "predicted": the extracted shape summary,
                "config": {"nr_obs", "nr_interior_knots", "u1_rel", "u2_rel"}.
        """
        predicted = self.shape_summaries_at(nr_obs, nr_interior_knots, u1_rel, u2_rel)
        truth = self.get_true_shape_summary(u1_rel, u2_rel)

        config = {"nr_obs": nr_obs, "nr_interior_knots": nr_interior_knots,
                  "u1_rel": u1_rel, "u2_rel": u2_rel}
        mismatches = []
        for i, (pred, true) in enumerate(zip(predicted, truth)):
            pred_states = [state for state, _ in pred]
            true_states = [state for state, _ in true]
            if pred_states != true_states:
                mismatches.append({"i": i, "true": true, "predicted": pred,
                                   "config": config})
        return mismatches

    def iterate_mismatches(self, nr_obs, nr_interior_knots, u1_rel, u2_rel):
        """Yield mismatch records one at a time (for clicking through in a notebook).

        Yields:
            mismatch record dicts (see find_mismatches).
        """
        yield from self.find_mismatches(nr_obs, nr_interior_knots, u1_rel, u2_rel)



class ModelEvaluator:
    """Assess a trained model's fit and shape extraction on a simulated test dataset.

    Args:
        inference_engine: An InferenceEngine object
        test_dataset: A simulated dataset to evaluate model performance
    """
    def __init__(self, inference_engine, test_dataset):
        self.inference_engine = inference_engine
        self.dataset = test_dataset

    def _get_individual_covariates(self):
        """Return the covariates X as a D-length list of individual feature dictionaries.
    
        self.dataset.X stores individual covariates as:
            {
                "size":  [person1, person2, person3, ...],
                "age":   [person1, person2, person3, ...],
                ...
            }
        We modify this into a list of feature values for individuals:  
            [
                {"size": float, "age": float, "weight": float, "dosage": float}, # Person 1
                {"size": float, "age": float, "weight": float, "dosage": float}, # Person 2
                ...
            ]

        Returns:
            A list containing feature dictionaries for alll individuals.
        """
        names = list(self.dataset.X.keys()) 
        
        return [{name: self.dataset.X[name][i] for name in names} for i in range(self.dataset.D)]

    def get_shape_summary_estimates(self):
        """Predicted shape summaries for every individual in the dataset.

        Returns:
            list of length D of shape summaries [(state, start_time), ...].
        """
        return [self.inference_engine.predict_mean_shape_summary(x) for x in self._get_individual_covariates()]


    def get_trajectory_estimates(self, evaluation_times):
        """Compute the continuous predicted curves for every individual in the dataset.

        Args:
            evaluation_times (np.ndarray or list): 1-D sequence of time points 
                (floats) in [0, T] at which to evaluate the mean trajectory. 
                This is used for plotting.

        Returns:
            np.ndarray: A 2D array of shape (D, len(evaluation_times)) containing 
                the predicted trajectories for all individuals.
        """
        return np.array([
            self.inference_engine.predict_trajectory_values(x, evaluation_times).detach().cpu().numpy()
            for x in self._get_individual_covariates()
        ])

    def get_true_trajectories(self, evaluation_times):
        """Compute the noise-free ground-truth trajectories for every test individual.

        Args:
            evaluation_times (shape (T,) np.ndarray): time points in [0, T] at which to 
                evaluate each individual's noise-free trajectory.

        Returns:
            np.ndarray of shape (D, T) where row i is individual i's noise-free
                trajectory at evaluation_times.
        """
        return np.array([
            compute_wilkerson(
                evaluation_times,
                self.dataset.X["size"][i], 
                self.dataset.params["g"][i],
                self.dataset.params["d"][i], 
            self.dataset.params["rho"][i],
            )
            for i in range(self.dataset.D)
        ])

    def _compute_RMSE(self, predicted, true):
        """Compute the root mean squared error (RMSE) 

        Compute the RMSE between all predicted and true (noise-free) 
        trajectories for all samples in a dataset. 

        Args:
            predicted: np.ndarray of shape (D, T); predicted trajectory values.
            true: np.ndarray of shape (D, T); ground-truth trajectory values.

        Returns:
            rmse (float): the root mean squared error, pooled over all D*T entries.
        """
        mse = np.mean((predicted - true) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def _compute_R_squared(self, predicted, true):
        """Compute the R-squared value for the predicted trajectory.

        Recall that R^2 = 1 - RSS/TSS

        Args:
            predicted: np.ndarray of shape (D, T); predicted trajectory values.
            true: np.ndarray of shape (D, T); ground-truth trajectory values.

        Returns:
            r_squared (float): r_squared value. 
        """
        rss = np.sum((true - predicted) ** 2)
        tss = np.sum((true - true.mean()) ** 2)
        r_squared = 1 - (rss/tss)
        return r_squared


    def compute_value_space_accuracy(self, evaluation_times):
        """Compute RMSE and R-squared values for the given model.

        For the trained model and given the ground truth data, compute the 
        root mean squared error and the R-squared score for the given model
        
        Args:
            evaluation_times: np.ndarray of shape (T,); time points at which
                to evaluate both predicted and true trajectories for scoring.

        Returns:
            dict with keys "rmse" (float) and "r2" (float); the value-space
                fit metrics for this model's predictions on the test dataset.
        """
        predicted = self.get_trajectory_estimates(evaluation_times)
        true = self.get_true_trajectories(evaluation_times)
        rmse = self._compute_RMSE(predicted, true)
        r_squared = self._compute_R_squared(predicted, true)
        return {"rmse": rmse, "r2": r_squared}






