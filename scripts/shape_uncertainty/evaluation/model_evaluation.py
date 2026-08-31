import torch
import numpy as np
from collections import defaultdict

from scripts.shape_uncertainty.shape_extraction.shape_distance import (
    mean_shape_sequence_match,
    mean_shape_summary_distance,
)

from scripts.shape_uncertainty.evaluation.evaluation_metrics import (
    compute_RMSE, compute_individual_RMSE,
    compute_R_squared, compute_individual_R_squared)


class ModelEvaluator:
    """Assess a trained model's fit and shape extraction on a simulated test dataset.

    Args:
        inference_engine: An InferenceEngine object
        test_dataset: A simulated dataset to evaluate model performance
    """
    def __init__(self, inference_engine, test_dataset):
        self.inference_engine = inference_engine
        self.dataset = test_dataset

        self.predicted = None

    
    def get_trajectory_estimates(self):
        """Compute predicted trajectories at each individual's own observation times.

        Args:
            evaluation_times (np.ndarray or list): 1-D sequence of time points 
                (floats) in [0, T] at which to evaluate the mean trajectory. 
                This is used for plotting.

         Returns:
            list of length D; element i is the model's predicted values at
                self.dataset.times[i], as an np.ndarray of shape (N_i,).
        """
        # Compute predictions only if they have not been computed before 
        if self.predicted is None:
            covariates = self.dataset.individual_covariates()
            times = self.dataset.times

            self.predicted = [
                self.inference_engine.predict_trajectory(x, t)[0]
                for x, t in zip(covariates, times)
            ]

        # Return the saved list of predicted trajectories
        return self.predicted

    def compute_value_space_metrics(self, predicted=None, target="observed"):
        """Compute pooled and per-individual RMSE and R-squared of predicted
        vs reference (noise-free or observed).

        Compare each individual's predicted trajectory against its stored
        reference trajectory Y_reference, both evaluated at that individual's own
        observation times. Options are:
         - "observed": we score against the held-out noisy observations. This is a check a 
            practitioner would run before deploying a model.
         - "latent": we score against the noise-free trajectories. This is a imulation-only
            diagnostic
        
        We then read the residuals in two ways:
            - We compute population-level metrics by pooling residuals
             across individuals. This weights each individual by the 
             scale of its outcomes. 
            - We compute individual level metrics and average across 
              individuals. This gives every individual equal weight. As a
              result small-amplitude individual that is fitted badly is 
              not masked by a large-amplitude individual that is fitted well
              as is the case with the population level metric
        
        Args:
            reference (str): "observed" or "latent". Defaults to "observed".

        Returns:
            dict with keys:
                "rmse" (float): RMSE pooled over all D * N_i residuals.
                "r2" (float): R-squared pooled over all D * N_i residuals,
                    taken about the population mean.
                "mean_individual_rmse" (float): (1 / D) * sum_i RMSE_i.
                "median_individual_rmse" (float): the median of RMSE_i.
                "mean_individual_r2" (float): (1 / D*) * sum_i R^2_i, averaged
                    over the D* individuals whose R-squared is defined.
                "median_individual_r2" (float): the median of R^2_i over the
                    same D* individuals.
                "individual_rmse" (np.ndarray): shape (D,); RMSE_i.
                "individual_r2" (np.ndarray): shape (D,); R^2_i, np.nan where
                    the individual's reference trajectory is constant.
            
        Note:
        The two returned arrays follow the dataset's individual ordering, 
        so they can be indexed alongside self.dataset to annotate a 
        figure with the metrics of the individual it plots.
        """
        # Step 0: Get preference trajectories at the dataset observations times
        if target == "latent":
            reference = self.dataset.Y_true
        elif target == "observed":
            reference = self.dataset.Y_noisy
        else:
            raise ValueError(f"target must be 'observed' or 'latent', got {target!r}")

        # Step 1: Get predicted trajectories at the dataset observations times if not provided
        if predicted is None:
            predicted = self.get_trajectory_estimates()

        # Step 2: Score each individual on its own observation times
        individual_rmse = compute_individual_RMSE(predicted, reference)
        individual_r_squared = compute_individual_R_squared(predicted, reference)

        # Step 3: Pool residuals across individuals (handles ragged N_i)
        predicted_all = np.concatenate([np.asarray(p) for p in predicted])
        reference_all = np.concatenate([np.asarray(y) for y in reference])

        # Compute pooled RMSE and R-squared
        rmse = compute_RMSE(predicted_all, reference_all)
        r_squared = compute_R_squared(predicted_all, reference_all)

        # Return value space metrics
        metrics = {
            "target": target,
            "rmse": rmse,
            "r2": r_squared,
            "mean_individual_rmse": float(np.mean(individual_rmse)),
            "median_individual_rmse": float(np.median(individual_rmse)),
            "mean_individual_r2": float(np.nanmean(individual_r_squared)),
            "median_individual_r2": float(np.nanmedian(individual_r_squared)),
            "individual_rmse": individual_rmse,
            "individual_r2": individual_r_squared,
        }

        return metrics

    def compute_shape_space_metrics(self, true_summaries):
        """Score the model's predicted shape summaries against the analytics summaries.

        Args:
            true_summaries: list of length D of true shape summaries,
                extracted at the same thresholds as the engine's, so both are
                expressed in one vocabulary.

        Returns:
            dict with keys "accuracy" (exact state-sequence match share, higher
                is better) and "distance" (mean shape-summary distance, lower is
                better).
        """
        predicted = self.inference_engine.predict_mean_shape_summary(self.dataset.X)
        return {
            "accuracy": mean_shape_sequence_match(predicted, true_summaries),
            "distance": mean_shape_summary_distance(predicted, true_summaries,
                                                    self.dataset.T),
        }



class EnsembleEvaluator:
    """Evaluate the performance of every member of an ensemble on a test dataset.

    Args:
        engines (list): A list of InferenceEngine objects, representing each 
            ensemble member.
        test_dataset (object): A simulated dataset used to evaluate model performance.
    """

    def __init__(self, engines, test_dataset):
        # Store the ensemble members and the evaluation dataset
        self.engines = engines
        self.dataset = test_dataset

    def compute_member_value_space_metrics(self, targets="observed"):
        """Score each ensemble member against the specified target reference(s).

        This calculates the R-squared and RMSE for every ensemble member. It is 
        optimised to perform exactly one forward pass per member, even when 
        scoring against multiple targets.

        Args:
            targets (str or iterable of str, optional): The reference target(s) 
                to score against. Accepts "observed", "latent", or a tuple 
                of both. Defaults to "observed".

        Returns:
            dict: A dictionary mapping the computed metric names (e.g., "observed_r2", 
                "latent_rmse") to a 1D numpy array of shape (M,), where M is the 
                number of ensemble members. Arrays preserve the engine order.
        """
        # Step 1: If the user passed a single word, wrap it in a list
        if isinstance(targets, str):
            targets = [targets]


        results = defaultdict(list)
        for engine in self.engines:
            # Step 2: Create a model evaluator for the given engine
            evaluator = ModelEvaluator(engine, self.dataset)
            
            # Step 3: Run the forward pass once per member
            predictions = evaluator.get_trajectory_estimates()
            
            # Step 4: Score the predictions against the specified target(s)
            for target in targets:
                scores = evaluator.compute_value_space_metrics(predicted=predictions, target=target)
                
                results[f"{target}_r2"].append(scores["r2"])
                results[f"{target}_rmse"].append(scores["rmse"])

        # Step 5: Convert the final accumulated lists into numpy arrays
        return {metric: np.array(values) for metric, values in results.items()}

    def summarise_member_metrics(self, member_metrics):
        """Calculates summary statistics across all ensemble members.

        Args:
            member_metrics (dict): A dictionary mapping metric names to numpy arrays 
                of scores, exactly as returned by `compute_member_value_space_metrics`.

        Returns:
            dict: A nested dictionary where each top-level key is a metric name 
                (e.g., "observed_r2"), mapping to an inner dictionary containing 
                its "mean", "median", and "std" (sample standard deviation) as floats.
        """
        return {
            metric_name: {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores, ddof=1))
            }
            for metric_name, scores in member_metrics.items()
        }