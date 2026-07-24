import torch
import numpy as np

class ModelEvaluator:
    """Assess a trained model's fit and shape extraction on a simulated test dataset.

    Args:
        inference_engine: An InferenceEngine object
        test_dataset: A simulated dataset to evaluate model performance
    """
    def __init__(self, inference_engine, test_dataset):
        self.inference_engine = inference_engine
        self.dataset = test_dataset

    
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
        covariates = self.dataset.individual_covariates()
        times = self.dataset.times
        return [
            self.inference_engine.predict_trajectory_values(x, t).detach().cpu().numpy()
            for x, t in zip(covariates, times)
        ]

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

    def compute_value_space_accuracy(self):
        """Compute RMSE and R-squared of predicted vs true noise-free values.

        Compares each individual's predicted trajectory against its stored
        noise-free trajectory Y_true, both evaluated at that individual's own
            observation times, then pools the residuals across all individuals.
        
        Returns:
            dict with keys "rmse" (float) and "r2" (float).
        """
        # Get predicted and true trajectories at the dataset observations times
        predicted = self.get_trajectory_estimates()
        true = self.dataset.Y_true

        # Pool residuals across individuals (handles ragged N_i)
        predicted_all = np.concatenate([np.asarray(p) for p in predicted])
        true_all = np.concatenate([np.asarray(y) for y in true])

        # Compute RMSE and R-squared
        rmse = self._compute_RMSE(predicted_all, true_all)
        r_squared = self._compute_R_squared(predicted_all, true_all)

        # Return value space metrics
        return {"rmse": rmse, "r2": r_squared}






