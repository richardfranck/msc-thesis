import numpy as np
import torch

from scripts.shape_uncertainty.spline_basis.bspline_basis import basis_matrix

class Normaliser:
    """Standardise a dataset's covariate dict into a fixed-order (D, M) tensor.
        and normalise the outcomes. 

    Attributes:
        eps: float, lower floor on each std to avoid division by zero for
            constant (zero-variance) covariates.
        names: tuple of covariate names in the fixed column order (None until fit).
        mean: np.ndarray of shape (M,); per-covariate training means (None until fit).
        std: np.ndarray of shape (M,); per-covariate training stds, floored at eps
            (None until fit).
        y_mean: float; mean of the outcome variable across all individuals and observations (None until fit_y).
        y_std: float; std of the outcome variable across all individuals and observations (None until fit_y).
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.names = None
        self.mean = None
        self.std = None
        self.y_mean = None
        self.y_std = None

    @property
    def is_fitted(self):
        """True once fit() has recorded the column order and statistics."""
        return self.names is not None

    @property
    def is_y_fitted(self):
        """True once fit_y() has recorded the outcome standardisation statistics."""
        return self.y_mean is not None

    def fit(self, dataset):
        """Learn feature order, feature means, and feature std devs from the TRAINING dataset.

        Args:
            dataset: a dataset with a covariate dict X mapping each covariate 
            name to an np.ndarray of shape (D,) holding that covariate's 
            value for all D samples.

        Returns:
            self (fitted), so calls can be chained.
        """
        # Step 1: Freeze the column order as the dataset's covariate keys.
        self.names = tuple(dataset.X.keys())
        
        # Step 2: Create a (D, M) matrix with columns in the fixed order.
        # Create a list of arrays in the fixed order
        ordered_columns_list = [np.asarray(dataset.X[name], dtype=float) for name in self.names]
        # Turn the list of arrays into a  (D, M) array
        ordered_X = np.column_stack(ordered_columns_list)

        # Step 3: Compute the per-covariate mean
        self.mean = ordered_X.mean(axis=0)

        # Step 4: Compute the per-covariate std. Floor the value at self.epsilon
        self.std = np.maximum(ordered_X.std(axis=0), self.epsilon)

        # Return the fitted Normaliser object
        return self

    def transform(self, X):
        """Map a covariate dict X to a normalised (D, M) tensor in the fitted order.

        Args:
            X: dict X mapping each covariate name to an np.ndarray of shape (D,) 
                holding that covariate's value for all D samples.

        Returns:
            torch.Tensor of shape (D, M); the standardised covariates, columns in
            the fitted order.
        """
        # Step 1: Check normaliser is fitted 
        if not self.is_fitted:
            raise RuntimeError("Normaliser must be fit before transform.")
        
        # Step 2: Standardise X with the stored training statistics and return an ordered tensor
        ordered_columns_list = [np.asarray(X[name], dtype=float) for name in self.names]
        ordered_X = np.column_stack(ordered_columns_list)
        Z = (ordered_X - self.mean) / self.std

        return torch.as_tensor(Z, dtype=torch.float64)

    def fit_transform(self, dataset):
        """Fit on dataset and then transform its covariates.

        Args:
            dataset: the training dataset to fit on and transform.

        Returns:
            torch.Tensor of shape (D, M); the standardised training covariates.
        """
        return self.fit(dataset).transform(dataset.X)

    def transform_one(self, x):
        """Map a single covariate dict x to a normalised (M,) tensor in the fitted order.

        This function is a single-vector counterpart of the transform method. 
        Note that while transform takes a dict of (D,) arrays and returns (D, M), 
        this method takes a dict of scalar covariate values for one individual 
        and returns the standardised (M,) vector, using the same stored training 
        statistics and fitted column order.

        Args:
            x: dict mapping each covariate name to a scalar value for one
                individual.

        Returns:
            torch.Tensor of shape (M,); the standardised covariates in the fitted
                order.
        """
        # Step 1: Check normaliser is fitted
        if not self.is_fitted:
            raise RuntimeError("Normaliser must be fit before transform_one.")

        # Step 2: Assemble the covariates in fitted order, standardise, return an ordered tensor
        ordered_vector = np.array([float(x[name]) for name in self.names])
        z = (ordered_vector - self.mean) / self.std

        return torch.as_tensor(z, dtype=torch.float64)

    def fit_y(self, dataset):
        """Learn target global mean and std from the TRAINING dataset's observations.

        Args:
            dataset: a dataset with a length-D list Y_noisy where
                Y_noisy[i] is the (N_i,) array of individual i's observations.

        Returns:
            self (fitted), so calls can be chained.
        """
        # Step 1: Combine all individual 1D arrays ((N_i,)) into one single 1D array
        all_y = np.concatenate(dataset.Y_noisy).astype(float)

        # Step 2: Compute the mean across all y in the dataset
        self.y_mean = float(all_y.mean())

        # Step 3: Compute the std across all y in the dataset (floor at self.epsilon)
        self.y_std = float(max(all_y.std(), self.epsilon))

        return self

    def transform_y(self, y):
        """Standardise a given outcome target array with the fitted stats.

        Apply z = (y - y_mean) / y_std using the statistics recorded by fit_y.

        Args:
            y: np.ndarray of target values.

        Returns:
            np.ndarray of the same shape; the standardised targets.
        """
        if not self.is_y_fitted:
            raise RuntimeError("Call fit_y before transform_y.")

        return (y - self.y_mean) / self.y_std

    def inverse_transform_y(self, y_norm):
        """Map standardised outcomes or predictions to original units.

        Args:
            y_norm: np.ndarray of standardised targets or predictions.

        Returns:
            np.ndarray of the same shape, expressed in the original target units.
        """
        if not self.is_y_fitted:
            raise RuntimeError("Call fit_y before inverse_transform_y.")

        return y_norm * self.y_std + self.y_mean


class Batch:
    """Model-ready training data for one set of individuals.

    Attributes:
        X: torch.Tensor of shape (D, M); the NORMALISED static covariates, in the
            column order fixed by the fitted normaliser. Used by the encoder and
            the mean-trajectory roughness term.
        X_list: list of length D; X_list[i] is the (M,) NORMALISED covariate
            vector for individual i (a row of Z, exposed per-individual for the
            per-individual marginal likelihood).
        Phi_list: list of length D; Phi_list[i] is the (N_i, B) B-spline design
            matrix at individual i's observation times.
        y_list: list of length D; y_list[i] is the (N_i,) tensor of individual
            i's observed noisy trajectory values.
    """
    def __init__(self, X, x_list, Phi_list, y_list):
        self.X = X
        self.x_list = x_list
        self.Phi_list = Phi_list
        self.y_list = y_list

    @property
    def D(self):
        """Number of individuals in the batch."""
        return len(self.y_list)


    def subbatch(self, indices):
        """Create a Batch object containing only selected individuals.

        Args:
            indices: sequence of ints (list, range, or numpy array); the
                individual indices to keep, in the desired order.

        Returns:
            Batch: a new Batch over the selected individuals, with X of shape
                (len(indices), M) and per-individual lists of that length.
        """
        ids = list(indices)

        X_sub = self.X[ids]
        x_list_sub = [self.x_list[i] for i in ids]
        Phi_list_sub = [self.Phi_list[i] for i in ids]
        y_list_sub = [self.y_list[i] for i in ids]

        return Batch(X_sub, x_list_sub, Phi_list_sub, y_list_sub)


def prepare_training_data(dataset, basis, normaliser):
    """Convert a SimulatedDataset into a model-ready Batch.

    Transform a SimulatedDataset instance in into a training ready Batch
    instance containing:
        - a normalised the covariate (D, M) tensor, 
        - per-individual covaritates in x_list
        - per individual B-spline design matrices Phi_i (evaluated at N_i)
        - per-individual outcomes in y_list
    
    Args:
        dataset: a SimulatedDataset; its model-visible tier (X, times, Y_noisy)
            is used (covariates, observation times, noisy observations).
        basis: the B-spline basis functions used to build each design matrix
            Phi_i = basis_matrix(times_i, basis).
        normaliser: a fitted covariate normaliser exposing a transform that maps
            the raw covariate dict to a normalised (D, M) tensor in
            COVARIATE_NAMES order.

    Returns:
        Batch: the prepared training data (normalised X, per-individual x_i,
            precomputed Phi_i, and observed y_i).
    """
    # Step 1: Fit the data normaliser on the dataset and transform
    if not normaliser.is_fitted:
        X = normaliser.fit_transform(dataset) # Fit & transform
    else:
        X = normaliser.transform(dataset.X) # Transform

    # Step 1b: Fit the normaliser on the first call, i.e. during training 
    if not normaliser.is_y_fitted:
        normaliser.fit_y(dataset)

    # Step 2: Get a list with each individuals's normalised covariate row
    x_list = [X[i] for i in range(X.shape[0])]

    # Step 3: Get each individual's design matrix Phi_i and outcomes y_i
    Phi_list = list()
    y_list = list()

    for i in range(dataset.D):
        # Get each individual's design matrix Phi_i
        observation_times = dataset.times[i] # (N_i,) np.ndarray
        Phi_i = basis_matrix(observation_times, basis) # (N_i, B) np.ndarray  
        Phi_i = torch.as_tensor(Phi_i, dtype=torch.float64) # (N_i, B) tensor
        Phi_list.append(Phi_i)

        # Get each individual's outcomes y_i
        y_i = normaliser.transform_y(dataset.Y_noisy[i]) # (N_i,) np.ndarray
        y_i = torch.as_tensor(y_i, dtype=torch.float64) # (N_i,) tensor
        y_list.append(y_i)

    return Batch(X, x_list, Phi_list, y_list)