import math
import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}


class Encoder(nn.Module):
    """Map static covariates to cubic B-spline coefficients.

    This class specifies a fully-connected network NN (h_theta) 
    that maps an individual's M static covariates to the B = K + 4
    cubic B-spline coefficients describing their mean trajectory 
    (K interior knots). The exact architecture (number and width 
    of hidden layers and activation) is determined via 
    hyperparameter tuning.

    Args:
        nr_covariates: Number M of static covariates.
        nr_basis: Number B of B-spline basis functions and output
            coefficients.
        hidden_sizes: Width of each hidden layer. Its length determines
            the network depth.
        activation: Activation function used in every hidden layer.
        dropout: Dropout probability used after each hidden-layer
            activation.
    """
    def __init__(self, nr_covariates, nr_basis, hidden_sizes=(32, 64, 32),
                 activation="relu", dropout=0.0):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Activation '{activation}' must be one of {set(ACTIVATIONS)}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must satisfy 0 <= dropout < 1.")

        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            raise ValueError("All hidden-layer sizes must be positive.")

        # Define the components of the neural network
        activation_class = ACTIVATIONS[activation]

        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = nr_covariates

        for hidden_dim in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.activations.append(activation_class())
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

            in_dim = hidden_dim

        # Define the output layer that produces the B spline coefficients.
        # No batch normalization, activation, or dropout after this layer.
        self.output = nn.Linear(in_dim, nr_basis)


    # Forward pass
    def forward(self, X):
        """Map a batch of covariates to spline coefficients.

        Args:
            X: Tensor of shape (batch, M), containing the normalized
                static covariates.

        Returns:
            Tensor of shape (batch, B), containing the spline coefficients
            h_theta(x).
        """
        layers = zip(
            self.hidden_layers,
            self.batch_norms,
            self.activations,
            self.dropouts,
        )

        # Pass data through the layers
        for linear, batch_norm, activation, dropout in layers:
            X = linear(X)
            X = batch_norm(X)
            X = activation(X)
            X = dropout(X)

        # Pass data through the output layer
        coeffs = self.output(X)
        return coeffs


class RandomEffectsModel(nn.Module, ABC):
    """Abstract base class for a spline forecasting model with random effects.

    This class defines the randome effects model structure that is shared by
    by all model variants, irrespective of their specific assumption regarding
    the distribution of the random effects. That is:
        y_i = Phi_i (h_theta(x_i) + u_i) + epsilon_i,

    Shared across subclasses are:
        1. the encoder
        2. the number of basis functions
        3. the trainable measurement-noise parameter log_sigma
        4. the NN coefficient predictions
        5. the measurement-noise standard deviation
        6. the measurement-noise variance

    Required from subclasses are:
        1. marginal log-likelihood  of y_i
        2. penalty for wiggles in the likelihood

    Args:
        encoder: an Encoder instance (the mean map h_theta).
        nr_basis: int, B, the spline coefficient dimension.
    """
    def __init__(self, encoder, nr_basis, sigma_init=0.1):
            super().__init__()
            self.encoder = encoder
            self.nr_basis = nr_basis
            # Unconstrained measurement-noise parameter (positivity via exp)
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))

    def get_coefficients(self, X):
        """Return the mean spline coefficients h_theta(x) for a batch of covariates.

        Args:
            X: torch.Tensor of shape (batch, M); normalised covariates.

        Returns:
            torch.Tensor of shape (batch, B); the mean coefficients.
        """
        return self.encoder(X)

    def get_noise_std(self):
        """Return the measurement-noise std. dev. sigma."""
        return torch.exp(self.log_sigma)

    def get_noise_variance(self):
        """Return th measurement-noise variance sigma^2."""
        return torch.exp(2.0 * self.log_sigma)

    @abstractmethod
    def marginal_log_likelihood(self, y_i, Phi_i, x_i):
        """Marginal log-likelihood of one individual's observations.

        Args:
            y_i: torch.Tensor of shape (N_i,); the individual's observations.
            Phi_i: torch.Tensor of shape (N_i, B); the spline design matrix at
                the individual's observation times.
            x_i: torch.Tensor of shape (M,); the individual's covariates.

        Returns:
            scalar torch.Tensor; log p(y_i | x_i) under the model.
        """
        raise NotImplementedError

    @abstractmethod
    def penalty(self, Omega, X, lambda_mean, lambda_re):
        """Total weighted smoothness penalty for this random-effects family.

        Args:
            Omega: torch.Tensor of shape (B, B); the spline penalty matrix.
            X: torch.Tensor of shape (D, M); the batch of covariates.
            lambda_mean: float, weight on the mean-trajectory roughness.
            lambda_re: float, weight on the random-effects roughness.

        Returns:
            scalar torch.Tensor; the total weighted smoothness penalty.
        """
        raise NotImplementedError


class GaussianModel(RandomEffectsModel):
    """Random-effects model with Gaussian random effect.

    In this model we assume the random effects u_i ~ N(0, Sigma). As a result,
        y_i | x_i ~ N( Phi_i h_theta(x_i),  Phi_i Sigma Phi_i^T + sigma^2 I ),
    such that the log-likelihood has a closed-form. To things to note are:

    In the log-likelihood, we parameterise Sigma through its Cholesky factor L 
    (Sigma = L L^T), where L is a lower triangual matrix. To ensure that Sigma 
    is symmetric positive-definite and optimisation problem is unconstrianed we
    parameterise the elements on the main diagonal as the exponensial.
    For example,
            [[l_11, 0   , 0    ],     [[exp(d_1), 0       , 0       ],
        L = [ l_21, l_22, 0    ],   = [a_21,      exp(d_2), 0       ],
            [ l_31, l_32, l_33]]      [a_31,      a_32    , exp(d_3)]]
    or generally,
        l_ij = exp(d_i)    if i == j,
        l_ij = a_ij        if i > j,
        l_ij = 0           if i < j.

    We solve for the unconstrained parameters:
        a_ij = l_ij        for i > j,
        d_i  = log(l_ii)   for i == j.

    We add a roughness penalty that penalises wiggles in the forecast trajectory
    as measured by the integral over the trajectories squared second derivatives. 
    Expected roughness splits into two components that we regularise separately:
        mean-trajectory roughness  h_theta(x_i)^T Omega h_theta(x_i)  [depends on theta]
        random-effects roughness   tr(Omega Sigma)                    [depends on Sigma]

    Args:
        encoder: an Encoder instance.
        nr_basis (int): Number B of spline basis functions and coefficient dimensions.
        sigma_init (float): Strictly positive initial value of the measurement-noise
            standard deviation sigma.
        sigma_cov_init (float): Strictly positive initial value initial random-effects
            standard deviations. 
    """

    def __init__(self, encoder, nr_basis, sigma_init=0.1, sigma_cov_init=0.1):
        super().__init__(encoder, nr_basis, sigma_init=sigma_init)
        

        if sigma_cov_init <= 0:
            raise ValueError("sigma_cov_init must be positive.")

        # Get indices of the row and column indices of L excluding the main diagonal.
        B = self.nr_basis
        self._row_indices, self._col_indices = torch.tril_indices(B, B, offset=-1) # # For 3x3 the indices are row = [1, 2, 2] and col = [0, 0, 1]. 

        # Initialise the trainable strictly lower-triangular elements a_ij.
        nr_offdiagonal_elements = self._row_indices.numel() # the number of a_ij parameters
        self.offdiag = nn.Parameter(torch.zeros(nr_offdiagonal_elements)) # vector of zeros of length nr_offdiagonal_elements

        # Initialise B trainable parameters d_i, initialised to log(sigma_cov_init) so that exp(d_i) equals sigma_cov_init.
        initial_log_diagonal = math.log(float(sigma_cov_init))
        self.log_diag = nn.Parameter(
            torch.full(size=(B,), fill_value=initial_log_diagonal)
        )
   
    def cholesky_factor(self):
        """Assemble the lower-triangular BxB Cholesky factor L with diag = exp(log-diag)."""
        B = self.nr_basis
        # Create matrix of zeros that matches the data type and hardware location of log_diag.
        L = torch.zeros(B, B, dtype=self.log_diag.dtype, device=self.log_diag.device)
        
        # Replace main diagonal with the elements of self.log_diag.
        L[range(B), range(B)] = torch.exp(self.log_diag)

        # Replace the lower triangular part with the elements a_ij of self.offdiag.
        L[self._tril_row, self._tril_col] = self.offdiag

        return L

    def sigma_matrix(self):
        """Return the (B, B) aleatoric covariance Sigma = L L^T (symmetric PD)."""
        L = self.cholesky_factor()
        return L @ L.T

    def marginal_log_likelihood(self, y_i, Phi_i, x_i):
        """Closed-form Gaussian marginal log-likelihood for one individual.

        With Gaussian random effects we have
            y_i | x_i ~ N(mu_i, V_i),
        where
            mu_i = Phi_i h_theta(x_i),
            V_i  = Phi_i Sigma Phi_i^T + sigma^2 I

        Args:
            y_i: torch.Tensor of shape (N_i,); the individual's observed
                trajectory values at its observation times.
            Phi_i: torch.Tensor of shape (N_i, B); the B-spline design matrix
                evaluated at the individual's observation times.
            x_i: torch.Tensor of shape (M,); the individual's static covariates.

        Returns:
            scalar torch.Tensor; the marginal log-likelihood log p(y_i | x_i).
        """
        # Assemble the random-effects covariance Sigma = L L^T.
        Sigma = self.sigma_matrix()

        # Get the mean spline coefficients h_theta(x_i). Since the encoder expects a batch, we add/remove a batch dimension.
        h = self.encoder(x_i.unsqueeze(0)).squeeze(0)  # (M,) -> (1, M) -> (1, B) -> (B,)

        # Get the mean trajectory mu_i = Phi_i h_theta(x_i) at the N_i observation times.
        mu_i = Phi_i @ h

        # Get the marginal covariance V_i = Phi_i Sigma Phi_i^T + sigma^2 I.
        N_i = y_i.shape[0]
        identity_N_i = torch.eye(N_i, dtype=y_i.dtype, device=y_i.device)
        V_i = Phi_i @ Sigma @ Phi_i.T + self.noise_variance() * identity_N_i

        # Create a multivariate normal distribution parameterised by the mean vector mu_i and the covariance matrix V_i.
        dist = torch.distributions.MultivariateNormal(mu_i, covariance_matrix=V_i)

        # Return log p(y_i|x_i) using the multivariate Gaussian avaluated at the observed y_i.
        return dist.log_prob(y_i)

    def mean_wiggle(self, X, Omega):
        """Mean-trajectory wiggle, averaged over the batch.

        Compute the integrated squared second derivative of each individual's MEAN
        predicted trajectory. That is, the mean over individuals of 
                    h_theta(x_i)^T Omega h_theta(x_i)
        Depends on theta only

        Args:
            X: torch.Tensor of shape (D, M); the batch of (normalised) covariates.
            Omega: torch.Tensor of shape (B, B); the spline penalty matrix
                (Omega = integral of phi''(t) phi''(t)^T dt).

        Returns:
            scalar torch.Tensor; the batch-averaged mean-trajectory wiggle.
        """
        # Mean coefficients for the whole batch
        H = self.encoder(X) # (D, B)

        # Quadratic form h_i^T Omega h_i for every individual and take the mean
        wiggle_per_individual = (H @ Omega * H).sum(dim=1) # (D,)
        mean_wiggle = wiggle_per_individual.mean() # scalar

        return mean_wiggle

    def re_wiggle(self, Omega):
        """Random-effects wiggle tr(Omega Sigma).

        The expected wigggle contributed by the random effects u_i ~ N(0, Sigma).
        Depends on Sigma only.

        Args:
            Omega: torch.Tensor of shape (B, B); the spline penalty matrix.

        Returns:
            scalar torch.Tensor; tr(Omega Sigma).
        """
        return torch.trace(Omega @ self.sigma_matrix())

    def penalty(self, Omega, X, lambda_mean, lambda_re):
        """Total weighted wiggle penalty with separate component knobs.

        Compute
            lambda_mean * mean_wiggle(X, Omega) + lambda_re * re_wiggle(Omega),

        Args:
            Omega: torch.Tensor of shape (B, B); the spline penalty matrix.
            X: torch.Tensor of shape (D, M); the batch of covariates.
            lambda_mean: float, weight on the mean-trajectory wiggles.
            lambda_re: float, weight on the random-effects wiggles.

        Returns:
            scalar torch.Tensor; the total weighted wiggle penalty.
        """
        return (lambda_mean * self.mean_wiggle(X, Omega)
                + lambda_re * self.re_wiggle(Omega))


class MixtureModel:
    pass

