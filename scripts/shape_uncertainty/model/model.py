import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


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
          nr_random_effect_outputs: Number of additional per-individual outputs emitted
            alongside the B coefficients. The Gaussian model takes one to model 
            the log scale of its random effects to model heteroscedastisity.
    """
    def __init__(self, nr_covariates, nr_basis, hidden_sizes=(32, 64, 32),
                 activation="relu", dropout=0.0, nr_random_effect_outputs=0):
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
        self.layer_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = nr_covariates

        for hidden_dim in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.activations.append(activation_class())
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

            in_dim = hidden_dim

        # Define the output layer that produces the B spline coefficients,
        # plus extra per-individual quantities (like s(x) for the random effects 
        # model). Callers slice the result.  
        # No batch normalization, activation, or dropout after this layer.
        self.nr_basis = nr_basis
        self.nr_random_effect_outputs = nr_random_effect_outputs
        self.output = nn.Linear(in_dim, nr_basis + nr_random_effect_outputs)


    # Forward pass
    def forward(self, X):
        """Map a batch of covariates to spline coefficients.

        Args:
            X: Tensor of shape (batch, M), containing the normalized
                static covariates.

        Returns:
            Tensor of shape (batch, B + nr_random_effect_outputs), containing the
            spline coefficients h_theta(x) in the first B columns, followed by any
            random-effects outputs the model asked for.
        """
        layers = zip(
            self.hidden_layers,
            self.layer_norms,
            self.activations,
            self.dropouts,
        )

        # Pass data through the layers
        for linear, layer_norm, activation, dropout in layers:
            X = linear(X)
            X = layer_norm(X)
            X = activation(X)
            X = dropout(X)

        # Pass data through the output layer
        coeffs = self.output(X)
        return coeffs


class RandomEffectsModel(nn.Module, ABC):
    """Abstract base class for a spline forecasting model with random effects.

    This class defines the random effects model structure that is shared by
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
    # How many outputs beyond the B coefficients this model needs from its encoder. 
    NR_RANDOM_EFFECT_OUTPUTS = 0

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
        """Return the measurement-noise variance sigma^2."""
        return torch.exp(2.0 * self.log_sigma)

    def mean_wiggle(self, X, Omega):
        """Compute the mean-trajectory wiggle, averaged over the batch.

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
        H = self.get_coefficients(X) # (D, B)

        # Quadratic form h_i^T Omega h_i for every individual and take the mean
        wiggle_per_individual = (H @ Omega * H).sum(dim=1) # (D,)
        mean_wiggle = wiggle_per_individual.mean() # scalar

        return mean_wiggle

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

    @abstractmethod
    def batch_log_likelihood(self, batch):
        """Compute the per-individual marginal log-likelihood for an entire batch.

        This method is the vectorised counterpart of marginal_log_likelihood. 
        Note that this requires all individuals to share the same observation 
        count N (observation times may still differ). If this is not the case, 
        the training pipeline uses theper-individual marginal_log_likelihood.

        Args:
            batch: a Batch of prepared training data.

        Returns:
            torch.Tensor of shape (D,); the per-individual marginal
                log-likelihoods.
        """
        raise NotImplementedError


class GaussianModel(RandomEffectsModel):
    """Random-effects model with Gaussian random effect.

    In this model we assume heteroscedastic random effects 

                    u_i ~ N(0, Sigma(x_i)),        Sigma(x) = s(x)^2 Sigma_0,

    where Sigma_0 = L L^T is a SHARED shape with unit trace and s(x) > 0 is a
    per-individual scale read off one extra encoder output. 
    Therefore,

        y_i | x_i ~ N( Phi_i h_theta(x_i),
                       s(x_i)^2 Phi_i Sigma_0 Phi_i^T + sigma^2 I ),
    such that the log-likelihood has a closed-form. 


    Things to note:

    1. Heteroscedasticity modelling:
       The heteroscedasticiy Sigma(x) is modelled such that only the MAGNITUDE  
       of the random effecs is individual-specific. That is, Sigma_0 is a shared
       shape and s(x) > 0 a per-individual scale, read off an extra encoder output. 

    2. Modelling of the shared shape Sigma_0
       In the log-likelihood, we parameterise Sigma_0 through its Cholesky factor L 
       (Sigma_0 = L L^T), where L is a lower triangual matrix. To ensure that Sigma_0 
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

       Sigma_0 is implemended to have a UNIT TRACE.

    3. The scaling factor s(x): 
       The scale splits into a global level and an individual deviation,
           log s(x) = a(x) + beta,      s(x) = exp(beta) * exp(a(x)),
       with a(x) the encoder's extra output and beta held outside the encoder.
       Weight decay reaches encoder parameters only, so it flattens a(x) toward
       homoscedasticity without also dragging the overall magnitude toward the
       arbitrary value s = 1. Detailed argument in __init__ below.
    
    We add a roughness penalty that penalises wiggles in the forecast trajectory
    as measured by the integral over the trajectories squared second derivatives. 
    Expected roughness splits into two components that we regularise separately:

        mean-trajectory roughness  h_theta(x_i)^T Omega h_theta(x_i)  [depends on theta]
        random-effects roughness   E[s(x)^2] tr(Omega Sigma_0)   [depends on Sigma_0, s]

    Args:
        encoder: an Encoder instance, built with NR_RANDOM_EFFECT_OUTPUTS extra outputs so
            that it emits the scale alongside the B coefficients.
        nr_basis (int): Number B of spline basis functions and coefficient
            dimensions.
        sigma_init (float): Strictly positive initial value of the
            measurement-noise standard deviation sigma.
        sigma_cov_init (float): Strictly positive initial random-effects SCALE,
            i.e. exp(beta) at initialisation. With unit-trace Sigma_0 this starts
            the model at tr(Sigma(x)) = sigma_cov_init^2.
    """
    # How many outputs beyond the B coefficients this model needs from its encoder. 
    NR_RANDOM_EFFECT_OUTPUTS = 1

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

        # Global level of the random-effects scale, held outside the encoder so
        # that weight decay cannot reach it.
        #
        # Write the encoder's scale output as a(x) = v^T g(x) + b, with g(x) the
        # last hidden layer. Decay shrinks both v and b. In the no-signal limit:
        #   v -> 0  gives a(x) -> b, constant across individuals. WANTED: absent a
        #           differentiating signal in x we should fall back to
        #           homoscedasticity.
        #   b -> 0  gives s(x) -> exp(0) = 1, so Sigma(x) -> Sigma_0. NOT wanted:
        #           it pins that fallback at an arbitrary magnitude, since nothing
        #           distinguishes s = 1 on a log scale.
        #
        # An undecayed beta separates the two. Decay still flattens a(x), but the
        # level exp(beta) is set by the data, so with no signal in x the model
        # reduces to Sigma(x) = exp(2 beta) Sigma_0 -- homoscedastic at the
        # magnitude the data imply. 
        self.log_scale_offset = nn.Parameter(
            torch.tensor(math.log(float(sigma_cov_init)))) # log_scale_offset initialised at sigma_cov_init=0.1


    def _coefficients_and_log_scale(self, X):
        """Run the encoder once and split its output into its two parts.

        The encoder emits B mean coefficients followed by one scale column. This
        method splits them.

        Args:
            X: torch.Tensor of shape (D, M); normalised covariates.

        Returns:
            tuple (coefficients, log_scale):
                coefficients: torch.Tensor (D, B); the mean coefficients
                    h_theta(x), one row per individual.
                log_scale: torch.Tensor (D,); log s(x), already carrying the
                    global level beta, so exp of it is the individual's scale.
        """
        outputs = self.encoder(X)

        coefficients = outputs[:, :self.nr_basis] # B spline coefficents. 
        log_scale = outputs[:, self.nr_basis] + self.log_scale_offset # log s(x) = a(x) + beta,

        return coefficients, log_scale

    def get_coefficients(self, X):
        """Return the mean spline coefficients h_theta(x).

        Since the encoder emits B + 1 columns, we overwrite the base implementation, 
        to split the encoder output and return the B coefficents.

        Args:
            X: torch.Tensor of shape (D, M); normalised covariates.

        Returns:
            torch.Tensor of shape (D, B); the mean coefficients.
        """
        return self._coefficients_and_log_scale(X)[0]

    def random_effect_scale(self, X):
        """Return each individual's random-effects scale s(x) > 0.

        Return the quantity that makes the model heteroscedastic.

        Args:
            X: torch.Tensor of shape (D, M); normalised covariates.

        Returns:
            torch.Tensor of shape (D,); strictly positive.
        """
        return torch.exp(self._coefficients_and_log_scale(X)[1])

    def cholesky_factor(self):
        """Assemble the lower-triangular BxB Cholesky factor L with diag = exp(log-diag).

        Note that we normalise L to by the Frobenius norm, so Sigma_0 = L L^T has unit trace.
        """
        B = self.nr_basis
        # Create matrix of zeros that matches the data type and hardware location of log_diag.
        L = torch.zeros(B, B, dtype=self.log_diag.dtype, device=self.log_diag.device)
        
        # Replace main diagonal with the elements of self.log_diag.
        L[range(B), range(B)] = torch.exp(self.log_diag)

        # Replace the lower triangular part with the elements a_ij of self.offdiag.
        L[self._row_indices, self._col_indices] = self.offdiag

        # Normalise L so Sigma_0 = L L^T has unit trace.
        # To achieve this we normalise L by the Frobenius norm. All magnitude this lives in s(x).
        normalised_L = L / torch.sqrt((L * L).sum())

        return normalised_L

    def sigma_0(self):
        """Return the shared Sigma_0 = L L^T.

        Sigma_0 carries the direction and relative structure of the random 
        effects, common to everyone. 
        The matrix is normalised to have unit trace. 

        Returns:
            torch.Tensor of shape (B, B); symmetric positive-definite, trace 1.
        """
        L = self.cholesky_factor()
        return L @ L.T

    def sigma_matrix_per_individual(self, X):
        """Return each individual's own random-effects covariance Sigma(x).

        Assembles Sigma(x) = s(x)^2 Sigma_0 from the shared shape and the
        individual scale. 
        Args:
            X: torch.Tensor of shape (D, M); normalised covariates.

        Returns:
            torch.Tensor of shape (D, B, B); one symmetric positive-definite
                covariance per individual.
        """
        scales = self.random_effect_scale(X)
        return scales[:, None, None] ** 2 * self.sigma_0() # Sigma(x) = s(x)^2 * Sigma_0

    def marginal_log_likelihood(self, y_i, Phi_i, x_i):
        """Closed-form Gaussian marginal log-likelihood for one individual.

        With Gaussian random effects we have
            y_i | x_i ~ N(mu_i, V_i),
        where
            mu_i = Phi_i h_theta(x_i),
            V_i  = s(x_i)^2 Phi_i Sigma_0 Phi_i^T + sigma^2 I

        Args:
            y_i: torch.Tensor of shape (N_i,); the individual's observed
                trajectory values at its observation times.
            Phi_i: torch.Tensor of shape (N_i, B); the B-spline design matrix
                evaluated at the individual's observation times.
            x_i: torch.Tensor of shape (M,); the individual's static covariates.

        Returns:
            scalar torch.Tensor; the marginal log-likelihood log p(y_i | x_i).
        """
        # Assemble the shared shape Sigma_0 = L L^T.
        Sigma_0 = self.sigma_0()

        # Get the mean spline coefficients h_theta(x_i). Since the encoder expects a batch, we add/remove a batch dimension.
        H, log_s = self._coefficients_and_log_scale(x_i.unsqueeze(0)) # (M,) -> (1, M) -> (1, B) and (1,)
        h = H.squeeze(0) # (1, B) -> (B,)
        s_squared = torch.exp(2.0 * log_s.squeeze(0))  # (1,) -> ()

        # Get the mean trajectory mu_i = Phi_i h_theta(x_i) at the N_i observation times.
        mu_i = Phi_i @ h

        # Get the marginal covariance V_i = s_i^2 Phi_i Sigma_0 Phi_i^T + sigma^2 I.
        N_i = y_i.shape[0]
        identity_N_i = torch.eye(N_i, dtype=y_i.dtype, device=y_i.device)
        V_i = (s_squared * (Phi_i @ Sigma_0 @ Phi_i.T)
               + self.get_noise_variance() * identity_N_i)

        # Create a multivariate normal distribution parameterised by the mean vector mu_i and the covariance matrix V_i.
        dist = torch.distributions.MultivariateNormal(mu_i, covariance_matrix=V_i)

        # Return log p(y_i|x_i) using the multivariate Gaussian avaluated at the observed y_i.
        return dist.log_prob(y_i)

    def re_wiggle(self, X, Omega):
        """Random-effects wiggle E[s(x)^2] tr(Omega Sigma_0).

        The expected wiggle contributed by the random effects u_i ~ N(0,
        Sigma(x_i)), averaged over the batch. 

        Args:
            X: torch.Tensor of shape (D, M); the batch of (normalised) covariates.
            Omega: torch.Tensor of shape (B, B); the spline penalty matrix.

        Returns:
            scalar torch.Tensor; E[s(x)^2] tr(Omega Sigma_0).
        """
        return (self.random_effect_scale(X) ** 2).mean() * torch.trace(Omega @ self.sigma_0())


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
                + lambda_re * self.re_wiggle(X, Omega))

    def batch_log_likelihood(self, batch):
        """Per-individual marginal log-likelihood (D,) for equal-N batches.

        The marginal_log_likelihood method above handles one individual
        at one time only. This permits it to handle batches where the
        observation counts differ across individuals but makes it slow.
        This function implements the same likelihood for an entire batch
        at once. However, it requires all individuals to share the same
        observation count N (observation times may differ, so each
        individual keeps its own marginal covariance V_d).

        Notice that unlike the MeanOnlyModel likelihood is NOT averaged here
        but is the joint multivariate-normal log-density per individual.

        Args:
            batch: the prepared training data (normalised X, per-individual x_i,
                precomputed Phi_i, and observed y_i).

        Returns:
            torch.Tensor: A 1D tensor of shape (D,) containing the marginal
                (joint) log-likelihood log p(y_d | x_d) for each individual.
        """
        # Step 1: Ensure the observation count N is identical across all individuals.
        Ns = {y.shape[0] for y in batch.y_list}
        if len(Ns) != 1:
            raise ValueError("batch_log_likelihood requires equal observation counts per individual.")

        # Step 2: Generate spline coefficient matrix and stack the design matrices.
        H, log_s = self._coefficients_and_log_scale(batch.X) # (D, B) and (D,)
        H = H.unsqueeze(-1) # (D, B) -> (D, B, 1)
        S_squared = torch.exp(2.0 * log_s).view(-1, 1, 1)  # (D, 1, 1)
        Phi = torch.stack(batch.Phi_list, dim=0)  # (D, N, B)

        # Step 3: Compute the mean trajectory (MU) using batch matrix multiplication.
        MU = torch.bmm(Phi, H).squeeze(-1)  # (D, N, 1) -> (D, N)

        # Step 4: Build each individual's marginal covariance
        #         V_d = s_d^2 Phi_d Sigma_0 Phi_d^T + sigma^2 I. Only the scale
        #         is per-individual; the shape Sigma_0 is shared.
        Sigma_0 = self.sigma_0()  # (B, B), unit trace
        N = Phi.shape[1]
        identity_N = torch.eye(N, dtype=Phi.dtype, device=Phi.device)  # (N, N)
        PhiSigma = Phi @ Sigma_0  # (D, N, B)
        V = (S_squared * torch.bmm(PhiSigma, Phi.transpose(1, 2))
             + self.get_noise_variance() * identity_N)  # (D, N, N)

        # Step 5: Create a batched multivariate normal parameterised by the means (MU)
        # and the per-individual marginal covariances (V).
        dist = torch.distributions.MultivariateNormal(MU, covariance_matrix=V)

        # Step 6: Stack the ground-truth outcomes into a single batch tensor.
        Y = torch.stack(batch.y_list, dim=0)  # (D, N)

        # Step 7: Compute the joint log-probability of the outcomes (Y) per individual.
        return dist.log_prob(Y)  # (D,)

class MeanOnlyModel(RandomEffectsModel):
    """A penalised-MSE model baseline.

    This class implements a MSE-loss baseline model with wiggle penalty. 
    We obtain the MSE model case from the random effects base class as a
    result of two choices:
        1. We set the random effects to u_i = 0. 
        2. We fix the noise variance sigma to be a constant. 
    Thus, we have 
            y_i | x_i ~ N(Phi_i h_theta(x_i), sigma^2 I).
    The NLL objective then is a scaled residual sum of squares plus a 
    log-noise constant. With sigma held constant, minimising this over theta 
    is equivalent to minimising the MSE objective.

    Args:
        encoder: an Encoder instance.
        nr_basis (int): number B of spline basis functions.
        sigma_init (float): strictly positive initial measurement-noise std.
    """
    def __init__(self, encoder, nr_basis):
        super().__init__(encoder, nr_basis, sigma_init=1/math.sqrt(2))
        
        self.log_sigma.requires_grad_(False)

    def marginal_log_likelihood(self, y_i, Phi_i, x_i):
        """MSE loss objective for one individual.

        This function implements the MSE loss objective for one individual 
        as used in the Timeview model. That is, the loss is AVERAGED over the N_i 
        observations (not summed). Additionally it is averaged over the D samples
        in negative_log_likelihood in optimisation_objective.py. This is done
        to match the TimeView MSE optimisation objective. 

        Args:
            y_i: torch.Tensor of shape (N_i,); the individual's observed
                trajectory values at its observation times.
            Phi_i: torch.Tensor of shape (N_i, B); the B-spline design matrix
                evaluated at the individual's observation times.
            x_i: torch.Tensor of shape (M,); the individual's static covariates.

        Returns:
            scalar torch.Tensor; the MSE loss (up to a constant)
        """
        # Get the mean spline coefficients h_theta(x_i).
        h = self.encoder(x_i.unsqueeze(0)).squeeze(0)  # (M,) -> (1, M) -> (1, B) -> (B,)

        # Get the mean trajectory mu_i = Phi_i h_theta(x_i) at the N_i observation times.
        mu_i = Phi_i @ h
        
        # Create independent univariate normal distributions centered at mu_i with a fixed std dev 
        # for each of the N_i observations of the given individual i.
        dist = torch.distributions.Normal(mu_i, self.get_noise_std())
        
        # Compute the log-likelihood for each of the N_i observations.
        log_prob = dist.log_prob(y_i) # (N_i,)

        # Average the log-likelihood over the N_i observations for this individual.
        log_prob_avg = log_prob.mean() # scalar

        return log_prob_avg


    def penalty(self, Omega, X, lambda_mean, lambda_re=0):
        """Compute a mean-trajectory roughness penalty. 

        Args:
            Omega: torch.Tensor of shape (B, B); the spline penalty matrix.
            X: torch.Tensor of shape (D, M); the batch of covariates.
            lambda_mean: float, weight on the mean-trajectory wiggles.
            lambda_re: float, weight on the random-effects wiggles (not used).

        Returns:
            scalar torch.Tensor; the mean trajectory wiggle penalty.
        """
        return lambda_mean * self.mean_wiggle(X, Omega)

    def batch_log_likelihood(self, batch):
        """Per-individual log-likelihood (D,) for equal-N batches.

        The marginal_log_likelihood method above handles one individual 
        at one time only. This permits it to handle batches where the
        observation counts differ across individuals but makes is slow. 
        This function implements the same likelihood for an entire batch
        at once. However, it requires all individuals to share the same 
        observation count N (observation times may differ). 

        Args:
            Batch: the prepared training data (normalised X, per-individual x_i,
            precomputed Phi_i, and observed y_i).

        Returns:
            Returns:
            torch.Tensor: A 1D tensor of shape (D,) containing the average 
                          log-likelihood for each individual in the batch.
        """
        # Step 1: Ensure the observation count N is identical across all individuals.
        Ns = {y.shape[0] for y in batch.y_list}
        if len(Ns) != 1:
            raise ValueError("batch_log_likelihood requires equal observation counts per individual.")

        # Step 2: Generate spline coefficent matix and stack feature matrices.
        H = self.encoder(batch.X).unsqueeze(-1) # (D, B) -> (D, B, 1)
        Phi = torch.stack(batch.Phi_list, dim=0) # (D, N, B)
   
        # Step 3: Compute the mean trajectory (MU) using batch matrix multiplication.
        MU  = torch.bmm(Phi, H).squeeze(-1) # (D, N, 1) -> (D, N)

        # Step 4: Create a normal distribution parameterised by the predicted means (MU)
        # and a globally shared noise standard deviation.
        dist = torch.distributions.Normal(MU, self.get_noise_std())

        # Step 5: Stack the ground-truth outcomes into a single batch tensor.
        Y = torch.stack(batch.y_list,  dim=0) # (D, N)

        # Step 6: Compute the log-probability of the outcomes (Y), averaged across N observations.
        return dist.log_prob(Y).mean(dim=1) # (D,)

class MixtureModel:
    pass

