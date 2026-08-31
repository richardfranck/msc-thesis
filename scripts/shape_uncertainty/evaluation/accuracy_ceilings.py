import numpy as np

from scripts.shape_uncertainty.evaluation.evaluation_metrics import (
    compute_R_squared, compute_individual_R_squared)

class AccuracyCeilings:

    def __init__(self, dataset, n_mc, seed):
        self.dataset = dataset
        self.n_mc = int(n_mc)
        self.seed = seed

        if not self.dataset.design["regular"]:
            raise ValueError("We require a the observation gird to be regular so every sample "
                "is observed at the same observation times.")
        self.times = self.dataset.times[0]

        self._parameters = None

    def _resampled_parameters(self):
        """Return the process parameters resampled at the dataset's covariates.

        Call the resample_parameters() method of theWilkerson DGP class and populates the
        self._parameters attribute.

        Intuition:
        We redraw the Wilkerson process parameters at a fixed set of covariates
        n_mc times. We do this as in the presence of unobserved heterogeneity,
        a covariate vector does not pin down the process parameters. Instead,
        they are controlled by the random latent factors (z_g, z_d) which induce
        a distribution over these parameters. We esitmate this distribution via
        Monte Carlo estimation.

        Returns:
            dict: Parameter name to np.ndarray of shape (D, n_mc), as returned
                by WilkersonDGP.resample_parameters, holding "g", "d" and "rho".
        """
        if self._parameters is None:
            rng = np.random.default_rng(self.seed)
            self._parameters = self.dataset.dgp.resample_parameters(
                self.dataset.X, self.n_mc, rng)

        return self._parameters

    
    def compute_conditional_curve_moments(self):
        """Estimate the mean and variance of the trajectory at fixed covariates.

        In the presence of unobserved heterogeneity, a a covariate vector mpas to a
        distribution over trajectories rather than one trajectory. The
        pointwise mean of that distribution is the best forecast available to any
        model that sees the covariates alone, and the pointwise variance is the
        variation that no such forecast can capture.
        We estimate both of these functions at at FIXED covariates as follows:

        Step 1: Resample the process parameters at the given covariates.
        Step 2: Evaluate the Wilkerson curve for every draw on the shared time
                grid, giving one trajectory per individual per draw.
        Step 3: Average across the draw axis at each time to obtain the
                conditional mean function.
        Step 4: Take the variance across the draw axis at each time to obtain the
                conditional variance function.

        Note as a correctness check that under a regime without unobserved
        heterogeneity the conditional variance, given the observed covariates
        is zero at every time.

        Returns:
            tuple: A tuple (conditional_mean, conditional_variance) of np.ndarray,
                each of shape (D, M), holding the mean and the variance across
                latent draws at each evaluation time for each individual.
        """
        times = self.times.copy()

        # Step 1: Resample the process parameters at the given covariates
        parameters = self._resampled_parameters()

        # Reformat the covariate size is a covariate which we need for curve computation
        size = np.asarray(self.dataset.X["size"], dtype=float)[:, None, None]

        # Step 2: Evaluate the Wilkerson curve for every draw on the shared time grid
        # we get one trajectory per individual per draw.
        curves = self.dataset.dgp._compute_wilkerson(
            times[None, None, :],           # shape (1, 1, M)
            size,                           # shape (D, 1, 1)
            parameters["g"][:, :, None],    # shape (D, n_mc, 1)
            parameters["d"][:, :, None],    # shape (D, n_mc, 1)
            parameters["rho"][:, :, None],  # shape (D, n_mc, 1)
        ) # shape (D, n_mc, M)

        # Step 3: Conditional mean function
        conditional_mean = curves.mean(axis=1) # Shape (D, M)

        # Step 4: Conditional variance function
        conditional_var = curves.var(axis=1) # Shape (D, M)

        return conditional_mean, conditional_var

    def _stack_evaluation_references(self):
        """Stack the noise-free and the observed outcomes into (D, M) matrices.

        Returns:
            tuple: A tuple (y_true, y_noisy) of np.ndarray of shape (D, M), holding
                the noise-free trajectory and the recorded noisy observations of
                every individual on the shared observation grid.
        """
        y_true = np.stack([np.asarray(y, dtype=float) for y in self.dataset.Y_true])
        y_noisy = np.stack([np.asarray(y, dtype=float) for y in self.dataset.Y_noisy])
        return y_true, y_noisy


    def compute_pooled_metrics(self, conditional_mean):
        """Score the conditional mean forecast, pooled over individuals and times.

        The R-squared ceiling is the R-squared of the conditional mean function.
        We compute the R-squared of this prediction once w.r.t. to the observed noisy
        outcomes and once w.r.t. the nose-free latents. Here we do so on a pooled-basis,
        producing one number for the entire dataset.

        Args:
            conditional_mean (np.ndarray): Shape (D, M); the expected trajectory
                for each individual across the shared time grid, computed via
                Monte Carlo over latent draws.

        Returns:
            dict: A dictionary containing pooled aggregate metrics:
                - "r2_ceiling_latent" (float): pooled R-squared ceiling against the
                  noise-free trajectories.
                - "r2_ceiling_observed" (float): pooled R-squared ceiling against
                  the recorded noisy observations.
        """
        y_true, y_noisy = self._stack_evaluation_references()
        r2_ceiling_latent = float(compute_R_squared(conditional_mean.ravel(), y_true.ravel()))
        r2_ceiling_observed = float(compute_R_squared(conditional_mean.ravel(), y_noisy.ravel()))

        return {
            "r2_ceiling_latent": r2_ceiling_latent,
            "r2_ceiling_observed": r2_ceiling_observed,
        }

    def compute_individual_metrics(self, conditional_mean):
        """Compute individual-specific curve matching metrics against true and noisy data.

        The R-squared ceiling is the R-squared of the conditional mean function.
        We compute the R-squared of this prediction once w.r.t. to the observed noisy
        outcomes and once w.r.t. the nose-free latents. Here we do so for each individual
        respectively, producing one number per individual.

        Args:
            conditional_mean (np.ndarray): Shape (D, M); the expected trajectory
                for each individual across the shared time grid, computed via
                Monte Carlo over latent draws.

        Returns:
            dict: A dictionary containing individual-specific ceilings:
                - "r2_ceiling_latent" (np.ndarray of shape (D,)): Individual R2
                  ceiling evaluated against noise-free true trajectories (Y_true).
                - "r2_ceiling_observed" (np.ndarray of shape (D,)): Individual R2
                  ceiling evaluated against recorded noisy observations (Y_noisy).
        """
        # Step 1: Stack individual curves into 2D matrices of Shape: (D, M)
        y_true, y_noisy = self._stack_evaluation_references()

        # Step 2: Evaluate the forecast per individual against both references
        forecast = list(conditional_mean)
        indiv_r2_latent = compute_individual_R_squared(forecast, list(y_true))
        indiv_r2_observed = compute_individual_R_squared(forecast, list(y_noisy))

        return {
            "r2_ceiling_latent": indiv_r2_latent,
            "r2_ceiling_observed": indiv_r2_observed,
        }

    def compute_signal_to_noise_ratio(self):
        """Compute the variance of the latent trajectories relative to the variance of the
        observation noise.

        The measurement-noise standard deviation sigma fixes how much of the
        observed variation is error rather than signal. The signal-to-noise ratio
        expresses the spread of the noise-free trajectories relative to that error,
            SNR = Var(y_true) / sigma^2,
        pooled over all individuals and observation times, and separately for each
        individual over its own observation times.

        Both variances are taken about the same centre as the corresponding
        R-squared: the pooled ratio about the grand mean over individuals and
        times, the per-individual ratio about that individual's temporal mean.

        Returns:
            dict: A dictionary containing:
                - "pooled" (float): Var of all latent values over all individuals
                and times, divided by sigma^2.
                - "per_individual" (np.ndarray of shape (D,)): Each individual's
                own variance over time, divided by sigma^2.

        Raises:
            ValueError: If sigma is zero, for which the ratio is undefined.
        """
        sigma = self.dataset.sigma
        if sigma <= 0.0:
            raise ValueError(
                "The SNR is defined for a strictly positive observation noise level only.")

        # Get a list whose (N_i,) elements are the values of the
        # outcome observations for individual i
        curves = [np.asarray(y, dtype=float) for y in self.dataset.Y_true]

        # Compute pooled SNR
        pooled_variance = float(np.concatenate(curves).var())
        pooled_snr = pooled_variance / (sigma ** 2)

        # Compute per-individual SNR
        per_indiv_variance = np.array([y.var() for y in curves])
        per_indiv_snr = per_indiv_variance / (sigma ** 2)

        # Return the aggregated metrics
        return {
            "snr_pooled": pooled_snr,
            "snr_per_individual": per_indiv_snr,
        }


    def _compute_variance_components(self):
        """Compute the explained and latent variance components at each time point.

        Conditioning on the observed covariates X and applying the law of total
        variance to the latent factors Z gives for the noise-free trajectory
            Var(y_latent) = Var_X( E_Z[y | X] )  +  E_X( Var_Z[y | X] ),
        where
            - E_Z[y | X] is the conditional mean function, m(t; X): the average
              trajectory over the latent distribution at fixed covariates,
              the best any covariate-only predictor can do;
            - Var_Z[y | X] is the conditional variance function, v(t; X): the
              spread of trajectories among individuals who share those
              covariates but differ in their latent draw;
        meaning,
            - the first term in Var(y_latent) is the variation the covariates account for; and
            - the second term in Var(y_latent) is the variation carried by unobserved heterogeneity.

        These components are the THEORETICAL decomposition. They are centred at
        each time on that time's cross-sectional mean, whereas compute_R_squared
        centres on the grand mean over individuals and times. The two therefore
        differ by the temporal variation of the population mean curve. The
        ceilings reported to the reader are computed by scoring the conditional
        mean forecast with compute_R_squared, so that they share a convention
        with the achieved values; the components here are reported alongside as
        the interpretable decomposition and as a correctness check.

        Returns:
            tuple: A tuple (explained_variance, latent_variance, total_variance)
                of np.ndarray, each of shape (M,), holding the covariate-explained,
                the latent and the total noise-free variance at each time.
        """
        # Step 1: Obtain conditional mean and variance functions across latent draws
        conditional_mean, conditional_variance = self._compute_conditional_curve_moments(self.times)

        # Step 2: Compute V_exp as the variance of the mean across individuals
        explained_variance = conditional_mean.var(axis=0) # Shape: (M,)

        # Step 3: Compute V_lat as the mean of the variance across individuals
        latent_variance = conditional_variance.mean(axis=0) # Shape: (M,)

        total_variance = explained_variance + latent_variance

        return explained_variance, latent_variance, total_variance
