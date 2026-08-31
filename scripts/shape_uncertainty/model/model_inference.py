import torch
import numpy as np

# Shape extraction utilities
from scripts.shape_uncertainty.shape_extraction.shape_summary import extract_shape_summary
from scripts.shape_uncertainty.spline_basis.bspline_basis import (
    basis_matrix,
    pointwise_sd_from_covariance,
)

# Uncertainty utilities
from scripts.shape_uncertainty.shape_extraction.shape_uncertainty import (
    compute_shape_uncertainty,
    medoid_summary,
)

from scripts.shape_uncertainty.shape_extraction.value_uncertainty import (
    compute_value_uncertainty,
)

def build_inference_engine(model, normaliser, knot_objects, shape_config):
    """Wrap a trained model and its normaliser in a populated InferenceEngine.

    Args:
        model: a trained RandomEffectsModel.
        normaliser: the covariate normaliser fitted alongside it.
        knot_objects: dict from build_knot_dictionary, supplying
            "basis_functions", "C" and "breakpoints".
        shape_config: dict with "zeta_rel", "upsilon_rel_1", "upsilon_rel_2",
            "upsilon_rel_prune" and "do_prune".

    Returns:
        populated InferenceEngine ready for prediction.
    """
    engine = InferenceEngine()
    engine.populate_attributes(
        model=model,
        normaliser=normaliser,
        basis_functions=knot_objects["basis_functions"],
        C=knot_objects["C"],
        breakpoints=knot_objects["breakpoints"],
        zeta_rel=shape_config["zeta_rel"],
        upsilon_rel_1=shape_config["upsilon_rel_1"],
        upsilon_rel_2=shape_config["upsilon_rel_2"],
        upsilon_rel_prune=shape_config["upsilon_rel_prune"],
        do_prune=shape_config["do_prune"],
    )
    return engine

class InferenceEngine:
    """Compute trajectory forecast from a covariate vector."""
    def __init__(self):

        # Initialise everything as empty placeholders
        self.model = None
        self.normaliser = None
        self.knot_objects = {
                "basis_functions": None,
                "C": None,
                "breakpoints": None,
                }
        self.shape_config = {
            "zeta_rel": 0.0,
            "upsilon_rel_1": None,
            "upsilon_rel_2": None,
            "upsilon_rel_prune": 0.0,
            "do_prune": False,
        }

        self.T = None

        self.populated = False

    def populate_attributes(self, model, normaliser, basis_functions, 
                            C=None , breakpoints=None, zeta_rel=0.0, upsilon_rel_1=0.0,
                            upsilon_rel_2=0.0, upsilon_rel_prune=0.0, do_prune=False):
        """ Populate the attributes of the InferenceEngine. """
        # Populate the dataset
        if self.model is None:
            self.model = model

        # Populate the normaliser
        if self.normaliser is None:
            self.normaliser = normaliser

        # Populate the knot objects dict
        if self.knot_objects["basis_functions"] is None:
            self.knot_objects["basis_functions"] = basis_functions
            self.knot_objects["C"] = C
            self.knot_objects["breakpoints"] = breakpoints

        # Populate thw shape threshold configurations
        if self.shape_config["upsilon_rel_1"] is None:
            self.shape_config["zeta_rel"] = zeta_rel
            self.shape_config["upsilon_rel_1"] = upsilon_rel_1
            self.shape_config["upsilon_rel_2"] = upsilon_rel_2
            self.shape_config["upsilon_rel_prune"] = upsilon_rel_prune
            self.shape_config["do_prune"] = do_prune

        # Store forecasting horizon
        self.T = None if breakpoints is None else breakpoints[-1]

        # Track whether the attributes have been filled yet
        self.populated = True

    ######################################################################
    # ------------------------- Point Prediction -------------------------
    ######################################################################

    def _get_mean_coefficients(self, X):
        """Predict mean spline coefficients h_theta(x) for a batch of individuals.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals

        Returns:
            W: A torch.Tensor of shape (D, B); the mean coefficients, 
                one row per individual, in the normaliser's fitted column order.
        """
        # Check attributes are populated
        if self.populated is False:
            raise RuntimeError("Call 'populate_attributes()' first.")
        
        # Get the mean spline coefficients h_theta(x)
        self.model.eval()
        Z = self.normaliser.transform(X) # (D, M)
        with torch.no_grad():
            W = self.model.get_coefficients(Z) # (D, B)
            return W

    def _extract_summaries(self, W):
        """Extract a shape summary from each of a stack of coefficient vectors.

        Args:
            W: np.ndarray of shape (N, B); one coefficient vector per row.

        Returns:
            list of length N of shape summaries, each a list of
                (state, start_time) tuples.
        """
        summaries = []
        for w in W:
            summaries.append(extract_shape_summary(
                w.reshape(-1, 1),
                self.knot_objects["C"],
                self.knot_objects["breakpoints"],
                zeta_rel=self.shape_config["zeta_rel"],
                upsilon_rel_1=self.shape_config["upsilon_rel_1"],
                upsilon_rel_2=self.shape_config["upsilon_rel_2"],
                upsilon_rel_prune=self.shape_config["upsilon_rel_prune"],
                do_prune=self.shape_config["do_prune"],
            ))
        return summaries

    def predict_mean_shape_summary(self, X):
        """Predict the mean-trajectory shape summary for a batch of individuals.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,).


        Returns:
            list of length D; element i is individual i's shape summary as a
                list of (state, start_time) tuples
        """
        # Step 1: Perform one forward pass for the whole batch
        W = self._get_mean_coefficients(X) # (D, B)
        W = W.detach().cpu().numpy()  # extract_shape_summary expects numpy

        # Step 2: Extract a summary per individual 
        summaries = self._extract_summaries(W)

        return summaries

    def predict_trajectory(self, X, times):
        """Predict mean trajectory values at `times` for one covariate dict x.

        We use this function to plot the trajectory of a given covariate vector. 

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.
            times: 1-D sequence of time points (floats) in [0, T] at which to 
                evaluate the mean trajectory. This is used for plotting.

        Returns:
            np.ndarray of shape (D, N); predicted values in original units.
        """
        # Step 1: Every individual's mean coefficients
        W = self._get_mean_coefficients(X) # (D, B)

        # Step 2: Evaluate the spline and return to original units
        Phi = torch.as_tensor(basis_matrix(np.asarray(times, float),
                                           self.knot_objects["basis_functions"]),
                            dtype=W.dtype) # (N, B)
        Y_norm = W @ Phi.T # (D, B) @ (B, N) -> (D, N)
        Y_norm = Y_norm.detach().cpu().numpy()

        # Step 3: Return to original units
        Y = self.normaliser.inverse_transform_y(Y_norm) # (D, N)
        return Y

    ######################################################################
    #######     Aleatoric uncertainty for a single model            ######
    ######################################################################

    @property
    def noise_std_hat(self):
        """Return the fitted measurement-noise std sigma in original y units.

        
        Returns:
            float; the fitted measurement-noise standard deviation.
        """
        with torch.no_grad():
            return float(self.model.get_noise_std()) * self.normaliser.y_std

    def _cholesky_factor(self):
        """Return the cholesky factor L of the random-effects covariance.

        Return the cholesky factor needed for smapling. 
        For sampling we require u ~ N(0, Sigma). The model parameterises Sigma
        through the cholesky factor L, with Sigma = L L^T.
        Let z ~ N(0, I), then we obtain u as 
                                u = L z
        and can draw samples from it.

        Returns:
            torch.Tensor of shape (B, B); the lower-triangular factor.
        """
        # Step 1: Check attributes are populated
        if self.populated is False:
            raise RuntimeError("Call 'populate_attributes()' first.")

        # Step 2: Raise an error if the model has no random effects (i.e. MeanOnly)
        if not hasattr(self.model, "cholesky_factor"):
            raise RuntimeError(
                f"{type(self.model).__name__} has no random-effects covariance.")

        # Step 3: Return the cholesky facotor
        with torch.no_grad():
            return self.model.cholesky_factor().detach()

    @property
    def sigma_0_hat(self):
        """Return the trained shared shape Sigma_0 = L L^T.

        Returns:
            np.ndarray of shape (B, B); symmetric positive-definite.
        """
        L = self._cholesky_factor()
        return (L @ L.T).cpu().numpy()  # NORMALISED unity

    def random_effect_scales(self, X):
        """Return each individual's fitted random-effects scale s(x).

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,).

        Returns:
            np.ndarray of shape (D,); strictly positive, in NORMALISED units.
        """
        self.model.eval()
        Z = self.normaliser.transform(X)
        with torch.no_grad():
            return self.model.random_effect_scale(Z).cpu().numpy() # NORMALISED unity

    def sigma_hat_per_individual(self, X):
        """Return each individual's fitted covariance Sigma(x) = s(x)^2 Sigma_0, in ORIGINAL y units.

        Assembles Sigma(x) = s(x)^2 Sigma_0 and converts it out of the model's
        normalised outcome space, so it can be set directly against a
        ground-truth covariance.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,).

        Returns:
            np.ndarray of shape (D, B, B), in the model's normalised units.
        """
        scales = self.random_effect_scales(X)
        scales = scales.reshape(-1, 1, 1)
        sigma_per_individual = scales ** 2 * self.sigma_0_hat # normalised units 
        sigma_per_individual = sigma_per_individual* self.normaliser.y_std ** 2
        return sigma_per_individual

    def _draw_random_effects(self, X, n_samples, rng):
        """Draw random-effect vectors u_i ~ N(0, Sigma(x_i)).

        Draws a specified number of random-effect vectors (n_samples) for each
        individual in X (e.g. for everyone in the test set). Draws are
        independent across individuals and across samples.

        The covariance is heteroscedastic, Sigma(x) = s(x)^2 Sigma_0, so a draw
        is built in two stages, shape first and magnitude second:

            u = s(x) * L z,     z ~ N(0, I),

        where L is the Cholesky factor of the SHARED shape (Sigma_0 = L L^T,
        with unit trace) and s(x) is the individual's own scale. The first stage
        gives every individual the same correlation structure; the second gives
        each one its own size.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.
            n_samples: int, n; draws per individual.
            rng: np.random.Generator or int seed.

        Returns:
            torch.Tensor of shape (D, n, B); the random-effect draws.
        """
        # Step 1: Accept a Generator or an int seed
        rng = np.random.default_rng(rng)

        # Step 2: Draw standard normals, one B-vector per (individual, sample)
        L = self._cholesky_factor() # (B, B)
        nr_individuals = len(next(iter(X.values()))) # scalar D
        nr_basis = L.shape[0] # scalar B
        z = rng.standard_normal((nr_individuals, n_samples, nr_basis)) # (D, n, B) i.i.d. N(0,1), numpy float64
        z = torch.as_tensor(z, dtype=L.dtype) # same values, cast to L's dtype

        # Step 3: Correlate draw through the Cholesky factor.
        # Here z is a row vector so z L^T gives Cov(u) = L L^T = Sigma_0
        u_unit_scale = z @ L.T # (D, n, B)

        # Step 4: Give each individual its own magnitude
        scales = torch.as_tensor(self.random_effect_scales(X), dtype=L.dtype) # (D,)

        return scales[:, None, None]  * u_unit_scale # (D, 1, 1) * (D, n, B) = (D, n, B)

    ######################################################################
    # ------------------------- Aleatoric Prediction ---------------------
    ######################################################################

    def draw_aleatoric_coefficients(self, X, n_samples, rng):
        """Draw coefficient vectors h_theta(x) + u for a batch of individuals.

        Under a model with random effects we obtain an indivdual's coefficient 
        vector as the individual's mean coefficients perturbed by random-effect 
        draws from the model's own trained covariance. 
        This function draws a clound of #n_samples random effect pertubations and
        returns the cloud of perturbed indivdual's coefficient vectors.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.
            n_samples: int, n; aleatoric draws per individual.
            rng: np.random.Generator.

        Returns:
            np.ndarray of shape (D, n, B); element [d, k] is individual d's
                k-th coefficient draw.
        """
        # H+U -> W[d, k] is individual d's k-th coefficient draw (of B coeffs), so each individual
                # has an (n, B) block of draws:
                #    W = [
                #       [[P0_S0], [P0_S1], ...],   # Person 0's n draws
                #       [[P1_S0], [P1_S1], ...],   # Person 1's n draws
                #       [[P2_S0], [P2_S1], ...],   # Person 2's n draws
                #    ]

        # Step 1: Get every individual's mean coefficients
        H = self._get_mean_coefficients(X) # (D, B)

        # Step 2: Draw the random effects for the whole batch
        U = self._draw_random_effects(X, n_samples, rng)  # (D, n, B)

        # Step 3: Perturb each individual's mean by each of its draws
        W = H.unsqueeze(1) + U # (D, 1, B) + (D, n, B) = (D, n, B)
        return W.detach().cpu().numpy()

    def predict_aleatoric_summaries(self, coefficients):
        """Get the aleatoric shape distribution for a batch of individuals.

        This is the counterpart to predict_mean_shape_summary for a MeanOnlyModel
        We obtain the cloud of summaries for each individual under the given model.
        The spread of the cloud is the irreducable aleatoric shape uncertainty. 
        It is the uncetainty inshapes that exists between individuals sharing 
        the same covariates.

        Args:
            coefficients: np.ndarray (D, n, B) from draw_aleatoric_coefficients

        Returns:
            list of length D; element d is that individual's list of n shape summaries.
        """
        # Extract shape summaries for every draw 
        # _extract_summaries loops over the rows of a 2-D (N, B) array, so the
        # (D, n, B) cloud must be collapsed to (D*n, B). We then have
        #    W = [
        #       [P0_S0], [P0_S1], ..., # rows 0 to n-1 are person 0
        #       [P1_S0], [P1_S1], ..., # rows n to 2n-1 are person 1
        #       [P2_S0], [P2_S1], ..., # rows 2n to 3n-1 are person 2
        #    ]
        # detach drops the autograd graph, cpu pulls the draws off any device,
        # and numpy hands extract_shape_summary the array type it expects.
        D, n, B = coefficients.shape  # (D, n, B)
        W = coefficients.reshape(D * n, B)
        summaries = self._extract_summaries(W) # summaries = [P0_S0, P0_S1, ..., P1_S0, P1_S1, ..., P2_S0, ...]

        # Step 3: Regroup into one cloud per individual
        # Undo the flattening by cutting the list into the D blocks of n
        # consecutive summaries. Individual d owns rows d*n up to but 
        # excluding (d+1)*n:
        #    [
        #       [P0_S0, P0_S1, ...],   # Person 0's aleatoric cloud
        #       [P1_S0, P1_S1, ...],   # Person 1's aleatoric cloud
        #       [P2_S0, P2_S1, ...],   # Person 2's aleatoric cloud
        #    ]
        return [summaries[d * n:(d + 1) * n] for d in range(D)]


    def predict_aleatoric_trajectories(self, coefficients, times):
        """Predict trajectory values of all aleatoric draws at `times` for one 
          covariate dict x.
       
        We use this function to plot the trajectory of a given covariate vector. 

        Args:
            coefficients: np.ndarray (D, n, B) from draw_aleatoric_coefficients
            times: 1-D sequence of N time points in [0, T].

        Returns:
            np.ndarray of shape (D, n, N); predicted values in original units.
        """
        # Step 1: Evaluate the spline basis once and apply it to every draw.
        Phi = basis_matrix(np.asarray(times, float), self.knot_objects["basis_functions"])                      # (N, B)
        Y_norm = (coefficients @ Phi.T)                                      # (D, n, N)

        # Step 2: Return to original units
        Y = self.normaliser.inverse_transform_y(Y_norm)
        return Y

    ######################################################################
    # -------------- Aleatoric Value-Space Uncertainty ------------------
    ######################################################################

    def predict_aleatoric_band(self, X, times):
        """Return the pointwise value-space aleatoric sd implied by the fitted Sigma(x).

        The conversion from coefficient-space covariance to value-space
        uncertainty is given by
            Var_j(t) = phi(t)^T Sigma_j phi(t)
        and
            SD_j(t) = sqrt(phi(t)^T Sigma_j phi(t)),
        where Sigma_j is the individual's random-effect covariance matrix and
        phi(t) is the basis-function vector evaluated at time t.

            DERIVATION:
                The random effects for individual i are
                        u_i ~ N(0, Sigma_i).
                The model is

                        y_i = phi(t)^T (h(x) + u_i)
                            = phi(t)^T h(x) + phi(t)^T u_i.
                Therefore, the contribution of the random effects to the outcome
                in value space is phi(t)^T u_i. For conciseness, let phi(t) = phi
                denote the basis functions evaluated at the observation times.
                        Var(phi^T u_i)
                            = E[(phi^T u_i)^2] - E[phi^T u_i]^2.
                Since phi is non-random and E[u_i] = 0,
                        E[phi^T u_i] = phi^T E[u_i] = 0,
                and therefore
                        Var(phi^T u_i)
                            = E[(phi^T u_i)^2]
                            = E[(phi^T u_i)(phi^T u_i)^T]
                            = E[phi^T u_i u_i^T phi]
                            = phi^T E[u_i u_i^T] phi
                            = phi^T Sigma_i phi.

                Thus, phi^T Sigma_i phi is the pointwise value-space aleatoric
                variance, and its square root is the corresponding standard
                deviation.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,).
            times: 1-D np.ndarray of shape (M,); the times at which to evaluate,
                shared by every individual.

        Returns:
            np.ndarray of shape (D, M); element [d, m] is individual d's
                aleatoric standard deviation at times[m], in original y units.
        """
        # Step 1: Assemble each individual's covariance, in original y units
        Sigma = self.sigma_hat_per_individual(X)                    # (D, B, B)

        # Step 2: Carry it into value space at the requested times
        return pointwise_sd_from_covariance(
            Sigma, times, self.knot_objects["basis_functions"])     # (D, M)



class UncertaintyEngine:
    """Quantify predictive uncertainty over shapes across an ensemble of models.

    This class operates on an ensemble of trained models, each wrapped as an 
    InferenceEngins. Based on this, we expplore for every individual in a 
    test set:

        1. What is the individual's shape uncertainty U in [0, 1], defined as the
           duration-weighted mean pairwise distance across the cloud. 

        2. What is the consensus summart, i.e. the shape summary (and trajectory)
            that best represents the cloud.

    Args:
        engines: sequence of InferenceEngine instances, the trained ensemble.
    """
    def __init__(self, engines):
        self.engines = engines

    @property
    def T(self):
        """The forecasting horizon.

        Returns:
            float; the right endpoint of [0, T].
        """
        horizons = {engine.T for engine in self.engines}
        if len(horizons) != 1:
            raise ValueError(f"Ensemble members disagree on the horizon: {horizons}.")

        return horizons.pop()

    def _build_epistemic_shape_summaries(self, X):
        """Build every individual's cloud of M epistemic shape summaries.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.

        Returns:
            list of length D; element i is the list of M shape summaries the
                ensemble assigns to individual i, in engine order.
        """
        # This is a list of lists. The outer list is of length M with one element per
        # engine. The inner lists are of dimension D, with one summary per person. 
        # summaries_by_model = [
        #    ["M0_P0", "M0_P1", "M0_P2"],  # Model 0's predictions
        #    ["M1_P0", "M1_P1", "M1_P2"],  # Model 1's predictions
        # ]
        summaries_by_model = [
            engine.predict_mean_shape_summary(X) for engine in self.engines
        ]

        # Group predictions by person
        # (
        #    (M0_P0, M1_P0, ...), # Person 0's predictions
        #    (M0_P1, M1_P1, ...), # Person 1's predictions
        #    (M0_P2, M1_P2, ...)  # Person 2's predictions
        # )
        summaries_by_individual = zip(*summaries_by_model)
        
        # Convert the inner tuples to lists
        summaries_by_individual = [
            list(prediction) for prediction in summaries_by_individual
        ] 

        return summaries_by_individual

    def _build_aleatoric_shape_summaries(self, X, n_samples, rng, member=0):
        """Build every individual's cloud of n aleatoric shape summaries.

        This cloud comes from one model. When this is nested with epistemic 
        uncertainty we may have multiple models (the ensemble). Each ensemble
        member will have its own estimate of the aleatoric uncertainty variance
        covariance matrix so which model is used must be indicated via the
        member variable. 

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.
            n_samples: int, n; aleatoric draws per individual.
            rng: np.random.Generator driving the draws.
            member: int; which ensemble member supplies Sigma. 
                Defaults to zero which is the case where we have no ensemble. 

        Returns:
            tuple (clouds, coefficients):
                clouds: list of length D; element i is the list of n shape
                    summaries drawn for individual i.
                coefficients: np.ndarray of shape (D, n, B); the draws those
                    summaries came from, returned so a caller can evaluate the
                    very same sample as trajectories rather than redrawing it.
        """
        engine = self.engines[member]
        coefficients = engine.draw_aleatoric_coefficients(X, n_samples, rng)
        summaries = engine.predict_aleatoric_summaries(coefficients)

        return summaries, coefficients

    def _compute_shape_uncertainty(self, shape_distribution, alpha=1/2, beta=1/4):
        """Score how much one cloud of summaries disagrees.

        Args:
            shape_distribution: list of M shape summaries over [0, T].
            alpha: float, slope disagreement weight.
            beta: float, curvature disagreement weight.

        Returns:
            Tuple (U, profile) where 
            - U is a float in [0, 1], zero for a unanimous ensemble 
            - profile is a dict with 
            profile = {
                'regions':            np.ndarray (n, 2) -- region boundaries R_k
                'lengths':            np.ndarray (n,)   -- region durations L_k
                'region_uncertainty': np.ndarray (n,)   -- regional uncertainty Q_k
                'state_histograms':   np.ndarray (n, V) -- state histograms n_k
                'state_indices': state_indices
            }
        """
        U, profile = compute_shape_uncertainty(shape_distribution, self.T, alpha, beta)
        return U, profile

    def _compute_consensus_summary(self, shape_distribution, profile, alpha=1/2, beta=1/4):
        """Select the summary that best represents one cloud.

        Args:
            shape_distribution: list of M shape summaries over [0, T].
            profile: dict as returned by _compute_shape_uncertainty for THIS
                cloud.
            alpha: float, slope disagreement weight.
            beta: float, curvature disagreement weight. 

        Returns:
            tuple (consensus, member_index):
                consensus: the selected shape summary.
                member_index: int; index for which ensemble member supplied consensus. 
        """
        consensus, member_index = medoid_summary(
            shape_distribution, self.T, profile, alpha, beta,
        )
        return consensus, member_index

    def _build_uncertainty_result(self, clouds, source, n_samples=None, alpha=1/2, beta=1/4):
        """Score every individual's cloud and select its consensus.

        Comoute the tuple (U, profile) for each cloud it is handed. The cloud 
        may come from any source, i.e. from an ensemble, aleatoric draws or a   
        nested combination.
        We also select the consensus summary for each cloud. 

        Args:
            clouds: list of length D; element i is that individual's list of
                shape summaries.
            source: str; "epistemic", "aleatoric" or "combined". Recorded on the
                result so that clouds measured under different regimes are not
                compared.
            n_samples: int or None; draws per individual, where the source has
                such a parameter.
            alpha: float, slope disagreement weight.
            beta: float, curvature disagreement weight.

        Returns:
            dict with keys:
                "U":                np.ndarray (D,); shape uncertainty per
                                    individual, all in [0, 1].
                "consensus":        list of length D; the shape summary reported
                                    for each individual.
                "selected_indices": np.ndarray (D,) of int; which element of the
                                    cloud supplied each consensus.
                "profiles":         list of length D of uncertainty profiles.
                "shapes":           list of length D of the clouds themselves.
                "T", "alpha", "beta", "source", "n_samples": the conditions
                                    everything was computed under.
        """
        # Step 1: Compute uncertainties + profile and consensus for all individuals
        D = len(clouds)
        U = np.empty(D, dtype=float)
        selected_indices = np.empty(D, dtype=int)
        profiles, consensus = [], []

        for i, shape_distribution in enumerate(clouds):
            U[i], profile = self._compute_shape_uncertainty(
                shape_distribution, alpha, beta)
            best_summary, selected_indices[i] = self._compute_consensus_summary(
                shape_distribution, profile, alpha, beta)

            profiles.append(profile)
            consensus.append(best_summary)

        # Step 2: Build the results dictionary
        return {"U": U,
                "consensus": consensus,
                "selected_indices": selected_indices,
                "profiles": profiles,
                "shapes": clouds,
                "T": self.T,
                "alpha": alpha,
                "beta": beta,
                "source": source,
                "n_samples": n_samples}


    def predict_with_epistemic_uncertainty(self, X, alpha=1/2, beta=1/4):
        """Quantify epistemic uncertainty and report a consensus for every individual.

        This serves as the entry for analyis. For each individual:
            - build the ensemble's cloud of M summaries,
            - compute the uncertainty and uncertainty profile
            - selects one consensus summary
    
        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.
            alpha: float, slope disagreement weight.
            beta: float, curvature disagreement weight.

        Returns:
            dict as described in _build_uncertainty_result()
        """
        # Step 1: Build every individual's cloud of M summaries
        clouds = self._build_epistemic_shape_summaries(X)

        # Step 2: Compute uncertainties + profile and consensus for all individuals
        predictions = self._build_uncertainty_result(clouds, source="epistemic",
                              n_samples=len(self.engines), alpha=alpha, beta=beta)

        return predictions 

    def predict_with_aleatoric_uncertainty(self, X, n_samples, rng, member=0, alpha=1/2, beta=1/4):
        """Quantify aleatoric uncertainty and report a consensus for every individual.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,).
            n_samples: int, n; aleatoric draws per individual. 
            rng: np.random.Generator driving the draws.
            member: int; which ensemble member supplies Sigma.
            alpha: float, slope disagreement weight.
            beta: float, curvature disagreement weight.

        Returns:
            dict as described in _build_uncertainty_result()
            + the coefficents of the aleatoric draws under "coefficients" (np.ndarray of shape (D, n, B))
        """
        clouds, coefficients = self._build_aleatoric_shape_summaries(X, n_samples, rng, member)

        predictions = self._build_uncertainty_result(
            clouds, source="aleatoric", n_samples=n_samples, alpha=alpha, beta=beta)

        # Add the coefficents that produced the aleatoric draws to the result
        predictions["coefficients"] = coefficients

        return predictions


    @staticmethod
    def rank_by_uncertainty(result):
        """Individual indices ordered from most to least uncertain.

        We rank individuals based on their associated uncertainty score U. 
        We then explore the individuals with the highest and lowest uncertainty
        level.

        Args:
            result: dict as returned by predict_with_uncertainty.

        Returns:
            np.ndarray of int; individual indices, most uncertain first. Ties are
                broken towards the smaller index.
        """
        order = np.argsort(-result["U"], kind="stable")
        return order 

    def predict_epistemic_trajectories(self, X, times):
        """Predict trajectory values of all ensemble members at `times` for one 
            covariate dict x (or an entire batch)
               
        We use this function for plotting.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,).
            times: 1-D sequence of N time points in [0, T].

        Returns:
            np.ndarray of shape (D, M, N); predicted values in original units.
        """
        # Step 1: Ask each member for ALL D individuals' mean trajectories, so
        #         every element of the list is one model's full (D, N) block:
        #            by_member = [
        #               [[M0_P0_curve], [M0_P1_curve], ...],   # Model 0
        #               [[M1_P0_curve], [M1_P1_curve], ...],   # Model 1
        #            ]
        by_member = [engine.predict_trajectory(X, times) # (D, N)
                     for engine in self.engines]           # M blocks

        # Step 2: Stack along a NEW middle axis so the cloud axis lands where
        #         predict_aleatoric_trajectories puts it, i.e. (D, cloud, N).
        #         Regrouped by individual, element [d] is that person's M curves:
        #            [
        #               [M0_P0_curve, M1_P0_curve, ...],   # Person 0's cloud
        #               [M0_P1_curve, M1_P1_curve, ...],   # Person 1's cloud
        #            ]
        #         axis=0 would instead give (M, D, N) -- grouped by model, which
        #         is the transpose of what a per-individual figure wants.
        return np.stack(by_member, axis=1) # M x (D, N) -> (D, M, N)

    ####################################################################
    # Implementation with aleatoric draws nested in epistemic ensemble
    ####################################################################

    def _build_combined_shape_summaries(self, X, n_samples, rng, keep_coefficients=False):
        """Build every individual's nested cloud of M x n shape summaries.

        To analyse the uncertainty profile of epistemic and aleatoric uncertainty,
        we need to build a nested cloud of shape summaries where,
            - an outer loop runs over ensemble members (epistemic, and 
            - an inner loop draws random effects from that member's own Sigma (aleatoric). 
        
        Note the rng is handed through the ensemble members in sequence, so no two
        members draw the same random effects.

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.
            n_samples: int, n; aleatoric draws per member, so each individual's
                cloud holds M * n summaries in total.
            rng: np.random.Generator driving the draws.
            keep_coefficients: bool; whether to also return the draws behind the
                clouds. Off by default because the nested cloud is large (multible GB)

        Returns:
            tuple (clouds, group_ids, coefficients):
                clouds: list of length D; element i is that individual's list of
                    M * n summaries, ordered member by member.
                group_ids: np.ndarray (M * n,) of int; the member index behind
                    each position in every cloud.
                coefficients: list of M arrays of shape (D, n, B), ordered member
                    by member to match group_ids, or None. Each block is in THAT
                    member's normalised units, since ensemble members carry their
                    own normalisers, so each must be evaluated by its own engine.
        """
        rng = np.random.default_rng(rng)
        # Step 1: For each ensemble member, draw their own n aleatoric summaries for every individual
        #            by_member = [
        #               [[M0_P0_S0, M0_P0_S1], [M0_P1_S0, M0_P1_S1], ...],  # Model 0
        #               [[M1_P0_S0, M1_P0_S1], [M1_P1_S0, M1_P1_S1], ...],  # Model 1
        #            ]
        # M0_P0_S0 = model 0, person 0, aleatoric draw 0.
        drawn = [
            self._build_aleatoric_shape_summaries(X, n_samples, rng, member=m)
            for m in range(len(self.engines))
        ]
        by_member = [clouds for clouds, _ in drawn]
        coefficients = [block for _, block in drawn] if keep_coefficients else None

        # Step 2: Pool the members' draws into one cloud per individual
        #  Regroup by individual and flatten the member axis away, i.e.
        #         (M, D, n) -> (D, M * n).
        #            [
        #               [M0_P0_S0, M0_P0_S1, ..., M1_P0_S0, M1_P0_S1, ...],  # Person 0
        #               [M0_P1_S0, M0_P1_S1, ..., M1_P1_S0, M1_P1_S1, ...],  # Person 1
        #            ]
        D = len(by_member[0])
        clouds = [
            [summary for member_clouds in by_member for summary in member_clouds[i]]
            for i in range(D)
        ]

        # Step 3: Label each position with the member that supplied it.
        #         Pairs sharing a group_id are the
        #         within-member (aleatoric) pairs; pairs across ids carry the
        #         between-member (epistemic) contribution to U.
        group_ids = np.repeat(np.arange(len(self.engines)), n_samples)

        return clouds, group_ids, coefficients

    def predict_with_combined_uncertainty(self, X, n_samples, rng, alpha=1/2, beta=1/4,  keep_coefficients=False):
        """Quantify epistemic and aleatoric uncertainty together for every individual.
    
        In this function we consider a case accounting for full predictive uncertainty
            - we have M ensemble members; and 
            - every ensemble member contributes n draws from its own 
              random-effects covariance.

        For each individual/for a given covariate vector, the pooled cloud carries 
        both epistemic and aleatoric uncertainty. 

        The cost of this run is: M * n * D shape extractions. 
        At M = 100 ensemble members, 
           n = 20 aleatoric draws, and 
           D = 1000 individuals 
        that is two million shape extractions. 

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,).
            n_samples: int, n; aleatoric draws per member
            rng: np.random.Generator driving the draws.
            alpha: float, slope disagreement weight.
            beta: float, curvature disagreement weight.

        Returns:
            dict as described in _build_uncertainty_result, with source
                "combined" and two extra keys:
                    "group_ids":    np.ndarray (M * n,) of int; the member behind
                                    each cloud position.
                    "n_per_member": int; the n that was requested, since
                                    "n_samples" records the pooled cloud size.

        Note:
        Note that the returned uncertainty can be decomposed into epistemic and
        aleatoric uncertainty. To enable this we return  "group_ids". Using these,
        we can identify the within-member (aleatoric) and between-member (epistemic) 
        contributions to U. 
        """
        # Step 1: Build the nested cloud and record which member supplied each draw
        clouds, group_ids, coefficients = self._build_combined_shape_summaries(
            X, n_samples, rng, keep_coefficients)


        # Step 2: Compute uncertainties + profile and consensus for all individuals
        predictions = self._build_uncertainty_result(
            clouds, source="combined", n_samples=len(group_ids),
            alpha=alpha, beta=beta)


        # Step 3: Attach what the uncertainty decomposition will need
        predictions["group_ids"] = group_ids
        predictions["n_per_member"] = n_samples
        if coefficients is not None:
            predictions["coefficients"] = coefficients

        return predictions


    def predict_combined_trajectories(self, result, times, indices=None):
        """Evaluate the stored combined draws at `times`.

        Args:
            result: dict from predict_with_combined_uncertainty, called with
                keep_coefficients=True.
            times: 1-D sequence of N time points in [0, T].
            indices: optional 1-D array-like of int; which individuals to
                evaluate. Defaults to all D.


        Returns:
            np.ndarray of shape (D, M * n, N); predicted values in original
                units, ordered member by member to match "group_ids".

        """
        if "coefficients" not in result:
            raise KeyError("result has no draws; call "
                           "predict_with_combined_uncertainty(..., "
                           "keep_coefficients=True)")


        # Step 1: Slice selected individuals
        blocks = result["coefficients"]
        if indices is not None:
            blocks = [block[indices] for block in blocks]

        # Step 2: Evaluate trajectories per engine
        by_member = [engine.predict_aleatoric_trajectories(block, times)
                     for engine, block in zip(self.engines, blocks)]

        # Step 3: Combine Results
        return np.concatenate(by_member, axis=1) # (len(indices), M * n, N)

    ####################################################################
    # Value-space uncertainty
    ####################################################################

    def _build_value_uncertainty_result(self, curves, times, references, source,
                                        n_samples=None, keep_curves=False):
        """Score every individual's cloud of trajectories in value space.

        This is the value-space counterpart of _build_uncertainty_result. 

        Given a clouds of trajectories, compute an uncertainty score as the mean
        pairwise absolute value-space distance. 

        Args:
            curves: np.ndarray of shape (D, M, N); every individual's cloud of M
                trajectories on a shared grid of N times, in original y units.
            times: np.ndarray of shape (N,); the evaluation grid, evenly spaced
                and running from 0 to the horizon.
            references: sequence of length D of int; the medoid trajectory of each
                clourd (as the shape-space result records in "selected_indices").
            source: str; "epistemic", "aleatoric" or "combined". Recorded on the
                result so that clouds measured under different regimes are not
                compared.
            n_samples: int or None; the cloud size M, where the source has such
                a parameter.
            keep_curves: bool; whether each profile keeps the trajectories it
                was scored from. Off by default due to data size.

        Returns:
            dict with keys:
                "V":          np.ndarray (D,); value-space uncertainty per
                              individual, each the fraction of that individual's
                              amplitude its cloud spans on average.
                "profiles":   list of length D of value-uncertainty profiles, as
                              returned by compute_value_uncertainty.
                "amplitudes": np.ndarray (D,); the A each profile was divided
                              by, read back from the profiles.
                "times":      np.ndarray (N,); the shared evaluation grid.
                "T", "source", "n_samples": the conditions everything was
                              computed under.
        """
        # Step 1: Score each individual's cloud against its own amplitude
        D = len(curves)
        V = np.empty(D, dtype=float)
        profiles = []

        for i in range(D):
            V[i], profile = compute_value_uncertainty(
                curves[i], times, references[i])

            # The profile carries its own cloud so that it can be decomposed later.
            if not keep_curves:
                profile.pop("curves")

            profiles.append(profile)

        # Step 2: Build the results dictionary
        return {"V": V,
                "profiles": profiles,
                "amplitudes": np.array([p["amplitude"] for p in profiles]),
                "times": times,
                "T": self.T,
                "source": source,
                "n_samples": n_samples}


    def predict_with_epistemic_value_uncertainty(self, X, times, references,
                                                keep_curves=False):
        """Quantify epistemic uncertainty in value space for every individual.

        This function is th value-space counterpart of predict_with_epistemic_uncertainty. 

        Args:
            X: dict mapping each covariate name to an np.ndarray of shape (D,)
                holding that covariate's value for all D individuals.
            times: np.ndarray of shape (N,); the evaluation grid, evenly spaced
                and running from 0 to the horizon.
            references: sequence of length D of int; the medoid trajectory of each
                clourd (as the shape-space result records in "selected_indices").
            keep_curves: bool; whether each profile keeps its own cloud.

        Returns:
            dict as described in _build_value_uncertainty_result, with source
                "epistemic".
        """
        # Step 1: Evaluate every member's trajectory for every individual
        curves = self.predict_epistemic_trajectories(X, times)   # (D, M, N)

        # Step 2: Score each individual's cloud
        return self._build_value_uncertainty_result(
            curves, times, references, source="epistemic",
            n_samples=len(self.engines), keep_curves=keep_curves)


    def predict_with_combined_value_uncertainty(self, result, times, references,
                                                indices=None, keep_curves=False):
        """Quantify combined uncertainty in value space for every individual.

        This function is the value-space counterpart of predict_with_combined_uncertainty, i.e. 
        just like predict_with_epistemic_value_uncertainty but now for both sources.

        Args:
            result: dict from predict_with_combined_uncertainty, called with
                keep_coefficients=True.
            times: np.ndarray of shape (N,); the evaluation grid, evenly spaced
                and running from 0 to the horizon.
            references: sequence of length D of int; the medoid trajectory of each
                clourd (as the shape-space result records in "selected_indices").
            indices: optional 1-D array-like of int; which individuals to score.
                Defaults to all D, which is only affordable in chunks.
            keep_curves: bool; whether each profile keeps its own cloud.

        Returns:
            dict as described in _build_value_uncertainty_result, with source
                "combined" and one extra key:
                    "group_ids": np.ndarray (M * n,) of int; the member behind
                                 each cloud position, carried through so the
                                 result can later be decomposed.
        """
        # Step 1: Evaluate trajectories
        curves = self.predict_combined_trajectories(result, times, indices)

        # Step 2: Score each individual's cloud
        predictions = self._build_value_uncertainty_result(
            curves, times, references, source="combined",
            n_samples=len(result["group_ids"]), keep_curves=keep_curves)

        # Step 3: Preserve group info
        predictions["group_ids"] = result["group_ids"]

        return predictions