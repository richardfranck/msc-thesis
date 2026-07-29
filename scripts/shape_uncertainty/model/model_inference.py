import torch
import numpy as np

# Shape extraction utilities
from scripts.shape_uncertainty.shape_extraction.shape_summary import extract_shape_summary
from scripts.shape_uncertainty.spline_basis.bspline_basis import basis_matrix

# Uncertainty utilities
from scripts.shape_uncertainty.shape_extraction.shape_uncertainty import (
    compute_shape_uncertainty,
    medoid_summary,
)
class InferenceEngine:
    """Compute trajectorfy forecast from a covariate vector.

    Args:
        model: a trained RandomEffectsModel (e.g. GaussianModel).
        normaliser: the fitted covariate normaliser (supplies transform_one).
        knot_objects: dict with "basis_functions", "C", "breakpoints".
        upsilon_rel_1: float, relative slope threshold for shape extraction.
        upsilon_rel_2: float, relative curvature threshold for shape extraction.
    """
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
            coeff_vector: A torch.Tensor of shape (D, B); the mean coefficients, 
                one row per individual, in the normaliser's fitted column order.
        """
        # Check attributes are populated
        if self.populated is False:
            raise RuntimeError("Call 'populate_attributes()' first.")
        
        # Get the mean spline coefficients h_theta(x)
        self.model.eval()
        Z = self.normaliser.transform(X) # (D, M)
        with torch.no_grad():
            coeff_vector = self.model.get_coefficients(Z) # (D, B)
            return coeff_vector


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

    def predict_trajectory_values(self, x, times):
        """Predict mean trajectory values at `times` for one covariate dict x.

        We use this function to plot the trajectory of a given covariate vector. 

        Args:
            x: dict mapping each covariate name to a scalar value for one
                individual.
            times: 1-D sequence of time points (floats) in [0, T] at which to 
                evaluate the mean trajectory. This is used for plotting.

        Returns:
            torch.Tensor of shape (len(times),);
        """
        # Step 1: Reformat individual covariate vector into batch dimension for _get_mean_coefficients
        X = {name: np.asarray([value], dtype=float) for name, value in x.items()} # (1, M)
        w = self._get_mean_coefficients(X)[0] # (B,)

        # Step 2: Evaluate the spline and return to original units
        Phi = torch.as_tensor(basis_matrix(np.asarray(times, float),
                                           self.knot_objects["basis_functions"]))
        y_norm = Phi @ w
        y_norm  = y_norm.detach().cpu().numpy()
        y = self.normaliser.inverse_transform_y(y_norm)
        return torch.as_tensor(y)

    # -------------- single-model aleatoric (deferred; needs Sigma_hat) --------------

    def sample_aleatoric_coefficients(self, X, n_samples, rng=None):
        """Draw coefficient vectors h_theta(x) + u, u ~ N(0, Sigma_hat), for one x.

        The aleatoric cloud of coefficient vectors under THIS model: the mean
        coefficients perturbed by random-effect draws from this model's trained
        covariance. The single-model building block that UncertaintyEngine nests
        over an ensemble.
        [Deferred: requires the trained model's Sigma_hat.]

        Args:
            x: dict of one individual's raw covariates.
            n_samples: int, number of aleatoric draws.
            rng: random generator.

        Returns:
            torch.Tensor of shape (n_samples, B); the perturbed coefficient draws.
        """
        pass

    def sample_aleatoric_shapes(self, x, n_samples, rng=None):
        """Aleatoric shape distribution for one x under THIS model.

        Extracts a shape summary from each aleatoric coefficient draw, yielding
        the within-model aleatoric shape distribution for a single covariate
        vector.
        [Deferred: requires Sigma_hat and the shape-distribution machinery.]

        Args:
            x: dict of one individual's raw covariates.
            n_samples: int, number of aleatoric draws.
            rng: random generator.

        Returns:
            list of n_samples shape summaries (the aleatoric shape cloud).
        """
        pass


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

    def _build_shape_summaries(self, X):
        """Build every individual's cloud of M shape summaries.

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

    def predict_with_uncertainty(self, X, alpha=1/2, beta=1/4):
        """Report the ensemble's shape uncertainty and consensus for every individual.

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
            dict with keys:
                "U": np.ndarray (D,); shape uncertainty 
                "profiles": list of length D of uncertainty profiles,
                "consensus": list of length D; the shape summary the
                    ensemble reports for each individual.
                "member_indices": np.ndarray (D,) of int; which member supplied
                                 each consensus.
        """
        # Step 1: Build every individual's cloud of M summaries
        shapes = self._build_shape_summaries(X)
        D = len(shapes)

        # Step 2: Compute uncertainties + profile and consensus for all individuals
        U = np.empty(D, dtype=float)
        member_indices = np.empty(D, dtype=int)
        profiles, consensus = [], []

        for i, shape_distribution in enumerate(shapes):
            U[i], profile = self._compute_shape_uncertainty(shape_distribution, alpha, beta)
            best_summary, member_indices[i] = self._compute_consensus_summary(shape_distribution, profile, alpha, beta)
            profiles.append(profile)
            consensus.append(best_summary)

        predictions = {"U": U, 
                        "consensus": consensus, 
                        "member_indices": member_indices,
                        "profiles": profiles,
        }

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





