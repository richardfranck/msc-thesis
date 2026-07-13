import torch
import numpy as np

# Shape extraction utilities
from scripts.shape_uncertainty.shape_extraction.shape_summary import extract_shape_summary
from scripts.shape_uncertainty.spline_basis.bspline_basis import basis_matrix


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
            "upsilon_rel_1": None,
            "upsilon_rel_2": None,
        }

        self.populated = False

    def populate_attributes(self, model, normaliser, basis_functions, 
                            C , breakpoints, upsilon_rel_1, upsilon_rel_2):
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
            self.shape_config["upsilon_rel_1"] = upsilon_rel_1
            self.shape_config["upsilon_rel_2"] = upsilon_rel_2

        # Track whether the attributes have been filled yet
        self.populated = True

    ######################################################################
    # ------------------------- Point Prediction -------------------------
    ######################################################################

    def _get_mean_coefficients(self, x):
        """Predict mean spline coefficients (B,) for one covariate dict x.

        Args:
            x: dict mapping each covariate name to a scalar value for one
                individual.

        Returns:
            coeff_vector: A torch.Tensor of shape (B,) of the mean coefficients.
        """
        # Check attributes are populated
        if self.populated is False:
            raise RuntimeError("Call 'populate_attributes()' first.")
        
        # Get the mean spline coefficients h_theta(x)
        self.model.eval()
        z = self.normaliser.transform_one(x).unsqueeze(0)      # (1, M)
        with torch.no_grad():
            coeff_vector =  self.model.get_coefficients(z).squeeze(0)        # (B,)
            return coeff_vector


    def predict_mean_shape_summary(self, x):
        """Predict the shape summary for the mean trajectory for one covariate dict x.

        Args:
            x: dict mapping each covariate name to a scalar value for one
                individual.

        Returns:
            a shape summary [(state, start_time), ...].
        """
        w = self._get_mean_coefficients(x).reshape(-1, 1)         # (B, 1)
        w = w.detach().cpu().numpy()  # extract_shape_summary expects numpy
        shape_summary = extract_shape_summary(
                            w, 
                            self.knot_objects["C"], 
                            self.knot_objects["breakpoints"], 
                            self.shape_config["upsilon_rel_1"],
                            self.shape_config["upsilon_rel_2"]
        )
        return shape_summary

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
        w = self._get_mean_coefficients(x)                        # (B,)
        Phi = torch.as_tensor(basis_matrix(np.asarray(times, float),
                                           self.knot_objects["basis_functions"]))
        return Phi @ w

    # -------------- single-model aleatoric (deferred; needs Sigma_hat) --------------

    def sample_aleatoric_coefficients(self, x, n_samples, rng=None):
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
    """Quantify predictive uncertainty over shapes for a single covariate vector.

    Orchestrates an ENSEMBLE of trained models (each wrapped as an
    InferenceEngine) to produce shape-uncertainty for one input. Handles the
    layers a single model cannot:
        - epistemic: variation ACROSS the ensemble (which data each model saw),
        - combined:  the NESTED draw -- for each model (epistemic), draw many
                     random effects u ~ N(0, Sigma_hat) (aleatoric, delegated to
                     each member engine's sample_aleatoric_shapes),
        - summarisation: modal shape + probability, and dispersion measures,
        - the user-facing prediction that surfaces the mean-vs-modal contrast.
    Aleatoric-only draws for a single model live on InferenceEngine; this class
    consumes those and adds the cross-model and summarisation machinery.

    Args:
        engines: sequence of InferenceEngine instances (the trained ensemble);
            each contributes its own Sigma_hat for the inner aleatoric draws.
    """

    def __init__(self, engines):
        self.engines = engines

    def sample_epistemic_shapes(self, x):
        """Epistemic shape distribution for one x across the ensemble.

        Extracts each member model's predicted MEAN shape, yielding the
        epistemic shape distribution (variation due to which data each model was
        trained on).
        [Deferred: requires the bootstrap/ensemble training infrastructure.]

        Args:
            x: dict of one individual's raw covariates.

        Returns:
            list of shape summaries, one per ensemble model.
        """
        pass

    def sample_combined_shapes(self, x, n_aleatoric, rng=None):
        """Full predictive shape distribution for one x: NESTED epistemic x aleatoric.

        Outer loop over the ensemble (epistemic); inner loop draws n_aleatoric
        random effects within each model via that member's
        sample_aleatoric_shapes (aleatoric). The pooled cloud mixes both sources
        and is decomposable (across-model-mean variation = epistemic; mean
        within-model spread = aleatoric). A NESTED draw, not a concatenation of
        two independent clouds.
        [Deferred: requires member Sigma_hat, the ensemble, and the
        shape-distribution machinery.]

        Args:
            x: dict of one individual's raw covariates.
            n_aleatoric: int, aleatoric draws per model.
            rng: random generator.

        Returns:
            the combined cloud of shape summaries (and/or a structure retaining
            the per-model grouping for decomposition).
        """
        pass

    def modal_shape(self, shape_distribution):
        """Most probable shape summary in a distribution, and its probability.

        Returns the modal composition (most frequent state sequence) and the
        fraction of the cloud exhibiting it.
        [Deferred: requires the shape-distribution machinery.]

        Args:
            shape_distribution: list of shape summaries (aleatoric, epistemic, or
                combined).

        Returns:
            (modal_summary, probability): the modal shape and its frequency in [0,1].
        """
        pass

    def shape_uncertainty(self, shape_distribution):
        """Dispersion measures over a shape distribution.

        Summarises a cloud of shape summaries into uncertainty measures (e.g.
        entropy over distinct compositions, per-transition timing dispersion).
        [Deferred: requires the shape-distribution machinery; relates to M7.]

        Args:
            shape_distribution: list of shape summaries.

        Returns:
            dict of dispersion measures.
        """
        pass

    def predict_with_uncertainty(self, x, n_aleatoric, rng=None):
        """User-facing prediction for one x: cloud, modal shape, and the contrast.

        Assembles the full per-input output using the NESTED combined draw:
            - the mean-coefficient trajectory (conventional point estimate),
            - a representative curve for the MODAL shape and its probability,
            - the combined epistemic x aleatoric shape cloud,
            - the shape-uncertainty measures.
        Surfaces the CONTRAST between the mean-coefficient curve and the modal-
        shape curve: agreement => clean prediction; divergence => multimodal /
        unrepresentative mean (the high-uncertainty case).
        [Deferred: requires sample_combined_shapes, modal_shape, shape_uncertainty.]

        Args:
            x: dict of one individual's raw covariates.
            n_aleatoric: int, aleatoric draws per model.
            rng: random generator.

        Returns:
            dict with the mean trajectory, the modal-shape trajectory and its
                probability, the shape cloud, and the uncertainty measures.
        """
        pass





