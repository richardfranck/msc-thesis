import numpy as np
import pandas as pd
import torch
import itertools

# B-Spline utilities
from scripts.shape_uncertainty.spline_basis.bspline_basis import (
    basis_matrix,
    build_knot_dictionary,
    pointwise_sd_from_covariance,
)

# Data generation and ground truth
from scripts.shape_uncertainty.simulated_data.data_generator import (
    WilkersonDGP,
    generate_simulated_dataset,
)
from scripts.shape_uncertainty.simulated_data.shape_ground_truth import (
    get_analytic_shape_summary,
    estimate_true_random_effects
)

# Model training and inference utilities
from scripts.shape_uncertainty.model.model_training import Tuner
from scripts.shape_uncertainty.model.model_inference import (
    InferenceEngine,
    build_inference_engine,
)
from scripts.shape_uncertainty.model.model_evaluation import ModelEvaluator

# Shape extraction utilities
from scripts.shape_uncertainty.shape_extraction.shape_summary import extract_shape_summary
from scripts.shape_uncertainty.shape_extraction.shape_distance import (
    mean_shape_sequence_match,
    mean_shape_summary_distance,
)


class ExtractionEvaluator:
    """Assess the faithfulness of shape extraction, independent of any model."""
    def __init__(self, dataset):
        self.dataset = dataset

    def _fit_ols_splines(self, nr_obs, nr_interior_knots):
        """Fit the best-possible spline to each individual's noise-free curve.

        Solve for the least-squares problem for the spline coefficients.
        Snice there is no model for mapping covariates X to spline coefficents
        and no observation noise here, the result is the best trajectory that
        can be achieved for a given number of observation points and spline
        dimension.

        Args:
            nr_obs: int; number of evenly spaced observation times on [0, T].
            nr_interior_knots: int; number of interior knots for the basis.

        Returns:
            dict with keys:
                "coefficients": np.ndarray of shape (D, B); one spline
                    coefficient vector per individual.
                "C": the monomial conversion tensor for the basis.
                "breakpoints": np.ndarray of the interior plus boundary knots.
        """
        # Step 1: Build the shared observation grid
        T = self.dataset.T
        observation_times = np.linspace(0.0, T, nr_obs)

        # Step 2: Build the spline basis 
        knot_objects = build_knot_dictionary(nr_interior_knots, T)
        Phi = basis_matrix(observation_times, knot_objects["basis_functions"])

        # Step 3: Evaluate every individual's noise-free curve on that grid
        Y_true = np.array(self.dataset.true_curves_at(observation_times))

        # Step 4: Solve for all individuals at once (Phi is shared)
        coefficients, _, _, _ = np.linalg.lstsq(Phi, Y_true.T, rcond=None)

        return {
            "coefficients": coefficients.T,
            "C": knot_objects["C"],
            "breakpoints": knot_objects["breakpoints"],
        }

    def evaluate_extraction(self, nr_obs, nr_interior_knots, zeta_rel=0.0,
                            u1_rel=0.0, u2_rel=0.0, upsilon_rel_prune=0.0,
                            do_prune=False):
        """Score shape extraction from best-possible splines against the truth.

        1. Fit a the splines at the given resolution of noise-free y using OLS,
        2. Extract a shape summary from each fitted spline, 
        2. Get the ground truth shape summary for each trajectory, 
        3. Grade the extracted shape summary against the ground truth shape summary. 
            - Note that both estimated and true summary use the same thresholds 
              so they describe shapes in one vocabulary
        4. Return the shape extraction accuracy and the generated summaries. 

        Args:
            nr_obs: int; number of evenly spaced observation times on [0, T].
            nr_interior_knots: int; number of interior knots for the basis.
            zeta_rel: float; relative transition de-duplication tolerance
                (applies to spline extraction only).
            u1_rel: float; relative slope significance threshold.
            u2_rel: float; relative curvature significance threshold.
            upsilon_rel_prune: float; relative pruning significance threshold.
            do_prune: bool; whether to apply slope then curvature pruning.

        Returns:
            dict with keys:
                "accuracy": float in [0, 1]; the exact-sequence-match share.
                "distance": float in [0, 1]; the mean shape-summary distance (0 = identical).
                "predicted": list of length D; summaries from the fitted splines.
                "truth": list of length D; analytic ground-truth summaries.
        """
        # Step 1: Fit the best-possible spline for every individual
        fit = self._fit_ols_splines(nr_obs, nr_interior_knots)

        # Step 2: Extract a shape summary from each fitted spline
        predicted = []
        for w in fit["coefficients"]:
            summary = extract_shape_summary(
                w.reshape(-1, 1), fit["C"], fit["breakpoints"],
                zeta_rel=zeta_rel,
                upsilon_rel_1=u1_rel,
                upsilon_rel_2=u2_rel,
                upsilon_rel_prune=upsilon_rel_prune,
                do_prune=do_prune,
            )
            predicted.append(summary)

        # Step 3: Get the analytic ground truth at the same thresholds
        truth = get_analytic_shape_summary(self.dataset, u1_rel, u2_rel,
                               upsilon_rel_prune, do_prune)

        # Step 4: Score the predicted state sequences against the truth
        accuracy = mean_shape_sequence_match(predicted, truth)
        distance = mean_shape_summary_distance(predicted, truth, self.dataset.T)

        return {"accuracy": accuracy, "distance": distance,
                "predicted": predicted, "truth": truth}

    # -------------------------------------------------------------------- #
    # --------------------- Shape Extraction Analysis -------------------- #
    # -------------------------------------------------------------------- #

    def sweep_observations(self, variable_nr_obs, fixed_nr_knots, zeta_rel, fixed_u1_rel, fixed_u2_rel, upsilon_rel_prune, do_prune):
        """Evaluate shape-extraction accuracy across varying observation densities.

        Evaluate how extraction accuracy varies as we vary the number of observations
        nr_obs. The other inputs (nr_knots, u1_rel, u2_rel) remain fixed.
        
        Note: We compare the analytic shape summart and the shape summary our extractor
        produces based on the data generating process. No model is involved. 

        Args:
            variable_nr_obs (list): The observation number counts over which to sweep.
            fixed_nr_knots (list): The interior knot counts (fixed). 
            zeta_rel (float): Relative transition de-duplication tolerance (fixed).
            fixed_u1_rel (float): Relative slope threshold (fixed).
            fixed_u2_rel (float): Relative curvature threshold (fixed).
            upsilon_rel_prune (float): Relative pruning threshold (fixed).
            do_prune (bool): Whether to apply slope then curvature pruning (fixed).

        Returns:
            list of (nr_obs, accuracy) pairs.
        """
        rows = []
        for n in variable_nr_obs:
            result = self.evaluate_extraction(
                n, fixed_nr_knots, zeta_rel=zeta_rel,
                u1_rel=fixed_u1_rel, u2_rel=fixed_u2_rel,
                upsilon_rel_prune=upsilon_rel_prune, do_prune=do_prune,
            )
            rows.append({
                "nr_obs": n,
                "accuracy": result["accuracy"],
                "distance": result["distance"],
            })
        return rows


    def sweep_knots(self, fixed_nr_obs, variable_nr_knots, zeta_rel, fixed_u1_rel, fixed_u2_rel, upsilon_rel_prune, do_prune):
        """Evaluate shape-extraction accuracy across varying spline complexities.

        Args:
            fixed_nr_obs (int): The observation count per individual (fixed).
            variable_nr_knots (list): The interior knot counts over which to sweep.
            zeta_rel (float): Relative transition de-duplication tolerance (fixed).
            fixed_u1_rel (float): Relative slope threshold (fixed).
            fixed_u2_rel (float): Relative curvature threshold (fixed).
            upsilon_rel_prune (float): Relative pruning threshold (fixed).
            do_prune (bool): Whether to apply slope then curvature pruning (fixed).


        Returns:
            list of dicts, one per swept value, each with keys
                "nr_interior_knots" (int), "accuracy" (float in [0, 1], higher
                is better) and "distance" (float in [0, 1], lower is better).
        """
        rows = []
        for k in variable_nr_knots:
            result = self.evaluate_extraction(
                fixed_nr_obs, k,
                zeta_rel=zeta_rel,
                u1_rel=fixed_u1_rel,
                u2_rel=fixed_u2_rel,
                upsilon_rel_prune=upsilon_rel_prune,
                do_prune=do_prune,
            )
            rows.append({
                "nr_interior_knots": k,
                "accuracy": result["accuracy"],
                "distance": result["distance"],
            })
        return rows



class GaussianModelEvaluator:
    """Assess how faithfully a fitted GaussianModel recovers the DGP's 
        random-effect covariance Sigma(x).

    The GaussianModel reports an individual-level covariance
                            Sigma(x) = s(x)^2 Sigma_0 
    over the spline coefficients. For different levels of unobserved heterogeneity 
    in the Wilkerson DGP, this class

        Step 1: Reconstructs the ground truth covariance Sigma_ture the DGP implies.
        Step 2: Fits models under a given pair of training penalites (lambda_mean, lambda_re)
                and obtaines associated estimates of the covariance Sigma. 
        Step 3: Scores the fitted covariance Sigma_hat against that ground truth Sigma_true
                Sigma is compared in value space via a comparison of the pointwise band sd 
                around the outcome vairable y:
                                    sqrt(phi(t)^T Sigma phi(t))
                which is how Sigma materialises for a model user. 

    Args:
        base_hyperparams: dict; the DGP hyperparameters {"g0", "d0", "rho0"}. 
        knot_objects: dict from build_knot_dictionary.
        shape_config: dict; the shape-extraction thresholds ("zeta_rel",
            "upsilon_rel_1", "upsilon_rel_2", "upsilon_rel_prune", "do_prune"),
        data_config: dict; the observation design and split sizes, with keys
            "D_train", "D_val", "D_test", "nr_obs", "sigma" and "regular".
        train_config: dict; the optimisation settings held fixed across the
            sweep, with keys "nr_epochs", "patience", "model_cls" and
            "model_kwargs".
        T: float; the end of the observation window.
        seed: int; the base seed for data generation and for the first fit of
            every configuration.
    """
    def __init__(self, base_hyperparams, knot_objects, shape_config,
                 data_config, train_config, T, seed=0):

        # Experimental design
        self.base_hyperparams = base_hyperparams
        self.knot_objects = knot_objects
        self.shape_config = shape_config
        self.data_config = data_config
        self.train_config = train_config
        self.T = float(T)
        self.seed = seed

        # The observation grid the regular design gives every individual
        self.observation_times = np.linspace(0.0, self.T, data_config["nr_obs"])

        # Monte Carlo settings for the ground truth
        self.truth_settings = {"n_ref": 200, "n_mc": 800}

        # Caches keyed by heterogeneity level
        self._splits = {}
        self._shape_summaries = {}
        self._covariance_truth = {}


    def _build_hyperparams(self, alpha_hetero):
        """Return the DGP hyperparameters at one heterogeneity level.

        Copies the shared base hyperparameters and sets both heterogeneity
        terms to the same value, so that a single scalar controls the size of
        the unobserved variation in the DGP.

        Args:
            alpha_hetero: float; the value taken by both alpha_g and alpha_d.

        Returns:
            dict; the base hyperparameters with "alpha_g" and "alpha_d" set to
                alpha_het.
        """
        hyperparams = dict(self.base_hyperparams)
        hyperparams["alpha_g"] = float(alpha_hetero)
        hyperparams["alpha_d"] = float(alpha_hetero)

        return hyperparams

    def get_dataset_split(self, alpha_hetero):
        """Return the test, validation and training split at one heterogeneity level.

        Regenerate the DGP at alpha_hetero,
        Draw a dataset individuals under the stored observation design
        Split the dataset into D_train, D_val, D_test 

        Args:
            alpha_hetero: float; the DGP's alpha_g = alpha_d.

        Returns:
            tuple of three SimulatedDatasets in the order (test, val, train)
        """
        # Step 0: Return the cached split if drawn before
        if alpha_hetero in self._splits:
            return self._splits[alpha_hetero]

        # Step 1: Rebuild the DGP at alpha_hetero
        dgp = WilkersonDGP(T=self.T,
                           hyperparams=self._build_hyperparams(alpha_hetero))

        # Step 2: Draw the dataset
        design = self.data_config
        D_total = design["D_train"] + design["D_val"] + design["D_test"]
        dataset = generate_simulated_dataset(
            dgp, 
            D=D_total,
            N=design["nr_obs"], 
            sigma=design["sigma"],
            regular=design["regular"], 
            include_endpoints=True, 
            seed=self.seed,
        )

        # Step 3: Split the data
        split = dataset.split_test_val_train(
            D_train=design["D_train"], 
            D_val=design["D_val"],
            D_test=design["D_test"],
        )
        self._splits[alpha_hetero] = split

        return split

    def _get_true_shape_summaries(self, alpha_hetero):
        """Return the analytic shape summaries of the test split.

        Read the ground truth analytic shape summaries from the DGP's parameters.
        This is used to score shape extraction performance. 

        Args:
            alpha_hetero: float; the DGP's alpha_g = alpha_d.

        Returns:
            list of length D_test; one analytic shape summary per test
                individual.
        """
        # Return the cached summaries if read before
        if alpha_hetero in self._shape_summaries:
            return self._shape_summaries[alpha_hetero]

        # Read the summaries off the DGP's parameters
        test = self.get_dataset_split(alpha_hetero).test
        summaries = get_analytic_shape_summary(
            test,
            self.shape_config["upsilon_rel_1"],
            self.shape_config["upsilon_rel_2"],
            self.shape_config["upsilon_rel_prune"],
            self.shape_config["do_prune"],
        )

        # Cache
        self._shape_summaries[alpha_hetero] = summaries

        return summaries

    
    # -------------------------- Ground truth -------------------------- #

    def get_covariance_truth(self, alpha_hetero, n_ref, n_mc):
        """Reconstruct the covariance the DGP implies, for a reference subset of the test split.

        Since DGP does not expose a covariance over spline coefficients, we 
        reconstruct it via Monte Carlo for a set of reference individuals as 
        follows:
        - Sraw `n_mc` latent pairs `(z_g, z_d) ~ N(0,1)`.
        - Obtain the ground truth trajectories for each latent pair.
        - Represent each curve in the model's spline coefficient space by OLS.
        - Comoute Sigma(x) as the empirical covariance of the coefficient vectors.
        - Convert each Sigma(x) into value space as the pointwise band standard dev,
          sqrt(phi(t)^T Sigma phi(t)).
          We also record the individual's true curve amplitude so that band widths 
          can be read on a scale-free basis.

        Args:
            alpha_hetero: float; the DGP's alpha_g = alpha_d.
            n_ref: int; how many test individuals to reconstruct the truth for.
            n_mc: int; latent draws used per individual.

        Returns:
            dict with keys:
                "indices": np.ndarray of shape (n_ref,); which test individuals
                    were selected.
                "Sigma": np.ndarray of shape (n_ref, B, B); the true
                    random-effect covariance of each one, in original y units.
                "band_sd": np.ndarray of shape (n_ref, N); the true pointwise
                    band sd on a shared evaluaton grid. 
                "amplitude": np.ndarray of shape (n_ref,); each individual's
                    true curve amplitude.
        """
        # Step 0: Record the settings
        self.truth_settings = {"n_ref": n_ref, "n_mc": n_mc}
        key = (alpha_hetero, n_ref, n_mc)
        if key in self._covariance_truth:
            return self._covariance_truth[key]

        test = self.get_dataset_split(alpha_hetero).test

        # Step 1: Select the reference individuals
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(test.D, size=n_ref, replace=False)

        # Step 2: Reconstruct Sigma(x) at each reference covariate point
        hyperparams = self._build_hyperparams(alpha_hetero)
        Sigma_list = []
        for i in indices:
            covariate_point = {name: float(column[i])
                               for name, column in test.X.items()}
            estimate = estimate_true_random_effects(
                covariate_point,
                hyperparams,
                self.knot_objects["basis_functions"],
                self.T,
                n_mc=n_mc,
                rng=self.seed,
            )
            Sigma_list.append(estimate["Sigma"])

        Sigma = np.stack(Sigma_list)        # list of (B, B) -> (n_ref, B, B)

        # Step 3: Carry Sigma into value space
        band_sd = pointwise_sd_from_covariance(
            Sigma, self.observation_times, self.knot_objects["basis_functions"])

        truth = {"indices": indices, "Sigma": Sigma, "band_sd": band_sd}
        self._covariance_truth[key] = truth

        return truth

    # ------------------------ Fitting one model ----------------------- #

    def _build_tuner(self, split, lambda_mean, lambda_re):
        """Build a Tuner on one split at one penalty pair.

        Args:
            split: Split namedtuple; supplies the train and val datasets.
            lambda_mean: float; the roughness penalty on the mean coefficients.
            lambda_re: float; the roughness penalty on the random effects.

        Returns:
            Tuner; configured with the shared basis and the stored optimisation
                settings.
        """
        return Tuner(
            train_dataset=split.train,
            val_dataset=split.val,
            basis_functions=self.knot_objects["basis_functions"],
            Omega=self.knot_objects["Omega"],
            nr_basis=self.knot_objects["nr_basis"],
            lambda_mean=lambda_mean,
            lambda_re=lambda_re,
            nr_epochs=self.train_config["nr_epochs"],
            patience=self.train_config["patience"],
            model_cls=self.train_config["model_cls"],
            model_kwargs=self.train_config["model_kwargs"],
        )

    def search_fixed_architecture(self, alpha_hetero, nr_trials):
        """Search the encoder architecture once, unpenalised, and return it
            for the whole sweep to reuse.

        Args:
            alpha_hetero: float; the heterogeneity level to search at.
            nr_trials: int; Optuna trials.

        Returns:
            dict; the best hyperparameters, in the form Tuner.run reports them
                and train_fixed_configuration consumes them.
        """
        # Step 1: Get the dataset split
        split = self.get_dataset_split(alpha_hetero)

        # Step 2: Search at zero penalty
        tuner = self._build_tuner(split, lambda_mean=0.0, lambda_re=0.0)

        # Step 3: Return the best parameter configuration
        return tuner.run(nr_trials=nr_trials, seed=self.seed)["best_params"]

    def _fit_penalised_model(self, alpha_hetero, hyperparas, lambda_mean, lambda_re, seed):
        """Train one GaussianModel at a fixed architecture and penalty pair.

        Args:
            alpha_hetero: float; the heterogeneity level to fit at.
            hyperparas: dict; the frozen architecture.
            lambda_mean: float; the roughness penalty on the mean coefficients.
            lambda_re: float; the roughness penalty on the random effects.
            seed: int; seeds both the weight initialisation and the shuffling.

        Returns:
            InferenceEngine wrapping the trained model, its normaliser, the
                shared basis and the shared shape_config.
        """
        # Seed torch's generator
        torch.manual_seed(seed)

        # Step 1: Train at the frozen architecture at the given penalty pair
        split = self.get_dataset_split(alpha_hetero)
        tuner = self._build_tuner(split, lambda_mean, lambda_re)
        fitted = tuner.train_fixed_configuration(
            hyperparas, rng=np.random.default_rng(seed))

        # Step 2: Wrap the trained model for prediction
        engine = build_inference_engine(
            fitted["model"], 
            fitted["normaliser"],
            self.knot_objects, 
            self.shape_config,
        )

        return engine

    # ------------------------ Scoring one fit ------------------------- #

    def _score_covariance_recovery(self, engine, alpha_hetero):
        """Score the fitted band_sd_fitted against the true band_sd_true.

        We report:
            1. To determine whether the level of uncertainty is correct, we 
               report 
                    median_band_error = median(band_sd_fitted/band_sd_true)
            2. To determine whether the spread of the heteroscedasticity is 
               correct, we report the dispersion ratio:
                     dispersion_ratio =
                            [P95(SD_hat_i) / median(SD_hat_i)]
                            / [P95(SD_true_i) / median(SD_true_i)]
            3. To determine whether indivdiduals are ordered correctly we compute
             the correlation between average band_sd_fitted and band_sd_true across 
            individuals
                        corr(avg_band_sd_fitted, corr(avg_band_sd_true)
        Args:
            engine: InferenceEngine; a trained model.
            alpha_hetero: float

        Returns:
            dict with keys:
                "band_error": float,
                "dispersion_ratio": float,
                "correlation": float,
        """
        # Step 1: Take the truth and the fitted covariance at the same individuals
        truth = self.get_covariance_truth(alpha_hetero, **self.truth_settings)
        test = self.get_dataset_split(alpha_hetero).test
        X_ref = {name: column[truth["indices"]] for name, column in test.X.items()}

        # Step 2: Reduce each individual's band to one number by averaging over
        # the observation grid
        sd_fitted = engine.predict_aleatoric_band(
            X_ref, self.observation_times).mean(axis=1)     # (n_ref,)
        sd_true = truth["band_sd"].mean(axis=1)             # (n_ref,)


        # Step 3: The comparison is a ratio throughout, so it needs a truth that
        #         is not identically zero.
        if np.any(sd_true <= 0.0):
            raise ValueError(f"alpha_hetero={alpha_hetero}. Use a small non-zero leve.")

        # Step 4: Compute the metrics to report for alpha > 0 cases
        band_error = float(np.median(sd_fitted / sd_true))

        dispersion_ratio = float(
            (np.percentile(sd_fitted, 95) / np.median(sd_fitted))
            / (np.percentile(sd_true, 95) / np.median(sd_true))
        )
        correlation = float(
            pd.Series(sd_fitted).corr(pd.Series(sd_true), method="spearman"))

        return {"band_error": band_error,
                "dispersion_ratio": dispersion_ratio,
                "correlation": correlation}



    def _score_mean_function_guards(self, engine, alpha_hetero):
        """Score the mean function.

        A penalty that improves Sigma by degrading the mean has not solved the
        problem. We therefore evaluate the performance of the prediction using
            1. mean_shape_accuracy
            2. mean_shape_distance
            3. mean_individual_r2_latent
            4. mean_individual_r2_observed

        Args:
            engine: InferenceEngine; a trained model.
            alpha_het: float; selects the cached split and its analytic
                summaries.

        Returns:
            dict with keys:
                "mean_shape_accuracy": float in [0, 1]
                "mean_shape_distance": float in [0, 1]
                "mean_individual_r2_latent": float; mean individual R^2 against
                    the latent outcomes at obs times.
                "mean_individual_r2_observed": float; mean individual R^2 against
                                    the observed outcomes at obs times.
        """
        # Step 1: Score the model's mean trajectories against the truth
        test = self.get_dataset_split(alpha_hetero).test
        evaluator = ModelEvaluator(engine, test)

        # Step 2: Evaluate shape-space metrics
        shape_metrics = evaluator.compute_shape_space_metrics(
            self._get_true_shape_summaries(alpha_hetero))

        # Step 3: Evaluate value-space metrics
        latent = evaluator.compute_value_space_metrics(target="latent")
        observed = evaluator.compute_value_space_metrics(target="observed")

        return {
            "mean_trajectory_shape_accuracy": shape_metrics["accuracy"],
            "mean_trajectory_shape_distance": shape_metrics["distance"],
            "mean_trajectory_individual_r2_latent": latent["mean_individual_r2"],
            "mean_trajectory_individual_r2_observed": observed["mean_individual_r2"],
        }

    def score_engine(self, engine, alpha_hetero):
        """Assemble the full scoring panel for one trained model.

        This function reports the results from the above two methods
            - _score_covariance_recovery
            - _score_mean_function_guards
        in a table. Additionally it reports fitted noise sd. 

        Args:
            engine: InferenceEngine; a trained model.
            alpha_hetero: float; selects the cached split and ground truth.

        Returns:
            dict; the union of _score_covariance_recovery and
                _score_mean_function_guards, plus "sigma_hat".
        """
        # Step 1: Get Sigma(x) evaluation
        results = self._score_covariance_recovery(engine, alpha_hetero)

        # Step 2: Get the shape-space and value-space evaluation
        results.update(self._score_mean_function_guards(engine, alpha_hetero))

        # Step 3: Get the estimate of the noise std dev
        results["sigma_hat"] = engine.noise_std_hat

        # Return a results dict 
        return results


    def fit_and_score(self, alpha_hetero, hyperparas, lambda_mean, lambda_re, seed):
        """Train one model at one configuration and return its scoring panel.

        Args:
            alpha_hetero: float; the heterogeneity level.
            hyperparas: dict; the frozen architecture.
            lambda_mean: float; the mean-coefficient roughness penalty.
            lambda_re: float; the random-effect roughness penalty.
            seed: int; the fit's seed.

        Returns:
            dict; the score_engine panel plus "alpha_hetero", "lambda_mean",
                "lambda_re" and "seed".
        """
        # Step 1: Train a model at a given configuration
        engine = self._fit_penalised_model(
            alpha_hetero, hyperparas, lambda_mean, lambda_re, seed)

        # Step 2: Return the configuration and its scroes
        return {
            "alpha_hetero": alpha_hetero,
            "lambda_mean": lambda_mean,
            "lambda_re": lambda_re,
            "seed": seed,
            **self.score_engine(engine, alpha_hetero),
        }

    # ---------------------------- Sweeps ------------------------------ #

    def sweep_penalty_grid(self, alpha_hetero, hyperparas, lambda_mean_grid,
                           lambda_re_grid, n_seeds, verbose=True):
        """Fit every penalty pair on the grid, several seeds each, 
           at ONE GIVEN heterogeneity level.

        Args:
            alpha_hetero: float; the heterogeneity level the grid is run at.
            hyperparas: dict; the frozen architecture.
            lambda_mean_grid: sequence of float; values of lambda_mean.
            lambda_re_grid: sequence of float; values of lambda_re.
            n_seeds: int; fits per cell.
            verbose: bool; print a per-cell summary line as the sweep runs.

        Returns:
            list of dicts; one fit_and_score row per fit, of length
                len(lambda_mean_grid) * len(lambda_re_grid) * n_seeds.
        """
        rows = []
        for lambda_mean, lambda_re in itertools.product(lambda_mean_grid, lambda_re_grid):
            # Step 1: Refit the cell under several seeds.
            cell = [
                self.fit_and_score(alpha_hetero, hyperparas,
                                   lambda_mean, lambda_re, self.seed + s)
                for s in range(n_seeds)
            ]
            rows.extend(cell)

            # Step 2: Report the cell as it completes as the code runs
            # The printed results are the average across the n_seeds for the given row.
            if verbose:
                band_error = [row["band_error"] for row in cell]
                r2 = [row["mean_trajectory_individual_r2_latent"] for row in cell]
                print(f"lambda_mean={lambda_mean:<8g} lambda_re={lambda_re:<8g} "
                      f"band_error={np.mean(band_error):7.3f} "
                      f"(sd {np.std(band_error, ddof=1):.3f})  "
                      f"r2={np.mean(r2):.4f} "
                      f"(sd {np.std(r2, ddof=1):.4f})", flush=True)

        return rows



    def sweep_heterogeneity_levels(self, alpha_grid, lambda_pairs, hyperparas,
                                   n_seeds, verbose=True):
        """Refit the penalty pairs across a list of heterogeneity levels.

        Args:
            alpha_grid: sequence of float; the heterogeneity levels, normally
                including 0.
            lambda_pairs: sequence of (lambda_mean, lambda_re) tuples; the
                shortlist to carry forward.
            hyperparas: dict; the frozen architecture.
            n_seeds: int; fits per (level, pair).
            verbose: bool; print a per-level summary line as the sweep runs.

        Returns:
            list of dicts; one fit_and_score row per fit, each carrying its
                "alpha_het" and the level's true band level for plotting.
        """
        rows = []
        for alpha_hetero in alpha_grid:
            # Step 1: Build this level's ground truth
            truth = self.get_covariance_truth(alpha_hetero, **self.truth_settings) # Sigma(x) true
            true_band_sd = float(np.median(truth["band_sd"].mean(axis=1))) # Value-space band true

            # Step 2: Refit every penalty pair for the data at this unobserved heterogeneity level
            for lambda_mean, lambda_re in lambda_pairs:
                cell = [self.fit_and_score(alpha_hetero, hyperparas, lambda_mean,
                                           lambda_re, self.seed + s)
                        for s in range(n_seeds)]
                rows.extend({**row, "true_band_sd": true_band_sd} for row in cell)

                if verbose:
                    print(f"alpha={alpha_hetero:<8g} "
                          f"lambda=({lambda_mean:g}, {lambda_re:g})  "
                          f"band_error={np.mean([r['band_error'] for r in cell]):7.3f}  "
                          f"true band sd={true_band_sd:.3e}", flush=True)

            # Step 3: Report the level as it completes
            if verbose:
                print(f"alpha_hetero={alpha_hetero:<8g} done  "
                      f"(median true band sd = {true_band_sd:.3e})", flush=True)

        return rows

    # --------------------- Reading the sweep -------------------------- #

    @staticmethod
    def aggregate_seed_replicates(rows, metrics, by=("lambda_mean", "lambda_re")):
        """Collapse the seed replicates into one row per configuration.

        Args:
            rows: list of dicts; the rows returned by either sweep.
            metrics: sequence of str; which columns to aggregate.
            by: sequence of str; the columns identifying a configuration.
                Defaults to the penalty pair; the heterogeneity sweep also
                groups by "alpha_het".

        Returns:
            pd.DataFrame indexed by the `by` columns, with a "<metric>_mean",
                "<metric>_sd" and "<metric>_count" column per metric.
        """
        # Step 1: One row per fit, with its configuration columns carried along
        df_sweep = pd.DataFrame(rows)

        # Step 2: Collapse the seeds. The standard deviation is kept alongside the mean
        aggregated = df_sweep.groupby(list(by))[list(metrics)].agg(
            ["mean", "std", "count"])

        # Step 3: Flatten ("band_error", "std") into "band_error_sd"
        aggregated.columns = [
            f"{metric}_{statistic}".replace("_std", "_sd")
            for metric, statistic in aggregated.columns
        ]

        return aggregated
