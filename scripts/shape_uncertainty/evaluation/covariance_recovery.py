import numpy as np
import pandas as pd
import torch
import itertools

# B-Spline utilities
from scripts.shape_uncertainty.spline_basis.bspline_basis import (
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
from scripts.shape_uncertainty.model.model_inference import build_inference_engine
from scripts.shape_uncertainty.evaluation.model_evaluation import ModelEvaluator


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
            "D_train", "D_val", "D_test", "nr_obs", "sigma", "regular" and
            "include_endpoints".
        train_config: dict; the optimisation settings held fixed across the
            sweep, with keys "nr_epochs", "patience", "model_cls" and
            "model_kwargs".
        T: float; the end of the observation window.
        n_ref: int; how many test individuals the Monte-Carlo ground truth is
            reconstructed for.
        n_mc: int; latent draws used per reference individual.
        seed: int; the base seed for data generation and for the first fit of
            every configuration.
    """
    def __init__(self, base_hyperparams, knot_objects, shape_config,
                 data_config, train_config, T, n_ref, n_mc, seed=0):

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
        self.truth_settings = {"n_ref": int(n_ref), "n_mc": int(n_mc)}

        # Caches keyed by heterogeneity level
        self._splits = {}
        self._shape_summaries = {}
        self._covariance_truth = {}


    def _build_hyperparams(self, alpha_g, alpha_d):
        """Return the DGP hyperparameters at one heterogeneity level.

        Copies the shared base hyperparameters and sets both heterogeneity
        terms to the same value, so that a single scalar controls the size of
        the unobserved variation in the DGP.

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.


        Returns:
             dict; the base hyperparameters with "alpha_g" and "alpha_d" set.
        """
        hyperparams = dict(self.base_hyperparams)
        hyperparams["alpha_g"] = float(alpha_g)
        hyperparams["alpha_d"] = float(alpha_d)

        return hyperparams

    def get_dataset_split(self, alpha_g, alpha_d):
        """Return the test, validation and training split at one heterogeneity level.

        Regenerate the DGP at (alpha_g, alpha_d),
        Draw a dataset individuals under the stored observation design
        Split the dataset into D_train, D_val, D_test 

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.

        Returns:
            tuple of three SimulatedDatasets in the order (test, val, train).
        """
        # Step 0: Return the cached split if drawn before
        key = (float(alpha_g), float(alpha_d))
        if key in self._splits:
            return self._splits[key]

        # Step 1: Rebuild the DGP at (alpha_g, alpha_d)
        dgp = WilkersonDGP(T=self.T,
                           hyperparams=self._build_hyperparams(alpha_g, alpha_d))

        # Step 2: Draw the dataset
        design = self.data_config
        D_total = design["D_train"] + design["D_val"] + design["D_test"]
        dataset = generate_simulated_dataset(
            dgp, 
            D=D_total,
            N=design["nr_obs"], 
            sigma=design["sigma"],
            regular=design["regular"], 
            include_endpoints=design["include_endpoints"],
            seed=self.seed,
        )

        # Step 3: Split the data
        split = dataset.split_test_val_train(
            D_train=design["D_train"], 
            D_val=design["D_val"],
            D_test=design["D_test"],
        )
        self._splits[key] = split

        return split

    def _get_true_shape_summaries(self, alpha_g, alpha_d):
        """Return the analytic shape summaries of the test split.

        Read the ground truth analytic shape summaries from the DGP's parameters.
        This is used to score shape extraction performance. 

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.

        Returns:
            list of length D_test; one analytic shape summary per test
                individual.
        """
        # Return the cached summaries if read before
        key = (float(alpha_g), float(alpha_d))
        if key in self._shape_summaries:
            return self._shape_summaries[key]

        # Read the summaries off the DGP's parameters
        test = self.get_dataset_split(alpha_g, alpha_d).test
        summaries = get_analytic_shape_summary(
            test,
            self.shape_config["upsilon_rel_1"],
            self.shape_config["upsilon_rel_2"],
            self.shape_config["upsilon_rel_prune"],
            self.shape_config["do_prune"],
        )

        # Cache
        self._shape_summaries[key] = summaries

        return summaries

    
    # -------------------------- Ground truth -------------------------- #

    def get_covariance_truth(self, alpha_g, alpha_d, n_ref, n_mc):
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

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.
            n_ref: int; we use n_ref=D.test, i.e. reconstruct the truth for the entire test set.
            n_mc: int; latent draws used per individual.

        Returns:
            dict with keys:
                "indices": np.ndarray of shape (n_ref,); which test individuals
                    were selected.
                "Sigma": np.ndarray of shape (n_ref, B, B); the true
                    random-effect covariance of each one, in original y units.
                "band_sd": np.ndarray of shape (n_ref, N); the true pointwise
                    band sd on a shared evaluaton grid. 
        """
        # Step 0: Return the cached truth if it was built before
        key = (float(alpha_g), float(alpha_d), n_ref, n_mc)
        if key in self._covariance_truth:
            return self._covariance_truth[key]

        test = self.get_dataset_split(alpha_g, alpha_d).test

        # Step 1: Select the reference individuals
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(test.D, size=n_ref, replace=False)

        # Step 2: Reconstruct Sigma(x) at each reference covariate point
        hyperparams = self._build_hyperparams(alpha_g, alpha_d)
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

    def search_fixed_architecture(self, alpha_g, alpha_d, nr_trials,
                                  lambda_mean, lambda_re):
        """Search the encoder architecture once and return it for reuse.

        The search runs under the same penalty pair the reported fit is trained
        with.

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.
            nr_trials: int; Optuna trials.
            lambda_mean: float; the roughness penalty on the mean coefficients.
            lambda_re: float; the roughness penalty on the random effects.

        Returns:
            dict; the best hyperparameters, in the form Tuner.run reports them
                and train_fixed_configuration consumes them.
        """
        # Step 1: Get the dataset split
        split = self.get_dataset_split(alpha_g, alpha_d)

        # Step 2: Search at the penalty pair the fit will use
        tuner = self._build_tuner(split, lambda_mean, lambda_re)

        # Step 3: Return the best parameter configuration
        return tuner.run(nr_trials=nr_trials, seed=self.seed)["best_params"]

    def fit_model(self, alpha_g, alpha_d, hyperparas, lambda_mean, lambda_re, seed):
        """Train one GaussianModel at a fixed architecture and penalty pair.

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.
            hyperparas: dict; the frozen architecture.
            lambda_mean: float; the roughness penalty on the mean coefficients.
            lambda_re: float; the roughness penalty on the random effects.
            seed: int; seeds both the weight initialisation and the shuffling.

        Returns:
            dict with keys:
                "engine": InferenceEngine wrapping the trained model, the
                    normaliser, the shared basis and the shared shape_config.
                "tuned": the train_fixed_configuration result, carrying the
                    model, the normaliser and the architecture, in the shape
                    save_run consumes.
        """
        # Seed torch's generator
        torch.manual_seed(seed)

        # Step 1: Train at the frozen architecture at the given penalty pair
        split = self.get_dataset_split(alpha_g, alpha_d)
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

        return {"engine": engine, "tuned": fitted}

    # ------------------------ Scoring one fit ------------------------- #

    def _score_covariance_recovery(self, engine, alpha_g, alpha_d):
        """Score the fitted covariance Sigma(x) against the covariance the DGP implies.

        Sigma natively lives in coefficient space. To make the fit of Sigma
        interpretable, we proceed as follows:

        Step 1: Compute the pointwise band around the outcome y that is implied
                by Sigma as the pointwise standard deviation:
                sqrt(phi(t)^T Sigma phi(t))
        Step 2: Reduce the pointwise band into a single figure for each
                individual by taking the average over the observation grid.
        Step 3: Compute the size of the pointwise band relative to an
                individual's own trajectory amplitude A (the ground truth
                amplitude).
        Step 4: Report the median, the 5th and the 95th percentile.

        Args:
            engine: InferenceEngine; a trained model.
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.

        Returns:
            dict of floats with keys 
                "band_rel_true_median", 
                "band_rel_true_p5",
                "band_rel_true_p95" 
                "band_rel_fitted_median", 
                "band_rel_fitted_p5",
                "band_rel_fitted_p95" 
        """
        # Step 1: Take the truth and the fitted covariance at the same individuals
        truth = self.get_covariance_truth(alpha_g, alpha_d, **self.truth_settings)
        test = self.get_dataset_split(alpha_g, alpha_d).test
        X_ref = {name: column[truth["indices"]] for name, column in test.X.items()}

        # Step 2: Carry each covariance into value space as the pointwise band
        #         sd, and average it over the observation grid
        sd_fitted = engine.predict_aleatoric_band(
            X_ref, self.observation_times).mean(axis=1)     # (n_ref,)
        sd_true = truth["band_sd"].mean(axis=1)             # (n_ref,)

        # Step 3: Express each band as a fraction of that same individual's
        #          predicted trajectory amplitude
        fitted_curves = engine.predict_trajectory(X_ref, self.observation_times)
        amplitude = np.ptp(fitted_curves, axis=1)          # (n_ref,)
        relative_true = sd_true / amplitude
        relative_fitted = sd_fitted / amplitude

        # Step 4: Summarise across the evaluated individuals
        return {
            "band_rel_true_median": float(np.median(relative_true)),
            "band_rel_true_p5": float(np.percentile(relative_true, 5)),
            "band_rel_true_p95": float(np.percentile(relative_true, 95)),
            "band_rel_fitted_median": float(np.median(relative_fitted)),
            "band_rel_fitted_p5": float(np.percentile(relative_fitted, 5)),
            "band_rel_fitted_p95": float(np.percentile(relative_fitted, 95)),
        }

    def _score_mean_function_guards(self, engine, alpha_g, alpha_d):
        """Score the mean function.

        A penalty that improves Sigma by degrading the mean has not solved the
        problem. We therefore evaluate the performance of the prediction using
            1. mean_shape_accuracy
            2. mean_shape_distance
            3. R^2 against the observed and against the latent outcomes, each
               read in two ways: pooled over all residuals, and median of individual
               scores.

        Args:
            engine: InferenceEngine; a trained model.
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.

        Returns:
            dict with keys:
                "mean_trajectory_shape_accuracy": float in [0, 1]
                "mean_trajectory_shape_distance": float in [0, 1]
                "mean_trajectory_pooled_r2_observed": float; R^2 pooled over all
                    D * N_i residuals against the observed outcomes.
                "mean_trajectory_median_r2_observed": float; median individual
                    R^2 against the observed outcomes.
                "mean_trajectory_pooled_r2_latent": float; R^2 pooled over all
                    D * N_i residuals against the latent outcomes.
                "mean_trajectory_median_r2_latent": float; median individual
                    R^2 against the latent outcomes.
        """
        # Step 1: Score the model's mean trajectories against the truth
        test = self.get_dataset_split(alpha_g, alpha_d).test
        evaluator = ModelEvaluator(engine, test)

        # Step 2: Evaluate shape-space metrics
        shape_metrics = evaluator.compute_shape_space_metrics(
            self._get_true_shape_summaries(alpha_g, alpha_d))

        # Step 3: Evaluate value-space metrics
        latent = evaluator.compute_value_space_metrics(target="latent")
        observed = evaluator.compute_value_space_metrics(target="observed")

        return {
            "mean_trajectory_shape_accuracy": shape_metrics["accuracy"],
            "mean_trajectory_shape_distance": shape_metrics["distance"],
            "mean_trajectory_pooled_r2_observed": float(observed["r2"]),
            "mean_trajectory_median_r2_observed": observed["median_individual_r2"],
            "mean_trajectory_pooled_r2_latent": float(latent["r2"]),
            "mean_trajectory_median_r2_latent": latent["median_individual_r2"],
        }

    def score_engine(self, engine, alpha_g, alpha_d):
        """Assemble the full scoring panel for one trained model.

        This function reports the results from the above two methods
            - _score_covariance_recovery
            - _score_mean_function_guards
        in a table. Additionally it reports fitted noise sd. 

        Args:
            engine: InferenceEngine; a trained model.
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.

        Returns:
            dict; the union of _score_covariance_recovery and
                _score_mean_function_guards, plus "sigma_hat".
        """
        # Step 1: Get Sigma(x) evaluation
        results = self._score_covariance_recovery(engine, alpha_g, alpha_d)

        # Step 2: Get the shape-space and value-space evaluation
        results.update(self._score_mean_function_guards(engine, alpha_g, alpha_d))

        # Step 3: Get the estimate of the noise std dev
        results["sigma_hat"] = engine.noise_std_hat

        # Return a results dict 
        return results


    def fit_and_score(self, alpha_g, alpha_d, hyperparas, lambda_mean, lambda_re, seed):
        """Train one model at one configuration and return its scoring panel.

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.
            hyperparas: dict; the frozen architecture.
            lambda_mean: float; the mean-coefficient roughness penalty.
            lambda_re: float; the random-effect roughness penalty.
            seed: int; the fit's seed.

        Returns:
            dict; the score_engine panel plus "alpha_g", "alpha_d", "lambda_mean",
                "lambda_re" and "seed".
        """
        # Step 1: Train a model at a given configuration
        engine = self.fit_model(alpha_g, alpha_d, hyperparas,
                                lambda_mean, lambda_re, seed)["engine"]

        # Step 2: Return the configuration and its scroes
        return {
            "alpha_g": alpha_g,
            "alpha_d": alpha_d,
            "lambda_mean": lambda_mean,
            "lambda_re": lambda_re,
            "seed": seed,
            **self.score_engine(engine, alpha_g, alpha_d),
        }

    # ---------------------------- Sweeps ------------------------------ #

    def sweep_penalty_grid(self, alpha_g, alpha_d, hyperparas, lambda_mean_grid,
                           lambda_re_grid, n_seeds, verbose=True):
        """Fit every penalty pair on the grid, several seeds each, 
           at ONE GIVEN heterogeneity level.

        Args:
            alpha_g: float; heterogeneity in the growth rate.
            alpha_d: float; heterogeneity in the decay rate.
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
                self.fit_and_score(alpha_g, alpha_d, hyperparas,
                                   lambda_mean, lambda_re, self.seed + s)
                for s in range(n_seeds)
            ]
            rows.extend(cell)

            # Step 2: Report the cell as it completes as the code runs
            # The printed results are the average across the n_seeds for the given row.
            if verbose:
                band_error = [row["band_error"] for row in cell]
                r2_columns = [
                    ("obs pooled", "mean_trajectory_pooled_r2_observed"),
                    ("obs median", "mean_trajectory_median_r2_observed"),
                    ("lat pooled", "mean_trajectory_pooled_r2_latent"),
                    ("lat median", "mean_trajectory_median_r2_latent"),
                ]
                r2_text = "  ".join(
                    f"{label}={np.mean([row[key] for row in cell]):>8.4f}"
                    for label, key in r2_columns
                )
                print(f"lambda_mean={lambda_mean:<8g} lambda_re={lambda_re:<8g} "
                      f"band_error={np.mean(band_error):7.3f} "
                      f"(sd {np.std(band_error, ddof=1):.3f})", flush=True)
                print(f"    r2:  {r2_text}", flush=True)

        return rows



    def sweep_heterogeneity_levels(self, alpha_pairs, lambda_pairs, hyperparas,
                                   n_seeds, verbose=True):
        """Refit the penalty pairs across a list of heterogeneity levels.

        Args:
            alpha_pairs: sequence of (alpha_g, alpha_d) tuples; the heterogeneity
                settings to sweep. 
            lambda_pairs: sequence of (lambda_mean, lambda_re) tuples; the
                shortlist to carry forward.
            hyperparas: dict; the frozen architecture.
            n_seeds: int; fits per (level, pair).
            verbose: bool; print a per-level summary line as the sweep runs.

        Returns:
            list of dicts; one fit_and_score row per fit, each carrying its
                "alpha_g", "alpha_d" and the setting's true band level for plotting.
        """
        rows = []
        for alpha_g, alpha_d in alpha_pairs:
            # Step 1: Build this level's ground truth
            truth = self.get_covariance_truth(alpha_g, alpha_d, **self.truth_settings) # Sigma(x) true
            true_band_sd = float(np.median(truth["band_sd"].mean(axis=1))) # Value-space band true

            # Step 2: Refit every penalty pair for the data at this unobserved heterogeneity level
            for lambda_mean, lambda_re in lambda_pairs:
                cell = [self.fit_and_score(alpha_g, alpha_d, hyperparas, lambda_mean,
                                           lambda_re, self.seed + s)
                        for s in range(n_seeds)]
                rows.extend({**row, "true_band_sd": true_band_sd} for row in cell)

                if verbose:
                    print(f"alpha=({alpha_g:g}, {alpha_d:g})  "
                          f"lambda=({lambda_mean:g}, {lambda_re:g})  "
                          f"band_error={np.mean([r['band_error'] for r in cell]):7.3f}  "
                          f"true band sd={true_band_sd:.3e}", flush=True)

            # Step 3: Report the level as it completes
            if verbose:
                print(f"alpha=({alpha_g:g}, {alpha_d:g})  done  "
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
                groups by "alpha_g" and "alpha_d".

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
