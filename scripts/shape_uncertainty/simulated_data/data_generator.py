import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple

class BaseDGP(ABC):
    """Abstract class for a data-generating process."""
    @abstractmethod
    def draw_population(self, D, rng):
        """Draw covariates, latent factors, and parameters for D individuals."""
        pass

    @abstractmethod
    def true_curves(self, times, X, params):
        """Get the the noise-free trajectories for every individual."""
        pass

    @abstractmethod
    def get_process_config(self):
        """Return the process definition as a dict."""
        pass

class WilkersonDGP(BaseDGP):
    """Define the Wilkerson toumour trajectory data-generating process.

    This class defines a data generating process. Once specified, we
    can generate a simulated dataset from this DGP. 

    Attributes:
        hyperparams: dict with keys "g0", "d0", "rho0", "alpha_g", "alpha_d".
        ranges: dict mapping covariate name to a (min, max) tuple.
        T: float; the time horizon [0, T] the process is defined on.
    """
    def __init__(self, hyperparams=None, ranges=None, T=1.0):
        """Define the datagenerating process process.

        Args:
            hyperparams: dict with keys "g0", "d0", "rho0", "alpha_g",
                "alpha_d". Set to a default when None. Setting
                alpha_g = alpha_d = 0 removes unobserved heterogeneity.
            ranges: dict mapping each of "size", "age", "weight", "dosage" to
                a (min, max) tuple. Set to a default when None.
            T: float; the time horizon.

        Raises:
            ValueError: if `ranges` or `hyperparams` have unexpected keys.
        """
        # Initialise uniform distribution ranges if not provided
        expected_cov = {"size", "age", "weight", "dosage"}
        if ranges is None:
            ranges = {
                "size": (0.1, 0.5), 
                "age": (20.0, 80.0),
                "weight": (40.0, 100.0), 
                "dosage": (0.0, 1.0),
            }
        elif set(ranges) != expected_cov:
            raise ValueError(f"Ranges keys {set(ranges)} must be {expected_cov}.")
        
        # Initialise hyperparams if not provided
        expected_hp = {"g0", "d0", "rho0", "alpha_g", "alpha_d"}
        if hyperparams is None:
            hyperparams = {
                    "g0": 2.0, 
                    "d0": 180.0, 
                    "rho0": 10.0,
                    "alpha_g": 0.0, 
                    "alpha_d": 0.0,
            }
        elif set(hyperparams) != expected_hp:
            raise ValueError(f"hyperparams keys {set(hyperparams)} must be {expected_hp}.")

        # Validate the time horizon
        if T <= 0:
            raise ValueError(f"T must be strictly positive.")

        self.hyperparams = hyperparams
        self.ranges = ranges
        self.T = float(T)


    def _sample_covariates(self, D, rng):
        """Draw observed static covariates for D samples.

        Each covariate is drawn independently from a UNIFORM distribution
        whose bounds are given by 'ranges'. These are the covariates the
        model is allowed to observe.

        Args:
            D: int, number of samples to draw.
            rng: np.random.Generator used for sampling (advances its state).

        Returns:
            X: dict mapping each covariate name to an np.ndarray of shape
                (D,) holding that covariate's value for all D samples.
        """
        X = dict()
        for name, (min_val, max_val) in self.ranges.items():
            X[name] = rng.uniform(low=min_val, high=max_val, size=D)

        return X

    def _sample_latents(self, D, rng):
        """Draw latent covariates for D samples.

        Each latent factor is drawn independently from a standard normal
        distribution N(0, 1). The latent covariate is unobserved for modelling 
        purposes. 

        Args:
            D: int, number of samples to draw.
            rng: np.random.Generator used for sampling (advances its state).

        Returns:
            Z: dict mapping each latent factor name ("z_g" growth, "z_d"
                decay) to an np.ndarray of shape (D,) of its values for all
                D samples.
        """
        Z = dict()
        Z["z_g"] = rng.normal(loc=0.0, scale=1.0, size=D)
        Z["z_d"] = rng.normal(loc=0.0, scale=1.0, size=D)
        
        return Z

    def _compute_parameters(self, covariates, latents):
        """Compute per-individual biological parameters (g, d, rho).

        Deterministic map from observed covariates and latent factors to
        the Wilkerson rate parameters, following:

            g   = g_0 * (age / 20)**0.5 * exp(alpha_g * z_g)
            d   = d_0 * (dosage / weight) * exp(-alpha_d * z_d)
            rho = 1 / (1 + exp(-dosage * rho_0))

        The latent factors enter g and d multiplicatively via an
        exponential (guaranteeing positivity of the rates); z_g raises
        growth and z_d, with its negative sign, weakens decay, so larger
        latent values correspond to more aggressive, less treatable
        disease. rho depends on observed covariates only.

        Args:
            covariates: dict mapping each covariate name (must include
                "age", "weight", "dosage") to an np.ndarray of shape (D,).
            latents: dict mapping each latent factor name ("z_g", "z_d")
                to an np.ndarray of shape (D,).

        Returns:
            parameters: dict mapping each biological parameter name
                ("g", "d", "rho") to an np.ndarray of shape (D,) holding
                that parameter's value for all D samples.
        """
        # Step 1: Extract input parameters

        # 1.1 Extract observed covariates
        age = covariates["age"] # Shape (D,)
        weight = covariates["weight"]
        dosage = covariates["dosage"]
        
        # 1.2 Extract latent factors
        z_g = latents["z_g"] # Shape (D,) 
        z_d = latents["z_d"]
        
        # 1.3 Extract hyperparameters
        g_0 = self.hyperparams["g0"] # scalar
        d_0 = self.hyperparams["d0"]
        rho_0 = self.hyperparams["rho0"]
        alpha_g = self.hyperparams["alpha_g"]
        alpha_d = self.hyperparams["alpha_d"]
        
        # Step 2: Compute output parameters
        g = g_0 * (age / 20.0)**0.5 * np.exp(alpha_g * z_g) 
        d = d_0 * (dosage / weight) * np.exp(-alpha_d * z_d)
        rho = 1.0 / (1.0 + np.exp(-dosage * rho_0)) # sigmoid curve based on dosage
        
        # Step 3: Return computed parameters as a dictionary 
        parameters = {
            "g": g,
            "d": d,
            "rho": rho
        }
        
        return parameters

    def draw_population(self, D, rng):
        """Draw covariates, latent factors, and parameters for D individuals: 

        Args:
            D: int; number of individuals to draw.
            rng: np.random.Generator; advanced in place.

        Returns:
            tuple (X, Z, params):
                X: dict of covariate name -> np.ndarray of shape (D,).
                Z: dict of latent name ("z_g", "z_d") -> np.ndarray (D,).
                params: dict of "g", "d", "rho" -> np.ndarray (D,).
        """
        # Step 1: Sample covariates
        X = self._sample_covariates(D, rng)
        # Step 2: Sample latent factors
        Z = self._sample_latents(D, rng)
        # Step 3: Compute parameters of the DGP
        params = self._compute_parameters(X, Z) # step 3
        return X, Z, params

    def _compute_wilkerson(self, t, size, g, d, rho):
        """Evaluate the Wilkerson tumour-trajectory at time(s) t.

        The noise-free trajectory is
            y(t) = size * (rho * exp(-d * t) + (1 - rho) * exp(g * t)),
        where rho is the treatment-sensitive fraction (decaying at rate d)
        and (1 - rho) the resistant fraction (growing at rate g). The
        normalisation is such that y(0) = size (the initial tumour volume);
        we omit the -1 of the original Wilkerson form so that 'size' is the
        starting value rather than a deviation scale.

        Args:
            t: time(s) at which to evaluate; scalar or np.ndarray.
                - t scalar, params scalar            -> scalar
                - t shape (N,), params scalar        -> one trajectory (N,)
                - t shape (1, N), params shape (D, 1)-> full matrix (D, N)
            size: initial tumour volume (y at t=0); scalar or np.ndarray.
            g: growth rate of the resistant fraction; scalar or np.ndarray.
            d: decay rate of the sensitive fraction; scalar or np.ndarray.
            rho: treatment-sensitive fraction in (0, 1); scalar or np.ndarray.

        Returns:
            y: the trajectory value(s), of the shape produced by
                broadcasting t against the parameters.
        """
        return size * (rho * np.exp(-d * t) + (1 - rho) * np.exp(g * t))


    def true_curves(self, times, X, params):
        """Evaluate the noise-free Wilkerson trajectory for every individual.

        Args:
            times: length-D sequence; times[i] is individual i's 1-D array of
                evaluation times. To evaluate every individual on one shared
                grid
            X: dict of covariate name to np.ndarray of shape (D,) each; must
                contain "size".
            params: dict of "g", "d", "rho" to np.ndarray of shape (D,) each.

        Returns:
            list of length D; element i is individual i's noise-free
                trajectory values as an np.ndarray of shape (len(times[i]),).
        """
        # Step 1: Determine the number of individuals
        D = len(params["g"])

        # Step 2: Evaluate each individual's noise-free trajectory
        Y_true = []
        for i in range(D):
            y_i = self._compute_wilkerson(
                times[i],
                X["size"][i],
                params["g"][i],
                params["d"][i],
                params["rho"][i],
            )
            Y_true.append(y_i)

        return Y_true

    def get_process_config(self):
        """Return the process definition as a dictionary.

        Returns:
            dict with the following keys:
                "dgp_name": str; the name of this DGP class, so a saved record
                    says which process produced it.
                "hyperparams": dict mapping each hyperparameter name to its
                    float value.
                "ranges": dict mapping each covariate name to a [min, max] list.
                "T": float; the time horizon.
        """
        # Step 1: Copy the hyperparameters
        hyperparams = dict(self.hyperparams)

        # Step 2: Copy the covariate ranges
        ranges = {}
        for name, bounds in self.ranges.items():
            min_val, max_val = bounds
            ranges[name] = [min_val, max_val]

        # Step 3: Assemble the configuration dictionary
        process_config = {
            "dgp_name": "WilkersonDGP",
            "hyperparams": hyperparams,
            "ranges": ranges,
            "T": self.T,
        }

        return process_config


class SimulatedDataset:
    """A simulated panel dataset with model-visible and ground-truth data.

    The dataset holds D individuals stored as a list-of-arrays indexed 
    by individual; observation counts and times may differ across individuals.
    We distringuish between three parts of data:
        1. Model-visible
            This data may be passed to a model.
        2. Ground-truth
            This data is withheld from a model and used to evaluate predictions.
        3. Metadata
            This is data on hyperparameters used for data generation.
    
    Model-visible
        X: dict mapping each covariate name to an np.ndarray of shape
            (D,) holding that covariate's value for all D samples.
        times: list of length D, where times[i] is a sorted np.ndarray
            of the observation times for individual i
        Y_noisy: list of length D, where Y_noisy[i] = Y_true[i] + noise,
            an np.ndarray of shape (N_i,)

    Ground-truth tier (withheld from the model):
        Z: dict mapping unobserved factors "z_g" and "z_d" to 
            an np.ndarray of shape (D,) of its values for all D samples.
        params: dict mapping each biological parameter name
            ("g", "d", "rho") to an np.ndarray of shape (D,) holding
            that parameter's value for all D samples.
        Y_true: list of length D, where Y_true[i] is an np.ndarray of
            shape (N_i,) holding individual i's noise-free trajectory
            values.
        true_shapes: list of length D; true_shapes[i] is the
            shape summary extracted from individual i's noise-free curve.
            None until computed.
        true_sigma: np.ndarray (B, B); the Monte-Carlo estimate
            of the aleatoric covariance induced on the spline
            coefficients. None until computed.

    Metadata:
        dgp: the data-generating process object (e.g. a WilkersonDGP) that
            produced this data, includes info like hyperparams, ranges, T. 
        design: dict of the  observation settings with keys:
            "D", "N", "sigma", "regular", "include_endpoints", "seed".
    """
    def __init__(self, X, times, Y_noisy, 
        Z, params, Y_true, true_shapes = None, true_sigma = None,
        dgp=None, design=None,
    ):

        # Model-visible data
        self.X = X
        self.times = times
        self.Y_noisy = Y_noisy

        # Ground-truth data
        self.Z = Z
        self.params = params
        self.Y_true = Y_true
        self.true_shapes = true_shapes
        self.true_sigma = true_sigma

        # Metadata
        self.dgp = dgp
        self.design = {} if design is None else design

    @property
    def D(self):
        """Return the number of samples D in the dataset."""
        return len(self.Y_noisy)

    @property
    def N(self):
        """Return the observation counts N_i for each sample in the dataset."""
        return [len(t) for t in self.times]

    @property
    def T(self):
        """Return the time horizon, from the data-generating process."""
        return self.dgp.T

    @property
    def sigma(self):
        """Return the measurement-noise std, from the draw design."""
        return self.design.get("sigma")

    @property
    def hyperparams(self):
        """Return the generative hyperparameters, from the process."""
        return self.dgp.hyperparams

    @property
    def gen_config(self):
        """Return process and design settings merged into one provenance dict."""
        return {**self.dgp.get_process_config(), **self.design}

    def model_inputs(self):
        """Return the model-visible data (X, times, Y_noisy)."""
        return self.X, self.times, self.Y_noisy

    def as_matrix(self):
        """Stack Y_noisy into a (D, N) array IF all N_i are equal.

        Raises ValueError if observation counts differ across
        individuals.
        """
        counts = self.N
        if len(set(counts)) != 1:
            raise ValueError(
                f"Cannot stack to a matrix: per-individual counts differ ({counts})."
            )
        return np.vstack(self.Y_noisy)

    def select_individuals(self, idx):
        """Return a new SimulatedDataset with only the selected individuals.

        Args:
            idx: 1-D array of int; the individual indices to keep, in
                the order they should appear in the returned dataset.

        Note:
        The function permits indices to repeat, in which case the corresponding
        individual appears more than once. We use this for bootstrap resampling.

        Returns:
            SimulatedDataset: a new dataset of len(indices) individuals
        """
        # Step 1: Slice the per-individual dictionaries
        X = {name: values[idx] for name, values in self.X.items()}
        Z = {name: values[idx] for name, values in self.Z.items()}
        params = {name: values[idx] for name, values in self.params.items()}

        # Step 2: Slice the per-individual lists
        times = [self.times[i] for i in idx]
        Y_noisy = [self.Y_noisy[i] for i in idx]
        Y_true = [self.Y_true[i] for i in idx]
        if self.true_shapes is None:
            true_shapes = None
        else:
            true_shapes = [self.true_shapes[i] for i in idx]

        # Step 3: Copy the design, recording the new number of individuals
        design = dict(self.design)
        design["D"] = len(idx)

        # Step 4: Build the new dataset
        return SimulatedDataset(
            X=X, times=times, Y_noisy=Y_noisy,
            Z=Z, params=params, Y_true=Y_true,
            true_shapes=true_shapes, true_sigma=self.true_sigma,
            dgp=self.dgp, design=design,
        )

    def take_first_d_individuals(self, d):
        """Return a dataset of the first d individuals.

        Args:
            d: int; the number of leading individuals to keep.

        Returns:
            SimulatedDataset with d individuals.
        """
        if d > self.D:
            raise ValueError
        return self.select_individuals(np.arange(d))

    def split_test_val_train(self, D_train, D_val, D_test):
        """Split the simulated dataset into test, validation, and training sets.

        The dataset is carved into three contiguous, non-overlapping blocks:
            - the first D_test individuals become the test set,
            - the next D_val individuals become the validation set,
            - the remaining D_train individuals become the training set.

        Args:
            D_train: int; the number of individuals in the training set.
            D_val: int; the number of individuals in the validation set.
            D_test: int; the number of individuals in the test set.

        Returns:
            Split: a namedtuple (test, val, train) of SimulatedDatasets, so
                both `test, val, train = ds.split_test_val_train(...)` and
                `s = ds.split_test_val_train(...); s.train` work.

        Raises:
            ValueError: if D_train + D_val + D_test does not equal the number
                of individuals in the dataset.
        """
        # Step 1: Define the container holding the three resulting subsets
        Split = namedtuple("Split", ["test", "val", "train"])

        # Step 2: Check the requested sizes account for every individual
        if D_train + D_val + D_test != self.D:
            raise ValueError(
                f"Split sizes must sum to the dataset size: D_train={D_train} "
                f"+ D_val={D_val} + D_test={D_test} = "
                f"{D_train + D_val + D_test}, but the dataset has {self.D}."
            )

        # Step 3: Carve the three contiguous, non-overlapping regions
        test = self.select_individuals(np.arange(0, D_test))
        val = self.select_individuals(np.arange(D_test, D_test + D_val))
        train = self.select_individuals(np.arange(D_test + D_val, self.D))

        return Split(test=test, val=val, train=train)

    def true_curves_at(self, times, indices=None):
        """Evaluate noise-free trajectories at arbitrary times.

        Unlike the stored Y_true, which holds each individual's trajectory
        only at that individual's own observation times, this function permits
        evaluating the ground truth curve at any time points. 
        This is used for plotting or for scoring a model'spredictions.

        Args:
            times: 1-D np.ndarray of times at which to evaluate, shared by
                every selected individual.
            indices: optional 1-D array-like of int; which individuals to
                evaluate. Defaults to all D individuals. Passing a small
                selection avoids evaluating the whole dataset when only a few
                curves are needed, such as plotting a handful of panels.

        Returns:
            list of np.ndarray; one noise-free trajectory per selected
                individual, in the order given by `indices`, each of shape
                (len(times),).
        """
        # Step 1: Restrict to the specified individuals, or use all of them
        if indices is None:
            selected = self
        else:
            selected = self.select_individuals(indices)

        # Step 2: Give every selected individual the same evaluation grid
        times_per_individual = [times] * selected.D

        # Step 3: Evaluate the noise-free trajectories via the process
        return self.dgp.true_curves(
            times_per_individual, selected.X, selected.params
        )

    def individual_covariates(self):
        """Return covariates as a D-length list of per-individual feature dicts.

        Converts the column-oriented X (covariate name to array over individuals)
        into a row-oriented list, one dict per individual, as required by the inference
        engine.

        self.dataset.X stores individual covariates as:
            {
                "size":  [person1, person2, person3, ...],
                "age":   [person1, person2, person3, ...],
                            ...
            }
        We modify this into a list of feature values for individuals:  
            [
            {"size": float, "age": float, "weight": float, "dosage": float}, # Person 1
            {"size": float, "age": float, "weight": float, "dosage": float}, # Person 2
                            ...
            ]

        Returns:
            list of length D; element i is {covariate_name: value} for individual i.
        """
        return [
            {name: np.asarray([column[i]], dtype=float)
             for name, column in self.X.items()}
            for i in range(self.D)
        ]

def _make_regular_observation_times(N, T):
    """Evenly spaced grid of N points on [0, T] (one individual)."""
    return np.linspace(0.0, T, N)

def _make_irregular_observation_times(N, T, rng, include_endpoints=True):
    """N sorted random times on [0, T] (one individual).

    When include_endpoints is True, 0 and T are anchored and
    the remaining N-2 interior times are drawn uniformly, so every
    individual has observations at both domain boundaries (keeping the
    spline edges constrained). Requires N >= 2 in that case.
    """
    if include_endpoints:
        if N < 2:
            raise ValueError("N must be >= 2 when include_endpoints is True.")
        interior = rng.uniform(0.0, T, size=N - 2)
        return np.concatenate([[0.0], np.sort(interior), [T]])
    return np.sort(rng.uniform(0.0, T, size=N))

def make_observation_times(D, N, T, regular=True, include_endpoints=True, rng=None):
    """Generate per-individual observation time grids on [0, T].

    Returns one time array per individual, as a length-D list, so that
    the number and spacing of observations may differ across
    individuals. In the regular case every individual shares the
    same evenly spaced grid; in the irregular case each individual's
    times are drawn at random within [0, T] and sorted.

    Args:
        D: int, number of individuals (length of the returned list).
        N: number of observations per individual. Either an int (the
            same count for all D individuals) or a length-D sequence of
            ints (per-individual counts N_i, for variable sampling).
        T: float, right end of the time domain [0, T].
        regular: if True, evenly spaced times on [0, T]; if False,
            times are drawn uniformly at random in [0, T] per individual
            (requires rng).
        include_endpoints: only used when regular is False. If True
            (default), 0 and T are anchored and the remaining N-2
            interior times are drawn uniformly, so the spline edges
            stay constrained; if False, all N times are random in
            [0, T] (boundaries may be unobserved to test edge uncertainty).
        rng: np.random.Generator, required when regular is False;
            ignored when regular is True.

    Returns:
        times: list of length D, where times[i] is a sorted np.ndarray
            of the observation times for individual i. In the regular,
            scalar-N case every element is an identical shared grid; in
            the variable or irregular case the arrays may differ in
            length and spacing across individuals.
    """
    # Standardize N input
    if np.isscalar(N):
        counts = [int(N)] * D
        is_scalar_N = True
    else:
        counts = list(N)
        if len(counts) != D:
            raise ValueError(f"len(N)={len(counts)} must equal D={D}.")
        is_scalar_N = False

    if not regular and rng is None:
        raise ValueError("rng is required when regular is False.")

    # This handles three cases:
    #   Case 1: Regular spaced observations + same number N for all samples D.
    #   Case 2: Regular spaced observations + different N_i for different samples
    #   Case 3: Irregular spaced observations + different N_i for different samples
    times = []
    for n_i in counts:
        if regular:
            times.append(_make_regular_observation_times(n_i, T))
        else:
            times.append(_make_irregular_observation_times(n_i, T, rng))
    return times

def add_noise(Y_true, sigma, rng):
    """Add i.i.d. Gaussian measurement noise to each trajectory.

    Adds independent N(0, sigma**2) noise to every observation,
    producing the noisy values the model sees. With sigma = 0 
    the trajectories are returned unchanged.

    Args:
        Y_true: list of length D, where Y_true[i] is an np.ndarray of
            shape (N_i,) holding individual i's noise-free trajectory
            values.
        sigma: float >= 0, standard deviation of the measurement noise.
        rng: np.random.Generator used to draw the noise (advances its
            state).

    Returns:
        Y_noisy: list of length D, where Y_noisy[i] = Y_true[i] + noise,
            an np.ndarray of shape (N_i,). A new list is returned;
            Y_true is not modified in place.
    """
    if sigma == 0:
        return [y.copy() for y in Y_true]
    return [y + rng.normal(0.0, sigma, size=y.shape) for y in Y_true]



def generate_simulated_dataset(dgp, D, N=20, sigma=0.0, regular=True,
                                include_endpoints=True, seed=None):
    """Generate a SimulatedDataset of D individuals from a data-generating process.

    Orchestrate the full generative process and return a
    SimulatedDataset holding both the model-visible data and the
    withheld ground truth. The pipeline is:

        1. Draw a population (X, Z, params)
        2. Build per-individual observation times on [0, T],
        3. Get noise-free trajectories at those times,
        3. Add i.i.d. N(0, sigma^2) measurement noise.

    Args:
        dgp: the data-generating process (e.g. a WilkersonDGP instance).
        D: int, number of individuals to generate.
        N: observation count per individual. Either an int (shared) or
            a length-D sequence of ints (variable N_i). Default 20.
        sigma: float >= 0, measurement-noise standard deviation.
            Default 0.0 (noise-free).
        regular: if True, evenly spaced observation times; if False,
            irregular times drawn per individual. Default True.
        include_endpoints: Whether or not to include endpoints at [0, T] only
            relevant when regular False
        seed: int or None, top-level seed for the run's rng.

    Returns:
        SimulatedDataset with the model-visible (X, times,
        Y_noisy), the ground-truth (Z, params, Y_true; true_shapes
        and true_sigma left as None), and a reference to `dgp`, and 
        a `design` record of D/N/sigma/regular/include_endpoints/seed
        for reproduction.
    """
    # Step 0: Initialise a random number generator
    rng = np.random.default_rng(seed)

    # Step 1: Draw observed covariates, latent factors and compute parameters
    X, Z, params = dgp.draw_population(D, rng)

    # Step 2: Build per-individual observation times on [0, T]
    times = make_observation_times(D, N, dgp.T, regular, include_endpoints, rng)

    # Step 3: Evaluate the noise-free Wilkerson trajectory at those times
    Y_true = dgp.true_curves(times, X, params)

    # Step 4: Add i.i.d. N(0, sigma^2) measurement noise
    Y_noisy = add_noise(Y_true, sigma, rng)

    # Step 5: Generate the SimulatedDataset instance and return it
    design = {
        "D": D, "N": N, "sigma": sigma, "regular": regular, 
        "include_endpoints": include_endpoints, "seed": seed,
    }

    dataset =  SimulatedDataset(
        X=X, times=times, Y_noisy=Y_noisy,
        Z=Z, params=params, Y_true=Y_true, true_shapes = None, true_sigma = None,
        dgp=dgp, design=design, 
    )

    return dataset

