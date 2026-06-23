
def sample_covariates(D, ranges, rng):
    """Draw observed static covariates for D samples.

    Each covariate is drawn independently from a UNIFORM distribution
    whose bounds are given by 'ranges'. These are the covariates the
    model is allowed to observe.

    Args:
        D: int, number of samples to draw.
        ranges: dict mapping covariate name to a (min, max) tuple, e.g.
            {"size": (0.1, 0.5), "age": (20, 80),
             "weight": (40, 100), "dosage": (0.0, 1.0)}.
        rng: np.random.Generator used for sampling (advances its state).

    Returns:
        X: dict mapping each covariate name to an np.ndarray of shape
            (D,) holding that covariate's value for all D samples.
    """
    X = dict()
    for name, (min_val, max_val) in ranges.items():
        X[name] = rng.uniform(low=min_val, high=max_val, size=D)

    return X


def sample_latents(D, rng):
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


def compute_parameters(covariates, latents, hyperparams):
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
        hyperparams: dict of float hyperparameters with keys
            "g0", "d0", "rho0", (strictly positive baseline constants) 
            and "alpha_g", "alpha_d" (non-negative latent strengths;
            zero disables the corresponding latent factor, e.g. for
            generating data with no aleatoric heterogeneity).

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
    g_0 = hyperparams["g0"] # scalar
    d_0 = hyperparams["d0"]
    rho_0 = hyperparams["rho0"]
    alpha_g = hyperparams["alpha_g"]
    alpha_d = hyperparams["alpha_d"]
    
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


def compute_wilkerson(t, size, g, d, rho):
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
        gen_config: dict capturing the full generative configuration used
            to produce this dataset, so it can be reproduced exactly.
            Keys: "D", "N", "T", "regular", "include_endpoints", "ranges",
            "hyperparams", "sigma", "seed".
    """
    def __init__(self,
        X, times, Y_noisy,
        Z, params, Y_true, true_shapes = None, true_sigma = None,
        gen_config=None,
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
        self.gen_config = {} if gen_config is None else gen_config

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
        """Time horizon, from the generative config."""
        return self.gen_config.get("T")

    @property
    def sigma(self):
        """Measurement-noise std, from the generative config."""
        return self.gen_config.get("sigma")

    @property
    def hyperparams(self):
        """Generative hyperparameters, from the generative config."""
        return self.gen_config.get("hyperparams", {})

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


def simulate_dataset(
    D,
    N=20,
    hyperparams=None,
    sigma=0.0,
    ranges=None,
    T=1.0,
    regular=True,
    include_endpoints=True,
    seed=None,
):
    """Generate a simulated panel dataset from the latent-factor Wilkerson model.

    Orchestrates the full generative process and returns a
    SimulatedDataset holding both the model-visible data and the
    withheld ground truth. The pipeline is:

        1. draw observed covariates (size, age, weight, dosage),
        2. draw independent latent factors (z_g, z_d),
        3. map covariates + latents to parameters (g, d, rho),
        4. build per-individual observation times on [0, T],
        5. evaluate the noise-free Wilkerson trajectory at those times,
        6. add i.i.d. N(0, sigma^2) measurement noise.

    A single random number generator (rng) is created from `seed` and 
    threaded through every stochastic step, so the whole dataset is 
    reproducible from `seed` together with the other generative arguments. 
        - Setting alpha_g = alpha_d = 0 (in hyperparams)
            removes unobserved heterogeneity; 
        - Setting sigma = 0 removes measurement noise.

    Args:
        D: int, number of individuals to generate.
        N: observation count per individual. Either an int (shared) or
            a length-D sequence of ints (variable N_i). Default 20.
        hyperparams: dict with keys "g0", "d0", "rho0", "alpha_g",
            "alpha_d". If None, the default Kacprzyk values are used.
        sigma: float >= 0, measurement-noise standard deviation.
            Default 0.0 (noise-free).
        ranges: dict mapping covariate name to (min, max). If None, the
            default Kacprzyk sampling ranges are used.
        T: float, time horizon [0, T]. Default 1.0.
        regular: if True, evenly spaced observation times; if False,
            irregular times drawn per individual. Default True.
        seed: int or None, top-level seed for the run's rng.

    Returns:
        SimulatedDataset with the model-visible (X, times,
        Y_noisy), the ground-truth (Z, params, Y_true; true_shapes
        and true_sigma left as None), and a gen_config dict recording
        the full generative configuration (D, N, T, regular,
        include_endpoints, ranges, hyperparams, sigma, seed) for
        reproduction.
    """
    # Step 0: Initialisations

    # -- Initialise a random number generator
    rng = np.random.default_rng(seed)

    # -- Initialise uniform distribution ranges if not provided
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
    
    # -- Initialise hyperparams if not provided
    expected_hp = {"g0", "d0", "rho0", "alpha_g", "alpha_d"}
    if hyperparams is None:
        hyperparams = {
                "g0": 2.0, 
                "d0": 180.0, 
                "rho0": 10.0,
                "alpha_g": 0.3, 
                "alpha_d": 0.3,
        }
    elif set(hyperparams) != expected_hp:
        raise ValueError(f"hyperparams keys {set(hyperparams)} must be {expected_hp}.")

    # Step 1: Draw observed covariates (size, age, weight, dosage)
    X = sample_covariates(D, ranges, rng)

    # Step 2: Draw independent latent factors (z_g, z_d)
    Z = sample_latents(D, rng)

    # Step 3: Map covariates + latents to parameters (g, d, rho)
    parameters = compute_parameters(X, Z, hyperparams)

    # Step 4: Build per-individual observation times on [0, T]
    times = make_observation_times(D, N, T, regular=regular, include_endpoints=include_endpoints, rng=rng)

    # Step 5: Evaluate the noise-free Wilkerson trajectory at those times
    Y_true = [
        compute_wilkerson(times[i], X["size"][i], parameters["g"][i],
                  parameters["d"][i], parameters["rho"][i])
        for i in range(D)
    ]

    # Step 6: Add i.i.d. N(0, sigma^2) measurement noise
    Y_noisy = add_noise(Y_true, sigma, rng)

    # Step 7: Generate the SimulatedDataset instance and return it
    gen_config = {
        "D": D, "N": N, "T": T,
        "regular": regular, "include_endpoints": include_endpoints,
        "ranges": ranges, "hyperparams": hyperparams,
        "sigma": sigma, "seed": seed,
    }

    dataset =  SimulatedDataset(
        X=X, times=times, Y_noisy=Y_noisy,
        Z=Z, params=parameters, Y_true=Y_true, true_shapes = None, true_sigma = None,
        gen_config=gen_config,
    )

    return dataset


def true_shape_summaries(true_params, size, C, breakpoints, T):
    pass

def estimate_true_sigma(covariates_ref, hyperparams, n_mc):
    pass

