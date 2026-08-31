"""Configurations for thesis experiments. 

The configurations are grouped as follows:
    1.  Time horizon and data-generating process
    2.  Dataset sizes
    3.  Observation design
    4.  Spline basis
    5.  Shape extraction
    6.  Shape distance
    7.  Regularisation
    8.  Optimisation, Monte-Carlo budgets and evaluation resolution
    9.  Seeds
    10. DGP regimes A, B and C
"""

from dataclasses import dataclass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Time horizon and data-generating process
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# The forecasting horizon [0, T]
T = 1.0

# Structural hyperparameters of the Wilkerson tumour-trajectory process
BASE_HYPERPARAMS = {
    "g0": 2.0,      # baseline growth rate of the treatment resistant tumour fraction
    "d0": 180.0,    # baseline decay rate of the treatment sensitive toumour fraction
    "rho0": 10.0,   # dosage sensitivity of the treatment-sensitive fraction
}

# Baseline unobserved heterogeneity, used wherever heterogeneity is switched on.
ALPHA_G = 0.00
ALPHA_D = 0.35

# Covariate ranges
COVARIATE_RANGES = {
    "size": (0.1, 0.5),
    "age": (20.0, 80.0),
    "weight": (40.0, 100.0),
    "dosage": (0.0, 1.0),
}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Dataset sizes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
D_TRAIN = 1700
D_VAL = 300
D_TEST = 1000
D_TOTAL = D_TRAIN + D_VAL + D_TEST


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Observation design
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Observations per individual (shared across all D individuals)
NR_OBS = 20

# Every individual is measured on the same evenlly paced grid on [0, T]
REGULAR = True
INCLUDE_ENDPOINTS = True

# Measurement-noise standard deviation.
SIGMA = 0.01


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. Spline basis
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Cubic B-spline basis on [0, T]. The coefficient dimension is
# B = NR_INTERIOR_KNOTS + 4 = 9, matching TimeView.
NR_INTERIOR_KNOTS = 5
NR_BASIS = NR_INTERIOR_KNOTS + 4


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5. Shape extraction
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings passed to extract_shape_summary.
#  - De-duplication at zeta_rel = 1e-6 collapses numerically coincident transition points;
#  - Flatness detection is off as curves considered in this repository are never linear; 
#  - Significance pruning is on at 2.5% of thetrajectory amplitude.
SHAPE_CONFIG = {
    "zeta_rel": 0.0,            # min transition spacing, as a fraction of T (off)
    "upsilon_rel_1": 0.0,       # relative slope flatness threshold (off)
    "upsilon_rel_2": 0.0,       # relative curvature flatness threshold (off)
    "upsilon_rel_prune": 0.025, # relative pruning threshold, fraction of A
    "do_prune": True,           # apply slope then curvature pruning
}

# The naive shape extraction setting: no de-duplication, no flatness, no pruning.
SHAPE_CONFIG_NAIVE = {
    "zeta_rel": 0.0,
    "upsilon_rel_1": 0.0,
    "upsilon_rel_2": 0.0,
    "upsilon_rel_prune": 0.0,
    "do_prune": False,
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6. Shape distance
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Weights of the pointwise shape-state distance. 
#  - Slope disagreement (alpha) is set to be twice as costly as a curvature-only disagreement (beta); 
#  - both are bounded by 1/2 so the distance is in [0, 1].
SHAPE_DISTANCE_CONFIG = {
    "alpha": 1 / 2,  # slope disagreement weight
    "beta": 1 / 4,   # curvature disagreement weight
}

# Fixed upper limits of the uncertainty panels' vertical axes. 
#  - Q_MAX is the largest value the pointwise shape dispersion Q(t) can take,
#    attained where the state distribution is uniform over the four curved
#    states. It is alpha + beta / 2, so 5/8 at the weights above.
#  - V_MAX is a display choice, not a bound
Q_MAX = SHAPE_DISTANCE_CONFIG["alpha"] + SHAPE_DISTANCE_CONFIG["beta"] / 2
V_MAX = 0.25


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7.Regularisation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MeanOnly Model Regularisation
#  - This model has no random effects, so lambda_re is zero structurally.
LAMBDA_MEAN_MEANONLY, LAMBDA_RE_MEANONLY = 1e-4, 0.0

# GaussianModel Model Regularisation
#  - This model leave the mean unpenalised 
LAMBDA_MEAN_GAUSSIAN, LAMBDA_RE_GAUSSIAN = 0.0, 1.0

PENALTIES = {
    "Gaussian": {"lambda_mean": LAMBDA_MEAN_GAUSSIAN,
                 "lambda_re": LAMBDA_RE_GAUSSIAN},
    "MeanOnly": {"lambda_mean": LAMBDA_MEAN_MEANONLY,
                 "lambda_re": LAMBDA_RE_MEANONLY},
}

# The unpenalised reference
LAMBDA_MEAN_UNPENALISED, LAMBDA_RE_UNPENALISED = 0.0, 0.0


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 8. Optimisation, Monte-Carlo budgets and evaluation resolution
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Optuna architecture search and training schedule.
NR_TRIALS = 100    # architecture search trials
NR_EPOCHS = 200    # max epochs per fit
PATIENCE = 10      # early-stopping patience, in epochs

# Aleatoric uncertainty 
N_DRAWS = 100

# Epistemic uncertainty
NR_MEMBERS = 100

# Ground-truth covariance estimation
N_REF = D_TEST  # we estimate the truth for every individual in the test set
N_MC = 800  # draws per individual

# Resolution of the evenly spaced grid on [0, T] that trajectories are evaluated
# on. This is the resolution parameter of the value-space uncertainty V. It is
# also the grid the figures draw their curves on.
N_DENSE = 200

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 9. Seeds
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Seed to fix data generation and architecture search
SEED = 42

# Seed used for evaluation-time draws
SEED_EVAL = 0

# Replicate count and the derived seeds. Replicates differ only in the
# encoder's initialisation; the data are held fixed.
N_SEEDS = 3
SEEDS = [SEED + s for s in range(N_SEEDS)]  # [42, 43, 44]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 10. DGP regimes: homogeneous and heterogeneous
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@dataclass(frozen=True)
class Regime:
    """A DGP regime specifies a level of observation noise and the level of unobserved heterogeneity.

    Attributes:
        name: str; the regime label, "homogeneous" or "heterogeneous".
        title: str; regime name for captions.
        sigma: float; measurement-noise standard deviation.
        alpha_g: float; latent scale on the growth rate.
        alpha_d: float; latent scale on the decay rate.
    """
    name: str
    title: str
    sigma: float
    alpha_g: float
    alpha_d: float

    @property
    def hyperparams(self):
        """Return the full WilkersonDGP hyperparameter dict for this regime."""
        return {**BASE_HYPERPARAMS,
                "alpha_g": self.alpha_g,
                "alpha_d": self.alpha_d}

    @property
    def design(self):
        """Return the generate_simulated_dataset keyword arguments."""
        return {"N": NR_OBS, "sigma": self.sigma, "regular": REGULAR,
                "include_endpoints": INCLUDE_ENDPOINTS, "seed": SEED}

    @property
    def has_noise(self):
        """Return whether observation noise is switched on."""
        return self.sigma > 0.0

    @property
    def has_heterogeneity(self):
        """Return whether unobserved heterogeneity is switched on."""
        return self.alpha_g > 0.0 or self.alpha_d > 0.0


REGIME_HOMOGENEOUS = Regime(
    name="homogeneous",
    title="Without unobserved heterogeneity",
    sigma=SIGMA,
    alpha_g=0.0,
    alpha_d=0.0,
)

REGIME_HETEROGENEOUS = Regime(
    name="heterogeneous",
    title="With unobserved heterogeneity",
    sigma=SIGMA,
    alpha_g=ALPHA_G,
    alpha_d=ALPHA_D,
)

REGIME_NOISELESS = Regime(
    name="noiseless",
    title="Noise-free",
    sigma=0.0,
    alpha_g=0.0,
    alpha_d=0.0,
)

REGIMES = {"homogeneous": REGIME_HOMOGENEOUS,
           "heterogeneous": REGIME_HETEROGENEOUS,
           "noiseless": REGIME_NOISELESS}