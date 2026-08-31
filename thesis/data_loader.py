from pathlib import Path
import sys

import numpy as np
import torch

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR.parent))
sys.path.insert(0, str(THESIS_DIR.parent / "scripts" / "shape_uncertainty"
                      / "real_data"))

import config

from real_dataset import load_real_dataset
from scripts.shape_uncertainty.simulated_data.data_generator import (
    WilkersonDGP, generate_simulated_dataset)
from scripts.shape_uncertainty.spline_basis.bspline_basis import build_knot_dictionary


# The cubic B-spline basis every model in the thesis is fitted in.
KNOT_CONFIG = {"nr_interior_knots": config.NR_INTERIOR_KNOTS, "T": config.T}

# The share of individuals going to training and to validation; the test split
# takes the remainder. TimeView splits its datasets the same way.
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15


def draw_simulated_split(regime):
    """Draw the Wilkerson dataset for one regime and split it.

    Args:
        regime: config.Regime; supplies the heterogeneity level and the
            observation design the population is drawn under.

    Returns:
        Split; the namedtuple (test, val, train) of SimulatedDatasets.
    """
    # Step 1: Fix the seeds the draw is taken under
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Step 2: Build the Wilkerson process at this regime's heterogeneity level
    dgp = WilkersonDGP(T=config.T, hyperparams=regime.hyperparams)

    # Step 3: Draw the full population under the shared observation design
    dataset = generate_simulated_dataset(dgp, D=config.D_TOTAL, **regime.design)

    # Step 4: Split it
    return dataset.split_test_val_train(
        D_train=config.D_TRAIN, D_val=config.D_VAL, D_test=config.D_TEST)


def load_real_split(name):
    """Load one deployment dataset and carve it into test, validation and train.

    The loader shuffles under a fixed seed before returning, so the contiguous
    blocks the split takes are not a block of the file's own ordering.

    Args:
        name: str; the dataset, "flchain" or "airfoil".

    Returns:
        tuple (dataset, split):
            dataset: RealDataset; all D individuals, whose gen_config records
                what the whole dataset was built from.
            split: Split; the namedtuple (test, val, train) of RealDatasets.
    """
    dataset = load_real_dataset(name, seed=config.SEED)

    return dataset, dataset.split_by_fractions(TRAIN_FRACTION, VAL_FRACTION)


def build_knot_objects():
    """Build the spline basis at the shared knot configuration.

    Returns:
        dict from build_knot_dictionary, carrying "basis_functions", "C",
            "breakpoints", "Omega" and "nr_basis".
    """
    return build_knot_dictionary(**KNOT_CONFIG)
