
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# The horizon every dataset is rescaled onto, matching config.T.
T_HORIZON = 1.0

# The range declared for each dataset's covariates. 
DATASET_SPECS = {
    "airfoil": {
        "feature_ranges": {
            "angle": (0, 22),
            "chord": (0.025, 0.30),
            "velocity": (31, 71),
            "thickness": (0.0004, 0.05),
        },
    },
    "flchain": {
        "feature_ranges": {
            "age": (50, 100),
            "sex": ["M", "F"],
            "creatinine": (0.4, 2.0),
            "kappa": (0.01, 5.0),
            "lambda": (0.04, 5.0),
            "flc.grp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "mgus": ["no", "yes"],
        },
    },
}


class RealDataset:
    """An observed panel dataset with model-visible data only.

    The dataset holds D individuals stored as a list-of-arrays indexed by
    individual; observation counts and times may differ across individuals.
    Unlike the simulated dataset there are only two parts of data, because no
    process is available to withhold a ground truth from the model:
        1. Model-visible
            This data may be passed to a model.
        2. Metadata
            This is data on where the observations came from.

    Model-visible
        X: dict mapping each covariate name to an np.ndarray of shape
            (D,) holding that covariate's value for all D samples.
        times: list of length D, where times[i] is a sorted np.ndarray
            of the observation times for individual i, rescaled onto [0, T].
        Y_noisy: list of length D, where Y_noisy[i] is an np.ndarray of shape
            (N_i,) holding individual i's observed outcomes. The name is kept
            from the simulated dataset because it is accurate here too: these
            are measurements, and so carry noise. It is Y_true that has no
            counterpart on real data.

    Metadata:
        name: str; which dataset this is, one of the keys of DATASET_SPECS.
        feature_ranges: dict mapping each original covariate name to its
            declared range, a tuple for a continuous covariate and a list of
            levels for a categorical one. Kept so the encoding a column
            received can be traced back to the covariate it came from.
        design: dict of the settings this dataset was built under, with keys
            "D", "T", "source", "options", "seed" and "scale_time_by".
    """

    def __init__(self, X, times, Y_noisy, name=None, feature_ranges=None,
                 design=None):

        # Model-visible data
        self.X = X
        self.times = times
        self.Y_noisy = Y_noisy

        # Metadata
        self.name = name
        self.feature_ranges = {} if feature_ranges is None else feature_ranges
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
        """Return the time horizon, from the draw design."""
        return self.design.get("T", T_HORIZON)

    @property
    def gen_config(self):
        """Return the settings describing where this dataset came from.

        The bootstrap ensemble writes this into run.json, so that a saved run
        records the data it was fitted on.
        """
        return dict(self.design)

    def select_individuals(self, idx):
        """Return a new RealDataset with only the selected individuals.

        Args:
            idx: 1-D array of int; the individual indices to keep, in
                the order they should appear in the returned dataset.

        Note:
        The function permits indices to repeat, in which case the corresponding
        individual appears more than once. We use this for bootstrap resampling.

        Returns:
            RealDataset: a new dataset of len(indices) individuals
        """
        # Step 1: Slice the per-individual dictionary
        X = {name: values[idx] for name, values in self.X.items()}

        # Step 2: Slice the per-individual lists
        times = [self.times[i] for i in idx]
        Y_noisy = [self.Y_noisy[i] for i in idx]

        # Step 3: Copy the design, recording the new number of individuals
        design = dict(self.design)
        design["D"] = len(idx)

        # Step 4: Build the new dataset
        return RealDataset(
            X=X, times=times, Y_noisy=Y_noisy,
            name=self.name, feature_ranges=self.feature_ranges, design=design,
        )

    def split_test_val_train(self, D_train, D_val, D_test):
        """Split the real dataset into test, validation, and training sets.

        The dataset is carved into three contiguous, non-overlapping blocks:
            - the first D_test individuals become the test set,
            - the next D_val individuals become the validation set,
            - the remaining D_train individuals become the training set.

        Contiguous blocks are only safe on a dataset whose individuals are in
        no meaningful order, which a file on disk generally is not. Call
        shuffle_individuals first, as load_real_dataset already does.

        Args:
            D_train: int; the number of individuals in the training set.
            D_val: int; the number of individuals in the validation set.
            D_test: int; the number of individuals in the test set.

        Returns:
            Split: a namedtuple (test, val, train) of RealDatasets, so
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

    def individual_covariates(self):
        """Return covariates as a D-length list of per-individual feature dicts.

        Converts the column-oriented X (covariate name to array over individuals)
        into a row-oriented list, one dict per individual, as required by the
        inference engine.

        self.X stores individual covariates as:
            {
                "age":   [person1, person2, person3, ...],
                "sex":   [person1, person2, person3, ...],
                            ...
            }
        We modify this into a list of feature values for individuals:
            [
            {"age": float, "sex": float, ...}, # Person 1
            {"age": float, "sex": float, ...}, # Person 2
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

    def shuffle_individuals(self, seed):
        """Return a new dataset with the individuals in a random order.

        A real dataset arrives in whatever order its file happens to use, which
        is often sorted by a covariate. Since split_test_val_train carves
        contiguous blocks, splitting such a dataset directly would hand the
        model a training set unlike its test set. Shuffling first removes that.

        Args:
            seed: int; the seed fixing the permutation.

        Returns:
            RealDataset holding the same individuals in a permuted order.
        """
        permutation = np.random.default_rng(seed).permutation(self.D)
        return self.select_individuals(permutation)

    def split_by_fractions(self, train_fraction, val_fraction):
        """Split the dataset by proportion rather than by count.

        Real datasets differ by more than an order of magnitude in size, so the
        split is easier to state as a proportion. The test set takes whatever
        the other two leave, so that every individual is used exactly once.

        Args:
            train_fraction: float in (0, 1); the share going to training.
            val_fraction: float in (0, 1); the share going to validation.

        Returns:
            Split: a namedtuple (test, val, train), as split_test_val_train
                returns.
        """
        D_train = int(round(train_fraction * self.D))
        D_val = int(round(val_fraction * self.D))
        D_test = self.D - D_train - D_val

        return self.split_test_val_train(D_train=D_train, D_val=D_val,
                                         D_test=D_test)

    def compute_horizon_coverage(self):
        """Return the share of the horizon each individual's observations span.

        Every individual is forecast over the whole of [0, T], because the model
        maps a covariate vector to a trajectory on that interval and nothing
        about an individual's own observation window enters. An individual whose
        observations cover a quarter of the horizon therefore has its shape
        summary decided mostly where it has no data, and its shape uncertainty
        will reflect that. This reports how much of the horizon each individual
        actually covers, so the reader can see it.

        Returns:
            np.ndarray of shape (D,) and dtype float; each in [0, 1].
        """
        spans = [t.max() - t.min() for t in self.times]
        return np.asarray(spans, dtype=float) / self.T


def _group_by_individual(frame, covariate_names):
    """Split a long-format frame into one trajectory per individual.

    Args:
        frame: pd.DataFrame carrying an "id" column, the covariate columns, and
            the "t" and "y" columns, one row per observation.
        covariate_names: list of str; the covariate columns to carry over.

    Returns:
        tuple (covariates, times, outcomes):
            covariates: pd.DataFrame of shape (D, M); one row per individual,
                taken from that individual's first observation.
            times: list of length D of np.ndarray, each sorted ascending.
            outcomes: list of length D of np.ndarray, aligned with times.
    """
    covariates, times, outcomes = [], [], []

    for _, rows in frame.groupby("id", sort=False):
        rows = rows.sort_values(by="t")
        covariates.append(rows.iloc[[0]][covariate_names])
        times.append(rows["t"].to_numpy(dtype=float))
        outcomes.append(rows["y"].to_numpy(dtype=float))

    return pd.concat(covariates, ignore_index=True), times, outcomes


def _read_airfoil():
    """Read the airfoil self-noise measurements.

    The file is one row per (configuration, frequency) with no identifier, so
    the 106 individuals have to be recovered: each configuration is a sweep
    upwards through the frequencies, and a new one begins wherever the
    frequency drops. Frequency is then divided by its own minimum of 200 Hz and
    read on a log axis, which is how the paper reports this dataset and which
    matters here because the frequencies are one-third-octave bands and so are
    spaced geometrically; on a linear axis most of the horizon would be empty.

    Returns:
        tuple (covariates, times, outcomes), as _group_by_individual returns.
    """
    columns = ["t", "angle", "chord", "velocity", "thickness", "y"]
    frame = pd.read_csv(DATA_DIR / "airfoil" / "airfoil_self_noise.dat",
                        sep="\t", header=None, names=columns)

    # Step 1: A falling frequency marks the start of the next configuration
    frame["id"] = (frame["t"].diff().fillna(-1.0) < 0).cumsum()

    # Step 2: Put the frequency on the axis the paper uses
    frame["t"] = np.log(frame["t"] / 200.0)

    return _group_by_individual(frame, list(DATASET_SPECS["airfoil"]
                                            ["feature_ranges"]))


def _read_flchain():
    """Read the survival curves derived from the free light chain study.

    This file already carries an identifier and a trajectory per subject, so
    only the time column needs rescaling. It is built by data/flchain/
    build_flchain.py, which repairs a covariate/trajectory mismatch in the file
    TimeView published; see that module for what the mismatch was.

    Returns:
        tuple (covariates, times, outcomes), as _group_by_individual returns.
    """
    frame = pd.read_csv(DATA_DIR / "flchain" / "flchain.csv")
    frame["t"] = frame["t"] / 5000.0

    return _group_by_individual(frame, list(DATASET_SPECS["flchain"]
                                            ["feature_ranges"]))


READERS = {"airfoil": _read_airfoil,
           "flchain": _read_flchain}


def _encode_covariates(frame, feature_ranges):
    """Turn the covariate frame into the numeric dict the model reads.

    The declared range of a covariate says how it should be encoded, which is
    the convention TimeView uses:
        - a tuple, such as ("age", (50, 100)), is continuous and passes through;
        - a list of two levels, such as ("sex", ["M", "F"]), is binary and
          becomes a single indicator, one for the second level;
        - a list of more than two levels becomes one indicator per level.
    The indicators are left unstandardised here; the Normaliser standardises
    every column alike when it is fitted on the training split.

    Args:
        frame: pd.DataFrame of shape (D, M); the covariates as read.
        feature_ranges: dict; each covariate's declared range.

    Returns:
        dict mapping each encoded covariate name to an np.ndarray of shape (D,)
            and dtype float.
    """
    X = {}
    for covariate, declared_range in feature_ranges.items():
        column = frame[covariate]

        # Case 1: a tuple declares an interval, so the covariate is continuous
        if isinstance(declared_range, tuple):
            X[covariate] = np.asarray(column, dtype=float)
            continue

        levels = list(declared_range)

        # Case 2: two levels need only one indicator to be told apart
        if len(levels) == 2:
            X[covariate] = np.asarray(column == levels[1], dtype=float)
            continue

        # Case 3: more than two levels get one indicator each
        for level in levels:
            X[f"{covariate}_{level}"] = np.asarray(column == level, dtype=float)

    return X


def _rescale_times_to_horizon(times, T=T_HORIZON):
    """Divide every observation time by the cohort's own maximum.

    Individuals share one horizon but need not span it: an airfoil sample
    covers only a stretch of the frequency range, a median of 65% of it.
    Dividing every individual by the same maximum lands the cohort on [0, T]
    while keeping those differences, which scaling each individual separately
    would erase. Zero is left where it is, since it is a real origin in both
    datasets.

    Args:
        times: list of length D of np.ndarray; the raw observation times.
        T: float; the right endpoint of the horizon.

    Returns:
        tuple (rescaled, scale):
            rescaled: list of length D of np.ndarray; times on [0, T].
            scale: float; the divisor, kept so the original units can be
                recovered.
    """
    scale = max(np.max(t) for t in times) / T

    return [np.asarray(t, dtype=float) / scale for t in times], float(scale)


def load_real_dataset(name, seed=0, T=T_HORIZON, **options):
    """Load one real dataset into the shape the model machinery reads.

    Args:
        name: str; a key of DATASET_SPECS, either "airfoil" or "flchain".
        seed: int; the seed fixing the shuffle applied before any split.
        T: float; the right endpoint of the horizon every dataset is put on.
        **options: passed to that dataset's reader. Neither reader currently
            takes any, so this is empty; it is kept because the design records
            it, and a saved run's provenance reports what it was built under.

    Returns:
        RealDataset holding all D individuals in a shuffled order.
    """
    # Step 1: Read the observations, one trajectory per individual
    frame, raw_times, outcomes = READERS[name](**options)

    # Step 2: Encode the covariates, expanding the categorical ones
    feature_ranges = DATASET_SPECS[name]["feature_ranges"]
    X = _encode_covariates(frame, feature_ranges)

    # Step 3: Put every individual on the shared horizon
    times, scale = _rescale_times_to_horizon(raw_times, T=T)

    # Step 4: Record what the dataset was built from
    design = {
        "D": len(outcomes),
        "T": T,
        "source": name,
        "options": dict(options),
        "seed": seed,
        "scale_time_by": scale,
    }

    dataset = RealDataset(
        X=X,
        times=times,
        Y_noisy=[np.asarray(y, dtype=float) for y in outcomes],
        name=name,
        feature_ranges=feature_ranges,
        design=design,
    )

    # Step 5: Break whatever order the file arrived in, before anything splits it
    return dataset.shuffle_individuals(seed)
