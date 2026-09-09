"""Rebuild flchain.csv so every survival curve sits beside its own covariates.

The file published with TimeView pairs each trajectory with the wrong subject's
covariates. TimeView dropped the 1350 subjects of the raw 7874-row study with
no recorded serum creatinine, fitted a random survival forest to the remaining
6524, and gave each curve an id numbered by its row in that reduced frame. The
covariates were then matched back on those ids as if they were the raw frame's
own row labels, which they are not: dropping the incomplete subjects shifted
every row after the first drop at raw row 15. The 1025 ids landing on a dropped
row drew blank covariates and were discarded, leaving 5499 subjects.

The mismatch is obvious since a forest is a deterministic function of its 
covariates, so the two subject pairs sharing a covariate vector
must share a curve, yet in the published file theirs differ by 0.19.

The repair is a relabelling. This is done in this file.

    python scripts/shape_uncertainty/real_data/data/flchain/build_flchain.py
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

# The raw Mayo Clinic study, as distributed with scikit-survival. It is fetched
# once and kept beside the built file so the rebuild needs no network again.
RAW_URL = ("https://raw.githubusercontent.com/sebp/scikit-survival/master/"
           "sksurv/datasets/data/flchain.arff")
RAW_PATH = DATA_DIR / "flchain.arff"

# The file TimeView published, and the repaired file written in its place. The
# published file is only ever read, so this script is safe to run twice.
PUBLISHED_PATH = DATA_DIR / "flchain_old.csv"
CORRECTED_PATH = DATA_DIR / "flchain.csv"

# The raw study columns, in the order the ARFF header declares them.
RAW_COLUMNS = ["age", "sex", "sample.yr", "kappa", "lambda", "flc.grp",
               "creatinine", "mgus", "futime", "death", "chapter"]

# The seven covariates TimeView models on, in the column order of the published
# file, so that the rebuilt file can stand in for it unchanged.
COVARIATES = ["age", "creatinine", "flc.grp", "kappa", "lambda", "mgus", "sex"]

# The types the published file stores those covariates as. The two categorical
# ones are left as read.
COVARIATE_TYPES = {"age": float, "creatinine": float, "flc.grp": int,
                   "kappa": float, "lambda": float}


def fetch_raw_study(path=RAW_PATH):
    """Download the raw study data unless it is already on disk.

    Args:
        path: pathlib.Path; where the ARFF file is kept.

    Returns:
        pathlib.Path; the path, now guaranteed to exist.
    """
    if not path.exists():
        print(f"Downloading the raw study data to {path.name}")
        urllib.request.urlretrieve(RAW_URL, path)

    return path


def read_raw_study(path=RAW_PATH):
    """Read the raw study data, one row per subject.

    Args:
        path: pathlib.Path; the ARFF file to read.

    Returns:
        pd.DataFrame of 7874 rows labelled 0-7873, holding RAW_COLUMNS, with a
            missing creatinine read as NaN.
    """
    lines = [line.strip().lower() for line in path.read_text().splitlines()]
    header_length = lines.index("@data") + 1

    return pd.read_csv(path, skiprows=header_length, header=None,
                       names=RAW_COLUMNS, na_values="?")


def locate_creatinine_complete_rows(raw):
    """Return the raw label of every subject the forest was fitted on.

    The forest saw the subjects with a recorded creatinine, and labelled their
    curves by position within that reduced frame. Element k of the result is
    therefore the raw row whose curve was published under id k.

    Args:
        raw: pd.DataFrame; the raw study data, as read_raw_study returns.

    Returns:
        np.ndarray of shape (6524,) and dtype int; raw labels in ascending
            order.
    """
    return raw.index[raw["creatinine"].notna()].to_numpy()


def reattach_covariates(published, raw, complete_rows):
    """Pair every published curve with the covariates that produced it.

    Args:
        published: pd.DataFrame; the published file, one row per observation,
            carrying "id", "t", "y" and the covariate columns.
        raw: pd.DataFrame; the raw study data, as read_raw_study returns.
        complete_rows: np.ndarray; the raw labels the forest was fitted on, as
            locate_creatinine_complete_rows returns.

    Returns:
        pd.DataFrame; the published identifiers, times and outcomes unchanged,
            with the covariate columns replaced.
    """
    source_row = complete_rows[published["id"].to_numpy()]

    corrected = published[["id", "t", "y"]].copy()
    for covariate in COVARIATES:
        corrected[covariate] = raw.loc[source_row, covariate].to_numpy()

    return corrected.astype(COVARIATE_TYPES)


def summarise_covariate_vectors(frame):
    """Reduce each subject's covariate vector to a single comparable string.

    Args:
        frame: pd.DataFrame; a long-format dataset carrying "id" and the
            covariate columns.

    Returns:
        pd.Series indexed by subject id; one string per subject, equal exactly
            where two subjects share a covariate vector.
    """
    per_subject = frame.groupby("id")[COVARIATES].first()

    return per_subject.astype(str).agg("|".join, axis=1)


def verify_correction(published, corrected):
    """Check that the rebuilt dataset repaired the pairing and nothing else.

    Three things must hold: the curves are untouched, they are still survival
    functions, and subjects sharing a covariate vector now share a curve. The
    last is the one that fails on the published file.

    Args:
        published: pd.DataFrame; the published file.
        corrected: pd.DataFrame; the rebuilt file, as reattach_covariates
            returns.

    Raises:
        ValueError: where any of the three checks fails.
    """
    # Step 1: Only covariates may differ; the curves are copied through
    for column in ["id", "t", "y"]:
        if not np.array_equal(corrected[column], published[column]):
            raise ValueError(f"The {column} column changed, but this script "
                             f"only repairs the covariate columns.")

    # Step 2: A survival function cannot increase
    curves = corrected.pivot(index="id", columns="t", values="y")
    if (np.diff(curves.to_numpy(), axis=1) > 0).any():
        raise ValueError("A curve increases somewhere on the horizon, so it "
                         "is no longer a survival function.")

    # Step 3: A deterministic forest gives equal covariates an equal curve
    signature = summarise_covariate_vectors(corrected)
    shared = signature.duplicated(keep=False)
    if not shared.any():
        raise ValueError("No two subjects share a covariate vector, so the "
                         "pairing cannot be checked.")

    for _, group in curves.loc[shared].groupby(signature.loc[shared]):
        if group.nunique().max() > 1:
            raise ValueError("Subjects sharing a covariate vector were given "
                             "different curves; the pairing is still wrong.")


def write_corrected_curves(corrected, path=CORRECTED_PATH):
    """Write the rebuilt dataset in place of the published one.

    Args:
        corrected: pd.DataFrame; the rebuilt file.
        path: pathlib.Path; where to write it.

    Returns:
        pathlib.Path; the path written to.
    """
    corrected.to_csv(path, index=False)
    print(f"Wrote {len(corrected)} rows for {corrected['id'].nunique()} "
          f"subjects to {path.name}")

    return path


def build_corrected_dataset():
    """Rebuild the corrected dataset from the published file and the raw study.

    Returns:
        pd.DataFrame; the rebuilt dataset that was written.
    """
    raw = read_raw_study(fetch_raw_study())
    published = pd.read_csv(PUBLISHED_PATH)

    complete_rows = locate_creatinine_complete_rows(raw)
    corrected = reattach_covariates(published, raw, complete_rows)

    verify_correction(published, corrected)

    write_corrected_curves(corrected)
    return corrected


if __name__ == "__main__":
    build_corrected_dataset()
