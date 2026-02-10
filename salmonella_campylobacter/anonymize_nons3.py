"""
Docstring for anonymize:

-----------------------------------------------------------
-----------------------------------------------------------
Goal:
Create an anonymization pipeline to handle a clean csv

-----------------------------------------------------------
-----------------------------------------------------------
To run:
1) Choose to run in the AWS Cloud (E2 with s3 connection) or
a local IDE with an ability to read and process python files.

2) Change the following variable names to match your work
- file_path => main file path to run within the document
- y_col => predictor column, and change accordingly

3) Follow the instructions in the corresponding section below
to complete the task

-----------------------------------------------------------
-----------------------------------------------------------
STEP 3 IF RUNNING IN PYTHON:
please pip install all necessary requirements
- python 3.10
- anjana
- numpy
- pandas

and please comment out the following import statements
- boto3
- bootcore
-----------------------------------------------------------
-----------------------------------------------------------
STEP 3 IF RUNNING IN AWS -> BEFORE USE IN E2:

Run the following in your bash upon starting an instance of an
E2 machine with access to the s3 read, write, and upload AIM role

export S3_INPUT_BUCKET="my-bucket"
export S3_INPUT_KEY="inputs/SalCampChicken_clean.csv"
export S3_HIER_KEY="inputs/hierarchies.pkl"
export S3_OUTPUT_BUCKET="my-bucket"
export S3_OUTPUT_PREFIX="outputs/anjana"

pip install any pip requirements needed
- designed and tested on python 3.10
- anjana
- numpy
- pandas

python3 your_script.py


-----------------------------------------------------------
-----------------------------------------------------------
To debug:
- check for any .lock files, and delete them, in your output directory if stopped
during processing or rerunning
- Check task manager. Should be running on max 82% RAM or memory before crashing
    - If CPU seems to be shorting: decrease the n_jobs variable to less than 6
    - If CPU is not powerful enough, or reaches spikes above 50%, it can crash CPU,
    so you may need to switch backend='loky' in parallelization
    - If RAM/Memory is overloading, switch backend='threading' in Parallelization

"""

# -----------------
# -- Imports ------
# -----------------

import numpy as np
import pandas as pd
import anjana
from anjana.anonymity import k_anonymity, l_diversity, t_closeness
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

import pickle
import glob
import os

import boto3
from bootcore.exceptions import ClientError
import tempfile

# ------------------------------------------------------------------------------
# -- Global Variables TO CHANGE ------------------------------------------------
# ------------------------------------------------------------------------------

# list of columns that are identifier columns
ident = [
    "Longitude",
    "Latitude",
    "EstablishmentID",
    "EstablishmentNumber",
    "EstablishmentName",
    "ProjectCode",
    "ProjectName",
    "FormID",
]
# The sensitive attribute (Usually going to be the y-column, aka the predictor value)
sens_att = "Salmon_or_camp_test"

# the CSV file path for the file want to anonymize
file_path = "data/SalCampChicken_clean.csv"

# write in any date columns here
date_columns = [
    "CollectionDate",
    "WeatherDate_Day0",
    "WeatherDate_Day1",
    "WeatherDate_Day2",
    "WeatherDate_Day3",
]
# write in any string, or other columns want to drop
cols_to_drop = [
    "State",
    "Weekday",
    "PrecipType_Day0",
    "PrecipType_Day1",
    "PrecipType_Day2",
    "PrecipType_Day3",
    "CampylobacterAnalysis30ml",
    "SalmonellaSPAnalysis",
]
# TEMP: Put strings in variable above, but eventual goal is to choose the type of processing (i.e. do a one-hot vectorization or switch to a numerical key)

# if want to fill nan or negative infinity variables, write True to the corresponding variable and then
fill_nan = False  # IF True, then goes to fill_nan_with variable, otherwise drops nan
# IF fill_nan is true, place here what you will fill it with
fill_nan_with = np.nan  # i.e. / suggusted -9

# if have negative inf (like from logs), and would like to fill them, do so here
fill_neg_inf = False  # IF True, then goes to fill_neg_inf_with variables, otherwise counts as neg inf
fill_neg_inf_with = np.nan  # i.e. / suggusted -9


# ------------------------------------------------------------------------------
# -- Global variables possible to change, but not encouraged -------------------
# ------------------------------------------------------------------------------

# global_variables
# Select the desired level of k, l and t
ks = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]  # k for k anonyminity. Also used in t-closeness and l-diversity
l_divs = [2, 3, 4, 5, 6]  # l-diversity setting to use
ts = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75]  # t-closeness setting to use
supp_levels = [10, 25, 50]  # Select the suppression limit allowed (%)


# ------------------------------------------------------------------------------
# -- Cloud Computing -----------------------------------------------------------
# ------------------------------------------------------------------------------

s3 = boto3.client("s3")


def s3_key_join(prefix: str, key: str) -> str:
    prefix = prefix.strip("/")
    key = key.lstrip("/")
    return f"{prefix}/{key}" if prefix else key


def s3_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def s3_download(bucket: str, key: str, local_path: str) -> None:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, local_path)


def s3_upload(local_path: str, bucket: str, key: str) -> None:
    s3.upload_file(local_path, bucket, key)


# output logic
def out_s3_key_for_task(task, output_prefix: str) -> str:
    kind, supp_level, k, l_div, t = task
    if kind == "k":
        rel = f"k/k_k={k}_suplevel={supp_level}_anon.parquet"
    elif kind == "l":
        rel = f"l/ldiv_k={k}_l={l_div}_suplevel={supp_level}_anon.parquet"
    elif kind == "t":
        rel = f"t/t_k={k}_t={t}_suplevel={supp_level}_anon.parquet"
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return s3_key_join(output_prefix, rel)


# ------------------------------------------------------------------------------
# -- Locking -------------------------------------------------------------------
# ------------------------------------------------------------------------------


# Locking stuctures for parallelization
def claim_lock(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(lock_path: Path):
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def out_path_for_task(task) -> Path:
    kind, supp_level, k, l_div, t = task
    if kind == "k":
        return Path(f"data/k/k_k={k}_suplevel={supp_level}_anon.parquet")
    if kind == "l":
        return Path(f"data/l/ldiv_k={k}_l={l_div}_suplevel={supp_level}_anon.parquet")
    if kind == "t":
        return Path(f"data/t/t_k={k}_t={t}_suplevel={supp_level}_anon.parquet")
    raise ValueError(f"Unknown kind: {kind}")


## -----------------------------------------------------------------------------
# -Set up ----------------------------------------------------------------------
# ------------------------------------------------------------------------------


def setup(file_path, date_columns, cols_to_drop):
    # reading in data
    data = pd.read_csv(file_path)
    print(data.head())

    # transforming catagorical data
    for date_col in date_columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data[f"ordinal_{date_col}"] = data[date_col].astype("int64") // 10**9

    # just doing the combined pos/neg of salmon and camp
    data["Salmon_or_camp_test"] = (data["CampylobacterAnalysis30ml"] == "Positive") | (
        data["SalmonellaSPAnalysis"] == "Positive"
    )

    # temp droppping salmonella and campylobacter outcome cols
    data = data.drop(columns=(cols_to_drop + date_columns))

    # filling nan values or negative inf values if the user requests it
    if fill_neg_inf:
        # filling all neg inf variables to log(1e-9) = -9
        data = data.replace(-np.inf, fill_nan_with)

    if fill_nan:
        data = data.fillna(fill_nan_with)

    return data


# ways to create intervals
# modified from source code:
def gener_intervals_with_floats(quasi_ident, inf, sup, step):
    values = np.arange(inf, sup + 1, step)
    interval = []
    for num in quasi_ident:
        lower = np.searchsorted(values, num)
        if lower == 0:
            lower = 1
        interval.append(f"[{values[lower - 1]}, {values[lower]})")

    print("interval for ", quasi_ident, " is ", interval)
    return interval


def assign_to_intervals(values, edges):
    """
    values: 1D array of numbers
    edges:  sorted 1D array of bin edges of length >= 2
    returns: list of interval strings [a,b) for each value
    """
    x = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)

    out = []
    n = len(edges)

    for num in x:
        if not np.isfinite(num):
            out.append(f"[{-np.inf}, {-9.0})")
            continue

        # idx is the "right edge index" of the bin
        idx = np.searchsorted(edges, num, side="right")

        # Clamp idx into [1, n-1] so edges[idx] is always valid
        if idx <= 0:
            idx = 1
        elif idx >= n:
            idx = n - 1

        out.append(f"[{edges[idx-1]}, {edges[idx]})")

    return out


def quantile_bins_with_tail(values, q=20, tail_q=0.01):
    """
    values = input original values
    tail q = percentile (0.01 = 1st percentile)
    q = granularity (number of bins created)
    """
    # defining the list to return
    to_return = []

    # creating low and high tail buckets
    x = values

    # defining lower and upper bounds to group all together into a box
    low_bound = np.quantile(
        x, tail_q
    )  # value at tail q's percentile (0.01 = 1st percentile)
    high_bound = np.quantile(
        x, 1 - tail_q
    )  # value at 100 - tail q's percentile (0.01 = 99th percentile)

    # adding lower bound first
    to_return.append(f"[{-np.inf}, {low_bound})")

    # Quantile edges inside the middle region
    mid = x[(x >= low_bound) & (x <= high_bound)]  # middle part
    if mid.size == 0:
        raise ValueError("quantile tails throwing error bc mid==0")
    probs = np.linspace(0, 1, q + 1)
    edges = np.quantile(mid, probs)
    edges = np.unique(edges)  # remove duplicates from ties

    if edges.size < 2:
        raise ValueError(
            f"quantile_bins_tail: Cannot form intervals: only {edges.size} unique edge(s)"
        )

    # adding quantile middle stuff
    to_return.extend(assign_to_intervals(values, edges))

    # adding upper bound (adding last to keep in order)
    to_return.append(f"[{high_bound}, {np.inf})")

    return to_return


def quantile_bins_tail_spikes(values, q=20, spike=-9, tail_q=0.01, tol=1e-3):
    """
    values = input original values
    tail q = percentile (0.01 = 1st percentile)
    q = granularity (number of bins created)

    spike= outlier that has a lot of weight (i.e. lots of -negative or 0 values)
    spike_label = label to call the spike
    tol = the fluff (error margin)
    """
    # defining the list to return
    to_return = []

    # creating low and high tail buckets
    x = values

    to_return.append(f"[{-np.inf}, {-8.0})")

    # getting the tail cuttoffs for the rest of the data
    low_bound = np.quantile(
        values, tail_q
    )  # value at tail q's percentile (0.01 = 1st percentile)
    high_bound = np.quantile(
        values, 1 - tail_q
    )  # value at 100 - tail q's percentile (0.01 = 99th percentile)

    # adding lower bound
    to_return.append(f"[{-8.0}, {low_bound})")

    # Quantile edges inside the middle region
    mid = x[(x >= low_bound) & (x <= high_bound)]  # middle part
    if mid.size == 0:
        raise ValueError(
            "spikes: Cannot compute quantiles: mid is empty (all NaN/non-finite or filtered out by tails)."
        )

    probs = np.linspace(0, 1, q + 1)
    edges = np.quantile(mid, probs)
    edges = np.unique(edges)  # remove duplicates from ties

    if edges.size < 2:
        raise ValueError(
            f"quantile_bins_tail_spikes: Cannot form intervals: only {edges.size} unique edge(s)"
        )

    # adding quantile middle stuff
    to_return.extend(assign_to_intervals(values, edges))

    # adding upper bound
    to_return.append(f"[{high_bound}, {np.inf})")
    return to_return


def mark_outliers_for_suppression(df, cols, z=4.0):
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        x = df[c].astype(float).to_numpy()
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        mask |= np.abs((x - mu) / (sd + 1e-12)) > z
    return mask


def combine_hierarchies():
    # reading from many overlapping pickle files
    # loading in all pickle files and saving them in 1 large picke file to reduce storage space

    # path to pickle files
    file_pattern = "data/hierarchies/*.pkl"
    hierarchies = {}

    for file_path in glob.glob(file_pattern):
        with open(file_path, "rb") as f:
            pkl_data = pickle.load(f)
            # taking the file name to append to the dictionary
            file_name_parts = str(f).split("_", 4)[-1]
            file_name_parts = file_name_parts.split(".")[0]

            # merges new data into combined_dict later files overwrite duplicate keys
            hierarchies[file_name_parts] = pkl_data

    # save the large dictionary back to a single file
    with open("data/hierarchies.pkl", "wb") as f:
        pickle.dump(hierarchies, f)


def develop_hierarchies(data, quasi_ident):
    # automatically creating heirerarchies

    q_vals = [40, 30, 20, 13, 10, 8, 6, 5, 4, 3]
    tail_qs = [0.01, 0.02, 0.05, 0.07, 0.1, 0.13, 0.2, 0.25, 0.3, 0.4]

    len_of_db = len(data)

    hierarchies = {}
    for identifier_col in quasi_ident:
        # checking if have already completed pickle file:
        if Path(f"data/hierarchies/hierarchies_ID_only_{identifier_col}.pkl").exists():
            print("already completed ", identifier_col)
            continue

        # functions want to test (in order): options: original_values, quantile_bins_with_tail, quantile_bins_tail_spikes, constant_intervals, suppression
        functions_to_test = [
            "original_values",
            "quantile_bins_with_tail",
            "quantile_bins_tail_spikes",
            "suppression",
        ]

        # if there is less than 8 distinct values, then this process should only do the original values
        num_unique = data[identifier_col].nunique()
        if num_unique <= 8:
            functions_to_test = ["original_values", "suppression"]

        # print(identifier_col)
        i = 0  # keeping track of current hierarchy index
        hierarchies[identifier_col] = {}

        # adding all options want to add
        for func_testing in functions_to_test:
            if func_testing == "original_values":
                hierarchies[identifier_col][i] = data[identifier_col].values
                # adding one to heirerarchy index
                i += 1
                # print("added i: ", i)

            elif func_testing == "quantile_bins_with_tail":
                for q in q_vals:
                    for tail_q in tail_qs:
                        try:
                            hierarchies[identifier_col][i] = quantile_bins_with_tail(
                                data[identifier_col].values, q=q, tail_q=tail_q
                            )
                            # adding one to heirerarchy index
                            i += 1
                            # print("added i: ", i)
                        except ValueError as e:
                            continue

            elif func_testing == "quantile_bins_tail_spikes":
                for q in q_vals:
                    for tail_q in tail_qs:
                        try:
                            hierarchies[identifier_col][i] = quantile_bins_tail_spikes(
                                data[identifier_col].values, q=q, tail_q=tail_q
                            )
                            # adding one to heirerarchy index
                            i += 1
                            # print("added i: ", i)
                        except ValueError as e:
                            continue

            # if func_testing == 'constant_intervals':
            #     # not done

            elif func_testing == "suppression":
                hierarchies[identifier_col][i] = np.array(
                    ["*"] * len_of_db
                )  # Suppression
                # adding one to heirerarchy index
                i += 1

            else:
                print("tried calling unknown function: ", func_testing)

            # saving hierarchy in a pickle file
            with open(
                f"data/hierarchies/hierarchies_ID_only_{identifier_col}.pkl", "wb"
            ) as f:
                pickle.dump(
                    hierarchies[identifier_col], f, protocol=pickle.HIGHEST_PROTOCOL
                )

    combine_hierarchies()


# this cell is entirely for debugging
def find_bad_hierarchy(hierarchies, quasi_ident):
    for qi, levels in hierarchies.items():
        if qi not in quasi_ident:
            print(f"Incorrect setup!, qi is not in quasi_identifiers, {qi}")
        # else:
        #     print(f"   qi seen: {qi} with len of levels {len(levels.keys())}")
        for lvl, arr in levels.items():
            try:
                _ = set(arr)
            except TypeError as e:
                print("\nBAD HIERARCHY FOUND")
                print("qi:", qi)
                print("level:", lvl)
                print("type(arr):", type(arr))
                # If it's array-like, inspect shape/dtype and first element type
                if hasattr(arr, "shape"):
                    print("shape:", arr.shape, "dtype:", arr.dtype)
                try:
                    first = arr[0]
                    print("type(arr[0]):", type(first))
                    # show a small preview
                    print("arr[:3]:", arr[:3])
                except Exception as e2:
                    print("could not index arr:", e2)
                raise


# this is for the s3 bucket
def s3_run_one(
    task,
    data,
    ident,
    quasi_ident,
    sens_att,
    hierarchies,
    output_bucket,
    output_prefix,
    local_out_root,
):
    # getting the s3 destination
    s3_key = out_s3_key_for_task(task, output_prefix)

    # local destination
    out = local_out_root / s3_key.replace("/", "_")
    out = out.with_suffix(".parquet")

    # locking mechanism
    lock = out.with_suffix(out.suffix + ".lock")

    # 1) already done
    if s3_exists(output_bucket, s3_key):
        return f"skip s3://{output_bucket}/{s3_key} (exists)"

    # 2) someone else is doing it
    if not claim_lock(lock):
        return f"skip {out.name} (already running)"

    try:
        kind, supp_level, k, l_div, t = task

        df = None

        if kind == "k":
            df = k_anonymity(data, ident, quasi_ident, k, supp_level, hierarchies)
        elif kind == "l":
            df = l_diversity(
                data, ident, quasi_ident, sens_att, k, l_div, supp_level, hierarchies
            )
        elif kind == "t":
            df = t_closeness(
                data, ident, quasi_ident, sens_att, k, t, supp_level, hierarchies
            )
        else:
            raise ValueError(
                kind, " is not a permissable kind, try using k, l, or t instead"
            )

        # getting the output
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")

        # upload and delete local to save disk space
        s3_upload(str(out), output_bucket, s3_key)
        out.unlink(missing_ok=True)

        return f"done {out.name}"

    # once function has completed, release the lock
    finally:
        release_lock(lock)


def run_one(task, data, ident, quasi_ident, sens_att, hierarchies):

    out = out_path_for_task(task)

    lock = out.with_suffix(out.suffix + ".lock")
    # try:
    #     lock.unlink()
    # except FileNotFoundError:
    #     print("this file did not need to be unlinked: ", out.name )

    # 1) already done
    if out.exists():
        return f"skip {out.name} (exists)"

    # 2) someone else is doing it
    if not claim_lock(lock):
        return f"skip {out.name} (already running)"

    try:
        kind, supp_level, k, l_div, t = task

        if kind == "k":
            df = k_anonymity(data, ident, quasi_ident, k, supp_level, hierarchies)
            df.to_parquet(out, index=False, compression="snappy")
            return f"done {out.name}"
        elif kind == "l":
            df = l_diversity(
                data, ident, quasi_ident, sens_att, k, l_div, supp_level, hierarchies
            )
            df.to_parquet(out, index=False, compression="snappy")
            return f"done {out.name}"
        elif kind == "t":
            df = t_closeness(
                data, ident, quasi_ident, sens_att, k, t, supp_level, hierarchies
            )
            df.to_parquet(out, index=False, compression="snappy")
            return f"done {out.name}"
        else:
            raise ValueError(
                kind, " is not a permissable kind, try using k, l, or t instead"
            )
    # once function has completed, release the lock
    finally:
        release_lock(lock)


def build_tasks(
    data, ident, quasi_ident, sens_att, hierarchies, output_bucket, output_prefix
):
    # build tasks
    tasks = []
    for supp_level in supp_levels:
        for k in ks:
            tasks.append(("k", supp_level, k, None, None))
        for k in ks:
            for l_div in l_divs:
                tasks.append(("l", supp_level, k, l_div, None))
            for t in ts:
                tasks.append(("t", supp_level, k, None, t))

    # NOTE IMPORTANT: keep n_jobs small if data is huge (RAM)
    # tried backend = loky, still slow and uses a ton of memory and ram, max out at 2 if run with loky

    # recommended to have 6 bc of CPU fluxuations and lower percentage use at 7, 8, 9
    local_out_root = Path("work_outputs")
    local_out_root.mkdir(exist_ok=True)

    results = Parallel(n_jobs=6, backend="threading", verbose=10)(
        delayed(s3_run_one)(
            task,
            data,
            ident,
            quasi_ident,
            sens_att,
            hierarchies,
            output_bucket,
            output_prefix,
            local_out_root,
        )
        for task in tasks
    )

    print("Finished:", sum(r.startswith("done") for r in results), "runs")

# ------------------------------------------------------------------------------
# - What running ---------------------------------------------------------------
# ------------------------------------------------------------------------------


def main():
    # configuring the s3 stuff
    input_bucket = os.environ["S3_INPUT_BUCKET"]
    input_key = os.environ["S3_INPUT_KEY"]
    hier_bucket = os.environ.get("S3_HIER_BUCKET", input_bucket)
    hier_key = os.environ["S3_HIER_KEY"]
    output_bucket = os.environ["S3_OUTPUT_BUCKET"]
    output_prefix = os.environ.get("S3_OUTPUT_PREFIX", "outputs").strip("/")

    # SOURCE: https://pypi.org/project/anjana/
    # date columns and cols to drop are noted as global variables 
    # to keep all user changes in one spot
    data = setup(
        file_path=file_path,
        date_columns=date_columns,
        cols_to_drop=cols_to_drop
    )

    # Handling the identifiers, quasi identifiers, and sensitive attributes
    non_quasi = ident + [sens_att]
    quasi_ident = list(set(data.columns) - set(non_quasi))

    # quick clean up
    # dropping nans that are in the quasi identifiers
    data = data.dropna(subset=quasi_ident)

    if not Path("data/hierarchies.pkl").exists():
        develop_hierarchies(data, quasi_ident)

    # if just want to load in the hierarchies that already exist
    hierarchies = None
    with open("data/hierarchies.pkl", "rb") as f:
        hierarchies = pickle.load(f)

    # quick bug testing to ensure everything is processing properly
    find_bad_hierarchy(hierarchies, quasi_ident)

    # ensuring the file has an index so can run consistent test cases 
    # (identify which columns are never dropped at each suppression level)
    try:
        if data["index"].empty():
            pass
    except Exception:
        data["index"] = data.index

    # running the anonymizations
    build_tasks(data, ident, quasi_ident, sens_att, hierarchies)


if __name__ == "__main__":
    main()
