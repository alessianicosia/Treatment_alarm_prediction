# ==================================================================================================
# ALARM PREDICTION CODE - TCN + TEMPORAL ATTENTION
# SINGLE RUN - RANDOM COMPLETE-TREATMENT TRAIN / VALIDATION / TEST SPLIT
# NO CROSS-VALIDATION
# ==================================================================================================
#
# Workflow:
#   1. Dataset loading and preprocessing
#   2. Treatment reconstruction from the 'time' column
#   3. Temporal sequence generation within each complete treatment
#   4. Random treatment-level Train / Validation / Test split (approximately 70 / 15 / 15)
#   5. Train-only imputation and standardization
#   6. TCN + Temporal Attention training with the original hyperparameters
#   7. Sample-level and event-level evaluation
#   8. Alarm-code-specific analysis and automatic output saving
#
# IMPORTANT:
# - A new treatment starts only when the time gap between consecutive rows is > 30 minutes.
# - Treatments are NEVER divided across Train, Validation and Test.
# - The split is random at treatment level; it is NOT required to be chronological.
# - Exact 70 / 15 / 15 percentages are not always possible because treatments are indivisible.
# - The TCN architecture, hyperparameters, training, metrics and downstream analyses are unchanged.
# ==================================================================================================

import os
import re
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import array, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# SEED, DEVICE AND GENERAL SETTINGS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------------------------------------
# Train / Validation / Test target proportions
# ------------------------------------------------------------
# These proportions are targeted at the TEMPORAL-SAMPLE level, but assignment is performed
# only at the COMPLETE-TREATMENT level. Therefore, the exact percentages may differ slightly.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Number of reproducible random treatment assignments explored. The best assignment is the one
# that most closely approximates the requested 70 / 15 / 15 temporal-sample proportions.
SPLIT_RANDOM_TRIALS = 2000

# ------------------------------------------------------------
# Treatment reconstruction rule
# ------------------------------------------------------------
# A new treatment begins whenever the gap between two consecutive values in the 'time' column
# is greater than 30 minutes. No other variable is used to define treatment boundaries.
TREATMENT_GAP_MINUTES = 30

# ORIGINAL temporal configuration of this TCN script
N_STEPS = 10
PREDICTION_GAP = 20
SAMPLING_SECONDS = 0.5

DELTA_T_SECONDS = PREDICTION_GAP * SAMPLING_SECONDS
INPUT_WINDOW_SECONDS = N_STEPS * SAMPLING_SECONDS

CSV_PATH = r"C:\Users\aless\Desktop\Alarm_paper/Definitive_FINISH_All_months_ready_for_training_.csv"
CODE_SOURCE_COLUMN = 'code'  # Nome reale della colonna nel CSV

PLOTS_DIR = "plots_tcn_attention"
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = [
    'dmparent',
    'dmchild',
    'pt4',
    'pt3',
    'delivPumpActuation',
    'pt5',
    'foamDetResult',
    'tmp',
    'diffFlow',
    'bld',
    'encoderDelivPump',
    'delFlow',
    'currentWeightLoss',
    'arterial_revolve',
    'ufPressureActuation',
    'condDO',
    'condDOnf',
    'pt6',
    'encoderUFPump',
    'Ctot',
    'infusion_revolve',
    'venous_revolve',
    'bmparent',
    'bmchild',
    'condTot',
    'condTotnf',
    'pVenous',
    'pPreFilt',
    'cond2',
    'ts1',
    'pPostInt',
    'SecondStepPumpSpeed',
    'pt8',
    'encoder1',
    'pArterial'
]


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def save_current_plot(filename, dpi=300):
    """Save current matplotlib figure and keep plt.show() behaviour."""
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(filename)).strip('_')
    if not safe_name:
        safe_name = 'plot'
    path = os.path.join(PLOTS_DIR, f"{safe_name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved: {path}")


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def convert_time_column(df, time_column='time'):
    """Convert time while preserving original ordering for equal timestamps."""
    df = df.copy()
    df['_original_order'] = np.arange(len(df))

    parsed = pd.to_datetime(
        df[time_column],
        format='%Y-%m-%d %H.%M.%S',
        errors='coerce'
    )

    if parsed.isna().any():
        parsed2 = pd.to_datetime(df.loc[parsed.isna(), time_column], errors='coerce')
        parsed.loc[parsed.isna()] = parsed2

    if parsed.isna().any():
        bad = df.loc[parsed.isna(), time_column].head(10).tolist()
        raise ValueError(
            "Some values in column 'time' cannot be converted. "
            f"Examples: {bad}"
        )

    df[time_column] = parsed
    return df


# ==================================================================================================
# TREATMENT IDENTIFICATION
# ==================================================================================================

def identify_sessions(df, time_column='time', gap_minutes=30):
    """
    Reconstruct complete treatments using only discontinuities in the time column.

    A new treatment starts whenever the time difference from the immediately preceding row
    is greater than ``gap_minutes``. With the current settings, this means a gap > 30 minutes.

    Notes
    -----
    - ``hash`` is NOT used to define the Train / Validation / Test treatment groups.
    - The original dataset order is preserved while treatment boundaries are detected.
    - ``session_id`` is intentionally retained as the column name because all existing
      downstream event-level functions use it. In this version, each ``session_id`` therefore
      corresponds exactly to one complete treatment.
    - ``treatment_id`` is stored as an explicit alias to make the split easy to audit.
    """
    df = df.copy()

    if time_column not in df.columns:
        raise ValueError("Column 'time' is not present in the dataset.")

    # Convert timestamps without changing row order.
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df = convert_time_column(df, time_column=time_column)
    elif '_original_order' not in df.columns:
        df['_original_order'] = np.arange(len(df))

    # Time difference from each row to the preceding row in the original dataset order.
    delta = df[time_column].diff()
    large_gap = delta > pd.Timedelta(minutes=gap_minutes)

    # The first row always starts treatment 1. Every subsequent gap larger than the threshold
    # starts a new complete treatment. No hash-based boundary is introduced.
    new_treatment = large_gap.copy()
    if len(new_treatment) > 0:
        new_treatment.iloc[0] = True

    df['treatment_id'] = new_treatment.cumsum().astype(int)
    df['session_id'] = df['treatment_id'].astype(int)
    df['time_gap_min'] = delta.dt.total_seconds() / 60.0

    print("\n" + "=" * 100)
    print("TREATMENT IDENTIFICATION")
    print("=" * 100)
    print(f"Treatment rule            : time gap > {gap_minutes} minutes")
    print(f"Number of treatments      : {df['treatment_id'].nunique()}")
    print("Treatment integrity rule  : one complete treatment -> one subset only")

    return df


def print_session_summary(df):
    summary = (
        df.groupby('session_id')
        .agg(
            hash=('hash', 'first'),
            start=('time', 'min'),
            end=('time', 'max'),
            n_rows=('session_id', 'size'),
            alarm_rows=('type', lambda x: int((x == 1).sum()))
        )
        .reset_index()
    )

    summary['duration_min'] = (
        (summary['end'] - summary['start']).dt.total_seconds() / 60.0
    )
    summary['treatment_hours_from_samples'] = (
        summary['n_rows'] * SAMPLING_SECONDS / 3600.0
    )

    print("\n" + "=" * 100)
    print("IDENTIFIED TREATMENTS")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\nTotal number of treatments: {len(summary)}")

    summary.to_csv('TCN_session_summary.csv', index=False)

    gap_table = df.loc[
        df['hash'].eq(df['hash'].shift(1)) & df['time_gap_min'].notna(),
        ['hash', 'time', 'time_gap_min', 'session_id']
    ].sort_values('time_gap_min', ascending=False)
    gap_table.head(200).to_csv('TCN_largest_within_hash_time_gaps.csv', index=False)

    return summary


def create_samples_from_sessions(df, session_ids, n_steps=10, prediction_gap=20):
    """
    Create sequences separately within each treatment.

    Returns X, y, groups, metadata. No temporal window can cross treatment boundaries.
    """
    X_list = []
    y_list = []
    group_list = []
    metadata_rows = []

    for session_id in session_ids:
        session_df = df[df['session_id'] == session_id].copy()

        sort_cols = []
        if 'time' in session_df.columns:
            sort_cols.append('time')
        if '_original_order' in session_df.columns:
            sort_cols.append('_original_order')
        if sort_cols:
            session_df = session_df.sort_values(sort_cols)

        session_df = session_df.reset_index(drop=True)

        if len(session_df) < prediction_gap + n_steps:
            print(
                f"WARNING: session {session_id} ignored because it contains "
                f"only {len(session_df)} rows."
            )
            continue

        historical_df = session_df.iloc[:-prediction_gap]
        alarm_type = session_df['type'].to_numpy()[prediction_gap:].reshape(-1, 1)

        stacked_features = [
            historical_df[col].to_numpy().reshape(-1, 1)
            for col in FEATURES
        ]

        dataset_session = hstack(stacked_features + [alarm_type])
        X_session, y_session = split_sequences(dataset_session, n_steps)

        if len(X_session) == 0:
            continue

        X_list.append(X_session)
        y_list.append(y_session)
        group_list.extend([session_id] * len(X_session))

        for i in range(len(X_session)):
            prediction_row_index = i + n_steps - 1
            target_row_index = prediction_row_index + prediction_gap

            metadata_rows.append({
                'treatment_id': int(session_id),
                'session_id': int(session_id),
                'sample_index_within_session': int(i),
                'prediction_row_index': int(prediction_row_index),
                'target_row_index': int(target_row_index),
                'prediction_time': session_df.loc[prediction_row_index, 'time']
                    if 'time' in session_df.columns else pd.NaT,
                'target_time': session_df.loc[target_row_index, 'time']
                    if 'time' in session_df.columns else pd.NaT,
                'target_code': session_df.loc[target_row_index, CODE_SOURCE_COLUMN]
                    if CODE_SOURCE_COLUMN in session_df.columns else np.nan,
            })

    if not X_list:
        raise ValueError("No sequence was created.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(int)
    groups = np.asarray(group_list)
    metadata = pd.DataFrame(metadata_rows)

    if not (len(X) == len(y) == len(groups) == len(metadata)):
        raise RuntimeError("Inconsistency among X, y, groups and metadata lengths.")

    return X, y, groups, metadata


# ==================================================================================================
# RANDOM COMPLETE-TREATMENT SPLIT
# ==================================================================================================

def treatment_based_split_indices(
    groups,
    y=None,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    n_random_trials=2000
):
    """
    Randomly assign COMPLETE treatments to Train / Validation / Test.

    Individual temporal samples are never assigned independently. All samples generated from
    one treatment always remain together in exactly one subset.

    Strategy
    --------
    1. Count the valid temporal sequences generated by each treatment.
    2. Randomly shuffle treatment IDs with a reproducible random seed.
    3. For each shuffled order, choose treatment boundaries that approximate 70 / 15 / 15
       as closely as possible in terms of the number of generated temporal samples.
    4. Repeat the random assignment ``n_random_trials`` times and retain the best candidate.
    5. When labels are supplied, prefer candidates containing both classes in Train,
       Validation and Test whenever this is possible.

    Important
    ---------
    - The split is RANDOM at treatment level and is NOT chronological.
    - Exact 70 / 15 / 15 proportions are not guaranteed because treatments are indivisible.
    - Treatment integrity is checked explicitly before returning the sample indices.
    """

    # ----------------------------------------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------------------------------------
    groups = np.asarray(groups).astype(int)
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)

    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be equal to 1.")

    if n_random_trials < 1:
        raise ValueError("n_random_trials must be at least 1.")

    if y is not None:
        y = np.asarray(y).astype(int).ravel()
        if len(y) != len(groups):
            raise ValueError("y and groups must contain the same number of samples.")

    # ----------------------------------------------------------------------------------------------
    # Treatment-level statistics
    # ----------------------------------------------------------------------------------------------
    treatment_ids, counts = np.unique(groups, return_counts=True)
    treatment_ids = treatment_ids.astype(int)
    counts = counts.astype(int)
    n_treatments = len(treatment_ids)

    if n_treatments < 3:
        raise ValueError(
            "At least 3 treatments containing valid temporal sequences are required to create "
            "Train, Validation and Test without splitting treatments."
        )

    total_samples = int(counts.sum())
    target_counts = ratios * total_samples

    # Optional class counts per treatment. These are used only to prefer assignments in which
    # all three subsets contain both class 0 and class 1.
    treatment_class_counts = None
    if y is not None:
        tmp = pd.DataFrame({
            'treatment_id': groups,
            'target': y,
        })
        class_table = pd.crosstab(tmp['treatment_id'], tmp['target'])
        class_table = class_table.reindex(index=treatment_ids, columns=[0, 1], fill_value=0)
        treatment_class_counts = class_table.to_numpy(dtype=int)

    # ----------------------------------------------------------------------------------------------
    # Reproducible random search for the best complete-treatment assignment
    # ----------------------------------------------------------------------------------------------
    rng = np.random.RandomState(random_state)

    best_valid = None       # Best ratio match with both classes in every subset.
    best_valid_score = np.inf
    best_any = None         # Fallback: best ratio match regardless of class composition.
    best_any_score = np.inf

    for _ in range(n_random_trials):
        # Only treatment IDs are shuffled. Samples belonging to a treatment remain inseparable.
        permutation = rng.permutation(n_treatments)
        ids_perm = treatment_ids[permutation]
        counts_perm = counts[permutation]
        cumulative_counts = np.cumsum(counts_perm)

        # ------------------------------ Train boundary ------------------------------
        # At least one complete treatment is left for Validation and one for Test.
        train_cut_candidates = np.arange(1, n_treatments - 1)
        train_cut = int(
            train_cut_candidates[
                np.argmin(
                    np.abs(
                        cumulative_counts[train_cut_candidates - 1] - target_counts[0]
                    )
                )
            ]
        )

        # --------------------------- Validation/Test boundary -----------------------
        # Target cumulative Train + Validation proportion = approximately 85%.
        second_cut_candidates = np.arange(train_cut + 1, n_treatments)
        train_val_target = target_counts[0] + target_counts[1]
        second_cut = int(
            second_cut_candidates[
                np.argmin(
                    np.abs(
                        cumulative_counts[second_cut_candidates - 1] - train_val_target
                    )
                )
            ]
        )

        train_treatments = ids_perm[:train_cut]
        val_treatments = ids_perm[train_cut:second_cut]
        test_treatments = ids_perm[second_cut:]

        # Actual sequence counts resulting from this complete-treatment assignment.
        train_n = int(counts_perm[:train_cut].sum())
        val_n = int(counts_perm[train_cut:second_cut].sum())
        test_n = int(counts_perm[second_cut:].sum())
        actual_counts = np.array([train_n, val_n, test_n], dtype=float)

        # Total normalized deviation from the requested 70 / 15 / 15 sequence proportions.
        ratio_score = float(np.sum(np.abs(actual_counts - target_counts)) / total_samples)

        candidate = (
            train_treatments.copy(),
            val_treatments.copy(),
            test_treatments.copy(),
        )

        if ratio_score < best_any_score:
            best_any_score = ratio_score
            best_any = candidate

        # Prefer assignments where every subset contains both classes.
        if treatment_class_counts is not None:
            cc_perm = treatment_class_counts[permutation]
            class_counts = np.vstack([
                cc_perm[:train_cut].sum(axis=0),
                cc_perm[train_cut:second_cut].sum(axis=0),
                cc_perm[second_cut:].sum(axis=0),
            ])
            all_subsets_have_both_classes = bool(np.all(class_counts > 0))
        else:
            all_subsets_have_both_classes = True

        if all_subsets_have_both_classes and ratio_score < best_valid_score:
            best_valid_score = ratio_score
            best_valid = candidate

    # Prefer a split containing both classes in all subsets. If no such split is found, use the
    # complete-treatment assignment that is simply closest to the requested 70 / 15 / 15 ratio.
    selected = best_valid if best_valid is not None else best_any
    if selected is None:
        raise RuntimeError("Unable to generate a treatment-based Train/Validation/Test split.")

    train_treatments, val_treatments, test_treatments = selected

    # ----------------------------------------------------------------------------------------------
    # Convert treatment assignments back to temporal-sample indices
    # ----------------------------------------------------------------------------------------------
    idx_train = np.flatnonzero(np.isin(groups, train_treatments))
    idx_val = np.flatnonzero(np.isin(groups, val_treatments))
    idx_test = np.flatnonzero(np.isin(groups, test_treatments))

    # ----------------------------------------------------------------------------------------------
    # Safety checks: treatment integrity and complete sample coverage
    # ----------------------------------------------------------------------------------------------
    train_set = set(train_treatments.tolist())
    val_set = set(val_treatments.tolist())
    test_set = set(test_treatments.tolist())

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError(
            "Split integrity error: at least one treatment appears in more than one subset."
        )

    assigned_indices = np.concatenate([idx_train, idx_val, idx_test])
    if len(assigned_indices) != len(groups) or len(np.unique(assigned_indices)) != len(groups):
        raise RuntimeError(
            "Split integrity error: samples are missing or duplicated across subsets."
        )

    return (
        idx_train,
        idx_val,
        idx_test,
        train_treatments,
        val_treatments,
        test_treatments,
    )


def check_classes(name, y):
    values, counts = np.unique(y, return_counts=True)
    print(f"\n{name}")
    print(f"Number of samples: {len(y)}")
    for value, count in zip(values, counts):
        print(f"  Class {value}: {count} ({100.0 * count / len(y):.2f}%)")
    if len(values) < 2:
        raise ValueError(f"{name} contains a single class.")


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

df = pd.read_csv(CSV_PATH)
df = df.rename(columns={'dT(CP)': 'dT'})
df['type'] = df['type'].replace({'alarm': 1, 'normal': 0, 'override': 0})

df = df.dropna(subset=['type']).copy()
df['type'] = pd.to_numeric(df['type'], errors='coerce')
df = df.dropna(subset=['type']).copy()
df['type'] = df['type'].astype(int)

missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(
        "The CSV is missing the following features:\n" + "\n".join(missing_features)
    )

for required_col in ['hash', 'time', CODE_SOURCE_COLUMN]:
    if required_col not in df.columns:
        raise ValueError(f"Required column '{required_col}' is not present in the CSV.")

df = convert_time_column(df, time_column='time')
df = identify_sessions(df, time_column='time', gap_minutes=TREATMENT_GAP_MINUTES)

if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
    sort_cols = ['time']
    if '_original_order' in df.columns:
        sort_cols.append('_original_order')
    df = df.sort_values(sort_cols).reset_index(drop=True)

session_summary = print_session_summary(df)


# ==================================================================================================
# CREATE TEMPORAL SAMPLES WITHIN EACH COMPLETE TREATMENT
# ==================================================================================================
#
# IMPORTANT:
# Temporal samples are created separately inside each treatment BEFORE the split. This guarantees
# that neither the N_STEPS input window nor the PREDICTION_GAP target can cross a treatment boundary.
# The resulting complete treatments are then randomized as indivisible groups across the three sets.

all_sessions = np.asarray(sorted(df['session_id'].unique()))

X_all, y_all, groups_all, meta_all = create_samples_from_sessions(
    df,
    all_sessions,
    n_steps=N_STEPS,
    prediction_gap=PREDICTION_GAP
)

y_all = y_all.astype(int).ravel()

check_classes("ALL TEMPORAL SAMPLES", y_all)

print("\nRaw complete-dataset shapes:")
print("X_all:", X_all.shape)
print("y_all:", y_all.shape)


# ==================================================================================================
# TRAIN / VALIDATION / TEST SPLIT - RANDOMIZED AT COMPLETE-TREATMENT LEVEL
# ==================================================================================================
# Fundamental rule:
#     one complete treatment -> one subset only
#
# Treatment IDs are randomized reproducibly using SEED. Several random assignments are explored,
# and the assignment closest to 70 / 15 / 15 in terms of temporal-sample counts is retained.
# The split is therefore RANDOM but NOT chronological, and no treatment can be shared between sets.
(
    idx_train,
    idx_val,
    idx_test,
    train_treatments,
    val_treatments,
    test_treatments,
) = treatment_based_split_indices(
    groups=groups_all,
    y=y_all,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    random_state=SEED,
    n_random_trials=SPLIT_RANDOM_TRIALS,
)

# ------------------------------------------------------------
# Build Train / Validation / Test arrays and aligned metadata
# ------------------------------------------------------------
X_train = X_all[idx_train]
y_train = y_all[idx_train]
groups_train = groups_all[idx_train]
meta_train = meta_all.iloc[idx_train].reset_index(drop=True)

X_val = X_all[idx_val]
y_val = y_all[idx_val]
groups_val = groups_all[idx_val]
meta_val = meta_all.iloc[idx_val].reset_index(drop=True)

X_test = X_all[idx_test]
y_test = y_all[idx_test]
groups_test = groups_all[idx_test]
meta_test = meta_all.iloc[idx_test].reset_index(drop=True)

# Existing downstream event-level functions use the variable names train_sessions, val_sessions
# and test_sessions. They are intentionally preserved; each now contains mutually exclusive
# complete treatment IDs.
train_sessions = np.sort(train_treatments)
val_sessions = np.sort(val_treatments)
test_sessions = np.sort(test_treatments)

# ------------------------------------------------------------
# Verify class composition
# ------------------------------------------------------------
check_classes("TRAIN", y_train)
check_classes("VALIDATION", y_val)
check_classes("TEST", y_test)

# ------------------------------------------------------------
# Split summary and treatment-integrity information
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("TRAIN / VALIDATION / TEST SUMMARY - COMPLETE TREATMENTS")
print("=" * 100)
print(
    f"TRAIN      : {len(y_train)} samples ({100.0 * len(y_train) / len(y_all):.2f}%) | "
    f"{len(train_treatments)} treatments"
)
print(
    f"VALIDATION : {len(y_val)} samples ({100.0 * len(y_val) / len(y_all):.2f}%) | "
    f"{len(val_treatments)} treatments"
)
print(
    f"TEST       : {len(y_test)} samples ({100.0 * len(y_test) / len(y_all):.2f}%) | "
    f"{len(test_treatments)} treatments"
)
print("Random treatment-level partition completed successfully.")
print("No treatment is shared across Train, Validation and Test.")

# Treatment IDs are printed for reproducibility and easy manual verification.
print(f"Train treatment IDs      : {sorted(train_treatments.tolist())}")
print(f"Validation treatment IDs : {sorted(val_treatments.tolist())}")
print(f"Test treatment IDs       : {sorted(test_treatments.tolist())}")

print("\nRaw shapes:")
print("TRAIN:", X_train.shape, y_train.shape)
print("VALIDATION:", X_val.shape, y_val.shape)
print("TEST:", X_test.shape, y_test.shape)

# ------------------------------------------------------------
# Save the split assignment for reproducibility
# ------------------------------------------------------------
# The historical output filename is retained so that the rest of the workflow remains unchanged.
# Its content now reflects a random COMPLETE-TREATMENT split rather than a sample-level split.
split_rows = []
for subset_name, subset_indices in [
    ('train', idx_train),
    ('validation', idx_val),
    ('test', idx_test),
]:
    temp = meta_all.iloc[subset_indices].copy().reset_index(drop=True)
    temp.insert(0, 'original_sample_index', subset_indices)
    temp.insert(1, 'subset', subset_name)
    split_rows.append(temp)

pd.concat(split_rows, ignore_index=True).to_csv(
    'TCN_random_sample_split.csv',
    index=False
)


# ============================================================
# STANDARDIZATION BY FEATURE USING TRAINING SET ONLY
# ============================================================
#
# TCN USES THE SAME 35 FEATURES AS THE RANDOM FOREST.
# NaN / +Inf / -Inf values, if present, are replaced using
# TRAIN-ONLY medians. Validation and Test never contribute
# to the imputation values.

n_features = X_train.shape[2]

if n_features != len(FEATURES):
    raise RuntimeError(
        f"Unexpected number of TCN features: X has {n_features}, "
        f"FEATURES contains {len(FEATURES)}."
    )


def report_non_finite_by_feature(X, subset_name):
    """Report NaN and Inf values feature by feature."""
    print("\n" + "-" * 100)
    print(f"NON-FINITE INPUT CHECK - {subset_name}")
    print("-" * 100)

    total_bad = 0
    rows = []

    for j, feature_name in enumerate(FEATURES):
        values = X[:, :, j]

        nan_count = int(np.isnan(values).sum())
        posinf_count = int(np.isposinf(values).sum())
        neginf_count = int(np.isneginf(values).sum())
        bad_count = nan_count + posinf_count + neginf_count
        total_bad += bad_count

        rows.append({
            'Subset': subset_name,
            'Feature': feature_name,
            'NaN': nan_count,
            'PosInf': posinf_count,
            'NegInf': neginf_count,
            'TotalNonFinite': bad_count
        })

        if bad_count > 0:
            print(
                f"{feature_name}: NaN={nan_count} | "
                f"+Inf={posinf_count} | -Inf={neginf_count} | "
                f"TOTAL={bad_count}"
            )

    if total_bad == 0:
        print("OK: no NaN or Inf values.")
    else:
        print(f"TOTAL non-finite values in {subset_name}: {total_bad}")

    return pd.DataFrame(rows)


nonfinite_train_df = report_non_finite_by_feature(X_train, "TRAIN")
nonfinite_val_df = report_non_finite_by_feature(X_val, "VALIDATION")
nonfinite_test_df = report_non_finite_by_feature(X_test, "TEST")

pd.concat(
    [nonfinite_train_df, nonfinite_val_df, nonfinite_test_df],
    ignore_index=True
).to_csv("TCN_nonfinite_input_diagnostic.csv", index=False)


# ------------------------------------------------------------
# TRAIN-ONLY MEDIAN IMPUTATION
# ------------------------------------------------------------
# One median is computed for each of the 35 RF-aligned TCN features.
# The same TRAIN-derived median is then used in Train, Validation and Test.

train_feature_medians = {}

for j, feature_name in enumerate(FEATURES):
    train_values = X_train[:, :, j]
    finite_train_values = train_values[np.isfinite(train_values)]

    if finite_train_values.size == 0:
        raise ValueError(
            f"Feature '{feature_name}' has no finite values in TRAIN."
        )

    median_value = float(np.median(finite_train_values))
    train_feature_medians[feature_name] = median_value

    for X_subset in (X_train, X_val, X_test):
        current = X_subset[:, :, j]
        bad_mask = ~np.isfinite(current)

        if bad_mask.any():
            current[bad_mask] = median_value


pd.DataFrame({
    'Feature': list(train_feature_medians.keys()),
    'TrainMedianUsedForImputation': list(train_feature_medians.values())
}).to_csv("TCN_train_feature_medians_for_imputation.csv", index=False)


# Hard check before scaling.
if not np.isfinite(X_train).all():
    raise RuntimeError("TRAIN still contains NaN/Inf after imputation.")
if not np.isfinite(X_val).all():
    raise RuntimeError("VALIDATION still contains NaN/Inf after imputation.")
if not np.isfinite(X_test).all():
    raise RuntimeError("TEST still contains NaN/Inf after imputation.")

print("\nOK: Train / Validation / Test contain only finite values before scaling.")


# ------------------------------------------------------------
# ORIGINAL STANDARDIZATION LOGIC - FIT ONLY ON TRAIN
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat = X_val.reshape((X_val.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

scaler.fit(X_train_flat)

X_train_std = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val_std = scaler.transform(X_val_flat).reshape(X_val.shape)
X_test_std = scaler.transform(X_test_flat).reshape(X_test.shape)


# Final hard check before sending arrays to PyTorch.
if not np.isfinite(X_train_std).all():
    raise RuntimeError("TRAIN contains NaN/Inf after StandardScaler.")
if not np.isfinite(X_val_std).all():
    raise RuntimeError("VALIDATION contains NaN/Inf after StandardScaler.")
if not np.isfinite(X_test_std).all():
    raise RuntimeError("TEST contains NaN/Inf after StandardScaler.")

print("OK: standardized Train / Validation / Test contain only finite values.")


# ============================================================
# TORCH DATASET
# ============================================================

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        # Conv1d expects (B, C, T), therefore transpose to (features, steps)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = SeqDataset(X_train_std, y_train)
val_ds = SeqDataset(X_val_std, y_val)
test_ds = SeqDataset(X_test_std, y_test)

BATCH_SIZE = 256

# Reproducible shuffle for training only
generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    generator=generator
)

# Evaluation loader preserves order, required for session/event-level analysis
train_eval_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)


# ============================================================
# MODEL: TCN + TEMPORAL ATTENTION
# ============================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.act1 = nn.PReLU()
        self.do1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.act2 = nn.PReLU()
        self.do2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act_out = nn.PReLU()

        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")
            nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.do1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.do2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.act_out(out + res)


class TemporalAttention(nn.Module):
    def __init__(self, in_ch, attn_dim=64):
        super().__init__()
        self.W = nn.Linear(in_ch, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        h_t = h.transpose(1, 2)
        score = torch.tanh(self.W(h_t))
        e = self.v(score).squeeze(-1)
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h_t).squeeze(1)
        return context, alpha


class TCNWithAttention(nn.Module):
    def __init__(self, in_ch, num_classes=2, channels=(64, 64, 128), kernel_size=3, drop=0.1):
        super().__init__()
        layers = []
        prev = in_ch
        dil = 1
        for ch in channels:
            layers.append(
                TemporalBlock(prev, ch, kernel_size=kernel_size, dilation=dil, dropout=drop)
            )
            prev = ch
            dil *= 2
        self.tcn = nn.Sequential(*layers)
        self.attn = TemporalAttention(in_ch=prev, attn_dim=64)
        self.bn = nn.BatchNorm1d(prev)
        self.head = nn.Linear(prev, num_classes)

        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.tcn(x)
        h = self.bn(h)
        ctx, alpha = self.attn(h)
        logits = self.head(ctx)
        return logits, alpha


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(y):
    cnt = Counter(y)
    classes = sorted(cnt.keys())
    total = sum(cnt.values())
    weights = {c: total / (len(classes) * cnt[c]) for c in classes}
    return torch.tensor([weights[c] for c in classes], dtype=torch.float32)


# ============================================================
# TRAIN / EVALUATION FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []

    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)

        # Do not silently continue if numerical instability appears.
        if not torch.isfinite(loss):
            finite_inputs = bool(torch.isfinite(xb).all().item())
            finite_logits = bool(torch.isfinite(logits).all().item())

            raise RuntimeError(
                f"Non-finite loss at training batch {batch_idx}. "
                f"Finite inputs={finite_inputs}, "
                f"finite logits={finite_logits}, "
                f"loss={loss.detach().cpu().item()}."
            )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_outputs(model, loader):
    """Return ordered y_true, y_pred and positive-class probabilities."""
    model.eval()
    all_y = []
    all_pred = []
    all_score = []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits, _ = model(xb)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        all_y.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_score.append(probs[:, 1].cpu().numpy())

    return (
        np.concatenate(all_y).astype(int),
        np.concatenate(all_pred).astype(int),
        np.concatenate(all_score).astype(float),
    )


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else np.nan


def count_false_alarm_episodes(y_true, y_pred, groups):
    """Count contiguous false-positive episodes within each treatment."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)

    total_episodes = 0
    for session_id in np.unique(groups):
        idx = np.where(groups == session_id)[0]
        fp_mask = (y_true[idx] == 0) & (y_pred[idx] == 1)
        if len(fp_mask) == 0:
            continue
        starts = fp_mask & np.r_[True, ~fp_mask[:-1]]
        total_episodes += int(starts.sum())

    return total_episodes


def treatment_hours(df, session_ids):
    """Sum treatment duration for the selected complete sessions."""
    if 'time' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['time']):
        n_rows = int(df['session_id'].isin(session_ids).sum())
        return (n_rows * SAMPLING_SECONDS) / 3600.0

    total_seconds = 0.0
    for session_id in session_ids:
        s = df.loc[df['session_id'] == session_id, 'time'].dropna()
        if len(s) == 0:
            continue
        duration = (s.max() - s.min()).total_seconds() + SAMPLING_SECONDS
        total_seconds += max(duration, 0.0)

    return total_seconds / 3600.0


def compute_metrics(y_true, y_pred, score, groups, df, session_ids):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = specificity_score(y_true, y_pred)

    if len(np.unique(y_true)) >= 2:
        auroc = roc_auc_score(y_true, score)
        auprc = average_precision_score(y_true, score)
    else:
        auroc = np.nan
        auprc = np.nan

    false_alarm_episodes = count_false_alarm_episodes(y_true, y_pred, groups)
    hours = treatment_hours(df, session_ids)
    false_alarms_per_hour = false_alarm_episodes / hours if hours > 0 else np.nan

    return {
        'Accuracy': float(accuracy),
        'F1': float(f1),
        'Precision': float(precision),
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'AUROC': float(auroc) if not pd.isna(auroc) else np.nan,
        'AUPRC': float(auprc) if not pd.isna(auprc) else np.nan,
        'FalseAlarmEpisodes': int(false_alarm_episodes),
        'TreatmentHours': float(hours),
        'FalseAlarmsPerHour': float(false_alarms_per_hour)
            if not pd.isna(false_alarms_per_hour) else np.nan,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
    }


def print_metrics(prefix, metrics):
    """Print ALL sample-level metrics as numeric values, tuning-style."""
    print(
        f"  - {prefix} | "
        f"Acc={metrics['Accuracy']:.4f} | "
        f"F1={metrics['F1']:.4f} | "
        f"P={metrics['Precision']:.4f} | "
        f"Sens={metrics['Sensitivity']:.4f} | "
        f"Spec={metrics['Specificity']:.4f} | "
        f"AUROC={metrics['AUROC']:.4f} | "
        f"AUPRC={metrics['AUPRC']:.4f} | "
        f"FA/h={metrics['FalseAlarmsPerHour']:.4f} | "
        f"FA episodes={metrics['FalseAlarmEpisodes']} | "
        f"Treatment hours={metrics['TreatmentHours']:.4f} | "
        f"TN={metrics['TN']} | FP={metrics['FP']} | "
        f"FN={metrics['FN']} | TP={metrics['TP']}"
    )


# ============================================================
# EVENT-LEVEL CLINICAL METRICS
# ============================================================

def _contiguous_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []

    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return list(zip(starts.tolist(), ends.tolist()))


def _clean_code(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == '' or value.lower() in {'nan', 'none'}:
        return np.nan
    return value


def extract_true_alarm_events(df, session_ids):
    """One true alarm episode = one contiguous run of type == 1 within a treatment."""
    rows = []

    for session_id in session_ids:
        s = df[df['session_id'] == session_id].copy()
        sort_cols = []
        if 'time' in s.columns:
            sort_cols.append('time')
        if '_original_order' in s.columns:
            sort_cols.append('_original_order')
        if sort_cols:
            s = s.sort_values(sort_cols)
        s = s.reset_index(drop=True)

        runs = _contiguous_true_runs(s['type'].to_numpy() == 1)

        for event_number, (start_idx, end_idx) in enumerate(runs, start=1):
            event_codes = [
                _clean_code(v)
                for v in s.loc[start_idx:end_idx, CODE_SOURCE_COLUMN].tolist()
            ]
            event_codes = [v for v in event_codes if not pd.isna(v)]
            unique_codes = list(dict.fromkeys(event_codes))

            # Primary code: first valid code occurring in the alarm episode.
            code = unique_codes[0] if unique_codes else 'UNKNOWN'
            all_codes = ' | '.join(unique_codes) if unique_codes else 'UNKNOWN'

            rows.append({
                'session_id': int(session_id),
                'event_number_within_session': int(event_number),
                'event_start_row': int(start_idx),
                'event_end_row': int(end_idx),
                'event_duration_rows': int(end_idx - start_idx + 1),
                'event_duration_seconds': float(
                    (end_idx - start_idx + 1) * SAMPLING_SECONDS
                ),
                'event_start_time': s.loc[start_idx, 'time']
                    if 'time' in s.columns else pd.NaT,
                'event_end_time': s.loc[end_idx, 'time']
                    if 'time' in s.columns else pd.NaT,
                'code': code,
                'codes_in_event': all_codes,
            })

    return pd.DataFrame(rows)


def extract_predicted_warning_episodes(pred, metadata):
    """One predicted warning episode = one contiguous run of predicted class 1."""
    pred = np.asarray(pred).astype(int)
    metadata = metadata.reset_index(drop=True).copy()

    if len(pred) != len(metadata):
        raise ValueError("pred and metadata must have the same length.")

    rows = []

    for session_id in sorted(metadata['session_id'].unique()):
        idx = np.flatnonzero(metadata['session_id'].to_numpy() == session_id)
        m = metadata.iloc[idx].reset_index(drop=True)
        p = pred[idx]

        runs = _contiguous_true_runs(p == 1)

        for warning_number, (start_local, end_local) in enumerate(runs, start=1):
            rows.append({
                'session_id': int(session_id),
                'warning_number_within_session': int(warning_number),
                'warning_start_prediction_row': int(
                    m.loc[start_local, 'prediction_row_index']
                ),
                'warning_end_prediction_row': int(
                    m.loc[end_local, 'prediction_row_index']
                ),
                'warning_start_time': m.loc[start_local, 'prediction_time'],
                'warning_end_time': m.loc[end_local, 'prediction_time'],
                'warning_duration_samples': int(end_local - start_local + 1),
                'warning_duration_seconds': float(
                    (end_local - start_local + 1) * SAMPLING_SECONDS
                ),
            })

    return pd.DataFrame(rows)


def evaluate_event_level(
    pred,
    metadata,
    df,
    session_ids,
    prediction_gap_rows=PREDICTION_GAP,
    sampling_seconds=SAMPLING_SECONDS
):
    """
    Successful detection: at least one predicted positive before event onset,
    within the nominal prediction horizon.
    """
    pred = np.asarray(pred).astype(int)
    metadata = metadata.reset_index(drop=True).copy()

    if len(pred) != len(metadata):
        raise ValueError("pred and metadata must have the same length.")

    true_events = extract_true_alarm_events(df, session_ids)
    warning_events = extract_predicted_warning_episodes(pred, metadata)

    event_detail_rows = []
    matched_warning_keys = set()

    for _, event in true_events.iterrows():
        sid = int(event['session_id'])
        start_row = int(event['event_start_row'])

        lower = start_row - int(prediction_gap_rows)
        upper = start_row - 1

        m_session = metadata[metadata['session_id'] == sid].copy()
        p_session = pred[m_session.index.to_numpy()]

        prediction_rows = m_session['prediction_row_index'].to_numpy(dtype=int)
        eligible_mask = (prediction_rows >= lower) & (prediction_rows <= upper)

        evaluable = bool(eligible_mask.any())
        detected = False
        earliest_prediction_row = np.nan
        lead_time_seconds = np.nan
        earliest_prediction_time = pd.NaT

        if evaluable:
            positive_mask = eligible_mask & (p_session == 1)

            if positive_mask.any():
                detected = True
                positive_rows = prediction_rows[positive_mask]
                earliest_prediction_row = int(positive_rows.min())
                lead_time_seconds = float(
                    (start_row - earliest_prediction_row) * sampling_seconds
                )

                pos_candidates = m_session.loc[
                    positive_mask,
                    ['prediction_row_index', 'prediction_time']
                ].sort_values('prediction_row_index')
                earliest_prediction_time = pos_candidates.iloc[0]['prediction_time']

        event_detail_rows.append({
            **event.to_dict(),
            'evaluable': evaluable,
            'detected_pre_alarm': detected,
            'nominal_prediction_horizon_seconds': float(
                prediction_gap_rows * sampling_seconds
            ),
            'earliest_warning_prediction_row': earliest_prediction_row,
            'earliest_warning_time': earliest_prediction_time,
            'lead_time_seconds': lead_time_seconds,
        })

        if not warning_events.empty and evaluable:
            w = warning_events[warning_events['session_id'] == sid]
            for widx, wr in w.iterrows():
                wstart = int(wr['warning_start_prediction_row'])
                wend = int(wr['warning_end_prediction_row'])
                overlaps = (wend >= lower) and (wstart <= upper)
                if overlaps:
                    matched_warning_keys.add(int(widx))

    event_details = pd.DataFrame(event_detail_rows)

    if warning_events.empty:
        warning_events = pd.DataFrame(columns=[
            'session_id', 'warning_number_within_session',
            'warning_start_prediction_row', 'warning_end_prediction_row',
            'warning_start_time', 'warning_end_time',
            'warning_duration_samples', 'warning_duration_seconds'
        ])

    warning_events = warning_events.copy()
    warning_events['matched_to_true_alarm'] = False
    if len(warning_events) > 0 and matched_warning_keys:
        warning_events.loc[
            warning_events.index.isin(matched_warning_keys),
            'matched_to_true_alarm'
        ] = True

    false_warning_events = warning_events[
        ~warning_events['matched_to_true_alarm']
    ].copy()

    n_total = len(event_details)
    if n_total > 0:
        evaluable_mask = event_details['evaluable'].astype(bool)
        detected_mask = evaluable_mask & event_details['detected_pre_alarm'].astype(bool)
        n_evaluable = int(evaluable_mask.sum())
        n_detected = int(detected_mask.sum())
    else:
        n_evaluable = 0
        n_detected = 0

    detection_rate = n_detected / n_evaluable if n_evaluable > 0 else np.nan

    lead_times = (
        event_details.loc[
            event_details['detected_pre_alarm'] == True,
            'lead_time_seconds'
        ].dropna().to_numpy(dtype=float)
        if len(event_details) > 0 else np.asarray([], dtype=float)
    )

    hours = treatment_hours(df, session_ids)
    n_sessions = len(np.unique(session_ids))
    n_false_warnings = len(false_warning_events)

    summary = {
        'AlarmEpisodesTotal': int(n_total),
        'AlarmEpisodesEvaluable': int(n_evaluable),
        'AlarmEpisodesDetected': int(n_detected),
        'AlarmEpisodesMissed': int(max(n_evaluable - n_detected, 0)),
        'AlarmEpisodeDetectionRate': float(detection_rate)
            if not pd.isna(detection_rate) else np.nan,
        'AlarmEpisodeDetectionPercent': float(100.0 * detection_rate)
            if not pd.isna(detection_rate) else np.nan,
        'PredictedWarningEpisodes': int(len(warning_events)),
        'FalseWarningEpisodes': int(n_false_warnings),
        'FalseWarningsPerSession': float(n_false_warnings / n_sessions)
            if n_sessions > 0 else np.nan,
        'FalseWarningsPerTreatmentHour': float(n_false_warnings / hours)
            if hours > 0 else np.nan,
        'LeadTimeN': int(len(lead_times)),
        'LeadTimeMeanSeconds': float(np.mean(lead_times)) if len(lead_times) else np.nan,
        'LeadTimeMedianSeconds': float(np.median(lead_times)) if len(lead_times) else np.nan,
        'LeadTimeStdSeconds': float(np.std(lead_times)) if len(lead_times) else np.nan,
        'LeadTimeMinSeconds': float(np.min(lead_times)) if len(lead_times) else np.nan,
        'LeadTimeQ25Seconds': float(np.percentile(lead_times, 25)) if len(lead_times) else np.nan,
        'LeadTimeQ75Seconds': float(np.percentile(lead_times, 75)) if len(lead_times) else np.nan,
        'LeadTimeMaxSeconds': float(np.max(lead_times)) if len(lead_times) else np.nan,
    }

    return summary, event_details, warning_events, false_warning_events


def print_event_metrics(prefix, metrics):
    """Print ALL event-level metrics as numeric values."""
    print(
        f"  - {prefix} EVENT | "
        f"Total={metrics['AlarmEpisodesTotal']} | "
        f"Evaluable={metrics['AlarmEpisodesEvaluable']} | "
        f"Detected={metrics['AlarmEpisodesDetected']} | "
        f"Missed={metrics['AlarmEpisodesMissed']} | "
        f"DetectionRate={metrics['AlarmEpisodeDetectionRate']:.4f} | "
        f"Detection%={metrics['AlarmEpisodeDetectionPercent']:.2f} | "
        f"PredictedWarnings={metrics['PredictedWarningEpisodes']} | "
        f"FalseWarnings={metrics['FalseWarningEpisodes']} | "
        f"FalseWarnings/session={metrics['FalseWarningsPerSession']:.4f} | "
        f"FalseWarnings/h={metrics['FalseWarningsPerTreatmentHour']:.4f} | "
        f"LeadN={metrics['LeadTimeN']} | "
        f"LeadMean={metrics['LeadTimeMeanSeconds']:.2f}s | "
        f"LeadMedian={metrics['LeadTimeMedianSeconds']:.2f}s | "
        f"LeadStd={metrics['LeadTimeStdSeconds']:.2f}s | "
        f"LeadMin={metrics['LeadTimeMinSeconds']:.2f}s | "
        f"LeadQ25={metrics['LeadTimeQ25Seconds']:.2f}s | "
        f"LeadQ75={metrics['LeadTimeQ75Seconds']:.2f}s | "
        f"LeadMax={metrics['LeadTimeMaxSeconds']:.2f}s"
    )


def code_summary(event_details, subset_label):
    """Summarize event-level detection by alarm code (source CSV column: code)."""
    columns = [
        'Code',
        f'Episodes in {subset_label}',
        'Detected episodes',
        'Detection rate (%)',
        'Median lead time (s)',
    ]

    if event_details.empty:
        return pd.DataFrame(columns=columns)

    evaluable = event_details[event_details['evaluable'] == True].copy()
    if evaluable.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for code, g in evaluable.groupby('code', dropna=False):
        detected = g[g['detected_pre_alarm'] == True]
        n_total = len(g)
        n_detected = len(detected)
        rate = 100.0 * n_detected / n_total if n_total > 0 else np.nan
        median_lead = detected['lead_time_seconds'].dropna().median()

        rows.append({
            'Code': code,
            f'Episodes in {subset_label}': int(n_total),
            'Detected episodes': int(n_detected),
            'Detection rate (%)': float(rate) if not pd.isna(rate) else np.nan,
            'Median lead time (s)': float(median_lead)
                if not pd.isna(median_lead) else np.nan,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        [f'Episodes in {subset_label}', 'Code'],
        ascending=[False, True]
    ).reset_index(drop=True)
    return out


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    save_current_plot(title)
    plt.show()


def plot_lead_time_distribution(event_details, title):
    if event_details.empty or 'lead_time_seconds' not in event_details.columns:
        return

    values = event_details.loc[
        event_details['detected_pre_alarm'] == True,
        'lead_time_seconds'
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins='auto', edgecolor='black')
    plt.xlabel('Prediction lead time (s)')
    plt.ylabel('Number of detected alarm episodes')
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot(title)
    plt.show()


def plot_metric_comparison(metrics_by_subset):
    metric_names = [
        'Accuracy', 'F1', 'Precision', 'Sensitivity',
        'Specificity', 'AUROC', 'AUPRC'
    ]
    plot_df = pd.DataFrame({
        subset: [metrics_by_subset[subset][m] for m in metric_names]
        for subset in metrics_by_subset
    }, index=metric_names)

    ax = plot_df.plot(kind='bar', figsize=(12, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('TCN+Attention - Train / Validation / Test metrics')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('TCN_Attention_Train_Validation_Test_metrics')
    plt.show()


def plot_false_alarms_per_hour(metrics_by_subset):
    labels = list(metrics_by_subset.keys())
    values = [metrics_by_subset[s]['FalseAlarmsPerHour'] for s in labels]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel('False alarm episodes / treatment hour')
    plt.title('TCN+Attention - False alarms per treatment hour')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('TCN_Attention_False_Alarms_Per_Hour')
    plt.show()


def plot_event_summary(val_event_metrics, test_event_metrics):
    labels = ['Validation', 'Test']

    # Detection rate
    plt.figure(figsize=(7, 5))
    plt.bar(labels, [
        val_event_metrics['AlarmEpisodeDetectionPercent'],
        test_event_metrics['AlarmEpisodeDetectionPercent']
    ])
    plt.ylabel('Detection rate (%)')
    plt.title('TCN+Attention - Alarm episode detection rate')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('TCN_Attention_Alarm_Episode_Detection_Rate')
    plt.show()

    # Detected vs missed
    detected = [
        val_event_metrics['AlarmEpisodesDetected'],
        test_event_metrics['AlarmEpisodesDetected']
    ]
    missed = [
        val_event_metrics['AlarmEpisodesMissed'],
        test_event_metrics['AlarmEpisodesMissed']
    ]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, detected, width, label='Detected')
    plt.bar(x + width / 2, missed, width, label='Missed')
    plt.xticks(x, labels)
    plt.ylabel('Number of evaluable alarm episodes')
    plt.title('TCN+Attention - Detected vs missed alarm episodes')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('TCN_Attention_Detected_vs_Missed_Alarm_Episodes')
    plt.show()

    # False warnings per session
    plt.figure(figsize=(7, 5))
    plt.bar(labels, [
        val_event_metrics['FalseWarningsPerSession'],
        test_event_metrics['FalseWarningsPerSession']
    ])
    plt.ylabel('False warning episodes / session')
    plt.title('TCN+Attention - False warnings per session')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('TCN_Attention_False_Warnings_Per_Session')
    plt.show()

    # False warnings per treatment hour
    plt.figure(figsize=(7, 5))
    plt.bar(labels, [
        val_event_metrics['FalseWarningsPerTreatmentHour'],
        test_event_metrics['FalseWarningsPerTreatmentHour']
    ])
    plt.ylabel('False warning episodes / treatment hour')
    plt.title('TCN+Attention - False warnings per treatment hour')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('TCN_Attention_False_Warnings_Per_Treatment_Hour')
    plt.show()


def plot_code_results(summary_df, subset_label):
    if summary_df.empty:
        print(f"No code event data available for {subset_label}.")
        return

    code_col = 'Code'
    episodes_col = f'Episodes in {subset_label.lower()}'

    # Detection rate by code
    plt.figure(figsize=(12, max(6, 0.35 * len(summary_df))))
    plt.barh(summary_df[code_col].astype(str), summary_df['Detection rate (%)'])
    plt.xlabel('Detection rate (%)')
    plt.ylabel('Code')
    plt.title(f'TCN+Attention - {subset_label} detection rate by alarm code')
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot(f'TCN_Attention_{subset_label}_Detection_Rate_By_Alarm_Code')
    plt.show()

    # Total vs detected episodes by code
    y = np.arange(len(summary_df))
    width = 0.38
    plt.figure(figsize=(12, max(6, 0.35 * len(summary_df))))
    plt.barh(y - width / 2, summary_df[episodes_col], height=width, label='Total evaluable episodes')
    plt.barh(y + width / 2, summary_df['Detected episodes'], height=width, label='Detected episodes')
    plt.yticks(y, summary_df[code_col].astype(str))
    plt.xlabel('Number of episodes')
    plt.ylabel('Code')
    plt.title(f'TCN+Attention - {subset_label} alarm episodes by code')
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot(f'TCN_Attention_{subset_label}_Alarm_Episodes_By_Code')
    plt.show()


# ============================================================
# ORIGINAL HYPERPARAMETERS - UNCHANGED
# ============================================================

CHANNELS = (64, 64, 128)
KERNEL_SIZE = 3
DROPOUT = 0.15
LR = 1e-3
EPOCHS = 60
PATIENCE = 10
WEIGHT_DECAY = 1e-4

TCN_PARAMS = {
    'channels': CHANNELS,
    'kernel_size': KERNEL_SIZE,
    'dropout': DROPOUT,
    'learning_rate': LR,
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'weight_decay': WEIGHT_DECAY,
    'batch_size': BATCH_SIZE,
    'n_steps': N_STEPS,
    'prediction_gap': PREDICTION_GAP,
    'seed': SEED,
}


# ============================================================
# MODEL / LOSS / OPTIMIZER

print("\n" + "=" * 100)
print("TCN CONFIGURATION")
print("=" * 100)
print(f"SEED = {SEED}")
print(f"N_STEPS = {N_STEPS}")
print(f"PREDICTION_GAP = {PREDICTION_GAP}")
print(f"Number of input features = {len(FEATURES)}")
print("FEATURES =", FEATURES)

# ============================================================

model = TCNWithAttention(
    in_ch=n_features,
    num_classes=2,
    channels=CHANNELS,
    kernel_size=KERNEL_SIZE,
    drop=DROPOUT
).to(DEVICE)

class_weights = compute_class_weights(y_train).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)
print("Class balance (Train):", Counter(y_train))
print("Class weights:", class_weights.detach().cpu().numpy())


# ============================================================
# EARLY STOPPING - ORIGINAL LOGIC
# ============================================================

best_f1 = -1.0
pat_count = 0
best_path = "best_tcn_attention.pt"

train_loss_history = []
val_acc_history = []
val_f1_history = []

for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_y_true_epoch, val_y_pred_epoch, _ = evaluate_outputs(model, val_loader)
    val_acc = accuracy_score(val_y_true_epoch, val_y_pred_epoch)
    val_f1 = f1_score(val_y_true_epoch, val_y_pred_epoch, pos_label=1, zero_division=0)

    train_loss_history.append(tr_loss)
    val_acc_history.append(val_acc)
    val_f1_history.append(val_f1)

    print(
        f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | "
        f"ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        pat_count = 0
        torch.save(model.state_dict(), best_path)
    else:
        pat_count += 1
        if pat_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best Val F1={best_f1:.4f}")
            break


# ============================================================
# FINAL EVALUATION OF THE SINGLE TRAINED MODEL
# ============================================================

model.load_state_dict(torch.load(best_path, map_location=DEVICE))

train_true, train_pred, train_score = evaluate_outputs(model, train_eval_loader)
val_true, val_pred, val_score = evaluate_outputs(model, val_loader)
test_true, test_pred, test_score = evaluate_outputs(model, test_loader)

train_metrics = compute_metrics(
    train_true, train_pred, train_score,
    groups_train, df, train_sessions
)
val_metrics = compute_metrics(
    val_true, val_pred, val_score,
    groups_val, df, val_sessions
)
test_metrics = compute_metrics(
    test_true, test_pred, test_score,
    groups_test, df, test_sessions
)

print("\n" + "=" * 100)
print("FINAL SAMPLE-LEVEL RESULTS - NUMERIC VALUES")
print("=" * 100)
print("\n>>> Model: TCN+Attention")
print_metrics("TRAIN", train_metrics)
print_metrics("VALIDATION", val_metrics)
print_metrics("TEST", test_metrics)
print(f"  - Params={TCN_PARAMS}")

print("\nClassification report (Train):")
print(classification_report(train_true, train_pred, digits=4, zero_division=0))
print("\nClassification report (Validation):")
print(classification_report(val_true, val_pred, digits=4, zero_division=0))
print("\nClassification report (Test):")
print(classification_report(test_true, test_pred, digits=4, zero_division=0))


# ============================================================
# EVENT-LEVEL VALIDATION / TEST
# ============================================================

val_event_metrics, val_event_details, val_warning_details, val_false_warnings = (
    evaluate_event_level(
        val_pred,
        meta_val,
        df,
        val_sessions,
        prediction_gap_rows=PREDICTION_GAP,
        sampling_seconds=SAMPLING_SECONDS
    )
)

test_event_metrics, test_event_details, test_warning_details, test_false_warnings = (
    evaluate_event_level(
        test_pred,
        meta_test,
        df,
        test_sessions,
        prediction_gap_rows=PREDICTION_GAP,
        sampling_seconds=SAMPLING_SECONDS
    )
)

print("\n" + "=" * 100)
print("EVENT-LEVEL RESULTS - NUMERIC VALUES")
print("=" * 100)
print_event_metrics("VALIDATION", val_event_metrics)
print_event_metrics("TEST", test_event_metrics)


# ============================================================
# CODE ANALYSIS - VALIDATION AND TEST
# ============================================================

val_code_table = code_summary(val_event_details, 'validation')
test_code_table = code_summary(test_event_details, 'test')

print("\n" + "=" * 100)
print("VALIDATION PERFORMANCE BY ALARM CODE")
print("=" * 100)
print(val_code_table.to_string(index=False))

print("\n" + "=" * 100)
print("TEST PERFORMANCE BY ALARM CODE")
print("=" * 100)
print(test_code_table.to_string(index=False))


# ============================================================
# CONFUSION MATRICES - TRAIN / VALIDATION / TEST
# ============================================================

plot_cm(train_true, train_pred, "TCN+Attention - Train Confusion Matrix")
plot_cm(val_true, val_pred, "TCN+Attention - Validation Confusion Matrix")
plot_cm(test_true, test_pred, "TCN+Attention - Test Confusion Matrix")


# ============================================================
# QUANTITATIVE RESULTS
# ============================================================
# All quantitative performance results are intentionally reported as
# NUMERIC VALUES in the console and saved to CSV.
# No bar/histogram plots are generated for F1, Precision, Sensitivity,
# Specificity, AUROC, AUPRC, treatment hours, FA/h, event-level metrics,
# lead time, or code-level detection.
#
# Confusion matrices are retained below/above as requested, and the original
# attention visualization is also retained. The code-identification results
# remain available as printed tables and CSV files.


# ============================================================
# ORIGINAL ATTENTION VISUALIZATION - NOW ALSO SAVED
# ============================================================

@torch.no_grad()
def visualize_attention(model, loader, n_samples=5):
    model.eval()
    xb, yb = next(iter(loader))
    xb = xb.to(DEVICE)
    logits, alpha = model(xb)
    alpha = alpha.detach().cpu().numpy()
    T = alpha.shape[1]
    ns = min(n_samples, alpha.shape[0])

    plt.figure(figsize=(8, 3 * ns))
    for i in range(ns):
        plt.subplot(ns, 1, i + 1)
        markerline, stemlines, baseline = plt.stem(range(T), alpha[i])
        plt.title(f"Sample {i} - Attention over time steps (0..{T - 1})")
        plt.xlabel("Time step")
        plt.ylabel("Weight")
    plt.tight_layout()
    save_current_plot('TCN_Attention_Weights_Test_Samples')
    plt.show()


visualize_attention(model, test_loader, n_samples=5)


# ============================================================
# SAVE ALL NUMERIC RESULTS
# ============================================================

sample_metrics_df = pd.DataFrame([
    {'Subset': 'Train', **train_metrics},
    {'Subset': 'Validation', **val_metrics},
    {'Subset': 'Test', **test_metrics},
])
sample_metrics_df.to_csv('TCN_sample_level_metrics.csv', index=False)

val_event_summary_df = pd.DataFrame([{'Subset': 'Validation', **val_event_metrics}])
test_event_summary_df = pd.DataFrame([{'Subset': 'Test', **test_event_metrics}])
pd.concat([val_event_summary_df, test_event_summary_df], ignore_index=True).to_csv(
    'TCN_event_level_metrics.csv', index=False
)

val_event_details.to_csv('TCN_validation_alarm_event_details.csv', index=False)
test_event_details.to_csv('TCN_test_alarm_event_details.csv', index=False)
val_warning_details.to_csv('TCN_validation_warning_event_details.csv', index=False)
test_warning_details.to_csv('TCN_test_warning_event_details.csv', index=False)
val_false_warnings.to_csv('TCN_validation_false_warning_events.csv', index=False)
test_false_warnings.to_csv('TCN_test_false_warning_events.csv', index=False)

val_code_table.to_csv(
    'TCN_validation_performance_by_code.csv', index=False
)
test_code_table.to_csv(
    'TCN_test_performance_by_code.csv', index=False
)

# Event-by-event files dedicated to code inspection
val_event_details.to_csv(
    'TCN_validation_alarm_event_details_by_code.csv', index=False
)
test_event_details.to_csv(
    'TCN_test_alarm_event_details_by_code.csv', index=False
)

training_history_df = pd.DataFrame({
    'Epoch': np.arange(1, len(train_loss_history) + 1),
    'TrainLoss': train_loss_history,
    'ValidationAccuracy': val_acc_history,
    'ValidationF1': val_f1_history,
})
training_history_df.to_csv('TCN_training_history.csv', index=False)

print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)
print("TCN_random_sample_split.csv")
print("TCN_sample_level_metrics.csv")
print("TCN_event_level_metrics.csv")
print("TCN_validation_alarm_event_details.csv")
print("TCN_test_alarm_event_details.csv")
print("TCN_validation_warning_event_details.csv")
print("TCN_test_warning_event_details.csv")
print("TCN_validation_false_warning_events.csv")
print("TCN_test_false_warning_events.csv")
print("TCN_validation_performance_by_code.csv")
print("TCN_test_performance_by_code.csv")
print("TCN_validation_alarm_event_details_by_code.csv")
print("TCN_test_alarm_event_details_by_code.csv")
print("TCN_training_history.csv")
print(f"Retained plots (confusion matrices + attention visualization) are saved in: {PLOTS_DIR}")

print("\nCurrent temporal configuration:")
print(f"N_STEPS = {N_STEPS} rows = {INPUT_WINDOW_SECONDS:.1f} s input window")
print(f"PREDICTION_GAP = {PREDICTION_GAP} rows = {DELTA_T_SECONDS:.1f} s nominal horizon")
print(f"SEED = {SEED}")
