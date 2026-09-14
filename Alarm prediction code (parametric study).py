# ==================================================================================================
# ALARM PREDICTION CODE: PARAMETRIC STUDY
# Random Forest - complete sample-level and event-level evaluation
# ==================================================================================================
#
# Workflow:
#   1. Dataset loading and preprocessing
#   2. Treatment reconstruction from time gaps and treatment-safe temporal sequence generation
#   3. Randomized Train / Validation / Test split by complete treatment (approximately 70 / 15 / 15)
#   4. Data reshape for Random Forest
#   5. Random Forest training with the selected configuration
#   6. Sample-level evaluation on Train / Validation / Test
#   7. Event-level evaluation on Validation / Test
#   8. Alarm-code-specific event analysis on Validation / Test
#   9. Confusion matrices, lead-time plots, feature importance
#  10. Automatic saving of summary and detailed CSV outputs
#
# IMPORTANT:
# - The features and model hyperparameters are kept exactly as in the original script supplied in the chat.
# - The only methodological change is the Train / Validation / Test partition: complete treatments
#   are kept intact and are never split across subsets.
# - The temporal configuration is N_STEPS = 10 and PREDICTION_GAP = 120 rows,
#   corresponding to a 60 s prediction horizon at 0.5 s sampling.
# - Treatments used for data partitioning are reconstructed ONLY from the 'time' column: a new
#   treatment starts when the gap from the previous row is greater than 30 minutes.
# - Session identifiers used by the event-level analysis retain the original reconstruction logic
#   and do NOT control the Train / Validation / Test split.
# - The binary prediction task remains alarm (1) vs normal/override (0).
#
# Additional sample-level metrics:
#   - Accuracy
#   - F1 Score
#   - Precision
#   - Sensitivity / Recall
#   - Specificity
#   - AUROC
#   - AUPRC
#   - False-alarm episodes per evaluated hour
#   - TN / FP / FN / TP
#
# Additional event-level metrics:
#   - Total / evaluable / detected / missed alarm episodes
#   - Alarm-episode detection rate (%)
#   - Predicted warning episodes
#   - False warning episodes
#   - False warnings per represented session
#   - False warnings per evaluated hour
#   - Lead-time distribution: N, mean, median, std, min, Q25, Q75, max
#   - Alarm-code-specific episodes, detections, detection rate and median lead time
#
# ==================================================================================================


# ==================================================================================================
# IMPORTS
# ==================================================================================================

from numpy import array, hstack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ==================================================================================================
# GENERAL SETTINGS
# ==================================================================================================

# ----------------------------------
# Input / output paths
# ----------------------------------
CSV_PATH = r"C:\Users\aless\Desktop\Alarm_paper\Definitive_FINISH_All_months_ready_for_training_.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------------
# Temporal configuration
# ----------------------------------
# Number of consecutive rows known to the model at prediction time.
# 30 rows x 0.5 s/row = 15 s input window.
N_STEPS = 10

# Number of rows between the current condition and the future condition predicted.
# 120 rows x 0.5 s/row = 60 s = 1 minute prediction horizon.
PREDICTION_GAP = 120

# Sampling interval of the original dataset.
SAMPLING_SECONDS = 0.5

# Derived values, used only for reporting.
PREDICTION_HORIZON_SECONDS = PREDICTION_GAP * SAMPLING_SECONDS
INPUT_WINDOW_SECONDS = N_STEPS * SAMPLING_SECONDS

# ----------------------------------
# Train / Validation / Test proportions
# ----------------------------------
# IMPORTANT: these proportions are targeted at the SAMPLE/SEQUENCE level, but the assignment
# itself is performed at the TREATMENT level. Therefore, exact 70 / 15 / 15 percentages are not
# always possible because an individual treatment is never divided between different subsets.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

# Number of reproducible random treatment assignments explored to find the split that best
# approximates the requested 70 / 15 / 15 sequence proportions.
SPLIT_RANDOM_TRIALS = 2000

# ----------------------------------
# Treatment reconstruction for dataset partitioning
# ----------------------------------
# A new treatment starts whenever the 'time' gap from the previous row is greater than 30 minutes.
# This treatment identifier controls BOTH sequence generation and Train / Validation / Test splitting.
TREATMENT_GAP_MINUTES = 30

# ----------------------------------
# Session reconstruction for event metrics only
# ----------------------------------
# This value and the original hash/time session logic are intentionally left unchanged because
# they are used only by the existing event-level metrics, not by the dataset partition.
SESSION_GAP_MINUTES = 60

# ----------------------------------
# Random Forest configuration
# ----------------------------------
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 30
RF_MIN_SAMPLES_SPLIT = 3
RF_MIN_SAMPLES_LEAF = 4
RF_MAX_FEATURES = 'sqrt'
RF_CLASS_WEIGHT = 'balanced'
RF_RANDOM_STATE = 42


# ==================================================================================================
# FEATURES
# ==================================================================================================
# Exact feature list from the original script supplied in the chat.

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
    'pArterial',
]


# ==================================================================================================
# BASIC UTILITIES
# ==================================================================================================

def save_current_plot(filename, dpi=300):
    """Save the current figure in PLOTS_DIR while still allowing plt.show()."""
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(filename)).strip('_')
    if not safe_name:
        safe_name = 'plot'
    path = os.path.join(PLOTS_DIR, f"{safe_name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Grafico salvato: {path}")


def split_sequences(sequences, n_steps):
    """Split a multivariate sequence into overlapping temporal samples."""
    X, y = list(), list()

    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break

        seq_x = sequences[i:end_ix, :-1]
        seq_y = sequences[end_ix - 1, -1]

        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)


def convert_time_column(df, time_column='time'):
    """Convert the time column without changing the original dataframe order."""
    df = df.copy()

    if '_original_order' not in df.columns:
        df['_original_order'] = np.arange(len(df))

    if time_column not in df.columns:
        return df

    if pd.api.types.is_datetime64_any_dtype(df[time_column]):
        return df

    parsed = pd.to_datetime(
        df[time_column],
        format='%Y-%m-%d %H.%M.%S',
        errors='coerce'
    )

    if parsed.isna().any():
        parsed_fallback = pd.to_datetime(
            df.loc[parsed.isna(), time_column],
            errors='coerce'
        )
        parsed.loc[parsed.isna()] = parsed_fallback

    if parsed.isna().any():
        bad = df.loc[parsed.isna(), time_column].head(10).tolist()
        raise ValueError(
            "Alcuni valori della colonna 'time' non possono essere convertiti. "
            f"Esempi: {bad}"
        )

    df[time_column] = parsed
    return df


# ==================================================================================================
# TREATMENT IDENTIFICATION FOR SEQUENCE GENERATION AND DATASET SPLITTING
# ==================================================================================================

def add_treatment_ids(df, time_column='time', gap_minutes=30):
    """
    Reconstruct complete treatments using only discontinuities in the time column.

    A new treatment begins whenever the time difference from the previous row is greater
    than ``gap_minutes``. With the current settings, this means a gap > 30 minutes.

    Important
    ---------
    - ``hash`` is NOT used to define treatments for Train / Validation / Test splitting.
    - Treatment reconstruction is deterministic and preserves the original row order.
    - Randomization is applied later, only when COMPLETE treatments are assigned to
      Train, Validation or Test.
    """
    if time_column not in df.columns:
        raise ValueError(
            f"The '{time_column}' column is required to reconstruct complete treatments."
        )

    df = convert_time_column(df, time_column=time_column)

    # Time difference between every row and the immediately preceding row.
    time_delta = df[time_column].diff()

    # The first row always starts treatment 1. Every gap larger than the selected threshold
    # starts a new treatment. No other variable is used to define the treatment boundary.
    new_treatment = time_delta > pd.Timedelta(minutes=gap_minutes)
    if len(new_treatment) > 0:
        new_treatment.iloc[0] = True

    df['treatment_id'] = new_treatment.cumsum().astype(int)

    print("\n" + "=" * 100)
    print("TREATMENT IDENTIFICATION FOR DATASET SPLITTING")
    print("=" * 100)
    print(f"Treatment rule            : time gap > {gap_minutes} minutes")
    print(f"Number of treatments      : {df['treatment_id'].nunique()}")
    print("Treatment integrity rule  : one complete treatment -> one subset only")

    return df


# ==================================================================================================
# SESSION IDENTIFICATION FOR EVENT-LEVEL METRICS
# ==================================================================================================

def add_session_ids_for_metrics(df, time_column='time', gap_minutes=60):
    """
    Reconstruct treatment/session identity only for the additional event metrics.

    Preferred rule when both 'hash' and 'time' are available:
        - new session when hash changes, OR
        - new session when the time gap exceeds gap_minutes.

    The original row order is preserved for the existing event-level calculations.
    This session identifier is independent from the treatment_id used for dataset splitting.
    """
    df = df.copy()

    if '_original_order' not in df.columns:
        df['_original_order'] = np.arange(len(df))

    if time_column in df.columns:
        df = convert_time_column(df, time_column=time_column)

    if 'hash' in df.columns and time_column in df.columns:
        df['hash'] = df['hash'].astype(str)

        hash_change = df['hash'].ne(df['hash'].shift(1))
        time_delta = df[time_column].diff()
        large_gap = time_delta > pd.Timedelta(minutes=gap_minutes)

        new_session = hash_change | large_gap
        if len(new_session) > 0:
            new_session.iloc[0] = True

        df['session_id'] = new_session.cumsum().astype(int)
        df['time_gap_min'] = time_delta.dt.total_seconds() / 60.0

    elif 'hash' in df.columns:
        codes, _ = pd.factorize(df['hash'].astype(str), sort=False)
        df['session_id'] = codes + 1
        df['time_gap_min'] = np.nan

    else:
        # Fallback used only if session identifiers cannot be reconstructed.
        # Event-level per-session interpretation is then limited.
        df['session_id'] = 1
        df['time_gap_min'] = np.nan

    print("\n" + "=" * 100)
    print("SESSION IDENTIFICATION FOR EVENT-LEVEL METRICS")
    print("=" * 100)
    print(f"Number of reconstructed sessions: {df['session_id'].nunique()}")

    if 'hash' in df.columns:
        print(f"Number of unique hash values: {df['hash'].nunique(dropna=True)}")

    return df


def print_session_summary(df):
    """Print and save a compact session summary used for event-level auditing."""
    if 'session_id' not in df.columns:
        return pd.DataFrame()

    agg_dict = {
        'n_rows': ('session_id', 'size'),
        'alarm_rows': ('type', lambda x: int((x == 1).sum())),
    }

    if 'hash' in df.columns:
        agg_dict['hash'] = ('hash', 'first')
    if 'time' in df.columns:
        agg_dict['start'] = ('time', 'min')
        agg_dict['end'] = ('time', 'max')

    summary = df.groupby('session_id').agg(**agg_dict).reset_index()

    if 'start' in summary.columns and 'end' in summary.columns:
        summary['duration_min'] = (
            (summary['end'] - summary['start']).dt.total_seconds() / 60.0
        )

    summary['duration_from_samples_h'] = (
        summary['n_rows'] * SAMPLING_SECONDS / 3600.0
    )

    print("\n" + "=" * 100)
    print("SESSION SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    summary.to_csv('session_summary_for_event_metrics.csv', index=False)
    return summary


# ==================================================================================================
# SEQUENCE GENERATION WITH METADATA
# ==================================================================================================

def create_splitted_df_with_metadata(df, n_steps, prediction_gap=PREDICTION_GAP):
    """
    Generate temporal sequences independently inside each complete treatment.

    The feature/target alignment is kept exactly as in the original script:
        - ``n_steps`` consecutive rows are used as model input;
        - the target is taken ``prediction_gap`` rows after the last input row.

    The ONLY difference is treatment safety: a temporal sequence is never allowed to
    include rows from two different treatments, and its future target must belong to the
    same treatment as the complete input window.

    All metadata fields required by the original sample-level, event-level and alarm-code
    analyses are retained unchanged.
    """
    if 'treatment_id' not in df.columns:
        raise ValueError(
            "The 'treatment_id' column is missing. Run add_treatment_ids() before sequence generation."
        )

    X_parts = []
    y_parts = []
    metadata_rows = []
    global_sample_index = 0
    skipped_treatments = 0

    # ``sort=False`` preserves the original order in which treatments occur in the dataset.
    # This ordering is used only to generate valid within-treatment sequences. It does NOT make
    # the subsequent Train / Validation / Test split chronological.
    for treatment_id, treatment_df in df.groupby('treatment_id', sort=False):
        treatment_df = treatment_df.copy()

        # At least ``prediction_gap + n_steps`` rows are required to create one valid sample.
        if len(treatment_df) < prediction_gap + n_steps:
            skipped_treatments += 1
            continue

        # Same feature/target alignment used by the original code, now applied separately
        # to each treatment so that no sequence can cross a treatment boundary.
        historical_df = treatment_df.iloc[:-prediction_gap]

        stacked_features = [
            historical_df[col].to_numpy().reshape(-1, 1)
            for col in FEATURES
        ]

        alarm_type = (
            treatment_df['type']
            .to_numpy()[prediction_gap:]
            .reshape(-1, 1)
        )

        dataset = hstack(stacked_features + [alarm_type])
        X_treatment, y_treatment = split_sequences(dataset, n_steps)

        if len(X_treatment) == 0:
            skipped_treatments += 1
            continue

        X_parts.append(X_treatment)
        y_parts.append(y_treatment)

        # Map treatment-local positions back to the GLOBAL dataframe row indices. This keeps
        # every downstream event-level calculation consistent with the original full dataframe.
        global_rows = treatment_df.index.to_numpy(dtype=int)

        for local_i in range(len(X_treatment)):
            input_start_local = local_i
            prediction_local = local_i + n_steps - 1
            target_local = prediction_local + prediction_gap

            input_start_row = int(global_rows[input_start_local])
            prediction_row_index = int(global_rows[prediction_local])
            target_row_index = int(global_rows[target_local])

            input_start_sid = int(df.loc[input_start_row, 'session_id'])
            prediction_sid = int(df.loc[prediction_row_index, 'session_id'])
            target_sid = int(df.loc[target_row_index, 'session_id'])

            same_session = (
                input_start_sid == prediction_sid == target_sid
            )

            metadata_rows.append({
                'global_sample_index': int(global_sample_index),
                'treatment_id': int(treatment_id),
                'input_start_row_index': input_start_row,
                'prediction_row_index': prediction_row_index,
                'target_row_index': target_row_index,
                'input_start_session_id': input_start_sid,
                'prediction_session_id': prediction_sid,
                'target_session_id': target_sid,
                'same_session': bool(same_session),
                'prediction_time': (
                    df.loc[prediction_row_index, 'time']
                    if 'time' in df.columns else pd.NaT
                ),
                'target_time': (
                    df.loc[target_row_index, 'time']
                    if 'time' in df.columns else pd.NaT
                ),
                'target_code': (
                    df.loc[target_row_index, 'code']
                    if 'code' in df.columns else np.nan
                ),
            })

            global_sample_index += 1

    if not X_parts:
        raise ValueError(
            "No treatment contains enough rows to generate temporal sequences with "
            f"N_STEPS={n_steps} and PREDICTION_GAP={prediction_gap}."
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    metadata = pd.DataFrame(metadata_rows)

    if not (len(X) == len(y) == len(metadata)):
        raise RuntimeError("X, y and metadata do not contain the same number of samples.")

    if skipped_treatments > 0:
        print(
            f"Treatments excluded from sequence generation because they are too short: "
            f"{skipped_treatments}"
        )

    print(f"Treatments represented by valid sequences: {metadata['treatment_id'].nunique()}")

    return X, y, metadata


def treatment_based_split_indices(
    metadata,
    y=None,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    n_random_trials=2000
):
    """
    Randomly assign COMPLETE treatments to Train / Validation / Test.

    This function never splits individual treatments. Randomization is performed only on
    treatment IDs; every temporal sequence generated from one treatment therefore belongs
    to exactly one subset.

    Strategy
    --------
    1. Count the valid temporal sequences generated by each treatment.
    2. Randomly shuffle treatment IDs using the fixed ``random_state``.
    3. For each random ordering, identify treatment boundaries that approximate 70 / 15 / 15
       as closely as possible in terms of the number of generated sequences.
    4. Repeat this process ``n_random_trials`` times and retain the best assignment.
    5. When labels are provided, prefer a candidate containing both classes in Train,
       Validation and Test whenever such a candidate is found.

    Notes
    -----
    - The split is NOT chronological.
    - Treatments from different dates can be assigned to the same subset.
    - Exact 70 / 15 / 15 proportions are not guaranteed because treatments are indivisible.
    """

    # ----------------------------------------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------------------------------------
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be equal to 1.")

    if 'treatment_id' not in metadata.columns:
        raise ValueError("metadata must contain the 'treatment_id' column.")

    if n_random_trials < 1:
        raise ValueError("n_random_trials must be at least 1.")

    if y is not None:
        y = np.asarray(y).astype(int)
        if len(y) != len(metadata):
            raise ValueError("y and metadata must contain the same number of samples.")

    # ----------------------------------------------------------------------------------------------
    # Treatment-level statistics
    # ----------------------------------------------------------------------------------------------
    # Number of generated temporal sequences belonging to each complete treatment.
    treatment_counts = metadata.groupby('treatment_id', sort=False).size()
    treatment_ids = treatment_counts.index.to_numpy(dtype=int)
    counts = treatment_counts.to_numpy(dtype=int)
    n_treatments = len(treatment_ids)

    if n_treatments < 3:
        raise ValueError(
            "At least 3 treatments containing valid sequences are required to create "
            "Train, Validation and Test without splitting treatments."
        )

    total_samples = int(counts.sum())
    target_counts = ratios * total_samples

    # Optional class counts per treatment. These are used only to prefer an assignment
    # in which Train, Validation and Test all contain both class 0 and class 1.
    treatment_class_counts = None
    if y is not None:
        tmp = pd.DataFrame({
            'treatment_id': metadata['treatment_id'].to_numpy(dtype=int),
            'target': y,
        })
        class_table = pd.crosstab(tmp['treatment_id'], tmp['target'])
        class_table = class_table.reindex(index=treatment_ids, columns=[0, 1], fill_value=0)
        treatment_class_counts = class_table.to_numpy(dtype=int)

    # ----------------------------------------------------------------------------------------------
    # Random search for the best complete-treatment assignment
    # ----------------------------------------------------------------------------------------------
    rng = np.random.RandomState(random_state)

    best_valid = None       # Best ratio match with both classes in all three subsets.
    best_valid_score = np.inf
    best_any = None         # Fallback: best ratio match irrespective of class composition.
    best_any_score = np.inf

    for _ in range(n_random_trials):
        # Only treatment IDs are shuffled. Individual temporal samples are never shuffled
        # independently between Train, Validation and Test.
        permutation = rng.permutation(n_treatments)
        ids_perm = treatment_ids[permutation]
        counts_perm = counts[permutation]
        cumulative_counts = np.cumsum(counts_perm)

        # ------------------------------ Train boundary ------------------------------
        # Leave at least one complete treatment for Validation and one for Test.
        train_cut_candidates = np.arange(1, n_treatments - 1)
        train_cut = int(
            train_cut_candidates[
                np.argmin(
                    np.abs(
                        cumulative_counts[train_cut_candidates - 1]
                        - target_counts[0]
                    )
                )
            ]
        )

        # --------------------------- Validation/Test boundary -----------------------
        # Target cumulative Train + Validation = approximately 85% of all sequences.
        second_cut_candidates = np.arange(train_cut + 1, n_treatments)
        train_val_target = target_counts[0] + target_counts[1]
        second_cut = int(
            second_cut_candidates[
                np.argmin(
                    np.abs(
                        cumulative_counts[second_cut_candidates - 1]
                        - train_val_target
                    )
                )
            ]
        )

        train_treatments = ids_perm[:train_cut]
        val_treatments = ids_perm[train_cut:second_cut]
        test_treatments = ids_perm[second_cut:]

        # Actual sequence counts produced by this complete-treatment assignment.
        train_n = int(counts_perm[:train_cut].sum())
        val_n = int(counts_perm[train_cut:second_cut].sum())
        test_n = int(counts_perm[second_cut:].sum())
        actual_counts = np.array([train_n, val_n, test_n], dtype=float)

        # Total absolute deviation from the requested 70 / 15 / 15 sequence proportions.
        ratio_score = float(np.sum(np.abs(actual_counts - target_counts)) / total_samples)

        candidate = (
            train_treatments.copy(),
            val_treatments.copy(),
            test_treatments.copy(),
            actual_counts.astype(int),
        )

        if ratio_score < best_any_score:
            best_any_score = ratio_score
            best_any = candidate

        # Prefer candidates where every subset contains both target classes.
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

    # Prefer a split containing both classes in all subsets. If no such candidate is found,
    # fall back to the complete-treatment assignment closest to 70 / 15 / 15.
    selected = best_valid if best_valid is not None else best_any
    if selected is None:
        raise RuntimeError("Unable to generate a treatment-based Train/Validation/Test split.")

    train_treatments, val_treatments, test_treatments, actual_counts = selected

    # ----------------------------------------------------------------------------------------------
    # Convert treatment assignments back to temporal-sample indices
    # ----------------------------------------------------------------------------------------------
    treatment_series = metadata['treatment_id']
    idx_train = np.flatnonzero(treatment_series.isin(train_treatments).to_numpy())
    idx_val = np.flatnonzero(treatment_series.isin(val_treatments).to_numpy())
    idx_test = np.flatnonzero(treatment_series.isin(test_treatments).to_numpy())

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
    if len(assigned_indices) != len(metadata) or len(np.unique(assigned_indices)) != len(metadata):
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
    """Print class balance for a subset."""
    values, counts = np.unique(y, return_counts=True)

    print(f"\n{name}")
    print(f"Number of samples: {len(y)}")

    for value, count in zip(values, counts):
        print(f"  Class {value}: {count} ({100.0 * count / len(y):.2f}%)")

    if len(values) < 2:
        raise ValueError(f"{name} contiene una sola classe.")


# ==================================================================================================
# SAMPLE-LEVEL METRIC UTILITIES
# ==================================================================================================

def specificity_score(y_true, y_pred):
    """Specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=['0', '1'])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else np.nan


def get_positive_probability(model, X):
    """Return the estimated probability of the positive alarm class."""
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)

    if '1' in classes:
        pos_idx = int(np.where(classes == '1')[0][0])
    elif 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        raise ValueError("La classe positiva 1 non è presente in model.classes_.")

    return np.asarray(proba)[:, pos_idx]


def _valid_metadata(metadata):
    """Keep only sequence samples fully contained within one reconstructed session."""
    if metadata is None or len(metadata) == 0:
        return pd.DataFrame()

    m = metadata.reset_index(drop=True).copy()
    if 'same_session' in m.columns:
        m = m[m['same_session'].astype(bool)].copy()
    return m


def count_false_alarm_episodes(y_true, y_pred, metadata):
    """
    Count distinct false-positive episodes among evaluated hold-out records.

    A new false-alarm episode starts when:
      - true class = 0 and predicted class = 1, and
      - the previous evaluated prediction is not an adjacent false positive
        in the original time series.

    This avoids counting every 0.5-s false-positive record as a separate alarm.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    metadata = metadata.reset_index(drop=True).copy()

    if not (len(y_true) == len(y_pred) == len(metadata)):
        raise ValueError("y_true, y_pred e metadata devono avere la stessa lunghezza.")

    work = metadata.copy()
    work['y_true'] = y_true
    work['y_pred'] = y_pred
    work = work[work['same_session'].astype(bool)].copy()

    total_episodes = 0

    for sid, g in work.groupby('prediction_session_id'):
        g = g.sort_values('prediction_row_index').reset_index(drop=True)
        if g.empty:
            continue

        rows = g['prediction_row_index'].to_numpy(dtype=int)
        fp_mask = (
            (g['y_true'].to_numpy() == '0')
            & (g['y_pred'].to_numpy() == '1')
        )

        previous_is_contiguous_fp = np.r_[
            False,
            fp_mask[:-1] & (rows[1:] == rows[:-1] + 1)
        ]

        starts = fp_mask & ~previous_is_contiguous_fp
        total_episodes += int(starts.sum())

    return total_episodes


def evaluated_hours(metadata, sampling_seconds=SAMPLING_SECONDS):
    """
    Effective duration represented by valid evaluated prediction records.

    For each evaluated subset, this denominator is based on the number of valid
    prediction records multiplied by the nominal sampling interval, preserving
    the original metric calculation.
    """
    m = _valid_metadata(metadata)
    return float(len(m) * sampling_seconds / 3600.0)


def represented_session_count(metadata):
    """Number of reconstructed sessions represented by valid evaluated records."""
    m = _valid_metadata(metadata)
    if m.empty:
        return 0
    return int(m['prediction_session_id'].nunique())


def calculate_metrics(model, X, y_true, y_pred, metadata):
    """Calculate all requested sample-level metrics."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    cm = confusion_matrix(y_true, y_pred, labels=['0', '1'])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label='1', zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label='1', zero_division=0)
    sensitivity = recall_score(y_true, y_pred, pos_label='1', zero_division=0)
    specificity = specificity_score(y_true, y_pred)

    score = get_positive_probability(model, X)
    y_binary = (y_true == '1').astype(int)

    if len(np.unique(y_binary)) == 2:
        auroc = roc_auc_score(y_binary, score)
        auprc = average_precision_score(y_binary, score)
    else:
        auroc = np.nan
        auprc = np.nan

    false_alarm_episodes = count_false_alarm_episodes(y_true, y_pred, metadata)
    hours = evaluated_hours(metadata)

    false_alarms_per_hour = (
        false_alarm_episodes / hours if hours > 0 else np.nan
    )

    return {
        'Accuracy': float(accuracy),
        'F1': float(f1),
        'Precision': float(precision),
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'AUROC': float(auroc) if np.isfinite(auroc) else np.nan,
        'AUPRC': float(auprc) if np.isfinite(auprc) else np.nan,
        'FalseAlarmEpisodes': int(false_alarm_episodes),
        'EvaluatedHours': float(hours),
        'FalseAlarmsPerHour': float(false_alarms_per_hour)
            if np.isfinite(false_alarms_per_hour) else np.nan,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
    }


def print_metrics(name, metrics):
    """Print a complete sample-level metric block."""
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    print(f"Accuracy             -> {metrics['Accuracy']:.4f}")
    print(f"F1-Score             -> {metrics['F1']:.4f}")
    print(f"Precision            -> {metrics['Precision']:.4f}")
    print(f"Sensitivity / Recall -> {metrics['Sensitivity']:.4f}")
    print(f"Specificity          -> {metrics['Specificity']:.4f}")
    print(f"AUROC                -> {metrics['AUROC']:.4f}")
    print(f"AUPRC                -> {metrics['AUPRC']:.4f}")
    print(f"Evaluated hours      -> {metrics['EvaluatedHours']:.4f}")
    print(f"False alarm episodes -> {metrics['FalseAlarmEpisodes']}")
    print(f"False alarms/h       -> {metrics['FalseAlarmsPerHour']:.4f}")
    print(
        f"TN={metrics['TN']} | FP={metrics['FP']} | "
        f"FN={metrics['FN']} | TP={metrics['TP']}"
    )


# ==================================================================================================
# EVENT-LEVEL CLINICAL METRICS
# ==================================================================================================

def _contiguous_true_runs(mask):
    """Return inclusive start/end positions of contiguous True runs."""
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0 or not mask.any():
        return []

    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])

    return list(zip(starts.tolist(), ends.tolist()))


def extract_true_alarm_events(df, session_ids=None):
    """
    Define a true alarm episode as a contiguous run of type == 1 within one
    reconstructed session.
    """
    rows = []

    if session_ids is None:
        session_ids = sorted(df['session_id'].unique())

    for session_id in session_ids:
        s = df[df['session_id'] == session_id].copy()
        s = s.sort_values('_original_order').reset_index(drop=False)

        # 'index' is the global dataframe row index because df is kept in original order.
        global_rows = s['index'].to_numpy(dtype=int)
        alarm_mask = s['type'].to_numpy(dtype=int) == 1
        runs = _contiguous_true_runs(alarm_mask)

        for event_number, (start_local, end_local) in enumerate(runs, start=1):
            start_global = int(global_rows[start_local])
            end_global = int(global_rows[end_local])

            rows.append({
                'session_id': int(session_id),
                'event_number_within_session': int(event_number),
                'event_start_row': start_global,
                'event_end_row': end_global,
                'event_duration_rows': int(end_local - start_local + 1),
                'event_duration_seconds': float(
                    (end_local - start_local + 1) * SAMPLING_SECONDS
                ),
                'event_start_time': (
                    s.loc[start_local, 'time'] if 'time' in s.columns else pd.NaT
                ),
                'event_end_time': (
                    s.loc[end_local, 'time'] if 'time' in s.columns else pd.NaT
                ),
            })

    return pd.DataFrame(rows)


def extract_predicted_warning_episodes(pred, metadata):
    """
    Define a warning episode as a contiguous run of predicted class 1 in the
    original row space within the same reconstructed session.

    Since the hold-out records are a subset of the original sequence samples,
    a new episode is forced whenever two evaluated prediction rows are not
    adjacent in the original timeline.
    """
    pred = np.asarray(pred).astype(str)
    metadata = metadata.reset_index(drop=True).copy()

    if len(pred) != len(metadata):
        raise ValueError("pred e metadata devono avere la stessa lunghezza.")

    metadata['pred'] = pred
    metadata = metadata[metadata['same_session'].astype(bool)].copy()

    rows_out = []

    for session_id, g in metadata.groupby('prediction_session_id'):
        g = g.sort_values('prediction_row_index').reset_index(drop=True)
        if g.empty:
            continue

        p = g['pred'].to_numpy()
        original_rows = g['prediction_row_index'].to_numpy(dtype=int)

        in_run = False
        run_start = None
        run_end = None
        warning_number = 0

        def close_run(start_idx, end_idx, number):
            gs = g.iloc[start_idx]
            ge = g.iloc[end_idx]
            return {
                'session_id': int(session_id),
                'warning_number_within_session': int(number),
                'warning_start_prediction_row': int(gs['prediction_row_index']),
                'warning_end_prediction_row': int(ge['prediction_row_index']),
                'warning_start_time': gs['prediction_time'],
                'warning_end_time': ge['prediction_time'],
                'warning_duration_samples': int(end_idx - start_idx + 1),
                'warning_duration_seconds': float(
                    (end_idx - start_idx + 1) * SAMPLING_SECONDS
                ),
            }

        for j in range(len(g)):
            is_positive = p[j] == '1'
            contiguous_with_previous = (
                j > 0 and original_rows[j] == original_rows[j - 1] + 1
            )

            if is_positive and (not in_run or not contiguous_with_previous):
                if in_run:
                    warning_number += 1
                    rows_out.append(close_run(run_start, run_end, warning_number))

                in_run = True
                run_start = j
                run_end = j

            elif is_positive and in_run and contiguous_with_previous:
                run_end = j

            elif not is_positive and in_run:
                warning_number += 1
                rows_out.append(close_run(run_start, run_end, warning_number))
                in_run = False
                run_start = None
                run_end = None

        if in_run:
            warning_number += 1
            rows_out.append(close_run(run_start, run_end, warning_number))

    return pd.DataFrame(rows_out)


def evaluate_event_level(
    pred,
    metadata,
    df,
    prediction_gap_rows=PREDICTION_GAP,
    sampling_seconds=SAMPLING_SECONDS
):
    """
    Event-level early-warning evaluation.

    Successful prediction of one true alarm episode:
      - at least one positive hold-out prediction exists before alarm onset,
      - it falls within the nominal pre-alarm window:
            event_start - prediction_gap_rows <= prediction_row < event_start

    Evaluable alarm episode:
      - at least one valid hold-out prediction record exists in that pre-alarm window.

    Lead time:
      - event_start_row - earliest positive prediction row, converted to seconds.

    False warning episode:
      - a predicted-positive warning episode not overlapping the pre-alarm window
        of any evaluable true alarm episode.
    """
    pred = np.asarray(pred).astype(str)
    metadata = metadata.reset_index(drop=True).copy()

    if len(pred) != len(metadata):
        raise ValueError("pred e metadata devono avere la stessa lunghezza.")

    metadata['pred'] = pred
    metadata_valid = metadata[metadata['same_session'].astype(bool)].copy()

    if metadata_valid.empty:
        empty_summary = {
            'AlarmEpisodesTotal': 0,
            'AlarmEpisodesEvaluable': 0,
            'AlarmEpisodesDetected': 0,
            'AlarmEpisodesMissed': 0,
            'AlarmEpisodeDetectionRate': np.nan,
            'AlarmEpisodeDetectionPercent': np.nan,
            'PredictedWarningEpisodes': 0,
            'FalseWarningEpisodes': 0,
            'FalseWarningsPerSession': np.nan,
            'FalseWarningsPerTreatmentHour': np.nan,
            'LeadTimeN': 0,
            'LeadTimeMeanSeconds': np.nan,
            'LeadTimeMedianSeconds': np.nan,
            'LeadTimeStdSeconds': np.nan,
            'LeadTimeMinSeconds': np.nan,
            'LeadTimeQ25Seconds': np.nan,
            'LeadTimeQ75Seconds': np.nan,
            'LeadTimeMaxSeconds': np.nan,
        }
        return empty_summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    represented_sessions = sorted(
        metadata_valid['prediction_session_id'].unique().astype(int)
    )

    true_events = extract_true_alarm_events(df, represented_sessions)
    warning_events = extract_predicted_warning_episodes(pred, metadata)

    event_detail_rows = []
    matched_warning_keys = set()

    for _, event in true_events.iterrows():
        sid = int(event['session_id'])
        start_row = int(event['event_start_row'])

        lower = start_row - int(prediction_gap_rows)
        upper = start_row - 1

        m_session = metadata_valid[
            metadata_valid['prediction_session_id'] == sid
        ].copy()

        prediction_rows = m_session['prediction_row_index'].to_numpy(dtype=int)
        eligible_mask = (prediction_rows >= lower) & (prediction_rows <= upper)

        evaluable = bool(eligible_mask.any())
        detected = False
        earliest_prediction_row = np.nan
        earliest_prediction_time = pd.NaT
        lead_time_seconds = np.nan

        if evaluable:
            p_session = m_session['pred'].to_numpy()
            positive_mask = eligible_mask & (p_session == '1')

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
            'evaluable': bool(evaluable),
            'detected_pre_alarm': bool(detected),
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

                if (wend >= lower) and (wstart <= upper):
                    matched_warning_keys.add(int(widx))

    event_details = pd.DataFrame(event_detail_rows)

    if warning_events.empty:
        warning_events = pd.DataFrame(columns=[
            'session_id',
            'warning_number_within_session',
            'warning_start_prediction_row',
            'warning_end_prediction_row',
            'warning_start_time',
            'warning_end_time',
            'warning_duration_samples',
            'warning_duration_seconds',
        ])

    warning_events = warning_events.copy()
    warning_events['matched_to_true_alarm'] = False

    if matched_warning_keys:
        warning_events.loc[
            warning_events.index.isin(matched_warning_keys),
            'matched_to_true_alarm'
        ] = True

    false_warning_events = warning_events[
        ~warning_events['matched_to_true_alarm']
    ].copy()

    n_total = int(len(event_details))

    if n_total > 0:
        evaluable_mask = event_details['evaluable'].astype(bool)
        detected_mask = (
            evaluable_mask
            & event_details['detected_pre_alarm'].astype(bool)
        )
        n_evaluable = int(evaluable_mask.sum())
        n_detected = int(detected_mask.sum())
    else:
        n_evaluable = 0
        n_detected = 0

    n_missed = int(max(n_evaluable - n_detected, 0))
    detection_rate = n_detected / n_evaluable if n_evaluable > 0 else np.nan

    if n_total > 0:
        lead_times = event_details.loc[
            event_details['detected_pre_alarm'] == True,
            'lead_time_seconds'
        ].dropna().to_numpy(dtype=float)
    else:
        lead_times = np.asarray([], dtype=float)

    hours = evaluated_hours(metadata, sampling_seconds=sampling_seconds)
    n_sessions = represented_session_count(metadata)
    n_false_warnings = int(len(false_warning_events))

    summary = {
        'AlarmEpisodesTotal': n_total,
        'AlarmEpisodesEvaluable': n_evaluable,
        'AlarmEpisodesDetected': n_detected,
        'AlarmEpisodesMissed': n_missed,
        'AlarmEpisodeDetectionRate': float(detection_rate)
            if np.isfinite(detection_rate) else np.nan,
        'AlarmEpisodeDetectionPercent': float(100.0 * detection_rate)
            if np.isfinite(detection_rate) else np.nan,
        'PredictedWarningEpisodes': int(len(warning_events)),
        'FalseWarningEpisodes': n_false_warnings,
        'FalseWarningsPerSession': float(n_false_warnings / n_sessions)
            if n_sessions > 0 else np.nan,
        'FalseWarningsPerTreatmentHour': float(n_false_warnings / hours)
            if hours > 0 else np.nan,
        'LeadTimeN': int(len(lead_times)),
        'LeadTimeMeanSeconds': float(np.mean(lead_times))
            if len(lead_times) else np.nan,
        'LeadTimeMedianSeconds': float(np.median(lead_times))
            if len(lead_times) else np.nan,
        'LeadTimeStdSeconds': float(np.std(lead_times))
            if len(lead_times) else np.nan,
        'LeadTimeMinSeconds': float(np.min(lead_times))
            if len(lead_times) else np.nan,
        'LeadTimeQ25Seconds': float(np.percentile(lead_times, 25))
            if len(lead_times) else np.nan,
        'LeadTimeQ75Seconds': float(np.percentile(lead_times, 75))
            if len(lead_times) else np.nan,
        'LeadTimeMaxSeconds': float(np.max(lead_times))
            if len(lead_times) else np.nan,
    }

    return summary, event_details, warning_events, false_warning_events


def print_event_metrics(prefix, metrics):
    """Print all event-level metrics."""
    print("\n" + "=" * 100)
    print(f"{prefix.upper()} EVENT-LEVEL METRICS")
    print("=" * 100)
    print(f"Total alarm episodes        -> {metrics['AlarmEpisodesTotal']}")
    print(f"Evaluable alarm episodes    -> {metrics['AlarmEpisodesEvaluable']}")
    print(f"Detected alarm episodes     -> {metrics['AlarmEpisodesDetected']}")
    print(f"Missed alarm episodes       -> {metrics['AlarmEpisodesMissed']}")
    print(
        f"Episode detection rate      -> "
        f"{metrics['AlarmEpisodeDetectionPercent']:.2f}%"
    )
    print(f"Predicted warning episodes  -> {metrics['PredictedWarningEpisodes']}")
    print(f"False warning episodes      -> {metrics['FalseWarningEpisodes']}")
    print(f"False warnings / session    -> {metrics['FalseWarningsPerSession']:.4f}")
    print(
        f"False warnings / hour       -> "
        f"{metrics['FalseWarningsPerTreatmentHour']:.4f}"
    )
    print(f"Lead-time observations      -> {metrics['LeadTimeN']}")
    print(f"Lead time mean (s)          -> {metrics['LeadTimeMeanSeconds']:.2f}")
    print(f"Lead time median (s)        -> {metrics['LeadTimeMedianSeconds']:.2f}")
    print(f"Lead time std (s)           -> {metrics['LeadTimeStdSeconds']:.2f}")
    print(f"Lead time min (s)           -> {metrics['LeadTimeMinSeconds']:.2f}")
    print(f"Lead time Q25 (s)           -> {metrics['LeadTimeQ25Seconds']:.2f}")
    print(f"Lead time Q75 (s)           -> {metrics['LeadTimeQ75Seconds']:.2f}")
    print(f"Lead time max (s)           -> {metrics['LeadTimeMaxSeconds']:.2f}")


def plot_lead_time_distribution(event_details, title):
    """Plot and save lead-time distribution for successfully detected episodes."""
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


# ==================================================================================================
# ALARM-CODE-SPECIFIC EVENT ANALYSIS
# ==================================================================================================

def extract_true_alarm_episodes_with_code(df, session_ids=None):
    """
    Extract true alarm episodes stratified by machine alarm code.

    A new code-specific episode starts when:
      - type changes from alarm to non-alarm,
      - or the alarm code changes while type remains alarm.
    """
    if 'code' not in df.columns:
        return pd.DataFrame(columns=[
            'session_id', 'code', 'event_start_row', 'event_end_row'
        ])

    rows = []

    if session_ids is None:
        session_ids = sorted(df['session_id'].unique())

    for session_id in session_ids:
        s = df[df['session_id'] == session_id].copy()
        s = s.sort_values('_original_order').reset_index(drop=False)

        global_rows = s['index'].to_numpy(dtype=int)
        type_arr = s['type'].to_numpy(dtype=int)
        code_arr = (
            s['code']
            .fillna('UNKNOWN')
            .astype(str)
            .str.strip()
            .to_numpy()
        )

        current_start_local = None
        current_code = None

        for local_idx in range(len(s)):
            is_alarm = type_arr[local_idx] == 1
            current_row_code = code_arr[local_idx]

            if not is_alarm:
                if current_start_local is not None:
                    rows.append({
                        'session_id': int(session_id),
                        'code': current_code,
                        'event_start_row': int(global_rows[current_start_local]),
                        'event_end_row': int(global_rows[local_idx - 1]),
                    })
                    current_start_local = None
                    current_code = None
                continue

            if current_start_local is None:
                current_start_local = local_idx
                current_code = current_row_code

            elif current_row_code != current_code:
                rows.append({
                    'session_id': int(session_id),
                    'code': current_code,
                    'event_start_row': int(global_rows[current_start_local]),
                    'event_end_row': int(global_rows[local_idx - 1]),
                })
                current_start_local = local_idx
                current_code = current_row_code

        if current_start_local is not None:
            rows.append({
                'session_id': int(session_id),
                'code': current_code,
                'event_start_row': int(global_rows[current_start_local]),
                'event_end_row': int(global_rows[-1]),
            })

    return pd.DataFrame(rows)


def build_code_detection_table(
    df,
    metadata,
    predictions,
    prediction_gap,
    subset_name,
    sampling_seconds=SAMPLING_SECONDS
):
    """
    Build the alarm-code-specific event table for one hold-out subset.

    Output columns:
      - Code
      - Episodes in validation/test
      - Detected episodes
      - Detection rate (%)
      - Median lead time (s)

    Only evaluable true alarm episodes are included in the denominator.
    """
    subset_name = str(subset_name).strip().lower()
    if subset_name not in {'validation', 'test'}:
        raise ValueError("subset_name deve essere 'validation' oppure 'test'.")

    episode_column = f'Episodes in {subset_name}'

    predictions = np.asarray(predictions).astype(str)
    metadata = metadata.reset_index(drop=True).copy()

    if len(predictions) != len(metadata):
        raise ValueError("predictions e metadata devono avere la stessa lunghezza.")

    metadata['prediction'] = predictions
    metadata = metadata[metadata['same_session'].astype(bool)].copy()

    if metadata.empty or 'code' not in df.columns:
        empty_table = pd.DataFrame(columns=[
            'Code', episode_column, 'Detected episodes',
            'Detection rate (%)', 'Median lead time (s)'
        ])
        return empty_table, pd.DataFrame()

    represented_sessions = sorted(
        metadata['prediction_session_id'].unique().astype(int)
    )

    true_episodes = extract_true_alarm_episodes_with_code(
        df,
        session_ids=represented_sessions
    )

    event_rows = []

    for _, event in true_episodes.iterrows():
        sid = int(event['session_id'])
        code = str(event['code'])
        start_row = int(event['event_start_row'])
        end_row = int(event['event_end_row'])

        lower = start_row - int(prediction_gap)
        upper = start_row - 1

        eligible = metadata[
            (metadata['prediction_session_id'] == sid)
            & (metadata['prediction_row_index'] >= lower)
            & (metadata['prediction_row_index'] <= upper)
        ].copy()

        evaluable = not eligible.empty
        detected = False
        earliest_prediction_row = np.nan
        lead_time_seconds = np.nan

        if evaluable:
            positive = eligible[eligible['prediction'] == '1'].copy()
            detected = not positive.empty

            if detected:
                earliest_prediction_row = int(
                    positive['prediction_row_index'].min()
                )
                lead_time_seconds = float(
                    (start_row - earliest_prediction_row) * sampling_seconds
                )

        event_rows.append({
            'Subset': subset_name,
            'Session ID': sid,
            'Code': code,
            'Event start row': start_row,
            'Event end row': end_row,
            'Evaluable': bool(evaluable),
            'Detected': bool(detected),
            'Earliest warning row': earliest_prediction_row,
            'Lead time (s)': lead_time_seconds,
        })

    event_details = pd.DataFrame(event_rows)

    if event_details.empty:
        empty_table = pd.DataFrame(columns=[
            'Code', episode_column, 'Detected episodes',
            'Detection rate (%)', 'Median lead time (s)'
        ])
        return empty_table, event_details

    evaluable_details = event_details[
        event_details['Evaluable'] == True
    ].copy()

    if evaluable_details.empty:
        empty_table = pd.DataFrame(columns=[
            'Code', episode_column, 'Detected episodes',
            'Detection rate (%)', 'Median lead time (s)'
        ])
        return empty_table, event_details

    grouped_rows = []

    for code, group in evaluable_details.groupby('Code', dropna=False):
        n_episodes = int(len(group))
        n_detected = int(group['Detected'].sum())

        detection_rate = (
            100.0 * n_detected / n_episodes if n_episodes > 0 else np.nan
        )

        detected_leads = group.loc[
            group['Detected'] == True,
            'Lead time (s)'
        ].dropna()

        median_lead = (
            float(detected_leads.median())
            if len(detected_leads) > 0 else np.nan
        )

        grouped_rows.append({
            'Code': code,
            episode_column: n_episodes,
            'Detected episodes': n_detected,
            'Detection rate (%)': float(detection_rate),
            'Median lead time (s)': median_lead,
        })

    table = pd.DataFrame(grouped_rows)
    table = table.sort_values(
        by=[episode_column, 'Code'],
        ascending=[False, True]
    ).reset_index(drop=True)

    return table, event_details


def print_code_table(table, subset_name):
    """Pretty-print one alarm-code table without changing numeric CSV values."""
    print("\n" + "=" * 110)
    print(f"RANDOM FOREST - {subset_name.upper()} PERFORMANCE BY ALARM CODE")
    print("=" * 110)

    if table.empty:
        print(f"Nessun episodio di allarme valutabile nel {subset_name} set.")
        return

    table_to_print = table.copy()

    table_to_print['Detection rate (%)'] = (
        table_to_print['Detection rate (%)']
        .map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
    )

    table_to_print['Median lead time (s)'] = (
        table_to_print['Median lead time (s)']
        .map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
    )

    print(table_to_print.to_string(index=False))


# ==================================================================================================
# TEST-ONLY ALARM-CODE REPORT FOR RANDOM FOREST
# ==================================================================================================
def build_rf_test_only_alarm_code_report(metadata, y_true, predictions, code_event_details):
    """
    Build the alarm-code report using ONLY the 15% TEST subset.

    Sample-level quantities:
      - Alarm samples in test: true alarm samples actually assigned to TEST;
      - Detected alarm samples in test: TEST alarm samples predicted as alarm;
      - Alarm-sample detection rate: detected TEST alarm samples / TEST alarm samples.

    Event-level quantities:
      - True alarm episodes are first reconstructed as contiguous code-specific runs
        in the original ordered dataset by build_code_detection_table().
      - Each complete true episode is assigned to TEST only when the sequence sample
        corresponding to the episode onset (event_start_row) belongs to TEST.
        This gives each full episode one and only one subset assignment and avoids
        counting all full-dataset episodes merely because TEST contains some points
        in their pre-alarm window.
      - Detected episodes and lead times are then calculated only for those episodes
        assigned to TEST, using TEST predictions only.
    """
    metadata = metadata.reset_index(drop=True).copy()
    y_true = np.asarray(y_true).astype(str)
    predictions = np.asarray(predictions).astype(str)

    if not (len(metadata) == len(y_true) == len(predictions)):
        raise ValueError("metadata, y_true e predictions devono avere la stessa lunghezza.")

    metadata['y_true'] = y_true
    metadata['prediction'] = predictions
    metadata['Code'] = (
        metadata['target_code']
        .fillna('UNKNOWN')
        .astype(str)
        .str.strip()
    )

    # ----------------------------------------------------------------------------------------------
    # SAMPLE-LEVEL: ONLY samples actually assigned to TEST
    # ----------------------------------------------------------------------------------------------
    alarm_test = metadata[metadata['y_true'] == '1'].copy()

    sample_rows = []
    if not alarm_test.empty:
        for code, g in alarm_test.groupby('Code', dropna=False):
            n_alarm_samples = int(len(g))
            n_detected_samples = int((g['prediction'] == '1').sum())

            sample_rows.append({
                'Code': str(code),
                'Alarm samples in test': n_alarm_samples,
                'Detected alarm samples in test': n_detected_samples,
                'Alarm-sample detection rate (%)': (
                    100.0 * n_detected_samples / n_alarm_samples
                    if n_alarm_samples > 0 else np.nan
                ),
            })

    sample_table = pd.DataFrame(sample_rows)

    # ----------------------------------------------------------------------------------------------
    # EVENT-LEVEL: complete contiguous episodes assigned ONLY to TEST
    # ----------------------------------------------------------------------------------------------
    # Each row of meta_test corresponds to one sequence sample actually assigned to TEST.
    # target_row_index identifies the real row whose class/code is the prediction target.
    test_target_rows = set(
        metadata['target_row_index'].dropna().astype(int).tolist()
    )

    episode_rows = []

    if code_event_details is not None and not code_event_details.empty:
        test_episodes = code_event_details[
            code_event_details['Event start row'].astype(int).isin(test_target_rows)
        ].copy()

        if not test_episodes.empty:
            for code, g in test_episodes.groupby('Code', dropna=False):
                detected_g = g[g['Detected'] == True].copy()
                leads = detected_g['Lead time (s)'].dropna().astype(float)

                episode_rows.append({
                    'Code': str(code),
                    'Alarm episodes in test': int(len(g)),
                    'Detected episodes': int(len(detected_g)),
                    'Mean lead time (s)': float(leads.mean()) if len(leads) else np.nan,
                    'Median lead time (s)': float(leads.median()) if len(leads) else np.nan,
                    'Min lead time (s)': float(leads.min()) if len(leads) else np.nan,
                    'Max lead time (s)': float(leads.max()) if len(leads) else np.nan,
                })

    episode_table = pd.DataFrame(episode_rows)

    # ----------------------------------------------------------------------------------------------
    # MERGE SAMPLE-LEVEL + EVENT-LEVEL TEST RESULTS
    # ----------------------------------------------------------------------------------------------
    expected_columns = [
        'Code',
        'Alarm samples in test',
        'Detected alarm samples in test',
        'Alarm-sample detection rate (%)',
        'Alarm episodes in test',
        'Detected episodes',
        'Mean lead time (s)',
        'Median lead time (s)',
        'Min lead time (s)',
        'Max lead time (s)',
    ]

    if sample_table.empty and episode_table.empty:
        return pd.DataFrame(columns=expected_columns)

    if sample_table.empty:
        final = episode_table.copy()
        final['Alarm samples in test'] = 0
        final['Detected alarm samples in test'] = 0
        final['Alarm-sample detection rate (%)'] = np.nan

    elif episode_table.empty:
        final = sample_table.copy()
        final['Alarm episodes in test'] = 0
        final['Detected episodes'] = 0
        final['Mean lead time (s)'] = np.nan
        final['Median lead time (s)'] = np.nan
        final['Min lead time (s)'] = np.nan
        final['Max lead time (s)'] = np.nan

    else:
        final = sample_table.merge(episode_table, on='Code', how='outer')

    # Integer count columns: missing means zero observations in TEST for that code.
    for col in [
        'Alarm samples in test',
        'Detected alarm samples in test',
        'Alarm episodes in test',
        'Detected episodes',
    ]:
        if col not in final.columns:
            final[col] = 0
        final[col] = final[col].fillna(0).astype(int)

    if 'Alarm-sample detection rate (%)' not in final.columns:
        final['Alarm-sample detection rate (%)'] = np.nan

    # Keep exactly the requested column order.
    final = final[expected_columns]

    final = final.sort_values(
        by=['Alarm samples in test', 'Code'],
        ascending=[False, True]
    ).reset_index(drop=True)

    return final

def print_rf_test_only_alarm_code_report(table):
    """Print only TEST sample counts, detected episodes and lead-time statistics."""
    print("\n" + "=" * 150)
    print("RANDOM FOREST - TEST-ONLY PERFORMANCE BY ALARM CODE")
    print("=" * 150)

    if table.empty:
        print("Nessun alarm sample disponibile nel test set.")
        return

    table_to_print = table.copy()
    float_cols = [
        'Alarm-sample detection rate (%)',
        'Mean lead time (s)',
        'Median lead time (s)',
        'Min lead time (s)',
        'Max lead time (s)',
    ]

    for col in float_cols:
        table_to_print[col] = table_to_print[col].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "NA"
        )

    print(table_to_print.to_string(index=False))


# ==================================================================================================
# CONFUSION MATRIX PLOT
# ==================================================================================================

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=['0', '1'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        annot_kws={'size': 14}
    )
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    save_current_plot(filename)
    plt.show()


# ==================================================================================================
# LOAD AND PREPROCESS DATA
# ==================================================================================================

print("\n" + "=" * 100)
print("LOAD AND PREPROCESS DATA")
print("=" * 100)

# Same dataset loading as the original script.
df = pd.read_csv(CSV_PATH)

# Same column rename as the original script.
df = df.rename(columns={'dT(CP)': 'dT'})

# Binary target definition.
# The target column may also contain missing values or values stored with
# different string formatting. We normalize the valid labels first and then
# remove only rows for which the binary target cannot be determined.
raw_type = df['type'].copy()

df['type'] = (
    df['type']
    .astype('string')
    .str.strip()
    .str.lower()
    .replace({
        'alarm': '1',
        'normal': '0',
        'override': '0',
    })
)

# Convert both textual labels and already-numeric 0/1 values safely.
df['type'] = pd.to_numeric(df['type'], errors='coerce')

invalid_target_mask = df['type'].isna()
if invalid_target_mask.any():
    n_invalid = int(invalid_target_mask.sum())
    invalid_examples = (
        raw_type.loc[invalid_target_mask]
        .astype('string')
        .drop_duplicates()
        .head(10)
        .tolist()
    )
    print(
        f"\nATTENZIONE: {n_invalid} righe con target 'type' mancante/non valido "
        "sono state escluse prima della creazione delle sequenze."
    )
    print(f"Esempi di valori esclusi: {invalid_examples}")
    df = df.loc[~invalid_target_mask].copy()

# At this point only valid binary labels remain.
df['type'] = df['type'].astype(int)
df = df.reset_index(drop=True)
df['_original_order'] = np.arange(len(df))

# Check that all original RF features are available.
missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(
        "Nel CSV mancano le seguenti feature:\n" + "\n".join(missing_features)
    )

# The 'code' column is needed only for the alarm-type-specific table.
if 'code' not in df.columns:
    print(
        "\nATTENZIONE: colonna 'code' non trovata. "
        "Le tabelle per tipologia di allarme non potranno essere calcolate."
    )

# Reconstruct COMPLETE treatments from the time column for sequence generation and splitting.
# A treatment changes ONLY when the gap from the previous row is greater than 30 minutes.
df = add_treatment_ids(
    df,
    time_column='time',
    gap_minutes=TREATMENT_GAP_MINUTES
)

# Reconstruct sessions only for the pre-existing event-level calculations.
# This original event-metric logic is intentionally kept separate from treatment-based splitting.
df = add_session_ids_for_metrics(
    df,
    time_column='time',
    gap_minutes=SESSION_GAP_MINUTES
)

session_summary = print_session_summary(df)


# ==================================================================================================
# CREATE TEMPORAL SEQUENCES
# ==================================================================================================

print("\n" + "=" * 100)
print("CREATE TEMPORAL SEQUENCES")
print("=" * 100)
print(f"N_STEPS = {N_STEPS} rows ({INPUT_WINDOW_SECONDS:.1f} s)")
print(
    f"PREDICTION_GAP = {PREDICTION_GAP} rows "
    f"({PREDICTION_HORIZON_SECONDS:.1f} s)"
)

X_all, y_all, meta_all = create_splitted_df_with_metadata(
    df,
    n_steps=N_STEPS,
    prediction_gap=PREDICTION_GAP
)

y_all = y_all.astype(int).astype(str)
all_indices = np.arange(len(y_all))

print(f"\nX_all shape: {X_all.shape}")
print(f"y_all shape: {y_all.shape}")
check_classes("ALL SAMPLES", y_all)


# ==================================================================================================
# TRAIN / VALIDATION / TEST SPLIT - RANDOMIZED AT COMPLETE-TREATMENT LEVEL
# ==================================================================================================
# The original sample-level stratified split is replaced by a grouped random split.
#
# Fundamental rule:
#     one complete treatment -> one subset only
#
# Treatment IDs are randomized reproducibly using SPLIT_SEED. The algorithm explores several
# random treatment assignments and keeps the one that best approximates 70 / 15 / 15 in terms
# of the number of generated temporal sequences. The split is therefore NOT chronological.
(
    idx_train,
    idx_val,
    idx_test,
    train_treatments,
    val_treatments,
    test_treatments,
) = treatment_based_split_indices(
    metadata=meta_all,
    y=y_all,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    random_state=SPLIT_SEED,
    n_random_trials=SPLIT_RANDOM_TRIALS,
)

# ----------------------------------
# Build Train / Validation / Test arrays
# ----------------------------------
X_train, y_train = X_all[idx_train], y_all[idx_train]
X_val, y_val = X_all[idx_val], y_all[idx_val]
X_test, y_test = X_all[idx_test], y_all[idx_test]

# Metadata must follow exactly the same temporal-sample indices used for X and y.
meta_train = meta_all.iloc[idx_train].reset_index(drop=True)
meta_val = meta_all.iloc[idx_val].reset_index(drop=True)
meta_test = meta_all.iloc[idx_test].reset_index(drop=True)

# ----------------------------------
# Verify class composition
# ----------------------------------
check_classes("TRAIN", y_train)
check_classes("VALIDATION", y_val)
check_classes("TEST", y_test)

# ----------------------------------
# Split summary and integrity information
# ----------------------------------
print("\n" + "=" * 100)
print("TRAIN / VALIDATION / TEST SUMMARY - COMPLETE TREATMENTS")
print("=" * 100)
print(
    f"TRAIN      : {len(y_train)} samples ({100 * len(y_train) / len(y_all):.2f}%) | "
    f"{len(train_treatments)} treatments"
)
print(
    f"VALIDATION : {len(y_val)} samples ({100 * len(y_val) / len(y_all):.2f}%) | "
    f"{len(val_treatments)} treatments"
)
print(
    f"TEST       : {len(y_test)} samples ({100 * len(y_test) / len(y_all):.2f}%) | "
    f"{len(test_treatments)} treatments"
)
print("Random treatment-level partition completed successfully.")
print("No treatment is shared across Train, Validation and Test.")

# Print treatment IDs for complete reproducibility and easy manual verification.
print(f"Train treatment IDs      : {sorted(train_treatments.tolist())}")
print(f"Validation treatment IDs : {sorted(val_treatments.tolist())}")
print(f"Test treatment IDs       : {sorted(test_treatments.tolist())}")

# ----------------------------------
# Save split assignment for reproducibility
# ----------------------------------
# The additional treatment_id column allows direct verification that every treatment appears
# in one and only one subset. No downstream model logic depends on this CSV file.
split_df = pd.DataFrame({
    'global_sample_index': np.concatenate([idx_train, idx_val, idx_test]),
    'subset': (
        ['train'] * len(idx_train)
        + ['validation'] * len(idx_val)
        + ['test'] * len(idx_test)
    )
}).sort_values('global_sample_index').reset_index(drop=True)

split_df['treatment_id'] = meta_all.iloc[
    split_df['global_sample_index'].to_numpy(dtype=int)
]['treatment_id'].to_numpy()

split_df.to_csv('sample_split.csv', index=False)


# ==================================================================================================
# RESHAPE FOR RANDOM FOREST
# ==================================================================================================

nsamples, nx, ny = X_train.shape
df_train_dataset = X_train.reshape((nsamples, nx * ny))

nsamples2, nx2, ny2 = X_val.shape
df_val_dataset = X_val.reshape((nsamples2, nx2 * ny2))

nsamples3, nx3, ny3 = X_test.shape
df_test_dataset = X_test.reshape((nsamples3, nx3 * ny3))

print("\n" + "=" * 100)
print("RESHAPED DATA")
print("=" * 100)
print("TRAIN      :", df_train_dataset.shape, y_train.shape)
print("VALIDATION :", df_val_dataset.shape, y_val.shape)
print("TEST       :", df_test_dataset.shape, y_test.shape)


# ==================================================================================================
# RANDOM FOREST MODEL
# ==================================================================================================
# Exact same configuration as the original script supplied in chat.

LR_T100 = RandomForestClassifier(
    n_estimators=137,
    max_depth=27,
    min_samples_split=4,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

print("\n" + "=" * 100)
print("RANDOM FOREST CONFIGURATION")
print("=" * 100)
print("n_estimators      = 137")
print("max_depth         = 27")
print("min_samples_split = 4")
print("min_samples_leaf  = 3")
print("max_features      = sqrt")
print("class_weight      = balanced")
print("random_state      = 42")


# ==================================================================================================
# TRAINING
# ==================================================================================================

print("\n" + "=" * 100)
print("TRAINING")
print("=" * 100)

LR_T100.fit(df_train_dataset, y_train)

print(
    "Random Forest Score on Training set -> ",
    LR_T100.score(df_train_dataset, y_train)
)


# ==================================================================================================
# TRAINING SET EVALUATION
# ==================================================================================================

pred_train = LR_T100.predict(df_train_dataset)

plot_confusion_matrix(
    y_train,
    pred_train,
    title='Train Confusion Matrix',
    filename='RandomForest_Train_Confusion_Matrix'
)

metrics_train = calculate_metrics(
    LR_T100,
    df_train_dataset,
    y_train,
    pred_train,
    meta_train
)

print_metrics("TRAINING METRICS", metrics_train)


# ==================================================================================================
# VALIDATION SET EVALUATION
# ==================================================================================================

print(
    "\nRandom Forest Classifier Score on Validation set -> ",
    LR_T100.score(df_val_dataset, y_val)
)

pred_val = LR_T100.predict(df_val_dataset)

plot_confusion_matrix(
    y_val,
    pred_val,
    title='Validation Confusion Matrix',
    filename='RandomForest_Validation_Confusion_Matrix'
)

metrics_val = calculate_metrics(
    LR_T100,
    df_val_dataset,
    y_val,
    pred_val,
    meta_val
)

print_metrics("VALIDATION METRICS", metrics_val)

# ----------------------------------
# Validation event-level analysis
# ----------------------------------
(
    val_event_metrics,
    val_event_details,
    val_warning_details,
    val_false_warnings
) = evaluate_event_level(
    pred_val,
    meta_val,
    df,
    prediction_gap_rows=PREDICTION_GAP,
    sampling_seconds=SAMPLING_SECONDS
)

print_event_metrics("Validation", val_event_metrics)

plot_lead_time_distribution(
    val_event_details,
    "RandomForest - Validation lead-time distribution"
)

# ----------------------------------
# Validation alarm-code table
# ----------------------------------
val_code_table, val_code_event_details = build_code_detection_table(
    df=df,
    metadata=meta_val,
    predictions=pred_val,
    prediction_gap=PREDICTION_GAP,
    subset_name='validation',
    sampling_seconds=SAMPLING_SECONDS
)

print_code_table(val_code_table, 'validation')


# ==================================================================================================
# TEST SET EVALUATION
# ==================================================================================================

print(
    "\nRandom Forest Classifier Score on Test set -> ",
    LR_T100.score(df_test_dataset, y_test)
)

pred_test = LR_T100.predict(df_test_dataset)

plot_confusion_matrix(
    y_test,
    pred_test,
    title='Test Confusion Matrix',
    filename='RandomForest_Test_Confusion_Matrix'
)

metrics_test = calculate_metrics(
    LR_T100,
    df_test_dataset,
    y_test,
    pred_test,
    meta_test
)

print_metrics("TEST METRICS", metrics_test)

# ----------------------------------
# Test event-level analysis
# ----------------------------------
(
    test_event_metrics,
    test_event_details,
    test_warning_details,
    test_false_warnings
) = evaluate_event_level(
    pred_test,
    meta_test,
    df,
    prediction_gap_rows=PREDICTION_GAP,
    sampling_seconds=SAMPLING_SECONDS
)

print_event_metrics("Test", test_event_metrics)

plot_lead_time_distribution(
    test_event_details,
    "RandomForest - Test lead-time distribution"
)

# ----------------------------------
# Test alarm-code table
# ----------------------------------
test_code_table_event, test_code_event_details = build_code_detection_table(
    df=df,
    metadata=meta_test,
    predictions=pred_test,
    prediction_gap=PREDICTION_GAP,
    subset_name='test',
    sampling_seconds=SAMPLING_SECONDS
)

# TEST-only table shown to the user: no total episode count from the full dataset.
test_code_table = build_rf_test_only_alarm_code_report(
    metadata=meta_test,
    y_true=y_test,
    predictions=pred_test,
    code_event_details=test_code_event_details
)

print_rf_test_only_alarm_code_report(test_code_table)


# ==================================================================================================
# FINAL SUMMARY TABLES
# ==================================================================================================

summary_columns = [
    'Model',
    # Sample-level metrics
    'Accuracy',
    'F1',
    'Precision',
    'Sensitivity',
    'Specificity',
    'AUROC',
    'AUPRC',
    'FalseAlarmEpisodes',
    'EvaluatedHours',
    'FalseAlarmsPerHour',
    'TN',
    'FP',
    'FN',
    'TP',
    # Event-level metrics
    'AlarmEpisodesTotal',
    'AlarmEpisodesEvaluable',
    'AlarmEpisodesDetected',
    'AlarmEpisodesMissed',
    'AlarmEpisodeDetectionRate',
    'AlarmEpisodeDetectionPercent',
    'PredictedWarningEpisodes',
    'FalseWarningEpisodes',
    'FalseWarningsPerSession',
    'FalseWarningsPerTreatmentHour',
    'LeadTimeN',
    'LeadTimeMeanSeconds',
    'LeadTimeMedianSeconds',
    'LeadTimeStdSeconds',
    'LeadTimeMinSeconds',
    'LeadTimeQ25Seconds',
    'LeadTimeQ75Seconds',
    'LeadTimeMaxSeconds',
]

validation_summary = pd.DataFrame([{
    'Model': 'RandomForest',
    **metrics_val,
    **val_event_metrics,
}])

test_summary = pd.DataFrame([{
    'Model': 'RandomForest',
    **metrics_test,
    **test_event_metrics,
}])

print("\n" + "=" * 120)
print("FINAL VALIDATION RESULTS")
print("=" * 120)
print(validation_summary[summary_columns].to_string(index=False))

print("\n" + "=" * 120)
print("FINAL TEST RESULTS")
print("=" * 120)
print(test_summary[summary_columns].to_string(index=False))


# ==================================================================================================
# SAVE NUMERIC RESULTS AND DETAILED EVENT TABLES
# ==================================================================================================

validation_summary[summary_columns].to_csv(
    'RF_validation_metrics_complete.csv',
    index=False
)

test_summary[summary_columns].to_csv(
    'RF_test_metrics_complete.csv',
    index=False
)

# True alarm episode details.
val_event_details.to_csv(
    'RF_validation_alarm_event_details.csv',
    index=False
)

test_event_details.to_csv(
    'RF_test_alarm_event_details.csv',
    index=False
)

# All predicted warning episodes.
val_warning_details.to_csv(
    'RF_validation_warning_event_details.csv',
    index=False
)

test_warning_details.to_csv(
    'RF_test_warning_event_details.csv',
    index=False
)

# False-warning-only tables.
val_false_warnings.to_csv(
    'RF_validation_false_warning_event_details.csv',
    index=False
)

test_false_warnings.to_csv(
    'RF_test_false_warning_event_details.csv',
    index=False
)

# Alarm-code summary tables and detailed code-specific episodes.
val_code_table.to_csv(
    'RF_validation_performance_by_code.csv',
    index=False
)

val_code_event_details.to_csv(
    'RF_validation_alarm_event_details_by_code.csv',
    index=False
)

test_code_table.to_csv(
    'RF_test_performance_by_code.csv',
    index=False
)

test_code_event_details.to_csv(
    'RF_test_alarm_event_details_by_code.csv',
    index=False
)

# Additional TEST-only file containing only episodes actually detected by the RF.
test_code_event_details[
    test_code_event_details['Detected'] == True
].to_csv(
    'RF_test_detected_alarm_event_details_by_code.csv',
    index=False
)


# ==================================================================================================
# ADDITIONAL RESULT PLOTS
# ==================================================================================================

# ----------------------------------
# Sample-level metrics: Train / Validation / Test
# ----------------------------------
sample_metric_names = [
    'Accuracy',
    'F1',
    'Precision',
    'Sensitivity',
    'Specificity',
    'AUROC',
    'AUPRC',
]

subset_metric_dict = {
    'Train': metrics_train,
    'Validation': metrics_val,
    'Test': metrics_test,
}

x = np.arange(len(sample_metric_names))
width = 0.25

plt.figure(figsize=(13, 6))
for j, (subset_name, metric_dict) in enumerate(subset_metric_dict.items()):
    values = [metric_dict[m] for m in sample_metric_names]
    plt.bar(x + (j - 1) * width, values, width=width, label=subset_name)

plt.xticks(x, sample_metric_names, rotation=30, ha='right')
plt.ylim(0, 1.05)
plt.ylabel('Score')
plt.title('Random Forest - Sample-level performance')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
save_current_plot('RandomForest_All_Sample_Level_Metrics')
plt.show()


# ----------------------------------
# False alarm episodes per evaluated hour
# ----------------------------------
plt.figure(figsize=(8, 5))
subset_names = list(subset_metric_dict.keys())
fa_values = [
    subset_metric_dict[s]['FalseAlarmsPerHour']
    for s in subset_names
]
plt.bar(subset_names, fa_values)
plt.ylabel('False alarm episodes / evaluated hour')
plt.title('Random Forest - False alarm episodes per evaluated hour')
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
save_current_plot('RandomForest_False_Alarms_Per_Hour')
plt.show()


# ----------------------------------
# Event-level detection rate: Validation / Test
# ----------------------------------
event_subsets = ['Validation', 'Test']
event_detection = [
    val_event_metrics['AlarmEpisodeDetectionPercent'],
    test_event_metrics['AlarmEpisodeDetectionPercent'],
]

plt.figure(figsize=(7, 5))
plt.bar(event_subsets, event_detection)
plt.ylabel('Detected alarm episodes (%)')
plt.ylim(0, 105)
plt.title('Random Forest - Event-level alarm detection')
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
save_current_plot('RandomForest_Event_Level_Detection_Percent')
plt.show()


# ----------------------------------
# Detected vs missed alarm episodes
# ----------------------------------
val_detected = val_event_metrics['AlarmEpisodesDetected']
val_missed = val_event_metrics['AlarmEpisodesMissed']
test_detected = test_event_metrics['AlarmEpisodesDetected']
test_missed = test_event_metrics['AlarmEpisodesMissed']

x_event = np.arange(2)
width_event = 0.35

plt.figure(figsize=(8, 5))
plt.bar(
    x_event - width_event / 2,
    [val_detected, test_detected],
    width_event,
    label='Detected'
)
plt.bar(
    x_event + width_event / 2,
    [val_missed, test_missed],
    width_event,
    label='Missed'
)
plt.xticks(x_event, ['Validation', 'Test'])
plt.ylabel('Number of evaluable alarm episodes')
plt.title('Random Forest - Detected vs missed alarm episodes')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
save_current_plot('RandomForest_Detected_vs_Missed_Alarm_Episodes')
plt.show()


# ----------------------------------
# False warnings per represented session
# ----------------------------------
plt.figure(figsize=(7, 5))
plt.bar(
    event_subsets,
    [
        val_event_metrics['FalseWarningsPerSession'],
        test_event_metrics['FalseWarningsPerSession'],
    ]
)
plt.ylabel('False warning episodes / session')
plt.title('Random Forest - False warning episodes per session')
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
save_current_plot('RandomForest_False_Warnings_Per_Session')
plt.show()


# ----------------------------------
# False warnings per evaluated hour
# ----------------------------------
plt.figure(figsize=(7, 5))
plt.bar(
    event_subsets,
    [
        val_event_metrics['FalseWarningsPerTreatmentHour'],
        test_event_metrics['FalseWarningsPerTreatmentHour'],
    ]
)
plt.ylabel('False warning episodes / evaluated hour')
plt.title('Random Forest - False warning episodes per evaluated hour')
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
save_current_plot('RandomForest_False_Warnings_Per_Hour')
plt.show()


# ----------------------------------
# Lead-time summary statistics
# ----------------------------------
lead_metric_names = [
    'LeadTimeMinSeconds',
    'LeadTimeQ25Seconds',
    'LeadTimeMeanSeconds',
    'LeadTimeMedianSeconds',
    'LeadTimeQ75Seconds',
    'LeadTimeMaxSeconds',
]

lead_labels = ['Min', 'Q25', 'Mean', 'Median', 'Q75', 'Max']
x_lead = np.arange(len(lead_labels))
lead_width = 0.35

val_lead = [val_event_metrics[m] for m in lead_metric_names]
test_lead = [test_event_metrics[m] for m in lead_metric_names]

plt.figure(figsize=(11, 6))
plt.bar(x_lead - lead_width / 2, val_lead, lead_width, label='Validation')
plt.bar(x_lead + lead_width / 2, test_lead, lead_width, label='Test')
plt.xticks(x_lead, lead_labels)
plt.ylabel('Lead time (s)')
plt.title('Random Forest - Lead-time summary')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
save_current_plot('RandomForest_Lead_Time_Summary')
plt.show()


# ----------------------------------
# Alarm-code-specific plots: Validation
# ----------------------------------
if not val_code_table.empty:
    val_plot_alarm = val_code_table.sort_values(
        'Detection rate (%)',
        ascending=True
    )

    plt.figure(figsize=(11, max(6, 0.35 * len(val_plot_alarm))))
    plt.barh(
        val_plot_alarm['Code'],
        val_plot_alarm['Detection rate (%)']
    )
    plt.xlabel('Detection rate (%)')
    plt.xlim(0, 105)
    plt.ylabel('Code')
    plt.title('Random Forest - Validation detection rate by alarm code')
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('RandomForest_Validation_Detection_Rate_By_Alarm_Code')
    plt.show()

    val_plot_counts = val_code_table.sort_values(
        'Episodes in validation',
        ascending=True
    )

    y_pos = np.arange(len(val_plot_counts))
    h = 0.38

    plt.figure(figsize=(11, max(6, 0.35 * len(val_plot_counts))))
    plt.barh(
        y_pos - h / 2,
        val_plot_counts['Episodes in validation'],
        height=h,
        label='Episodes in validation'
    )
    plt.barh(
        y_pos + h / 2,
        val_plot_counts['Detected episodes'],
        height=h,
        label='Detected episodes'
    )
    plt.yticks(y_pos, val_plot_counts['Code'])
    plt.xlabel('Number of episodes')
    plt.ylabel('Code')
    plt.title('Random Forest - Validation alarm episodes by code')
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('RandomForest_Validation_Episodes_By_Alarm_Code')
    plt.show()


# ----------------------------------
# Alarm-code-specific plots: Test
# ----------------------------------
if not test_code_table_event.empty:
    test_plot_alarm = test_code_table_event.sort_values(
        'Detection rate (%)',
        ascending=True
    )

    plt.figure(figsize=(11, max(6, 0.35 * len(test_plot_alarm))))
    plt.barh(
        test_plot_alarm['Code'],
        test_plot_alarm['Detection rate (%)']
    )
    plt.xlabel('Detection rate (%)')
    plt.xlim(0, 105)
    plt.ylabel('Code')
    plt.title('Random Forest - Test detection rate by alarm code')
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('RandomForest_Test_Detection_Rate_By_Alarm_Code')
    plt.show()

    test_plot_counts = test_code_table_event.sort_values(
        'Episodes in test',
        ascending=True
    )

    y_pos = np.arange(len(test_plot_counts))
    h = 0.38

    plt.figure(figsize=(11, max(6, 0.35 * len(test_plot_counts))))
    plt.barh(
        y_pos - h / 2,
        test_plot_counts['Episodes in test'],
        height=h,
        label='Episodes in test'
    )
    plt.barh(
        y_pos + h / 2,
        test_plot_counts['Detected episodes'],
        height=h,
        label='Detected episodes'
    )
    plt.yticks(y_pos, test_plot_counts['Code'])
    plt.xlabel('Number of episodes')
    plt.ylabel('Code')
    plt.title('Random Forest - Test alarm episodes by code')
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_current_plot('RandomForest_Test_Episodes_By_Alarm_Code')
    plt.show()


# ==================================================================================================
# FEATURE IMPORTANCE
# ==================================================================================================
# Original flattened-feature importance plot retained.

importances = LR_T100.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(
    range(df_train_dataset.shape[1]),
    importances[indices]
)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
save_current_plot('RandomForest_Feature_Importances')
plt.show()


# ==================================================================================================
# FINAL OUTPUT LIST AND METRIC DEFINITIONS
# ==================================================================================================

print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)
print(" - sample_split.csv")
print(" - session_summary_for_event_metrics.csv")
print(" - RF_validation_metrics_complete.csv")
print(" - RF_test_metrics_complete.csv")
print(" - RF_validation_alarm_event_details.csv")
print(" - RF_test_alarm_event_details.csv")
print(" - RF_validation_warning_event_details.csv")
print(" - RF_test_warning_event_details.csv")
print(" - RF_validation_false_warning_event_details.csv")
print(" - RF_test_false_warning_event_details.csv")
print(" - RF_validation_performance_by_code.csv")
print(" - RF_validation_alarm_event_details_by_code.csv")
print(" - RF_test_performance_by_code.csv")
print(" - RF_test_alarm_event_details_by_code.csv")
print(f" - plots/*.png")

print("\n" + "=" * 100)
print("METRIC DEFINITIONS")
print("=" * 100)
print("Sensitivity = Recall of the alarm class.")
print("Specificity = TN / (TN + FP).")
print("AUROC = Area Under the Receiver Operating Characteristic curve.")
print("AUPRC = Average Precision / area-summary of the Precision-Recall curve.")
print(
    "FalseAlarmEpisodes = contiguous false-positive prediction episodes among "
    "valid evaluated records; non-adjacent original rows start a new episode."
)
print(
    "FalseAlarmsPerHour = FalseAlarmEpisodes divided by the effective evaluated "
    "hold-out duration represented by valid prediction records."
)
print(
    "AlarmEpisodeDetectionPercent = percentage of evaluable distinct true alarm "
    "episodes with at least one positive prediction before alarm onset within the "
    "predefined prediction horizon."
)
print(
    "FalseWarningsPerSession = unmatched predicted-positive warning episodes divided "
    "by the number of reconstructed sessions represented by the hold-out records."
)
print(
    "FalseWarningsPerTreatmentHour = unmatched predicted-positive warning episodes "
    "divided by the effective evaluated hold-out duration."
)
print(
    "LeadTime*Seconds = distribution of the time between the earliest useful warning "
    "and the start of the corresponding true alarm episode."
)
print(
    f"Current temporal configuration: ns={N_STEPS} rows "
    f"({INPUT_WINDOW_SECONDS:.1f} s), Delta_t={PREDICTION_GAP} rows "
    f"({PREDICTION_HORIZON_SECONDS:.1f} s)."
)