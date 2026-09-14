# ==================================================================================================
# ALARM PREDICTION - COMPARISON OF 10 MACHINE LEARNING MODELS
# ==================================================================================================
#
# Analysis workflow:
#   1. Dataset loading and preprocessing
#   2. Temporal sequence generation
#   3. Randomized Train / Validation / Test partition by complete treatment (approximately 70 / 15 / 15)
#   4. Evaluation of 10 hyperparameter configurations for each model
#   5. Selection of the best configuration according to Validation F1 Score
#   6. 5-fold cross-validation of the selected configuration for each model
#   7. Final fit and independent Validation / Test evaluation
#   8. Sample-level and event-level performance analysis
#   9. Automatic saving of figures, configurations, metrics and detailed outputs
#
# Models evaluated:
#   - Random Forest
#   - Extra Trees
#   - AdaBoost
#   - HistGradientBoosting
#   - Logistic Regression
#   - Ridge Classifier
#   - XGBoost
#   - LightGBM
#   - Multi-Layer Perceptron (MLP)
#   - K-Nearest Neighbors (KNN)
#
# Performance metrics:
#   - Accuracy
#   - F1 Score
#   - Precision
#   - Sensitivity (Recall of the alarm class)
#   - Specificity
#   - AUROC
#   - AUPRC
#   - False-alarm episodes per evaluated hour
#   - TN / FP / FN / TP
#
# Additional event-level metrics:
#   - Number and percentage of distinct alarm episodes detected
#   - False-warning episodes
#   - False warnings per session
#   - False warnings per evaluated hour
#   - Prediction lead-time distribution and summary statistics
#
# IMPORTANT:
# The configuration that proceeds after Phase 1 is selected according to
# the maximum F1 Score obtained on the Validation set, consistently with the
# reference workflow. The other metrics are reported for characterization only.
# ==================================================================================================



# ==================================================================================================
# IMPORTS
# ==================================================================================================

# General-purpose libraries
from numpy import array, hstack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
warnings.filterwarnings("ignore")

# Scikit-learn utilities
from sklearn.base import clone
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Performance metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Machine-learning models
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==================================================================================================
# GENERAL SETTINGS
# ==================================================================================================

# ---------------------- Reproducibility ----------------------
# Fixed seed used throughout the analysis whenever a reproducible operation is required.
RANDOM_STATE = 42

# ---------------------- Temporal configuration ----------------------
# N_STEPS: number of consecutive rows used as input by the model.
# PREDICTION_GAP: number of rows between the last known row and the future target condition.
# SAMPLING_SECONDS: nominal acquisition interval of the dataset.
N_STEPS = 10
PREDICTION_GAP = 20
SAMPLING_SECONDS = 0.5

# Derived temporal quantities, used only for printing/documentation.
INPUT_WINDOW_SECONDS = N_STEPS * SAMPLING_SECONDS
PREDICTION_HORIZON_SECONDS = PREDICTION_GAP * SAMPLING_SECONDS

# ---------------------- Train / Validation / Test proportions ----------------------
# The split is performed at TREATMENT level, not at individual-sample level.
# Complete treatments are randomly assigned to Train / Validation / Test while targeting
# approximately 70 / 15 / 15 of the generated temporal sequences.
#
# IMPORTANT:
#   - a treatment is NEVER divided across different subsets;
#   - treatment assignment is random but reproducible through RANDOM_STATE;
#   - exact percentages can differ slightly from 70 / 15 / 15 because treatments are indivisible.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Number of random treatment-level assignments explored when looking for the split that
# best approximates the requested sample proportions. This affects ONLY the dataset split.
SPLIT_RANDOM_TRIALS = 2000

# ---------------------- Treatment reconstruction for dataset partitioning ----------------------
# A new treatment starts when the time gap from the previous row is greater than 30 minutes.
# This identifier controls sequence generation and the Train / Validation / Test split.
TREATMENT_GAP_MINUTES = 30

# ---------------------- Session reconstruction for event-level metrics ----------------------
# Session identity is reconstructed only to evaluate contiguous alarm/warning events and lead times.
# This logic is intentionally left unchanged from the original script.
SESSION_GAP_MINUTES = 60

# ---------------------- Input / output paths ----------------------
CSV_PATH = r"C:\Users\dvella\Desktop\Alarm_paper\Definitive_FINISH_All_months_ready_for_training_.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ==================================================================================================
# INPUT FEATURES
# ==================================================================================================
# 35 input variables used.
# The temporal window therefore contains N_STEPS x len(FEATURES) values before flattening.
FEATURES = [
    'dmchild', 'dmparent', 'pt4', 'pt3', 'delivPumpActuation', 'pt5',
    'foamDetResult', 'tmp', 'diffFlow', 'bld', 'encoderDelivPump', 'delFlow',
    'currentWeightLoss', 'arterial_revolve', 'ufPressureActuation', 'condDO',
    'condDOnf', 'encoderUFPump', 'pt6', 'Ctot', 'infusion_revolve', 'bmparent',
    'venous_revolve', 'bmchild', 'condTot', 'cond2', 'pt8', 'pVenous',
    'pPreFilt', 'encoder1', 'ts3', 'ts1', 'ts2', 'SecondStepPumpSpeed',
    'airDetAnalogSensor'
]


# ==================================================================================================
# BASIC FUNCTIONS - SEQUENCE GENERATION AND TEMPORAL METADATA
# ==================================================================================================
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
    """Convert time while preserving the original row order for ties."""
    df = df.copy()
    if '_original_order' not in df.columns:
        df['_original_order'] = np.arange(len(df))

    if pd.api.types.is_datetime64_any_dtype(df[time_column]):
        return df

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
            "Some values in the 'time' column cannot be converted. "
            f"Examples: {bad}"
        )

    df[time_column] = parsed
    return df


def add_treatment_ids(df, time_column='time', gap_minutes=30):
    """
    Reconstruct complete treatments from the time column.

    A new treatment begins whenever the time difference from the previous row
    is greater than ``gap_minutes``. Treatment reconstruction itself is deterministic:
    randomization is applied later, only when complete treatments are assigned to
    Train / Validation / Test.
    """
    if time_column not in df.columns:
        raise ValueError(
            f"The '{time_column}' column is required to reconstruct treatments."
        )

    df = convert_time_column(df, time_column=time_column)
    delta = df[time_column].diff()
    new_treatment = delta > pd.Timedelta(minutes=gap_minutes)
    new_treatment.iloc[0] = True
    df['treatment_id'] = new_treatment.cumsum().astype(int)

    print(
        f"\nTreatments reconstructed from '{time_column}' "
        f"(gap > {gap_minutes} min): {df['treatment_id'].nunique()}"
    )
    return df


def add_session_ids_for_metrics(df, time_column='time', gap_minutes=60):
    """
    Reconstruct session_id ONLY for event-level/episode metrics.

    This function does not control train/validation/test partitioning.
    Partitioning is controlled separately by treatment_id.

    Preferred logic (when hash + time are available):
      new session when hash changes OR time gap > gap_minutes.

    Fallbacks:
      - if only hash exists: each hash is treated as a session;
      - otherwise: all rows are treated as one pseudo-session.
    """
    df = df.copy()
    if '_original_order' not in df.columns:
        df['_original_order'] = np.arange(len(df))

    if time_column in df.columns:
        df = convert_time_column(df, time_column=time_column)

    if 'hash' in df.columns and time_column in df.columns:
        df['hash'] = df['hash'].astype(str)
        # Preserve global dataset order. Session boundaries are detected on that order.
        hash_change = df['hash'].ne(df['hash'].shift(1))
        delta = df[time_column].diff()
        large_gap = delta > pd.Timedelta(minutes=gap_minutes)
        new_session = hash_change | large_gap
        new_session.iloc[0] = True
        df['session_id'] = new_session.cumsum().astype(int)

    elif 'hash' in df.columns:
        codes, _ = pd.factorize(df['hash'].astype(str), sort=False)
        df['session_id'] = codes + 1

    else:
        df['session_id'] = 1

    print(f"\nSessions reconstructed ONLY for event-level metrics: {df['session_id'].nunique()}")
    print("NOTE: these sessions are used only for event-level metrics.")
    return df


def create_splitted_df_with_metadata(df, n_steps, prediction_gap=20):
    """
    Generate temporal sequences independently inside each complete treatment.

    The feature/target alignment is exactly the same as in the original script:
    the model receives ``n_steps`` consecutive rows and the target is taken
    ``prediction_gap`` rows after the last input row. The only change is that a
    sequence is never allowed to cross a treatment boundary.
    """
    if 'treatment_id' not in df.columns:
        raise ValueError(
            "The 'treatment_id' column is missing. "
            "Run add_treatment_ids() first."
        )

    X_parts, y_parts, metadata_rows = [], [], []
    global_sample_index = 0
    skipped_treatments = 0

    # sort=False preserves the chronological order in which treatments appear in the dataset.
    for treatment_id, treatment_df in df.groupby('treatment_id', sort=False):
        treatment_df = treatment_df.copy()

        # At least prediction_gap + n_steps rows are needed to create one sample.
        if len(treatment_df) < prediction_gap + n_steps:
            skipped_treatments += 1
            continue

        historical_df = treatment_df.iloc[:-prediction_gap]
        stacked_features = [
            historical_df[col].to_numpy().reshape((len(historical_df), 1))
            for col in FEATURES
        ]
        alarm_type = treatment_df['type'].to_numpy()[prediction_gap:].reshape(
            (len(treatment_df) - prediction_gap, 1)
        )
        dataset = hstack(stacked_features + [alarm_type])
        X_treatment, y_treatment = split_sequences(dataset, n_steps)

        if len(X_treatment) == 0:
            skipped_treatments += 1
            continue

        X_parts.append(X_treatment)
        y_parts.append(y_treatment.astype(int))

        # Map treatment-local sequence positions back to global dataframe row indices.
        global_rows = treatment_df.index.to_numpy(dtype=int)
        for local_i in range(len(X_treatment)):
            prediction_local_index = local_i + n_steps - 1
            target_local_index = prediction_local_index + prediction_gap
            prediction_row_index = int(global_rows[prediction_local_index])
            target_row_index = int(global_rows[target_local_index])

            pred_sid = int(df.loc[prediction_row_index, 'session_id'])
            target_sid = int(df.loc[target_row_index, 'session_id'])

            metadata_rows.append({
                'global_sample_index': int(global_sample_index),
                'treatment_id': int(treatment_id),
                'prediction_row_index': prediction_row_index,
                'target_row_index': target_row_index,
                'prediction_session_id': pred_sid,
                'target_session_id': target_sid,
                'same_session': bool(pred_sid == target_sid),
                'prediction_time': df.loc[prediction_row_index, 'time']
                    if 'time' in df.columns else pd.NaT,
                'target_time': df.loc[target_row_index, 'time']
                    if 'time' in df.columns else pd.NaT,
            })
            global_sample_index += 1

    if not X_parts:
        raise ValueError(
            "No treatment contains enough rows to generate sequences "
            f"con N_STEPS={n_steps} e PREDICTION_GAP={prediction_gap}."
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0).astype(int)
    metadata = pd.DataFrame(metadata_rows)

    if skipped_treatments > 0:
        print(
            f"Treatments excluded from sequence generation because they are too short: "
            f"{skipped_treatments}"
        )

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
    Randomly split COMPLETE treatments into Train / Validation / Test.

    The function operates only on ``treatment_id`` groups. Individual temporal
    sequences are never independently shuffled between subsets. Therefore, every
    sequence generated from the same treatment is assigned to exactly one of:
    Train, Validation or Test.

    Strategy
    --------
    1. Count how many valid temporal sequences belong to each treatment.
    2. Randomly shuffle the treatment IDs using ``random_state``.
    3. For each random ordering, choose two treatment boundaries that approximate
       the requested 70 / 15 / 15 sample proportions as closely as possible.
    4. Repeat the procedure ``n_random_trials`` times and retain the best split.
    5. If ``y`` is provided, prefer candidate splits containing both classes in
       Train, Validation and Test whenever such a split is possible.

    Notes
    -----
    The split is NOT chronological. Treatments from different dates can therefore
    appear in the same subset. However, each treatment always remains intact.
    Exact 70 / 15 / 15 proportions are not guaranteed because treatments have
    different lengths and cannot be divided.
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
    # Number of generated temporal sequences belonging to each treatment.
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

    # Optional class counts per treatment. These are used ONLY to prefer splits
    # in which all three subsets contain both class 0 and class 1.
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

    best_valid = None       # Best split with both classes in all subsets.
    best_valid_score = np.inf
    best_any = None         # Fallback: best ratio match regardless of class presence.
    best_any_score = np.inf

    for _ in range(n_random_trials):
        # Random treatment order. Only treatment IDs are shuffled; samples are not split.
        permutation = rng.permutation(n_treatments)
        ids_perm = treatment_ids[permutation]
        counts_perm = counts[permutation]
        cumulative_counts = np.cumsum(counts_perm)

        # ------------------------------ Train boundary ------------------------------
        # Keep at least one complete treatment for Validation and one for Test.
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
        # The second boundary targets Train + Validation = 85% of all samples.
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

        # Actual sample counts generated by this complete-treatment assignment.
        train_n = int(counts_perm[:train_cut].sum())
        val_n = int(counts_perm[train_cut:second_cut].sum())
        test_n = int(counts_perm[second_cut:].sum())
        actual_counts = np.array([train_n, val_n, test_n], dtype=float)

        # Score = total absolute deviation from the desired 70 / 15 / 15 proportions.
        # Dividing by total_samples makes the score independent of dataset size.
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

        # If labels are available, check whether all subsets contain both classes.
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

    # Prefer a split containing both classes in all subsets. If this is not possible
    # within the explored assignments, fall back to the split closest to 70 / 15 / 15.
    selected = best_valid if best_valid is not None else best_any
    if selected is None:
        raise RuntimeError("Unable to generate a treatment-based Train/Validation/Test split.")

    train_treatments, val_treatments, test_treatments, actual_counts = selected

    # ----------------------------------------------------------------------------------------------
    # Convert treatment assignments back to sample indices
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
    values, counts = np.unique(y, return_counts=True)
    print(f"\n{name}")
    print(f"Number of samples: {len(y)}")
    for value, count in zip(values, counts):
        print(f"  Class {value}: {count} ({100.0 * count / len(y):.2f}%)")
    if len(values) < 2:
        raise ValueError(f"{name} contains only one class.")


# ==================================================================================================
# SAMPLE-LEVEL PERFORMANCE METRICS
# ==================================================================================================
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else np.nan


def get_continuous_score(model, X):
    """Continuous score for AUROC/AUPRC."""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        if 1 not in classes:
            raise ValueError("Positive class 1 is not present in model.classes_.")
        pos_idx = int(np.where(classes == 1)[0][0])
        return np.asarray(proba)[:, pos_idx]

    if hasattr(model, 'decision_function'):
        score = np.asarray(model.decision_function(X))
        if score.ndim == 2:
            classes = np.asarray(model.classes_)
            pos_idx = int(np.where(classes == 1)[0][0])
            score = score[:, pos_idx]
        return score.ravel()

    return np.asarray(model.predict(X), dtype=float).ravel()


def _valid_metadata_for_episode_metrics(metadata):
    """Exclude sequence samples whose prediction/target cross a reconstructed session boundary."""
    if metadata is None or len(metadata) == 0:
        return pd.DataFrame()
    m = metadata.reset_index(drop=True).copy()
    if 'same_session' in m.columns:
        m = m[m['same_session'].astype(bool)].copy()
    return m


def count_false_alarm_episodes(y_true, y_pred, metadata):
    """
    Count contiguous false-positive episodes within reconstructed sessions.

    Two hold-out samples are considered temporally contiguous only when their
    original prediction_row_index values differ by exactly 1. This avoids
    joining non-adjacent samples into the same false-positive episode.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metadata = metadata.reset_index(drop=True).copy()

    if not (len(y_true) == len(y_pred) == len(metadata)):
        raise ValueError("y_true, y_pred e metadata devono avere la stessa lunghezza.")

    work = metadata.copy()
    work['y_true'] = y_true
    work['y_pred'] = y_pred
    work = work[work.get('same_session', True) == True].copy()

    total_episodes = 0
    for sid, g in work.groupby('prediction_session_id'):
        g = g.sort_values('prediction_row_index').reset_index(drop=True)
        rows = g['prediction_row_index'].to_numpy(dtype=int)
        fp = ((g['y_true'].to_numpy() == 0) & (g['y_pred'].to_numpy() == 1))

        if len(g) == 0:
            continue

        prev_contiguous_fp = np.r_[
            False,
            fp[:-1] & (rows[1:] == rows[:-1] + 1)
        ]
        starts = fp & ~prev_contiguous_fp
        total_episodes += int(starts.sum())

    return total_episodes


def evaluated_hours_from_samples(metadata, sampling_seconds=SAMPLING_SECONDS):
    """
    Effective evaluated sample-time in hours.

    The hold-out subset may contain only part of each reconstructed treatment
    trajectory. Therefore the denominator is the number
    of valid evaluated prediction records multiplied by the sampling interval.
    """
    m = _valid_metadata_for_episode_metrics(metadata)
    return float(len(m) * sampling_seconds / 3600.0)


def evaluate_model(model, X, y, metadata):
    """Compute all requested sample-level metrics on a hold-out set."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(model.predict(X)).astype(int)
    score = np.asarray(get_continuous_score(model, X), dtype=float)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred, pos_label=1, zero_division=0)
    precision = precision_score(y, pred, pos_label=1, zero_division=0)
    sensitivity = recall_score(y, pred, pos_label=1, zero_division=0)
    specificity = specificity_score(y, pred)

    if len(np.unique(y)) >= 2:
        auroc = roc_auc_score(y, score)
        auprc = average_precision_score(y, score)
    else:
        auroc = np.nan
        auprc = np.nan

    false_alarm_episodes = count_false_alarm_episodes(y, pred, metadata)
    hours = evaluated_hours_from_samples(metadata)
    false_alarms_per_hour = false_alarm_episodes / hours if hours > 0 else np.nan

    return {
        'Accuracy': float(accuracy),
        'F1': float(f1),
        'Precision': float(precision),
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'AUROC': float(auroc) if np.isfinite(auroc) else np.nan,
        'AUPRC': float(auprc) if np.isfinite(auprc) else np.nan,
        'FalseAlarmEpisodes': int(false_alarm_episodes),
        'TreatmentHours': float(hours),
        'FalseAlarmsPerHour': float(false_alarms_per_hour)
            if np.isfinite(false_alarms_per_hour) else np.nan,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'pred': pred,
        'score': score,
    }


def print_metrics(prefix, metrics):
    print(
        f"{prefix:<11} -> "
        f"Acc={metrics['Accuracy']:.4f} | "
        f"F1={metrics['F1']:.4f} | "
        f"P={metrics['Precision']:.4f} | "
        f"Sens={metrics['Sensitivity']:.4f} | "
        f"Spec={metrics['Specificity']:.4f} | "
        f"AUROC={metrics['AUROC']:.4f} | "
        f"AUPRC={metrics['AUPRC']:.4f} | "
        f"FA/h={metrics['FalseAlarmsPerHour']:.4f} | "
        f"FA episodes={metrics['FalseAlarmEpisodes']} | "
        f"Eval hours={metrics['TreatmentHours']:.4f} | "
        f"TN={metrics['TN']} | FP={metrics['FP']} | "
        f"FN={metrics['FN']} | TP={metrics['TP']}"
    )


# ==================================================================================================
# EVENT-LEVEL PERFORMANCE METRICS
# ==================================================================================================
def _contiguous_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return list(zip(starts.tolist(), ends.tolist()))


def extract_true_alarm_events(df, session_ids=None):
    """Distinct true alarm episode = contiguous run of type==1 within one session."""
    rows = []

    if session_ids is None:
        session_ids = sorted(df['session_id'].unique())

    for session_id in session_ids:
        s = df[df['session_id'] == session_id].copy()
        s = s.sort_values('_original_order').reset_index(drop=False)

        # Map session-local positions back to GLOBAL dataframe row indices.
        global_rows = s['index'].to_numpy(dtype=int)
        runs = _contiguous_true_runs(s['type'].to_numpy() == 1)

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
                'event_start_time': s.loc[start_local, 'time']
                    if 'time' in s.columns else pd.NaT,
                'event_end_time': s.loc[end_local, 'time']
                    if 'time' in s.columns else pd.NaT,
            })

    return pd.DataFrame(rows)


def extract_predicted_warning_episodes(pred, metadata):
    """
    Warning episode = contiguous run of predicted class 1 in ORIGINAL row space.

    A new episode is forced whenever the
    next hold-out prediction row is not exactly the next original row.
    """
    pred = np.asarray(pred).astype(int)
    metadata = metadata.reset_index(drop=True).copy()

    if len(pred) != len(metadata):
        raise ValueError("pred e metadata devono avere la stessa lunghezza.")

    metadata['pred'] = pred
    metadata = metadata[metadata.get('same_session', True) == True].copy()
    rows_out = []

    for session_id, g in metadata.groupby('prediction_session_id'):
        g = g.sort_values('prediction_row_index').reset_index(drop=True)
        if g.empty:
            continue

        p = g['pred'].to_numpy(dtype=int)
        original_rows = g['prediction_row_index'].to_numpy(dtype=int)

        in_run = False
        run_start = None
        run_end = None
        warning_number = 0

        for j in range(len(g)):
            is_pos = p[j] == 1
            contiguous_with_previous = (
                j > 0 and original_rows[j] == original_rows[j - 1] + 1
            )

            if is_pos and (not in_run or not contiguous_with_previous):
                # Close old run if a gap started a new positive run.
                if in_run:
                    warning_number += 1
                    gs = g.iloc[run_start]
                    ge = g.iloc[run_end]
                    rows_out.append({
                        'session_id': int(session_id),
                        'warning_number_within_session': int(warning_number),
                        'warning_start_prediction_row': int(gs['prediction_row_index']),
                        'warning_end_prediction_row': int(ge['prediction_row_index']),
                        'warning_start_time': gs['prediction_time'],
                        'warning_end_time': ge['prediction_time'],
                        'warning_duration_samples': int(run_end - run_start + 1),
                        'warning_duration_seconds': float(
                            (run_end - run_start + 1) * SAMPLING_SECONDS
                        ),
                    })
                in_run = True
                run_start = j
                run_end = j

            elif is_pos and in_run and contiguous_with_previous:
                run_end = j

            elif not is_pos and in_run:
                warning_number += 1
                gs = g.iloc[run_start]
                ge = g.iloc[run_end]
                rows_out.append({
                    'session_id': int(session_id),
                    'warning_number_within_session': int(warning_number),
                    'warning_start_prediction_row': int(gs['prediction_row_index']),
                    'warning_end_prediction_row': int(ge['prediction_row_index']),
                    'warning_start_time': gs['prediction_time'],
                    'warning_end_time': ge['prediction_time'],
                    'warning_duration_samples': int(run_end - run_start + 1),
                    'warning_duration_seconds': float(
                        (run_end - run_start + 1) * SAMPLING_SECONDS
                    ),
                })
                in_run = False
                run_start = None
                run_end = None

        if in_run:
            warning_number += 1
            gs = g.iloc[run_start]
            ge = g.iloc[run_end]
            rows_out.append({
                'session_id': int(session_id),
                'warning_number_within_session': int(warning_number),
                'warning_start_prediction_row': int(gs['prediction_row_index']),
                'warning_end_prediction_row': int(ge['prediction_row_index']),
                'warning_start_time': gs['prediction_time'],
                'warning_end_time': ge['prediction_time'],
                'warning_duration_samples': int(run_end - run_start + 1),
                'warning_duration_seconds': float(
                    (run_end - run_start + 1) * SAMPLING_SECONDS
                ),
            })

    return pd.DataFrame(rows_out)


def evaluate_event_level(
    pred,
    metadata,
    df,
    prediction_gap_rows=PREDICTION_GAP,
    sampling_seconds=SAMPLING_SECONDS
):
    """
    Event-level evaluation on the hold-out prediction records.

    A true alarm episode is evaluable only if at least one hold-out prediction
    record from that same reconstructed session falls in the pre-alarm window.

    Successful detection:
        event_start - prediction_gap_rows <= prediction_row < event_start
        and predicted class == 1.

    Lead time uses the EARLIEST positive hold-out prediction in that window.
    """
    pred = np.asarray(pred).astype(int)
    metadata = metadata.reset_index(drop=True).copy()
    if len(pred) != len(metadata):
        raise ValueError("pred e metadata devono avere la stessa lunghezza.")

    metadata['pred'] = pred
    metadata_valid = metadata[metadata.get('same_session', True) == True].copy()

    represented_sessions = sorted(metadata_valid['prediction_session_id'].unique())
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
        lead_time_seconds = np.nan
        earliest_prediction_time = pd.NaT

        if evaluable:
            p_session = m_session['pred'].to_numpy(dtype=int)
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

    if len(event_details) > 0:
        evaluable_mask = event_details['evaluable'].astype(bool)
        detected_mask = evaluable_mask & event_details['detected_pre_alarm'].astype(bool)
        n_total = len(event_details)
        n_evaluable = int(evaluable_mask.sum())
        n_detected = int(detected_mask.sum())
    else:
        n_total = 0
        n_evaluable = 0
        n_detected = 0

    detection_rate = n_detected / n_evaluable if n_evaluable > 0 else np.nan

    if len(event_details) > 0 and 'detected_pre_alarm' in event_details.columns:
        lead_times = event_details.loc[
            event_details['detected_pre_alarm'] == True,
            'lead_time_seconds'
        ].dropna().to_numpy(dtype=float)
    else:
        lead_times = np.asarray([], dtype=float)

    hours = evaluated_hours_from_samples(metadata, sampling_seconds=sampling_seconds)
    n_sessions = len(represented_sessions)
    n_false_warnings = len(false_warning_events)

    summary = {
        'AlarmEpisodesTotal': int(n_total),
        'AlarmEpisodesEvaluable': int(n_evaluable),
        'AlarmEpisodesDetected': int(n_detected),
        'AlarmEpisodesMissed': int(max(n_evaluable - n_detected, 0)),
        'AlarmEpisodeDetectionRate': float(detection_rate)
            if np.isfinite(detection_rate) else np.nan,
        'AlarmEpisodeDetectionPercent': float(100.0 * detection_rate)
            if np.isfinite(detection_rate) else np.nan,
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
    det_pct = metrics['AlarmEpisodeDetectionPercent']
    med_lt = metrics['LeadTimeMedianSeconds']
    print(
        f"{prefix} EVENT-LEVEL -> "
        f"alarm episodes detected={metrics['AlarmEpisodesDetected']}/"
        f"{metrics['AlarmEpisodesEvaluable']} "
        f"({det_pct:.2f}% if defined) | "
        f"false warnings/session={metrics['FalseWarningsPerSession']:.4f} | "
        f"false warnings/h={metrics['FalseWarningsPerTreatmentHour']:.4f} | "
        f"lead time median={med_lt:.2f} s"
    )


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


# ==================================================================================================
# MODEL AND HYPERPARAMETER UTILITIES
# ==================================================================================================
def _materialize_numeric(spec):
    t = spec.get('type')
    if t not in ('int', 'float'):
        raise ValueError("spec['type'] must be 'int' or 'float'.")

    vmin = spec['min']
    vmax = spec['max']
    allow_none = spec.get('allow_none', False)

    if t == 'int':
        if 'step' in spec:
            vals = list(range(int(vmin), int(vmax) + 1, int(spec['step'])))
        else:
            num = int(spec.get('num', 5))
            vals = list(np.linspace(int(vmin), int(vmax), num=num, dtype=int))
            vals = sorted(list(dict.fromkeys(vals)))
    else:
        num = int(spec.get('num', 5))
        scale = spec.get('scale', 'linear')
        if scale == 'log':
            vmin_eff = max(vmin, 1e-12)
            vals = list(np.logspace(np.log10(vmin_eff), np.log10(vmax), num=num))
        else:
            vals = list(np.linspace(vmin, vmax, num=num))
        vals = [float(f"{v:.6f}") for v in vals]

    if allow_none:
        vals = vals + [None]
    return vals


def materialize_grid_from_ranges(range_spec: dict) -> dict:
    out = {}
    for param, spec in range_spec.items():
        if isinstance(spec, dict) and 'type' in spec:
            out[param] = _materialize_numeric(spec)
        elif isinstance(spec, list):
            out[param] = spec[:]
        else:
            raise ValueError(f"Invalid specification for {param}: {spec}")
    return out


def build_pipe(model):
    model_copy = clone(model)
    return Pipeline([('classifier', model_copy)]) if not isinstance(model_copy, Pipeline) else model_copy


def sample_param_configs(param_grid_lists, n_samples=10, random_state=42):
    """Sample a fixed number of reproducible configurations from the parameter grid."""
    grid = list(ParameterGrid(param_grid_lists))
    if len(grid) <= n_samples:
        return grid
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(grid), size=n_samples, replace=False)
    return [grid[i] for i in idx]


# ==================================================================================================
# DATASET LOADING AND PREPROCESSING
# ==================================================================================================
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={'dT(CP)': 'dT'})
df['type'] = df['type'].replace({'alarm': 1, 'normal': 0, 'override': 0})
df['type'] = pd.to_numeric(df['type'], errors='coerce')
df = df.dropna(subset=['type']).copy().reset_index(drop=True)
df['type'] = df['type'].astype(int)
df['_original_order'] = np.arange(len(df))

missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(
        "The following required features are missing from the CSV:\n" + "\n".join(missing_features)
    )

# Reconstruct complete treatments from time gaps. This identifier is used ONLY
# to keep treatments intact during sequence generation and the 70/15/15 split.
df = add_treatment_ids(
    df,
    time_column='time',
    gap_minutes=TREATMENT_GAP_MINUTES
)

# Session IDs are added ONLY for the extra event/episode metrics.
# Their original reconstruction logic is intentionally unchanged.
df = add_session_ids_for_metrics(
    df,
    time_column='time',
    gap_minutes=SESSION_GAP_MINUTES
)

X_all, y_all, meta_all = create_splitted_df_with_metadata(
    df,
    n_steps=N_STEPS,
    prediction_gap=PREDICTION_GAP
)
y_all = y_all.astype(int)
all_indices = np.arange(len(y_all))

print("\nDataset sequences:", X_all.shape, y_all.shape)
check_classes("ALL SAMPLES", y_all)


# ==================================================================================================
# TRAIN / VALIDATION / TEST SPLIT - RANDOMIZED AT TREATMENT LEVEL
# ==================================================================================================
# The original sample-level random split is replaced by a GROUPED random split:
# complete treatments are randomly assigned to Train, Validation or Test.
#
# Key rule:
#     one treatment -> one subset only
#
# The algorithm explores several reproducible random treatment assignments and selects
# the one that best approximates 70 / 15 / 15 in terms of generated temporal sequences.
# The chronological order of treatments is NOT used to define the three subsets.
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
    random_state=RANDOM_STATE,
    n_random_trials=SPLIT_RANDOM_TRIALS,
)

# --------------------------------------------------------------------------------------------------
# Build Train / Validation / Test arrays
# --------------------------------------------------------------------------------------------------
X_train, y_train = X_all[idx_train], y_all[idx_train]
X_val, y_val = X_all[idx_val], y_all[idx_val]
X_test, y_test = X_all[idx_test], y_all[idx_test]

# Metadata must follow the exact same sample indices as X and y.
meta_train = meta_all.iloc[idx_train].reset_index(drop=True)
meta_val = meta_all.iloc[idx_val].reset_index(drop=True)
meta_test = meta_all.iloc[idx_test].reset_index(drop=True)

# --------------------------------------------------------------------------------------------------
# Verify class composition
# --------------------------------------------------------------------------------------------------
check_classes("TRAIN", y_train)
check_classes("VALIDATION", y_val)
check_classes("TEST", y_test)

# --------------------------------------------------------------------------------------------------
# Split summary
# --------------------------------------------------------------------------------------------------
print("\nTRAIN / VALIDATION / TEST SPLIT")
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

# Print treatment IDs assigned to each subset for full reproducibility and traceability.
print(f"Train treatment IDs      : {sorted(train_treatments.tolist())}")
print(f"Validation treatment IDs : {sorted(val_treatments.tolist())}")
print(f"Test treatment IDs       : {sorted(test_treatments.tolist())}")

# --------------------------------------------------------------------------------------------------
# Save split assignment
# --------------------------------------------------------------------------------------------------
# Save the subset assigned to every generated temporal sequence. The treatment_id column
# makes it easy to verify externally that each treatment belongs to one subset only.
split_df = pd.DataFrame({
    'global_sample_index': np.concatenate([idx_train, idx_val, idx_test]),
    'subset': (
        ['train'] * len(idx_train)
        + ['validation'] * len(idx_val)
        + ['test'] * len(idx_test)
    ),
}).sort_values('global_sample_index').reset_index(drop=True)

split_df['treatment_id'] = meta_all.iloc[
    split_df['global_sample_index'].to_numpy(dtype=int)
]['treatment_id'].to_numpy()

split_df.to_csv('sample_split.csv', index=False)


# ==================================================================================================
# RESHAPE TEMPORAL WINDOWS FOR CONVENTIONAL ML MODELS
# ==================================================================================================
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat = X_val.reshape((X_val.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

print("\nShapes after flattening:")
print("TRAIN      :", X_train_flat.shape, y_train.shape)
print("VALIDATION :", X_val_flat.shape, y_val.shape)
print("TEST       :", X_test_flat.shape, y_test.shape)


# ==================================================================================================
# MODEL DEFINITIONS
# ==================================================================================================
# Same model definitions/order as the second reference script.
models = {
    # ---------------------- Random Forest ----------------------
    'RandomForest': (
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
        None
    ),

    # ---------------------- Extra Trees ----------------------
    'ExtraTrees': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', ExtraTreesClassifier(class_weight='balanced', random_state=RANDOM_STATE))
    ]), None),

    # ---------------------- AdaBoost ----------------------
    'AdaBoost': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', AdaBoostClassifier(random_state=RANDOM_STATE))
    ]), None),

    # ---------------------- HistGradientBoosting ----------------------
    'HistGradientBoosting': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    ]), None),

    # ---------------------- Logistic Regression ----------------------
    'LogisticRegression': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE
        ))
    ]), None),

    # ---------------------- Ridge Classifier ----------------------
    'RidgeClassifier': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RidgeClassifier())
    ]), None),

    # ---------------------- XGBoost ----------------------
    'XGBoost': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            tree_method='hist'
        ))
    ]), None),

    # ---------------------- LightGBM ----------------------
    'LightGBM': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE))
    ]), None),

    # ---------------------- Multi-Layer Perceptron ----------------------
    'MLP': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(max_iter=600, random_state=RANDOM_STATE))
    ]), None),

    # ---------------------- K-Nearest Neighbors ----------------------
    'KNN': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]), None),
}


# ==================================================================================================
# HYPERPARAMETER SEARCH RANGES
# ==================================================================================================
# Same ranges as the second reference script.
param_ranges = {
    # ---------------------- Random Forest ----------------------
    # n_estimators: number of trees; max_depth: maximum tree depth;
    # min_samples_split/min_samples_leaf: minimum samples controlling tree growth;
    # max_features: number of features considered at each split.
    'RandomForest': {
        'classifier__n_estimators': {'type': 'int', 'min': 50, 'max': 400, 'num': 9},
        'classifier__max_depth': {'type': 'int', 'min': 3, 'max': 40, 'num': 7, 'allow_none': True},
        'classifier__min_samples_split': {'type': 'int', 'min': 2, 'max': 10, 'num': 9},
        'classifier__min_samples_leaf': {'type': 'int', 'min': 1, 'max': 5, 'num': 5},
        'classifier__max_features': ['sqrt', 'log2']
    },
    # ---------------------- Extra Trees ----------------------
    # Same tree-complexity controls explored for the Extra Trees ensemble.
    'ExtraTrees': {
        'classifier__n_estimators': {'type': 'int', 'min': 50, 'max': 400, 'num': 9},
        'classifier__max_depth': {'type': 'int', 'min': 3, 'max': 40, 'num': 7, 'allow_none': True},
        'classifier__min_samples_split': {'type': 'int', 'min': 2, 'max': 10, 'num': 9},
        'classifier__min_samples_leaf': {'type': 'int', 'min': 1, 'max': 5, 'num': 5},
        'classifier__max_features': ['sqrt', 'log2']
    },
    # ---------------------- AdaBoost ----------------------
    # n_estimators: number of boosting stages; learning_rate: contribution of each stage;
    # algorithm: boosting algorithm considered.
    'AdaBoost': {
        'classifier__n_estimators': {'type': 'int', 'min': 50, 'max': 400, 'num': 11},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'num': 8, 'scale': 'log'},
        'classifier__algorithm': ['SAMME', 'SAMME.R']
    },
    # ---------------------- HistGradientBoosting ----------------------
    # max_iter: boosting iterations; learning_rate: shrinkage factor;
    # max_depth: tree depth; l2_regularization: regularization strength.
    'HistGradientBoosting': {
        'classifier__max_iter': {'type': 'int', 'min': 100, 'max': 300, 'num': 9},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'num': 6, 'scale': 'log'},
        'classifier__max_depth': {'type': 'int', 'min': 3, 'max': 20, 'num': 4, 'allow_none': True},
        'classifier__l2_regularization': {'type': 'float', 'min': 0.0, 'max': 0.2, 'num': 5}
    },
    # ---------------------- Logistic Regression ----------------------
    # C: inverse regularization strength; penalty: regularization type; solver: optimizer.
    'LogisticRegression': {
        'classifier__C': {'type': 'float', 'min': 0.01, 'max': 10.0, 'num': 8, 'scale': 'log'},
        'classifier__penalty': ['l2', 'none'],
        'classifier__solver': ['lbfgs', 'saga']
    },
    # ---------------------- Ridge Classifier ----------------------
    # alpha: L2 regularization strength; solver: numerical optimization method.
    'RidgeClassifier': {
        'classifier__alpha': {'type': 'float', 'min': 0.1, 'max': 20.0, 'num': 8, 'scale': 'log'},
        'classifier__solver': ['auto', 'lsqr', 'sparse_cg']
    },
    # ---------------------- XGBoost ----------------------
    # n_estimators: number of boosting trees; learning_rate: shrinkage factor;
    # max_depth: tree depth; subsample: fraction of samples used by each tree.
    'XGBoost': {
        'classifier__n_estimators': {'type': 'int', 'min': 50, 'max': 400, 'num': 7},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'num': 6, 'scale': 'log'},
        'classifier__max_depth': {'type': 'int', 'min': 3, 'max': 40, 'num': 6},
        'classifier__subsample': {'type': 'float', 'min': 0.6, 'max': 1.0, 'num': 5},
    },
    # ---------------------- LightGBM ----------------------
    # n_estimators: boosting iterations; learning_rate: shrinkage factor;
    # max_depth: maximum depth; num_leaves: maximum number of leaves per tree.
    'LightGBM': {
        'classifier__n_estimators': {'type': 'int', 'min': 50, 'max': 400, 'num': 7},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'num': 6, 'scale': 'log'},
        'classifier__max_depth': {'type': 'int', 'min': 3, 'max': 40, 'num': 3, 'allow_none': True},
        'classifier__num_leaves': {'type': 'int', 'min': 30, 'max': 260, 'num': 8},
    },
    # ---------------------- Multi-Layer Perceptron ----------------------
    # hidden_layer_sizes: network architecture; activation: hidden-layer activation;
    # learning_rate_init: initial optimization learning rate.
    'MLP': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__learning_rate_init': {'type': 'float', 'min': 0.0005, 'max': 0.02, 'num': 6, 'scale': 'log'},
    },
    # ---------------------- K-Nearest Neighbors ----------------------
    # n_neighbors: number of neighbors; weights: neighbor weighting strategy;
    # metric: distance function.
    'KNN': {
        'classifier__n_neighbors': {'type': 'int', 'min': 3, 'max': 31, 'step': 2},
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }
}


# ==================================================================================================
# PREPARATION OF DISCRETE HYPERPARAMETER GRIDS
# ==================================================================================================
param_grids_discrete = {
    name: materialize_grid_from_ranges(param_ranges[name])
    for name in param_ranges
}


# ==================================================================================================
# PHASE 1 - HOLDOUT VALIDATION: 10 CONFIGURATIONS PER MODEL
# ==================================================================================================
# Each model is evaluated on 10 reproducibly sampled hyperparameter configurations.
# All metrics are computed for every trial, but ONLY Validation F1 Score is used
# to determine which configuration proceeds to the subsequent analysis.
pos_label = 1
per_model_trials = {}
best_configs = {}
best_models = {}
all_boxplot_data = {}

print("\n========== PHASE 1: HOLDOUT VAL, 10 CONFIGURATIONS ==========")

for name, (model, _) in models.items():
    print(f"\n>>> Model: {name}")
    pgrid_lists = param_grids_discrete[name]
    trials = sample_param_configs(pgrid_lists, n_samples=10, random_state=RANDOM_STATE)

    results = []

    for i, params in enumerate(trials, 1):
        pipe_try = None
        try:
            pipe_try = build_pipe(model).set_params(**params)
            pipe_try.fit(X_train_flat, y_train)

            m = evaluate_model(pipe_try, X_val_flat, y_val, meta_val)
            result = {
                'params': params,
                **{k: v for k, v in m.items() if k not in ['pred', 'score']}
            }
            results.append(result)

            print(
                f"  - Trial {i:02d} | "
                f"Acc={m['Accuracy']:.4f} | "
                f"F1={m['F1']:.4f} | "
                f"P={m['Precision']:.4f} | "
                f"Sens={m['Sensitivity']:.4f} | "
                f"Spec={m['Specificity']:.4f} | "
                f"AUROC={m['AUROC']:.4f} | "
                f"AUPRC={m['AUPRC']:.4f} | "
                f"FA/h={m['FalseAlarmsPerHour']:.4f} | "
                f"Params={params}"
            )

        except Exception as e:
            print(f"  - Trial {i:02d} FAILED | Params={params} | Error: {e}")

    if not results:
        raise RuntimeError(f"All configurations for {name} failed.")

    per_model_trials[name] = results

    # BEST = maximum validation F1, exactly as the second script.
    best_idx = int(np.nanargmax([r['F1'] for r in results]))
    best_result = results[best_idx]
    best_params = best_result['params']
    best_configs[name] = best_params

    best_models[name] = build_pipe(model).set_params(**best_params).fit(
        X_train_flat, y_train
    )

    print(
        f"  -> Best (no-CV) {name}: "
        f"Acc={best_result['Accuracy']:.4f} | "
        f"F1={best_result['F1']:.4f} | "
        f"P={best_result['Precision']:.4f} | "
        f"Sens={best_result['Sensitivity']:.4f} | "
        f"Spec={best_result['Specificity']:.4f} | "
        f"AUROC={best_result['AUROC']:.4f} | "
        f"AUPRC={best_result['AUPRC']:.4f} | "
        f"FA/h={best_result['FalseAlarmsPerHour']:.4f}"
    )


# ==================================================================================================
# PHASE 1 - BOXPLOTS OF ALL METRICS ACROSS THE 10 CONFIGURATIONS
# ==================================================================================================
print("\n>>> Boxplots (no CV) across the 10 trials for each model: all metrics")

plot_metrics = [
    'Accuracy',
    'F1',
    'Precision',
    'Sensitivity',
    'Specificity',
    'AUROC',
    'AUPRC',
    'FalseAlarmsPerHour',
]

all_boxplot_data_metrics = {
    metric: {
        m: [r[metric] for r in per_model_trials[m]]
        for m in per_model_trials
    }
    for metric in plot_metrics
}


def _plot_no_cv_box(metric_name, datadict):
    df_plot = pd.DataFrame({k: pd.Series(v) for k, v in datadict.items()})
    plt.figure(figsize=(14, 6))
    df_plot.boxplot()
    plt.title(
        f"{metric_name} distribution (Validation) - 10 trials without CV per model",
        fontsize=14
    )
    plt.ylabel(f"{metric_name} su Validation", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_current_plot(f"Validation_10_trials_boxplot_{metric_name}")
    plt.show()


for metric in plot_metrics:
    _plot_no_cv_box(metric, all_boxplot_data_metrics[metric])


# ==================================================================================================
# PHASE 2 - 5-FOLD CROSS-VALIDATION OF THE BEST CONFIGURATION PER MODEL
# ==================================================================================================
print("\n========== PHASE 2: 5-FOLD CV OF THE BEST CONFIGURATION (FIT ON TRAIN) ==========")

cv_metric_names = [
    'Accuracy',
    'F1',
    'Precision',
    'Sensitivity',
    'Specificity',
    'AUROC',
    'AUPRC',
    'FalseAlarmsPerHour',
]

cv_scores_dict = {metric: {} for metric in cv_metric_names}
cv_results_summary = []
cv_rows = []

# Five-fold stratified cross-validation applied to the training subset.
# The same fold structure is reused across models for a consistent comparison.
skf = StratifiedKFold(n_splits=5, shuffle=False)
cv_splits = list(skf.split(X_train_flat, y_train))

for name, _best_model in best_models.items():
    fold_metric_values = {metric: [] for metric in cv_metric_names}

    for fold_idx, (fit_idx, eval_idx) in enumerate(cv_splits, start=1):
        fold_model = build_pipe(models[name][0]).set_params(**best_configs[name])
        fold_model.fit(X_train_flat[fit_idx], y_train[fit_idx])

        fold_meta = meta_train.iloc[eval_idx].reset_index(drop=True)
        m = evaluate_model(
            fold_model,
            X_train_flat[eval_idx],
            y_train[eval_idx],
            fold_meta
        )

        for metric in cv_metric_names:
            fold_metric_values[metric].append(m[metric])

    row = {'Model': name}
    for metric in cv_metric_names:
        scores = np.asarray(fold_metric_values[metric], dtype=float)
        cv_scores_dict[metric][name] = scores
        row[f'{metric}_mean'] = float(np.nanmean(scores))
        row[f'{metric}_std'] = float(np.nanstd(scores))

    cv_rows.append(row)
    cv_results_summary.append((name, row['F1_mean'], row['F1_std']))

    print(
        f"{name:>22} | "
        f"Acc={row['Accuracy_mean']:.4f}±{row['Accuracy_std']:.4f} | "
        f"F1={row['F1_mean']:.4f}±{row['F1_std']:.4f} | "
        f"P={row['Precision_mean']:.4f}±{row['Precision_std']:.4f} | "
        f"Sens={row['Sensitivity_mean']:.4f}±{row['Sensitivity_std']:.4f} | "
        f"Spec={row['Specificity_mean']:.4f}±{row['Specificity_std']:.4f} | "
        f"AUROC={row['AUROC_mean']:.4f}±{row['AUROC_std']:.4f} | "
        f"AUPRC={row['AUPRC_mean']:.4f}±{row['AUPRC_std']:.4f} | "
        f"FA/h={row['FalseAlarmsPerHour_mean']:.4f}±{row['FalseAlarmsPerHour_std']:.4f}"
    )

cv_results_summary.sort(key=lambda x: x[1], reverse=True)
print("\n>>> Classifica (CV-5 su TRAIN) per F1 medio:")
for rank, (name, mean_cv, std_cv) in enumerate(cv_results_summary, 1):
    print(
        f"{rank:2d}. {name:>22} | mean={mean_cv:.4f} ± {std_cv:.4f} | "
        f"params={best_configs[name]}"
    )


def _plot_cv5_box(metric_name, datadict):
    print(
        f"\n>>> {metric_name} boxplot across the 5 folds (CV=5) "
        f"for the best configuration of each model"
    )
    df_plot = pd.DataFrame({k: pd.Series(v) for k, v in datadict.items()})
    plt.figure(figsize=(14, 6))
    df_plot.boxplot()
    plt.title(
        f"{metric_name} on 5-fold CV - best configuration per model",
        fontsize=14
    )
    plt.ylabel(f"{metric_name} (fold-wise)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    for i, model_name in enumerate(df_plot.columns, start=1):
        y_mean = df_plot[model_name].mean()
        plt.scatter(i, y_mean, s=80, marker='o', color='black', zorder=3)
    plt.scatter([], [], s=80, marker='o', color='black', label='Media CV-5')
    plt.legend(loc='best')
    plt.tight_layout()
    save_current_plot(f"CV5_boxplot_{metric_name}")
    plt.show()


for metric in cv_metric_names:
    _plot_cv5_box(metric, cv_scores_dict[metric])


# ==================================================================================================
# FINAL FIT AND VALIDATION / TEST EVALUATION
# ==================================================================================================
print("\n========== FINAL FIT AND VALIDATION/TEST ==========")

validation_rows = []
test_rows = []
validation_event_details_all = []
test_event_details_all = []
validation_warning_details_all = []
test_warning_details_all = []

for name, (model, _) in models.items():
    print(f"\n\n--- {name} ---")

    # Same final-fit logic as the second script: best validation-F1 config,
    # fitted on TRAIN, then evaluated on Validation and Test.
    final_model = build_pipe(model).set_params(**best_configs[name]).fit(
        X_train_flat, y_train
    )

    # ----------------------------------------------------------------------------------------------
    # VALIDATION SET
    # ----------------------------------------------------------------------------------------------
    val_metrics = evaluate_model(final_model, X_val_flat, y_val, meta_val)
    print_metrics("Validation", val_metrics)

    val_event_metrics, val_event_details, val_warning_details, val_false_warnings = (
        evaluate_event_level(
            val_metrics['pred'],
            meta_val,
            df,
            prediction_gap_rows=PREDICTION_GAP,
            sampling_seconds=SAMPLING_SECONDS
        )
    )
    print_event_metrics("Validation", val_event_metrics)

    val_event_details = val_event_details.copy()
    if not val_event_details.empty:
        val_event_details.insert(0, 'Model', name)
        val_event_details.insert(1, 'Subset', 'validation')
        validation_event_details_all.append(val_event_details)

    val_warning_details = val_warning_details.copy()
    if not val_warning_details.empty:
        val_warning_details.insert(0, 'Model', name)
        val_warning_details.insert(1, 'Subset', 'validation')
        validation_warning_details_all.append(val_warning_details)

    plot_lead_time_distribution(
        val_event_details,
        f"{name} - Validation lead-time distribution"
    )

    validation_rows.append({
        'Model': name,
        **{k: v for k, v in val_metrics.items() if k not in ['pred', 'score']},
        **val_event_metrics
    })

    cm_val = confusion_matrix(y_val, val_metrics['pred'], labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} - Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    save_current_plot(f"{name}_Validation_Confusion_Matrix")
    plt.show()

    # ----------------------------------------------------------------------------------------------
    # TEST SET
    # ----------------------------------------------------------------------------------------------
    test_metrics = evaluate_model(final_model, X_test_flat, y_test, meta_test)
    print_metrics("Test", test_metrics)

    test_event_metrics, test_event_details, test_warning_details, test_false_warnings = (
        evaluate_event_level(
            test_metrics['pred'],
            meta_test,
            df,
            prediction_gap_rows=PREDICTION_GAP,
            sampling_seconds=SAMPLING_SECONDS
        )
    )
    print_event_metrics("Test", test_event_metrics)

    test_event_details = test_event_details.copy()
    if not test_event_details.empty:
        test_event_details.insert(0, 'Model', name)
        test_event_details.insert(1, 'Subset', 'test')
        test_event_details_all.append(test_event_details)

    test_warning_details = test_warning_details.copy()
    if not test_warning_details.empty:
        test_warning_details.insert(0, 'Model', name)
        test_warning_details.insert(1, 'Subset', 'test')
        test_warning_details_all.append(test_warning_details)

    plot_lead_time_distribution(
        test_event_details,
        f"{name} - Test lead-time distribution"
    )

    test_rows.append({
        'Model': name,
        **{k: v for k, v in test_metrics.items() if k not in ['pred', 'score']},
        **test_event_metrics
    })

    cm_test = confusion_matrix(y_test, test_metrics['pred'], labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} - Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    save_current_plot(f"{name}_Test_Confusion_Matrix")
    plt.show()


# ==================================================================================================
# FINAL TABLES AND AUTOMATIC SAVING OF RESULTS
# ==================================================================================================
validation_df = pd.DataFrame(validation_rows).sort_values('F1', ascending=False)
test_df = pd.DataFrame(test_rows).sort_values('F1', ascending=False)
cv_df = pd.DataFrame(cv_rows).sort_values('F1_mean', ascending=False)

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
    'TreatmentHours',
    'FalseAlarmsPerHour',
    'TN', 'FP', 'FN', 'TP',
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
    'LeadTimeMaxSeconds'
]

print("\n" + "=" * 100)
print("FINAL VALIDATION RESULTS")
print("=" * 100)
print(validation_df[summary_columns].to_string(index=False))

print("\n" + "=" * 100)
print("FINAL TEST RESULTS")
print("=" * 100)
print(test_df[summary_columns].to_string(index=False))

print("\n" + "=" * 100)
print("CV-5 RESULTS")
print("=" * 100)
print(cv_df.to_string(index=False))

# Main summary files
validation_df[summary_columns].to_csv(
    'validation_metrics.csv', index=False
)
test_df[summary_columns].to_csv(
    'test_metrics.csv', index=False
)
cv_df.to_csv(
    'cv5_metrics.csv', index=False
)

# Save best hyperparameters selected STRICTLY by validation F1.
best_config_rows = []
for model_name in models:
    row = {'Model': model_name}
    row.update(best_configs[model_name])
    best_config_rows.append(row)
pd.DataFrame(best_config_rows).to_csv('best_configs_by_validation_F1.csv', index=False)

# Save all 10 trial results for every model.
trial_rows = []
for model_name, results in per_model_trials.items():
    for trial_number, r in enumerate(results, start=1):
        row = {
            'Model': model_name,
            'SuccessfulTrialNumber': trial_number,
            'F1': r['F1'],
            'Precision': r['Precision'],
            'Sensitivity': r['Sensitivity'],
            'Specificity': r['Specificity'],
            'AUROC': r['AUROC'],
            'AUPRC': r['AUPRC'],
            'FalseAlarmEpisodes': r['FalseAlarmEpisodes'],
            'TreatmentHours': r['TreatmentHours'],
            'FalseAlarmsPerHour': r['FalseAlarmsPerHour'],
            'TN': r['TN'], 'FP': r['FP'], 'FN': r['FN'], 'TP': r['TP'],
        }
        for p_name, p_value in r['params'].items():
            row[p_name] = p_value
        trial_rows.append(row)
pd.DataFrame(trial_rows).to_csv('phase1_all_trials.csv', index=False)

# Event-level detailed outputs
if validation_event_details_all:
    pd.concat(validation_event_details_all, ignore_index=True).to_csv(
        'validation_alarm_event_details.csv', index=False
    )
if test_event_details_all:
    pd.concat(test_event_details_all, ignore_index=True).to_csv(
        'test_alarm_event_details.csv', index=False
    )
if validation_warning_details_all:
    pd.concat(validation_warning_details_all, ignore_index=True).to_csv(
        'validation_warning_event_details.csv', index=False
    )
if test_warning_details_all:
    pd.concat(test_warning_details_all, ignore_index=True).to_csv(
        'test_warning_event_details.csv', index=False
    )

print("\nSaved files:")
print("  - sample_split.csv")
print("  - phase1_all_trials.csv")
print("  - best_configs_by_validation_F1.csv")
print("  - validation_metrics.csv")
print("  - test_metrics.csv")
print("  - cv5_metrics.csv")
print("  - validation_alarm_event_details.csv")
print("  - test_alarm_event_details.csv")
print("  - validation_warning_event_details.csv")
print("  - test_warning_event_details.csv")
print(f"  - all plots in: {PLOTS_DIR}/")

print("\nNOTE ON MODEL SELECTION:")
print(
    "For each model, the configuration that proceeds after tuning is selected "
    "EXCLUSIVELY according to the maximum F1 Score on the Validation set, exactly "
    "as in the reference code. No other metric is used for this decision."
)

print("\nNOTE ON DATA SPLIT:")
print(
    "Train/Validation/Test are created by randomly assigning COMPLETE treatments "
    "to the three subsets. The target proportions are approximately 70/15/15 in "
    "terms of generated temporal sequences. Treatments are never split, and the "
    "assignment is reproducible through RANDOM_STATE."
)

print("\nNOTE ON ADDITIONAL METRICS:")
print("Sensitivity = Recall of the alarm class.")
print("Specificity = TN / (TN + FP).")
print("AUROC = area under the ROC curve.")
print("AUPRC = Average Precision, a summary of the precision-recall curve.")
print(
    "FalseAlarmsPerHour = number of contiguous false-positive prediction episodes "
    "divided by the equivalent evaluated hours in the subset."
)
print(
    "Event-level metrics are calculated using the original temporal indices and "
    "the reconstructed sessions used exclusively for event-level analysis."
)
print(
    f"Temporal configuration: PREDICTION_GAP={PREDICTION_GAP} rows "
    f"({PREDICTION_GAP * SAMPLING_SECONDS:.1f} s), "
    f"N_STEPS={N_STEPS} rows ({N_STEPS * SAMPLING_SECONDS:.1f} s)."
)