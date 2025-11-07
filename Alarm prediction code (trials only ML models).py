# ====================== ALARM PREDICTION CODE ======================
# TEST 10 ML MODELS
from numpy import array, hstack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, make_scorer
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ====================== FUNCTIONS ======================
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def create_splitted_df(df, n_steps):
    historical_df = df[:-20]
    features = [
        'dmchild', 'dmparent', 'pt4', 'pt3', 'delivPumpActuation', 'pt5', 'foamDetResult', 'tmp', 'diffFlow', 'bld',
        'encoderDelivPump', 'delFlow', 'currentWeightLoss', 'arterial_revolve', 'ufPressureActuation', 'condDO',
        'condDOnf', 'encoderUFPump', 'pt6', 'Ctot', 'infusion_revolve', 'bmparent', 'venous_revolve', 'bmchild',
        'condTot', 'cond2', 'pt8', 'pVenous', 'pPreFilt', 'encoder1', 'ts3', 'ts1', 'ts2',
        'SecondStepPumpSpeed', 'airDetAnalogSensor'
    ]
    stacked_features = [historical_df[col].to_numpy().reshape((len(historical_df), 1)) for col in features]
    alarm_type = df['type'].to_numpy()[20:].reshape((len(df)-20, 1))
    dataset = hstack(stacked_features + [alarm_type])
    X, y = split_sequences(dataset, n_steps)
    return X, y


# ====================== UPLOAD AND PREPROCESSING ======================
df = pd.read_csv(r"dataset.csv")
df = df.rename(columns={'dT(CP)': 'dT'})
df['type'] = df['type'].replace({'alarm': '1', 'normal': '0', 'override': '0'})

X_all, y_all = create_splitted_df(df, 10)
y_all = y_all.astype(int)

# ====================== TRAIN / VAL / TEST SPLIT ======================
X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)  # val ~ 15% del totale (=> 70/15/15)

# ====================== RESHAPE ======================
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat   = X_val.reshape((X_val.shape[0], -1))
X_test_flat  = X_test.reshape((X_test.shape[0], -1))


# ====================== MODELS ======================
models = {
    'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=42), None),

    'ExtraTrees': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', ExtraTreesClassifier(class_weight='balanced', random_state=42))
    ]), None),

    'AdaBoost': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', AdaBoostClassifier(random_state=42))
    ]), None),

    'HistGradientBoosting': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ]), None),

    'LogisticRegression': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
    ]), None),

    'RidgeClassifier': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RidgeClassifier())
    ]), None),

    'XGBoost': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method='hist'
        ))
    ]), None),

    'LightGBM': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', LGBMClassifier(class_weight='balanced', random_state=42))
    ]), None),

    'MLP': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(max_iter=600, random_state=42))
    ]), None),

    'KNN': (Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]), None),
}


# ====================== RANGE FOR HYPERPARAMETERS ======================
param_ranges = {
    'RandomForest': {
        'classifier__n_estimators': {'type': 'int', 'min': 100, 'max': 300, 'num': 9},
        'classifier__max_depth': {'type': 'int', 'min': 10, 'max': 40, 'num': 7, 'allow_none': True},
        'classifier__min_samples_split': {'type': 'int', 'min': 2, 'max': 10, 'num': 9},
        'classifier__min_samples_leaf': {'type': 'int', 'min': 1, 'max': 5, 'num': 5},
        'classifier__max_features': ['sqrt', 'log2']
    },
    'ExtraTrees': {
        'classifier__n_estimators': {'type': 'int', 'min': 100, 'max': 300, 'num': 9},
        'classifier__max_depth': {'type': 'int', 'min': 10, 'max': 40, 'num': 7, 'allow_none': True},
        'classifier__min_samples_split': {'type': 'int', 'min': 2, 'max': 10, 'num': 9},
        'classifier__min_samples_leaf': {'type': 'int', 'min': 1, 'max': 5, 'num': 5},
        'classifier__max_features': ['sqrt', 'log2']
    },
    'AdaBoost': {
        'classifier__n_estimators': {'type': 'int', 'min': 50, 'max': 300, 'num': 11},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'num': 8, 'scale': 'log'},
        'classifier__algorithm': ['SAMME', 'SAMME.R']
    },
    'HistGradientBoosting': {
        'classifier__max_iter': {'type': 'int', 'min': 100, 'max': 300, 'num': 9},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.2, 'num': 6, 'scale': 'log'},
        'classifier__max_depth': {'type': 'int', 'min': 5, 'max': 20, 'num': 4, 'allow_none': True},
        'classifier__l2_regularization': {'type': 'float', 'min': 0.0, 'max': 0.2, 'num': 5}
    },
    'LogisticRegression': {
        'classifier__C': {'type': 'float', 'min': 0.01, 'max': 10.0, 'num': 8, 'scale': 'log'},
        'classifier__penalty': ['l2', 'none'],
        'classifier__solver': ['lbfgs', 'saga']
    },
    'RidgeClassifier': {
        'classifier__alpha': {'type': 'float', 'min': 0.1, 'max': 20.0, 'num': 8, 'scale': 'log'},
        'classifier__solver': ['auto', 'lsqr', 'sparse_cg']
    },
    'XGBoost': {
        'classifier__n_estimators': {'type': 'int', 'min': 100, 'max': 400, 'num': 7},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.2, 'num': 6, 'scale': 'log'},
        'classifier__max_depth': {'type': 'int', 'min': 3, 'max': 8, 'num': 6},
        'classifier__subsample': {'type': 'float', 'min': 0.6, 'max': 1.0, 'num': 5},
        'classifier__colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0, 'num': 5}
    },
    'LightGBM': {
        'classifier__n_estimators': {'type': 'int', 'min': 100, 'max': 400, 'num': 7},
        'classifier__learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.2, 'num': 6, 'scale': 'log'},
        'classifier__max_depth': {'type': 'int', 'min': 5, 'max': 15, 'num': 3, 'allow_none': True},
        'classifier__num_leaves': {'type': 'int', 'min': 31, 'max': 255, 'num': 8},
        'classifier__subsample': {'type': 'float', 'min': 0.6, 'max': 1.0, 'num': 5},
        'classifier__colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0, 'num': 5}
    },
    'MLP': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__learning_rate_init': {'type': 'float', 'min': 0.0005, 'max': 0.02, 'num': 6, 'scale': 'log'},
        'classifier__alpha': {'type': 'float', 'min': 5e-5, 'max': 5e-3, 'num': 6, 'scale': 'log'}
    },
    'KNN': {
        'classifier__n_neighbors': {'type': 'int', 'min': 3, 'max': 31, 'step': 2},
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }
}


# ====================== UTILS FOR MATERIALIZING CONTINUOUS RANGES ======================
def _materialize_numeric(spec):
    t = spec.get('type')
    if t not in ('int', 'float'):
        raise ValueError("spec['type'] deve essere 'int' o 'float'.")
    vmin = spec['min']; vmax = spec['max']
    allow_none = spec.get('allow_none', False)

    if t == 'int':
        if 'step' in spec:
            vals = list(range(int(vmin), int(vmax) + 1, int(spec['step'])))
        else:
            num = int(spec.get('num', 5))
            vals = list(np.linspace(int(vmin), int(vmax), num=num, dtype=int))
            vals = sorted(list(dict.fromkeys(vals)))
    else:  # float
        num = int(spec.get('num', 5))
        scale = spec.get('scale', 'linear')
        if scale == 'log':
            # evita log10(0)
            vmin_eff = max(vmin, 1e-12)
            vals = list(np.logspace(np.log10(vmin_eff), np.log10(vmax), num=num))
        else:
            vals = list(np.linspace(vmin, vmax, num=num))
        vals = [float(f"{v:.6f}") for v in vals]  # <-- fix parentesi

    if allow_none:
        vals = vals + [None]
    return vals


def materialize_grid_from_ranges(range_spec: dict) -> dict:
    out = {}
    for param, spec in range_spec.items():
        if isinstance(spec, dict) and 'type' in spec:
            out[param] = _materialize_numeric(spec)
        elif isinstance(spec, list):
            out[param] = spec[:]  # copia
        else:
            raise ValueError(f"Spec non valido per {param}: {spec}")
    return out


def build_pipe(model):
    return Pipeline([('classifier', model)]) if not isinstance(model, Pipeline) else model


def sample_param_configs(param_grid_lists, n_samples=10, random_state=42):
    grid = list(ParameterGrid(param_grid_lists))
    if len(grid) <= n_samples:
        return grid
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(grid), size=n_samples, replace=False)
    return [grid[i] for i in idx]


# ====================== PREPARATION OF DISCRETE GRIDS FROM CONTINUOUS RANGES ======================
param_grids_discrete = {name: materialize_grid_from_ranges(param_ranges[name]) for name in param_ranges}


# ====================== PHASE 1: HOLDOUT VAL, 10 randomic configuration combinations ======================
pos_label = 1
scorer = make_scorer(f1_score, pos_label=pos_label)

per_model_trials = {}
best_configs = {}
best_models = {}
all_boxplot_data = {}

print("\n========== PHASE 1: HOLDOUT VAL, 10 randomic configuration combinations) ==========")
for name, (model, _) in models.items():
    print(f"\n>>> Modello: {name}")
    pgrid_lists = param_grids_discrete[name]
    trials = sample_param_configs(pgrid_lists, n_samples=10, random_state=42)
    results = []
    f1_list = []

    for i, params in enumerate(trials, 1):
        pipe_try = build_pipe(model).set_params(**params)
        pipe_try.fit(X_train_flat, y_train)
        val_pred = pipe_try.predict(X_val_flat)
        f1 = f1_score(y_val, val_pred, pos_label=pos_label)
        prec = precision_score(y_val, val_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_val, val_pred, pos_label=pos_label)
        results.append((params, f1, prec, rec))
        f1_list.append(f1)
        print(f"  - Trial {i:02d} | F1={f1:.4f} | P={prec:.4f} | R={rec:.4f} | Params={params}")

    per_model_trials[name] = results
    all_boxplot_data[name] = f1_list

    # migliore in F1 su validation
    best_idx = int(np.argmax([r[1] for r in results]))
    best_params, best_f1, best_p, best_r = results[best_idx]
    best_configs[name] = best_params
    best_models[name] = build_pipe(model).set_params(**best_params).fit(X_train_flat, y_train)
    print(f"  -> Best (no-CV) {name}: F1={best_f1:.4f} | P={best_p:.4f} | R={best_r:.4f}")


# ====================== BOXPLOT (F1, Precision, Recall) ======================
print("\n>>> Boxplot (no-CV) delle 10 prove per ciascun modello: F1 / Precision / Recall")
all_boxplot_data_metrics = {
    'F1':        {m: [r[1] for r in per_model_trials[m]] for m in per_model_trials},
    'Precision': {m: [r[2] for r in per_model_trials[m]] for m in per_model_trials},
    'Recall':    {m: [r[3] for r in per_model_trials[m]] for m in per_model_trials},
}

def _plot_no_cv_box(metric_name, datadict):
    df = pd.DataFrame({k: pd.Series(v) for k, v in datadict.items()})
    plt.figure(figsize=(14, 6))
    df.boxplot()
    plt.title(f"Distribuzione {metric_name} (Validation) — 10 prove senza CV per modello", fontsize=14)
    plt.ylabel(f"{metric_name} su Validation", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

_plot_no_cv_box('F1',        all_boxplot_data_metrics['F1'])
_plot_no_cv_box('Precision', all_boxplot_data_metrics['Precision'])
_plot_no_cv_box('Recall',    all_boxplot_data_metrics['Recall'])


# ====================== PHASE 2: 5-FOLD CV ON THE BEST CONFIGURATION PER MODEL ======================
print("\n========== FASE 2: 5-FOLD CV sulla migliore configurazione (fit su TRAIN) ==========")
cv_results_summary = []

scorer_f1  = make_scorer(f1_score, pos_label=pos_label)
scorer_pr  = make_scorer(precision_score, pos_label=pos_label, zero_division=0)
scorer_rec = make_scorer(recall_score, pos_label=pos_label)

cv_scores_dict = {'F1': {}, 'Precision': {}, 'Recall': {}}

for name, best_model in best_models.items():
    cv_f1  = cross_val_score(best_model, X_train_flat, y_train, cv=5, scoring=scorer_f1, n_jobs=1)
    cv_pr  = cross_val_score(best_model, X_train_flat, y_train, cv=5, scoring=scorer_pr, n_jobs=1)
    cv_rec = cross_val_score(best_model, X_train_flat, y_train, cv=5, scoring=scorer_rec, n_jobs=1)

    cv_scores_dict['F1'][name]        = cv_f1
    cv_scores_dict['Precision'][name] = cv_pr
    cv_scores_dict['Recall'][name]    = cv_rec

    mean_cv = float(np.mean(cv_f1))
    std_cv  = float(np.std(cv_f1))
    cv_results_summary.append((name, mean_cv, std_cv))
    print(f"{name:>22} | F1 CV-5: mean={mean_cv:.4f} ± {std_cv:.4f}")

cv_results_summary.sort(key=lambda x: x[1], reverse=True)
print("\n>>> Classifica (CV-5 su TRAIN) per F1 medio:")
for rank, (name, mean_cv, std_cv) in enumerate(cv_results_summary, 1):
    print(f"{rank:2d}. {name:>22} | mean={mean_cv:.4f} ± {std_cv:.4f} | params={best_configs[name]}")

def _plot_cv5_box(metric_name, datadict):
    print(f"\n>>> Boxplot {metric_name} delle 5 fold (CV=5) della migliore configurazione per ciascun modello")
    df = pd.DataFrame({k: pd.Series(v) for k, v in datadict.items()})
    plt.figure(figsize=(14, 6))
    df.boxplot()
    plt.title(f"{metric_name} su CV=5 — migliore configurazione per modello", fontsize=14)
    plt.ylabel(f"{metric_name} (fold-wise)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    # puntino nero sulla media delle 5 fold per ogni modello
    for i, model_name in enumerate(df.columns, start=1):
        y_mean = df[model_name].mean()
        plt.scatter(i, y_mean, s=80, marker='o', color='black', zorder=3)
    plt.scatter([], [], s=80, marker='o', color='black', label='Media CV-5')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

_plot_cv5_box('F1',        cv_scores_dict['F1'])
_plot_cv5_box('Precision', cv_scores_dict['Precision'])
_plot_cv5_box('Recall',    cv_scores_dict['Recall'])


# ====================== FINAL FIT AND VALIDATION/TEST ======================
print("\n========== FINAL FIT AND VALIDATION/TEST ==========")
for name, (model, _) in models.items():
    print(f"\n\n--- {name} ---")
    final_model = build_pipe(model).set_params(**best_configs[name]).fit(X_train_flat, y_train)

    # ===== Validation =====
    val_pred = final_model.predict(X_val_flat)
    val_acc = final_model.score(X_val_flat, y_val)
    val_f1  = f1_score(y_val, val_pred, pos_label=pos_label)
    val_p   = precision_score(y_val, val_pred, pos_label=pos_label, zero_division=0)
    val_r   = recall_score(y_val, val_pred, pos_label=pos_label)
    print(f"Validation  -> Acc={val_acc:.4f} | F1={val_f1:.4f} | P={val_p:.4f} | R={val_r:.4f}")
    cm_val = confusion_matrix(y_val, val_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} - Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ===== Test =====
    test_pred = final_model.predict(X_test_flat)
    test_acc = final_model.score(X_test_flat, y_test)
    test_f1  = f1_score(y_test, test_pred, pos_label=pos_label)
    test_p   = precision_score(y_test, test_pred, pos_label=pos_label, zero_division=0)
    test_r   = recall_score(y_test, test_pred, pos_label=pos_label)
    print(f"Test        -> Acc={test_acc:.4f} | F1={test_f1:.4f} | P={test_p:.4f} | R={test_r:.4f}")
    cm_test = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} - Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()