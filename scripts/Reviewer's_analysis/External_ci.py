"""
external_validation_bootstrap_ci.py   (Reviewer 1 R1.1 / R1 minor, Reviewer 2 R2.3)

Fully coherent with the real microbiome benchmarking pipeline.

Key design decisions matching the original exactly:
  - Hyperparameter tuning: inner 3-fold GridSearchCV with identical search
    spaces as the original pipeline
  - Final model: trained on ALL training data using aggregated best params
    from CV folds (same strategy as evaluate_single_combination_with_external)
  - MicrobiomeScaler: exact copy including explicit log branch in fit()
  - MicrobiomeFeatureSelector: exact copy including alpha param, differential
    method with BH correction, and all six FS methods
  - AbbreviationMapper: exact copy with full column detection logic
  - Feature count: computed from min(task n_features) across all tasks,
    matching build_microbiome_configs() exactly
  - Per-task external validation: each geo/sequencer combination is evaluated
    separately, not silently dropped to first match
  - Bootstrap CI: stratified resampling on predictions from the correctly
    tuned final model
  - PPR with dual Wilson + bootstrap CI for disease-only cohorts (LC)

Three outputs per disease × feature retention:
  (i)   AUC with 95% bootstrap CI (CRC, CD — where healthy controls exist)
  (ii)  PPR with 95% bootstrap CI + Wilson exact CI (LC — no healthy controls)
  (iii) ROC curve with CI band (CRC, CD)
"""

import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.stats.multitest import multipletests
from scipy import stats

import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)
import random
random.seed(42)


# ============================================================================
# CONFIGURATION — edit these
# ============================================================================

BASE_DIR = Path(r"E:\NAFLD\minimize nafld")

TRAINING_FOLDER = BASE_DIR / "msp5"
MAPPING_FILE    = BASE_DIR / "abbreviation_mapping.xlsx"
EXTERNAL_FOLDER = BASE_DIR / "External Validation"

# Disease abbreviations as used in sheet-naming convention.
DISEASES = {
    "CRC": "D5",
    "CD":  "D4",
    "LC":  "D9",
}

# Fixed maximin-selected config — must match Methods Section 5.6 exactly.
# Feature retention percentage is varied across FEATURE_PERCENTAGES.
FIXED_CONFIG = dict(
    model             = "XGBoost",
    scaling           = "standard",
    feature_selection = "random_forest",
    balancing         = "none",
)
FEATURE_PERCENTAGES = [0.10, 0.30, 0.50]

# Nested CV settings — must match original pipeline exactly
N_OUTER_FOLDS  = 5
N_INNER_FOLDS  = 3
N_BOOTSTRAP    = 2000
RANDOM_STATE   = 42
MIN_SAMPLES    = 25   # training task minimum
EXTERNAL_MIN_SAMPLES = 2

OUTPUT_DIR = Path("external_validation_ci_results")

# Search spaces — identical to original get_search_spaces()
SEARCH_SPACES = {
    'RandomForest': {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 10],
        'clf__max_features': ['sqrt', 'log2'],
        'clf__class_weight': ['balanced', None]
    },
    'XGBoost': {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.01, 0.1],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
        'clf__reg_alpha': [0, 0.1],
        'clf__reg_lambda': [1, 10]
    },
    'LightGBM': {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.01, 0.1],
        'clf__num_leaves': [31, 63],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
        'clf__reg_alpha': [0, 0.1],
        'clf__reg_lambda': [1, 10]
    },
    'LogisticRegression': {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__penalty': ['l1', 'l2'],
        'clf__class_weight': ['balanced', None]
    },
    'SVM_RBF': {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__gamma': [0.001, 0.01, 0.1],
        'clf__class_weight': ['balanced', None]
    },
    'MLP': {
        'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'clf__activation': ['relu', 'tanh'],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate_init': [0.001, 0.01]
    }
}


# ============================================================================
# PIPELINE COMPONENTS — exact copies from real microbiome pipeline
# ============================================================================

class MicrobiomeScaler(BaseEstimator, TransformerMixin):
    """Exact copy from real pipeline including explicit log branch in fit()."""
    def __init__(self, method='none'):
        self.method = method
        self.scaler_ = None

    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scaler_ = StandardScaler().fit(X)
        elif self.method == 'robust':
            self.scaler_ = RobustScaler().fit(X)
        elif self.method == 'log':
            self.scaler_ = None   # log handled in transform; explicit branch preserved
        else:
            self.scaler_ = None
        return self

    def transform(self, X):
        if self.method == 'log':
            min_nonzero = X[X > 0].min() if np.any(X > 0) else 1e-6
            pseudocount = min_nonzero / 2
            return np.log1p(X + pseudocount)
        elif self.scaler_ is not None:
            return self.scaler_.transform(X)
        return X


class MicrobiomeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Exact copy from real pipeline.
    All six methods preserved including differential with BH correction.
    alpha parameter preserved.
    """
    def __init__(self, method='none', n_features=100, alpha=0.05):
        self.method = method
        self.n_features = n_features
        self.alpha = alpha       # preserved from original
        self.selector_ = None
        self.selected_features_ = None
        self.feature_scores_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        if self.method == 'none':
            self.selected_features_ = np.arange(X.shape[1])
            return self

        if self.method == 'variance':
            self.feature_scores_ = np.var(X, axis=0)

        elif self.method == 'f_classif':
            self.selector_ = SelectKBest(f_classif, k='all').fit(X, y)
            self.feature_scores_ = self.selector_.scores_

        elif self.method == 'mutual_info':
            self.selector_ = SelectKBest(mutual_info_classif, k='all').fit(X, y)
            self.feature_scores_ = self.selector_.scores_

        elif self.method == 'differential':
            # BH correction preserved — identical to original
            p_values, effect_sizes = [], []
            for i in range(X.shape[1]):
                a, b = X[y == 0, i], X[y == 1, i]
                try:
                    stat, p = stats.mannwhitneyu(b, a, alternative='two-sided')
                    n1, n0 = len(b), len(a)
                    eff = 1 - (2 * stat) / (n1 * n0) if (n1 * n0) > 0 else 0
                except Exception:
                    p, eff = 1.0, 0.0
                p_values.append(p)
                effect_sizes.append(abs(eff))
            reject, p_corr, _, _ = multipletests(
                p_values, alpha=self.alpha, method='fdr_bh'
            )
            self.feature_scores_ = np.array(effect_sizes) * (1 - np.array(p_corr))

        elif self.method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
            rf.fit(X, y)
            self.feature_scores_ = rf.feature_importances_

        else:
            raise ValueError(f"Unknown FS method: {self.method}")

        self.feature_scores_ = np.nan_to_num(self.feature_scores_, nan=0.0)
        k = min(self.n_features, len(self.feature_scores_))
        self.selected_features_ = np.argsort(self.feature_scores_)[::-1][:k]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        return X[:, self.selected_features_]


# ============================================================================
# MODEL REGISTRY — identical to real pipeline's get_models()
# ============================================================================

def get_base_model(model_name: str):
    """Returns fresh base model with same default params as real pipeline."""
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(
            random_state=42, eval_metric='logloss',
            n_jobs=1, verbosity=0, use_label_encoder=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            random_state=42, verbose=-1, n_jobs=1, force_col_wise=True
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42, max_iter=2000, solver='liblinear'
        ),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]


# ============================================================================
# PIPELINE BUILDER — identical to real pipeline's build_pipeline()
# ============================================================================

def build_pipeline(scaling: str, feature_selection: str, n_features: int,
                   balancing: str, model_name: str) -> Pipeline:
    """
    Exact replica of real pipeline's build_pipeline().
    Step order: scaler -> feature_selector -> balancer -> clf
    """
    steps = []
    if scaling != 'none':
        steps.append(('scaler', MicrobiomeScaler(method=scaling)))
    if feature_selection != 'none':
        steps.append(('feature_selector', MicrobiomeFeatureSelector(
            method=feature_selection, n_features=n_features
        )))
    if balancing == 'smote':
        steps.append(('balancer', SMOTE(random_state=42, k_neighbors=5)))
    elif balancing == 'undersample':
        steps.append(('balancer', RandomUnderSampler(random_state=42)))
    steps.append(('clf', get_base_model(model_name)))
    return Pipeline(steps=steps)


# ============================================================================
# ABBREVIATION MAPPER — exact copy from real pipeline
# ============================================================================

class AbbreviationMapper:
    """Exact copy from real pipeline including full column detection logic."""
    def __init__(self, mapping_file: str):
        self.mapping_file = mapping_file
        self.abbr_to_full = {}
        self.full_to_abbr = {}
        self.healthy_abbrs = set()
        self.healthy_keywords = {'healthy', 'control', 'ctrl', 'normal', 'health'}
        self._load_mapping()

    def _load_mapping(self):
        print(f"Loading abbreviation mapping from: {self.mapping_file}")
        try:
            xl_file = pd.ExcelFile(self.mapping_file)
            all_sheets = []
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(self.mapping_file, sheet_name=sheet_name)
                all_sheets.append(df)
            mapping_df = pd.concat(all_sheets, ignore_index=True)
        except Exception as e:
            print(f"ERROR: Could not load mapping file: {e}")
            return

        # Full column detection logic from original — not stripped to positional
        possible_full_names = ['Original', 'Full Name', 'Name', 'Full', 'Fullname']
        possible_abbr_names = ['Abbreviation', 'Abbr', 'Code', 'Short', 'Abbreviations']

        full_name_col = None
        abbr_col = None
        for col in mapping_df.columns:
            col_str = str(col).strip()
            if col_str in possible_full_names:
                full_name_col = col
            if col_str in possible_abbr_names:
                abbr_col = col

        # Fallback to positional only if named detection failed
        if full_name_col is None or abbr_col is None:
            cols = mapping_df.columns.tolist()
            if len(cols) >= 3:
                full_name_col = cols[1]
                abbr_col = cols[2]
            elif len(cols) >= 2:
                full_name_col = cols[0]
                abbr_col = cols[1]

        if full_name_col is None or abbr_col is None:
            print("  ERROR: Could not identify mapping columns!")
            return

        print(f"  Using columns: Full='{full_name_col}', Abbr='{abbr_col}'")
        for _, row in mapping_df.iterrows():
            try:
                full_name = str(row[full_name_col]).strip()
                abbr = str(row[abbr_col]).strip()
                if full_name and abbr and full_name != 'nan' and abbr != 'nan':
                    self.abbr_to_full[abbr] = full_name
                    self.full_to_abbr[full_name] = abbr
                    if any(k in full_name.lower() for k in self.healthy_keywords):
                        self.healthy_abbrs.add(abbr)
            except Exception:
                continue
        print(f"  Loaded {len(self.abbr_to_full)} mappings")
        print(f"  Healthy abbreviations: {self.healthy_abbrs}")

    def get_full_name(self, abbr: str) -> str:
        return self.abbr_to_full.get(abbr, abbr)

    def is_healthy(self, abbr: str) -> bool:
        if abbr in self.healthy_abbrs:
            return True
        full_name = self.get_full_name(abbr)
        return any(k in full_name.lower() for k in self.healthy_keywords)

    def parse_sheet_name(self, sheet_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        parts = str(sheet_name).strip().split('_')
        if len(parts) != 3:
            return None, None, None
        return parts[0].strip(), parts[1].strip(), parts[2].strip()

    def get_sheet_info(self, sheet_name: str) -> Optional[Dict]:
        disease_abbr, geography_abbr, sequencer_abbr = self.parse_sheet_name(sheet_name)
        if disease_abbr is None:
            return None
        return {
            'disease_abbr': disease_abbr,
            'disease_full': self.get_full_name(disease_abbr),
            'geography_abbr': geography_abbr,
            'geography_full': self.get_full_name(geography_abbr),
            'sequencer_abbr': sequencer_abbr,
            'sequencer_full': self.get_full_name(sequencer_abbr),
            'is_healthy': self.is_healthy(disease_abbr)
        }


# ============================================================================
# DATA LOADERS — exact copies from real pipeline
# ============================================================================

def load_classification_tasks_from_excel(
    excel_file: str,
    mapper: AbbreviationMapper,
    min_samples: int = 25
) -> List[Dict]:
    """
    Exact copy from real pipeline.
    Preserves project field assignment for compatibility.
    """
    print(f"\nLoading: {excel_file}")
    try:
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names
        print(f"  Found {len(sheet_names)} sheets")
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    parsed_sheets = {}
    for sheet in sheet_names:
        sheet_info = mapper.get_sheet_info(sheet)
        if sheet_info is None:
            continue
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet, index_col=0)
            df = df.T
            sheet_info['data'] = df
            parsed_sheets[sheet] = sheet_info
        except Exception as e:
            print(f"    ERROR loading {sheet}: {e}")
            continue

    classification_tasks = []
    disease_groups = {}
    for sheet_name, sheet_info in parsed_sheets.items():
        if not sheet_info['is_healthy']:
            key = (sheet_info['disease_abbr'],
                   sheet_info['geography_abbr'],
                   sheet_info['sequencer_abbr'])
            disease_groups[key] = sheet_info

    for (disease_abbr, geo_abbr, seq_abbr), disease_info in disease_groups.items():
        matching_healthy = []
        for sheet_name, sheet_info in parsed_sheets.items():
            if (sheet_info['is_healthy'] and
                    sheet_info['geography_abbr'] == geo_abbr and
                    sheet_info['sequencer_abbr'] == seq_abbr):
                matching_healthy.append(sheet_info)
        if not matching_healthy:
            continue

        disease_df = disease_info['data'].copy()
        healthy_df = pd.concat([h['data'] for h in matching_healthy], axis=0)

        common_features = list(set(disease_df.columns) & set(healthy_df.columns))
        if not common_features:
            continue

        disease_df = disease_df[common_features].fillna(0).astype(float)
        healthy_df = healthy_df[common_features].fillna(0).astype(float)

        X = np.vstack([disease_df.values, healthy_df.values])
        y = np.array([1] * len(disease_df) + [0] * len(healthy_df))

        n_disease = np.sum(y == 1)
        n_healthy = np.sum(y == 0)
        if n_disease < min_samples or n_healthy < min_samples:
            continue

        task = {
            'X': X, 'y': y,
            'feature_names': common_features,
            'task_name': f"{disease_abbr}_{geo_abbr}_{seq_abbr}",
            'disease_abbr': disease_abbr,
            'disease_full': disease_info['disease_full'],
            'geography_abbr': geo_abbr,
            'geography_full': disease_info['geography_full'],
            'sequencer_abbr': seq_abbr,
            'sequencer_full': disease_info['sequencer_full'],
            'n_disease': int(n_disease),
            'n_healthy': int(n_healthy),
            'n_features': int(len(common_features))
        }
        classification_tasks.append(task)
        print(f"    ✅ {task['task_name']}: "
              f"{n_disease}D + {n_healthy}H, {len(common_features)} features")

    return classification_tasks


def load_external_validation_data(
    external_folder: str,
    mapper: AbbreviationMapper,
    training_feature_names: List[str],
    disease_abbr: str,
    min_samples: int = 5
) -> Optional[Dict]:
    """Exact copy from real pipeline."""
    external_path = Path(external_folder)
    if not external_path.exists():
        print(f"    ⚠️  External folder not found: {external_folder}")
        return None

    excel_files = (list(external_path.glob("*.xlsx")) +
                   list(external_path.glob("*.xls")))
    if not excel_files:
        print(f"    ⚠️  No Excel files in external folder")
        return None

    print(f"    Searching for disease '{disease_abbr}' in "
          f"{len(excel_files)} external files...")

    for excel_file in excel_files:
        try:
            xls = pd.ExcelFile(excel_file)
            disease_sheets = []
            healthy_sheets = []

            for sheet_name in xls.sheet_names:
                sheet_info = mapper.get_sheet_info(sheet_name)
                if sheet_info is None:
                    continue
                try:
                    df = pd.read_excel(
                        excel_file, sheet_name=sheet_name, index_col=0
                    ).T
                    sheet_info['data'] = df
                    sheet_info['sheet_name'] = sheet_name
                    if sheet_info['disease_abbr'] == disease_abbr:
                        disease_sheets.append(sheet_info)
                    elif sheet_info['is_healthy']:
                        healthy_sheets.append(sheet_info)
                except Exception:
                    continue

            if not disease_sheets:
                continue

            print(f"      Found in {excel_file.name}: "
                  f"{len(disease_sheets)} disease sheets, "
                  f"{len(healthy_sheets)} healthy sheets")

            def align_to_training(df, feats):
                available = [f for f in feats if f in df.columns]
                if len(available) < len(feats) * 0.5:
                    print(f"        ⚠️  Only "
                          f"{len(available)}/{len(feats)} features available")
                    return None
                aligned = pd.DataFrame(0.0, index=df.index, columns=feats)
                for f in available:
                    aligned[f] = df[f].values
                return aligned.fillna(0).astype(float)

            disease_dfs = [
                align_to_training(s['data'], training_feature_names)
                for s in disease_sheets
            ]
            disease_dfs = [d for d in disease_dfs if d is not None]
            if not disease_dfs:
                continue

            disease_df = pd.concat(disease_dfs, axis=0)
            X_disease = disease_df.values
            y_disease = np.ones(len(disease_df))

            X_healthy = None
            has_healthy = False

            if healthy_sheets:
                healthy_dfs = [
                    align_to_training(s['data'], training_feature_names)
                    for s in healthy_sheets
                ]
                healthy_dfs = [d for d in healthy_dfs if d is not None]
                if healthy_dfs:
                    healthy_df = pd.concat(healthy_dfs, axis=0)
                    X_healthy = healthy_df.values
                    has_healthy = True

            if has_healthy:
                X = np.vstack([X_disease, X_healthy])
                y = np.concatenate([y_disease, np.zeros(len(X_healthy))])
            else:
                X = X_disease
                y = y_disease

            n_disease = int(np.sum(y == 1))
            n_healthy = int(np.sum(y == 0)) if has_healthy else 0

            if n_disease < min_samples:
                print(f"        ⚠️  Insufficient disease samples: {n_disease}")
                continue

            result = {
                'X': X, 'y': y,
                'feature_names': training_feature_names,
                'n_disease': n_disease,
                'n_healthy': n_healthy,
                'has_healthy': has_healthy,
                'source_file': str(excel_file.name),
                'disease_abbr': disease_abbr,
                'can_compute_all_metrics': (
                    has_healthy and n_healthy >= min_samples
                )
            }
            status = "disease+healthy" if has_healthy else "disease only"
            print(f"      ✅ Loaded: {n_disease}D + {n_healthy}H ({status})")
            return result

        except Exception as e:
            print(f"        Error processing {excel_file.name}: {e}")
            continue

    print(f"      ❌ No external validation data found for '{disease_abbr}'")
    return None


# ============================================================================
# HYPERPARAMETER TUNING VIA NESTED CV + FINAL MODEL TRAINING
# Matches evaluate_single_combination_with_external() exactly
# ============================================================================

def tune_and_train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaling: str,
    feature_selection: str,
    n_features: int,
    balancing: str,
    model_name: str,
    search_space: Dict,
    n_outer: int = N_OUTER_FOLDS,
    n_inner: int = N_INNER_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Replicates the exact two-phase strategy from evaluate_single_combination_with_external:

    Phase 1 — Nested CV to identify best hyperparameters:
      - Outer: StratifiedKFold(n_splits=n_outer, shuffle=True)
      - Inner: StratifiedKFold(n_splits=min(n_inner, min_class_count), shuffle=True)
      - GridSearchCV on inner folds, scoring='roc_auc'
      - Best params collected per fold and aggregated by majority vote

    Phase 2 — Final model:
      - Clone base model, set aggregated best params
      - Fit on ALL training data
      - This is the model used for external prediction

    Returns the fitted final pipeline.
    """
    base_model   = get_base_model(model_name)
    outer_cv     = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=random_state
    )
    space        = search_space.get(model_name, {})
    best_params_list = []

    print(f"    Phase 1: nested CV hyperparameter tuning "
          f"({n_outer}-outer × {n_inner}-inner GridSearchCV)...")

    for fold_idx, (tr_idx, te_idx) in enumerate(
        outer_cv.split(X_train, y_train)
    ):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

        pipe = build_pipeline(scaling, feature_selection, n_features,
                              balancing, model_name)

        # Same min-class guard as real pipeline
        min_class_tr  = (int(np.bincount(y_tr).min())
                         if np.bincount(y_tr).size > 1 else 2)
        n_inner_safe  = max(2, min(n_inner, min_class_tr))
        inner_cv = StratifiedKFold(
            n_splits=n_inner_safe, shuffle=True, random_state=random_state
        )

        try:
            search = GridSearchCV(
                pipe,
                param_grid=space,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=1,
                error_score='raise'
            )
            search.fit(X_tr, y_tr)
            best_params_list.append(search.best_params_)
        except Exception as e:
            print(f"      Fold {fold_idx+1} GridSearchCV failed ({e}); "
                  f"using defaults")
            best_params_list.append({})

    # Aggregate best params by majority vote — identical to real pipeline
    best_params_aggregated = {}
    if best_params_list and any(best_params_list):
        all_params: Dict[str, List] = {}
        for params in best_params_list:
            for key, val in params.items():
                all_params.setdefault(key, []).append(val)
        for key, values in all_params.items():
            try:
                best_params_aggregated[key] = max(
                    set(values), key=values.count
                )
            except Exception:
                best_params_aggregated[key] = values[0]

    print(f"    Phase 2: final model training on all {len(X_train)} samples...")
    print(f"    Aggregated best params: {best_params_aggregated}")

    # Clone and set aggregated params — same as real pipeline
    final_model = clone(base_model)
    if best_params_aggregated:
        clf_params = {
            k.replace('clf__', ''): v
            for k, v in best_params_aggregated.items()
            if k.startswith('clf__')
        }
        try:
            final_model.set_params(**clf_params)
        except Exception as e:
            print(f"    ⚠️  Could not set params ({e}); using defaults")

    final_pipe = build_pipeline(
        scaling, feature_selection, n_features, balancing, model_name
    )
    # Replace the clf step with the tuned model
    final_pipe.steps[-1] = ('clf', final_model)
    final_pipe.fit(X_train, y_train)

    return final_pipe


# ============================================================================
# BOOTSTRAP CI FUNCTIONS
# ============================================================================

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE
) -> Tuple[float, float, float, List[float]]:
    """
    Stratified bootstrap 95% CI for AUC.
    Stratified resampling guarantees both classes are present in each
    bootstrap sample — critical for tiny cohorts like CD (4 cases, 14 controls).

    Returns: (point_auc, ci_lo, ci_hi, all_bootstrap_aucs)
    """
    rng       = np.random.default_rng(seed)
    point_auc = roc_auc_score(y_true, y_prob)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_aucs = []
    for _ in range(n_bootstrap):
        bp  = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn  = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:
            continue

    lo = float(np.percentile(boot_aucs, 2.5))
    hi = float(np.percentile(boot_aucs, 97.5))
    return float(point_auc), lo, hi, boot_aucs


def bootstrap_ppr_ci(
    y_pred_disease: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI for positive prediction rate (PPR) on disease-only cohort.
    Used for LC where no healthy controls exist and AUC is not computable.
    y_pred_disease: binary predictions for disease samples only (1 = predicted disease).
    """
    rng       = np.random.default_rng(seed)
    point_ppr = float(np.mean(y_pred_disease))
    n         = len(y_pred_disease)
    boot_pprs = [
        np.mean(y_pred_disease[rng.integers(0, n, size=n)])
        for _ in range(n_bootstrap)
    ]
    lo = float(np.percentile(boot_pprs, 2.5))
    hi = float(np.percentile(boot_pprs, 97.5))
    return point_ppr, lo, hi


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval — exact binomial CI.
    Provides a cross-check for PPR in disease-only cohorts.
    """
    if n == 0:
        return np.nan, np.nan
    z      = stats.norm.ppf(1 - alpha / 2)
    p      = k / n
    denom  = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half   = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ============================================================================
# FEATURE COUNT COMPUTATION
# Matches build_microbiome_configs() exactly:
# uses min(n_features across all tasks) as the base, then computes pct × base.
# ============================================================================

def compute_n_features(all_tasks: List[Dict], feature_pct: float) -> int:
    """
    Computes the feature count for the given retention percentage using the
    same logic as build_microbiome_configs():
      max_features = min(task['n_features'] for task in all_tasks)
      n_features   = max(5, int(max_features * pct))

    This ensures the n_features value here is identical to what was used
    when the original pipeline built its config grid.
    """
    max_features = min(task['n_features'] for task in all_tasks)
    return max(5, int(max_features * feature_pct))


# ============================================================================
# CORE: EXTERNAL CI FOR ONE TASK × ONE FEATURE PERCENTAGE
# ============================================================================

def run_external_ci_for_task(
    task: Dict,
    ext_data: Dict,
    feature_pct: float,
    config: Dict,
    all_tasks: List[Dict],
) -> Optional[Dict]:
    """
    For one training task and one feature retention percentage:
      1. Compute n_features using the same min-across-tasks logic as the
         original pipeline (not from the individual task's feature count).
      2. Run nested CV to identify best hyperparameters (Phase 1).
      3. Train final model on all training data with tuned params (Phase 2).
      4. Predict on external data.
      5. Compute bootstrap CI.

    This exactly replicates what the real pipeline did during benchmarking.
    """
    n_feat   = compute_n_features(all_tasks, feature_pct)
    X_train  = task['X']
    y_train  = task['y']
    X_ext    = ext_data['X']
    y_ext    = ext_data['y']

    print(f"\n  Config: {config['model']}, scaling={config['scaling']}, "
          f"fs={config['feature_selection']}, "
          f"n_features={n_feat} ({int(feature_pct*100)}%), "
          f"balancing={config['balancing']}")

    try:
        final_pipe = tune_and_train_final_model(
            X_train, y_train,
            scaling          = config['scaling'],
            feature_selection= config['feature_selection'],
            n_features       = n_feat,
            balancing        = config['balancing'],
            model_name       = config['model'],
            search_space     = SEARCH_SPACES,
            n_outer          = N_OUTER_FOLDS,
            n_inner          = N_INNER_FOLDS,
            random_state     = RANDOM_STATE,
        )
    except Exception as e:
        print(f"    ✗ Model training failed: {e}")
        return None

    # Predict on external cohort
    try:
        y_prob_ext = final_pipe.predict_proba(X_ext)[:, 1]
    except Exception:
        try:
            y_prob_ext = final_pipe.decision_function(X_ext)
        except Exception as e:
            print(f"    ✗ Prediction failed: {e}")
            return None
    y_pred_ext = (y_prob_ext >= 0.5).astype(int)

    result = {
        "disease":              task['disease_abbr'],
        "disease_full":         task['disease_full'],
        "task_name":            task['task_name'],
        "feature_pct":          feature_pct,
        "n_features_used":      n_feat,
        "n_features_total_min": compute_n_features(all_tasks, 1.0),
        "n_train_disease":      task['n_disease'],
        "n_train_healthy":      task['n_healthy'],
        "n_external_disease":   ext_data['n_disease'],
        "n_external_healthy":   ext_data['n_healthy'],
        "has_healthy_controls": ext_data['has_healthy'],
        "source_file":          ext_data['source_file'],
        "model":                config['model'],
        "scaling":              config['scaling'],
        "feature_selection":    config['feature_selection'],
        "balancing":            config['balancing'],
    }

    if ext_data['has_healthy'] and len(np.unique(y_ext)) > 1:
        # Full AUC with stratified bootstrap CI
        point_auc, lo, hi, boot_aucs = bootstrap_auc_ci(y_ext, y_prob_ext)
        result.update({
            "metric_type":    "AUC",
            "point_estimate": round(point_auc, 4),
            "CI_lo_95":       round(lo, 4),
            "CI_hi_95":       round(hi, 4),
            "CI_width":       round(hi - lo, 4),
            # Stored separately for plotting; not written to CSV
            "_y_true":        y_ext,
            "_y_prob":        y_prob_ext,
            "_boot_aucs":     boot_aucs,
        })
        print(f"    AUC = {point_auc:.4f} "
              f"[95% CI {lo:.4f}–{hi:.4f}]  (width={hi-lo:.4f})")
        if hi - lo > 0.35:
            print(f"    ⚠️  Wide CI (>{0.35:.2f}) expected due to small n; "
                  f"report as exploratory only")

    else:
        # Disease-only cohort (e.g. LC): PPR + dual CI
        disease_mask = y_ext == 1
        ppr_point, ppr_lo, ppr_hi = bootstrap_ppr_ci(y_pred_ext[disease_mask])
        k           = int(np.sum(y_pred_ext[disease_mask]))
        n           = int(np.sum(disease_mask))
        wlo, whi    = wilson_ci(k, n)
        result.update({
            "metric_type":         "PPR (no healthy controls — AUC not computable)",
            "point_estimate":      round(ppr_point, 4),
            "CI_lo_95":            round(ppr_lo, 4),
            "CI_hi_95":            round(ppr_hi, 4),
            "CI_width":            round(ppr_hi - ppr_lo, 4),
            "CI_lo_95_wilson":     round(wlo, 4),
            "CI_hi_95_wilson":     round(whi, 4),
            "k_correctly_predicted": k,
            "n_disease_total":     n,
        })
        print(f"    PPR = {ppr_point:.4f} "
              f"[95% bootstrap CI {ppr_lo:.4f}–{ppr_hi:.4f}] "
              f"(Wilson: [{wlo:.4f}, {whi:.4f}])")
        print(f"    {k}/{n} disease samples predicted positive")

    return result


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_roc_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    boot_aucs: List[float],
    disease_label: str,
    pct: float,
    n_train: int,
    n_ext: int,
    out: Path
):
    """ROC curve with approximate CI band from bootstrap distribution."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_point   = roc_auc_score(y_true, y_prob)
    lo, hi      = np.percentile(boot_aucs, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2980b9", lw=2.2,
            label=f"AUC = {auc_point:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    ax.fill_between(
        fpr,
        np.clip(tpr - (hi - lo) / 2, 0, 1),
        np.clip(tpr + (hi - lo) / 2, 0, 1),
        color="#2980b9", alpha=0.12,
        label="Approx. CI band"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"{disease_label} External Validation ROC\n"
        f"feature retention={int(pct*100)}%  |  "
        f"train={n_train} samples  |  ext={n_ext} samples",
        fontsize=10
    )
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✓ ROC curve → {out}")


def plot_ci_barplot(df: pd.DataFrame, out: Path):
    """Forest-plot style: point estimate + CI per disease × feature %."""
    df = df.copy()
    df["label"] = (
        df["disease"] + " (" +
        (df["feature_pct"] * 100).astype(int).astype(str) + "%)"
    )
    df = df.sort_values(["disease", "feature_pct"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.55)))
    y_pos  = np.arange(len(df))
    colors = [
        "#27ae60" if "AUC" in mt else "#e67e22"
        for mt in df["metric_type"]
    ]

    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row["CI_lo_95"], row["CI_hi_95"]], [i, i],
                color=colors[i], lw=2.5, solid_capstyle="round")
        ax.plot(row["point_estimate"], i, "o",
                color=colors[i], markersize=8,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.text(
            row["CI_hi_95"] + 0.015, i,
            f"{row['point_estimate']:.3f} "
            f"[{row['CI_lo_95']:.3f}, {row['CI_hi_95']:.3f}]",
            va="center", fontsize=8
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.axvline(0.5, color="grey", linestyle="--", lw=0.8, alpha=0.5,
               label="Chance (AUC=0.5)")
    ax.set_xlim(0, 1.42)
    ax.set_xlabel(
        "AUC with 95% CI (green) or "
        "Positive Prediction Rate with 95% CI (orange)",
        fontsize=9
    )
    ax.set_title(
        "External Validation — Point Estimates with 95% Bootstrap CI\n"
        "Tuned pipeline (nested CV + GridSearchCV), all feature-retention levels",
        fontsize=11
    )
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ CI forest plot → {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print("=" * 75)
    print("  EXTERNAL VALIDATION — 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("  Coherent with real microbiome benchmarking pipeline")
    print(f"  Base dir: {BASE_DIR}")
    print("=" * 75)

    mapper = AbbreviationMapper(str(MAPPING_FILE))

    # Load all training tasks (with project field, matching real pipeline)
    project_files = (
        list(TRAINING_FOLDER.glob("*.xlsx")) +
        list(TRAINING_FOLDER.glob("*.xls"))
    )
    if not project_files:
        print(f"  ✗ No training Excel files found in {TRAINING_FOLDER}")
        return

    all_tasks = []
    for pf in project_files:
        project_name = pf.stem   # preserved from real pipeline
        tasks = load_classification_tasks_from_excel(
            str(pf), mapper, min_samples=MIN_SAMPLES
        )
        for t in tasks:
            t['project'] = project_name
        all_tasks.extend(tasks)

    if not all_tasks:
        print("  ✗ No valid training tasks found")
        return

    print(f"\n  Loaded {len(all_tasks)} training tasks:")
    for t in all_tasks:
        print(f"    {t['task_name']}: {t['n_disease']}D + {t['n_healthy']}H, "
              f"{t['n_features']} features")

    # Report the feature count that will be used for each retention level
    print(f"\n  Feature selection base "
          f"(min across tasks): "
          f"{min(t['n_features'] for t in all_tasks)} features")
    for pct in FEATURE_PERCENTAGES:
        n = compute_n_features(all_tasks, pct)
        print(f"    {int(pct*100)}% retention → {n} features")

    all_records   = []
    plot_payloads = []   # (y_true, y_prob, boot_aucs, disease_label, pct, n_train, n_ext)

    for disease_label, disease_abbr in DISEASES.items():
        print(f"\n{'─'*65}")
        print(f"  {disease_label} (abbr='{disease_abbr}')")
        print(f"{'─'*65}")

        # All matching training tasks (not just first — handle multiple geo/seq)
        matching_tasks = [
            t for t in all_tasks if t['disease_abbr'] == disease_abbr
        ]
        if not matching_tasks:
            print(f"  ✗ No training task for disease_abbr='{disease_abbr}'")
            continue

        if len(matching_tasks) > 1:
            print(f"  ℹ️  {len(matching_tasks)} training tasks found for "
                  f"'{disease_abbr}' (multiple geo/sequencer combinations). "
                  f"Running external validation for each separately.")

        for task in matching_tasks:
            print(f"\n  Training task: {task['task_name']} "
                  f"({task['n_disease']}D + {task['n_healthy']}H)")

            ext_data = load_external_validation_data(
                str(EXTERNAL_FOLDER), mapper,
                task['feature_names'], disease_abbr,
                min_samples=EXTERNAL_MIN_SAMPLES
            )
            if ext_data is None:
                print(f"  ✗ No external data found for '{disease_abbr}'")
                continue

            status = ("with healthy controls"
                      if ext_data['has_healthy']
                      else "DISEASE ONLY (no healthy controls — PPR only)")
            print(f"  External: {ext_data['n_disease']}D + "
                  f"{ext_data['n_healthy']}H from "
                  f"{ext_data['source_file']} ({status})")

            for pct in FEATURE_PERCENTAGES:
                print(f"\n  ── feature retention = {int(pct*100)}% ──")
                res = run_external_ci_for_task(
                    task, ext_data, pct, FIXED_CONFIG, all_tasks
                )
                if res is None:
                    continue

                # Extract plot payloads before stripping arrays
                if res.get("metric_type") == "AUC":
                    plot_payloads.append((
                        res.pop("_y_true"),
                        res.pop("_y_prob"),
                        res.pop("_boot_aucs"),
                        disease_label,
                        pct,
                        task['n_disease'] + task['n_healthy'],
                        ext_data['n_disease'] + ext_data['n_healthy'],
                    ))

                all_records.append(res)

    if not all_records:
        print("\n  ✗ No results generated. Check DISEASES abbreviations and paths.")
        return

    # Save main results table
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(
        OUTPUT_DIR / "external_validation_ci_all_diseases.csv", index=False
    )

    print(f"\n{'='*75}")
    print("  SUMMARY TABLE")
    print(f"{'='*75}")
    display_cols = [
        "disease", "task_name", "feature_pct", "metric_type",
        "point_estimate", "CI_lo_95", "CI_hi_95",
        "n_external_disease", "n_external_healthy"
    ]
    available = [c for c in display_cols if c in results_df.columns]
    print(results_df[available].to_string(index=False))

    # ROC curves
    for (y_true, y_prob, boot_aucs,
         disease_label, pct, n_train, n_ext) in plot_payloads:
        plot_roc_with_ci(
            y_true, y_prob, boot_aucs,
            disease_label, pct, n_train, n_ext,
            OUTPUT_DIR / f"{disease_label}_{int(pct*100)}pct_roc_curve.png"
        )

    # Forest plot
    plot_ci_barplot(
        results_df,
        OUTPUT_DIR / "external_validation_ci_barplot.png"
    )

    # Rebuttal-ready text
    with open(OUTPUT_DIR / "external_validation_ci_summary.txt", "w") as f:
        f.write(
            "Rebuttal text (Reviewer 1 R1.1, Reviewer 2 R2.3):\n"
            "All metrics computed using the maximin-selected pipeline "
            "(XGBoost, standard scaling, random-forest feature selection, "
            "no class balancing) with hyperparameters tuned via nested CV "
            "(5-fold outer × 3-fold inner GridSearchCV) and final model "
            "trained on all training data — identical to the procedure "
            "used in the original benchmarking pipeline.\n\n"
        )
        for disease_label in DISEASES:
            sub = results_df[results_df["disease"] == disease_label]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                task_info = (
                    f"train={row['n_train_disease']}D+{row['n_train_healthy']}H, "
                    f"ext={row['n_external_disease']}D+{row['n_external_healthy']}H"
                )
                if row["metric_type"] == "AUC":
                    line = (
                        f"{disease_label} external validation "
                        f"(task={row['task_name']}, "
                        f"feature retention={int(row['feature_pct']*100)}%, "
                        f"{task_info}): "
                        f"AUC = {row['point_estimate']:.3f} "
                        f"[95% bootstrap CI "
                        f"{row['CI_lo_95']:.3f}–{row['CI_hi_95']:.3f}]"
                    )
                    if row["CI_width"] > 0.35:
                        line += (
                            f". NOTE: wide CI (width={row['CI_width']:.3f}) "
                            f"due to small external cohort (n="
                            f"{row['n_external_disease']+row['n_external_healthy']}); "
                            f"reported as exploratory only (Tier 3)."
                        )
                    f.write(line + "\n")
                else:
                    f.write(
                        f"{disease_label} external evaluation "
                        f"(task={row['task_name']}, "
                        f"feature retention={int(row['feature_pct']*100)}%, "
                        f"n={row['n_disease_total']}, "
                        f"no healthy controls available — AUC not computable): "
                        f"positive prediction rate = {row['point_estimate']:.3f} "
                        f"[95% bootstrap CI "
                        f"{row['CI_lo_95']:.3f}–{row['CI_hi_95']:.3f}; "
                        f"Wilson exact CI "
                        f"{row['CI_lo_95_wilson']:.3f}–{row['CI_hi_95_wilson']:.3f}]. "
                        f"{row['k_correctly_predicted']}/{row['n_disease_total']} "
                        f"disease samples predicted positive. "
                        f"Reported as exploratory positive-prediction-rate "
                        f"assessment only (Tier 3).\n"
                    )
            f.write("\n")

    print(f"\n  ✓ Rebuttal text → "
          f"{OUTPUT_DIR / 'external_validation_ci_summary.txt'}")

    print(f"\n{'='*75}")
    print(f"  COMPLETE — all outputs in: {OUTPUT_DIR}/")
    print(f"  ├── external_validation_ci_all_diseases.csv")
    print(f"  ├── external_validation_ci_summary.txt")
    print(f"  ├── external_validation_ci_barplot.png")
    for disease_label in DISEASES:
        for pct in FEATURE_PERCENTAGES:
            print(f"  ├── {disease_label}_{int(pct*100)}pct_roc_curve.png")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()