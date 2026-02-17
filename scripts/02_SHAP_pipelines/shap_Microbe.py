"""


SHAP ANALYSIS FOR MICROBIOME PIPELINE (SINGLE CONFIG, SINGLE TASK)

- You provide:
    * results_file: path to ONE task's results (the results.csv from that task)
    * input_folder: folder with original microbiome project Excel files
    * mapping_file: abbreviation mapping Excel used in main pipeline
    * config_name: EXACT config_name to analyze
    * model_name: (optional) model_name to restrict to
- Script will:
    * Rebuild X, y, feature_names for that task from the original Excel
    * Parse config_name → scaling, feature_selection, n_features, balancing
    * Load best_params from results_file
    * Train pipeline on ALL samples with saved hyperparameters
    * Compute SHAP on ALL samples
    * Validation includes:
        - Cross-validation performance metrics
        - Permutation test for model significance
        - Bootstrap confidence intervals for metrics
    * Save:
        - selected_configurations.csv
        - config_XX_[config]_[model]_SHAP.csv
        - ALL_SHAP_VALUES.csv
        - FEATURE_IMPORTANCE_SUMMARY.csv
        
"""

import os
import warnings
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

from scipy import stats
from statsmodels.stats.multitest import multipletests

import shap

warnings.filterwarnings("ignore")

# Fixed random seed (match main pipeline)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)


# ============================================================================
# ABBREVIATION MAPPING (COPIED FROM MAIN MICROBIOME PIPELINE)
# ============================================================================

class AbbreviationMapper:
    """Handle abbreviation mapping from Excel file."""
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
        
        if full_name_col is None or abbr_col is None:
            cols = mapping_df.columns.tolist()
            if len(cols) >= 3:
                full_name_col = cols[1]; abbr_col = cols[2]
            elif len(cols) >= 2:
                full_name_col = cols[0]; abbr_col = cols[1]
        
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
            except:
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


def load_classification_tasks_from_excel(
    excel_file: str,
    mapper: AbbreviationMapper,
    min_samples: int = 1
) -> List[Dict]:
    """
    Load multi-task classification from Excel.
    (Same logic as in main microbiome pipeline, but min_samples default = 1)
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
            df = df.T  # Transpose: samples as rows, features as columns
            sheet_info['data'] = df
            parsed_sheets[sheet] = sheet_info
        except Exception as e:
            print(f"    ERROR loading {sheet}: {e}")
            continue
    
    classification_tasks = []
    disease_groups = {}
    for sheet_name, sheet_info in parsed_sheets.items():
        if not sheet_info['is_healthy']:
            key = (sheet_info['disease_abbr'], sheet_info['geography_abbr'], sheet_info['sequencer_abbr'])
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
        
        # Zero imputation
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
        print(f"    Task: {task['task_name']}: {n_disease}D + {n_healthy}H, {len(common_features)} features")
    
    return classification_tasks


# ============================================================================
# PREPROCESSING TRANSFORMERS (MATCH MAIN PIPELINE)
# ============================================================================

class MicrobiomeScaler(BaseEstimator, TransformerMixin):
    """Scaling - same logic as in main pipeline."""
    def __init__(self, method='none'):
        self.method = method
        self.scaler_ = None
        
    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scaler_ = StandardScaler().fit(X)
        elif self.method == 'robust':
            self.scaler_ = RobustScaler().fit(X)
        elif self.method == 'log':
            self.scaler_ = None
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
        else:
            return X


class MicrobiomeFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection - same logic as in main pipeline."""
    def __init__(self, method='none', n_features=100, alpha=0.05):
        self.method = method
        self.n_features = n_features
        self.alpha = alpha
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
            p_values, effect_sizes = [], []
            for i in range(X.shape[1]):
                a, b = X[y == 0, i], X[y == 1, i]
                try:
                    stat, p = stats.mannwhitneyu(b, a, alternative='two-sided')
                    n1, n0 = len(b), len(a)
                    eff = 1 - (2*stat)/(n1*n0) if (n1*n0) > 0 else 0
                except:
                    p, eff = 1.0, 0.0
                p_values.append(p)
                effect_sizes.append(abs(eff))
            reject, p_corr, _, _ = multipletests(p_values, alpha=self.alpha, method='fdr_bh')
            self.feature_scores_ = np.array(effect_sizes) * (1 - np.array(p_corr))
            
        elif self.method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=1)
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
# CONFIG PARSER (SAME NAMING LOGIC AS METABOLOMICS)
# ============================================================================

def parse_microbiome_config(config_name: str) -> Dict[str, Any]:
    """
    Parse config_name like:
        'standard_differential_50f(10%)_smote'
        'NoScale_AllFeatures(100%)_NoBalance'
        'robust_f_classif_150f(30%)_undersample'
    Returns dict: scaling, feature_selection, n_features, balancing
    """
    config = {
        'scaling': 'none',
        'feature_selection': 'none',
        'n_features': 1000,  # default (will be clipped by data)
        'balancing': 'none'
    }
    
    parts = config_name.split('_')
    
    # 1. Scaling
    if 'NoScale' in parts:
        config['scaling'] = 'none'
    elif 'standard' in parts:
        config['scaling'] = 'standard'
    elif 'robust' in parts:
        config['scaling'] = 'robust'
    elif 'log' in parts:
        config['scaling'] = 'log'
    
    # 2. Feature selection
    if 'AllFeatures' in config_name:
        config['feature_selection'] = 'none'
    elif 'mutual' in config_name or 'info' in config_name:
        config['feature_selection'] = 'mutual_info'
    elif 'classif' in config_name or 'f_classif' in parts:
        config['feature_selection'] = 'f_classif'
    elif 'random' in parts and 'forest' in parts:
        config['feature_selection'] = 'random_forest'
    elif 'differential' in parts:
        config['feature_selection'] = 'differential'
    elif 'variance' in parts:
        config['feature_selection'] = 'variance'
    
    # 3. n_features from "XXXf(YY%)"
    for part in parts:
        if 'f(' in part and '%' in part:
            try:
                num_str = part.split('f(')[0]
                config['n_features'] = int(num_str)
            except:
                pass
    
    # 4. Balancing
    if 'NoBalance' in parts:
        config['balancing'] = 'none'
    elif 'smote' in parts:
        config['balancing'] = 'smote'
    elif 'adasyn' in parts:
        config['balancing'] = 'adasyn'
    elif 'undersample' in parts:
        config['balancing'] = 'undersample'
    
    return config


# ============================================================================
# MODELS (MATCH MAIN PIPELINE)
# ============================================================================

def get_model_instance(model_name: str):
    models = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric='logloss', n_jobs=1, 
            verbosity=0, use_label_encoder=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            random_state=RANDOM_SEED, verbose=-1, n_jobs=1, force_col_wise=True
        ),
        'LogisticRegression': LogisticRegression(
            random_state=RANDOM_SEED, max_iter=2000, solver='liblinear'
        ),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED),
        'MLP': MLPClassifier(max_iter=1000, random_state=RANDOM_SEED, early_stopping=True)
    }
    return models.get(model_name)


# ============================================================================
# BALANCING (ADAPTIVE, SAFE)
# ============================================================================

def get_balancer(balancing_method: str, y: np.ndarray):
    """Adaptive balancing (smote/adasyn/undersample) using class counts."""
    if balancing_method == 'none':
        return None
    
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return None
    
    min_class_size = counts.min()
    
    if balancing_method == 'undersample':
        return RandomUnderSampler(random_state=RANDOM_SEED)
    
    if balancing_method == 'smote':
        if min_class_size < 2:
            return None
        k_neighbors = min(5, min_class_size - 1)
        return SMOTE(random_state=RANDOM_SEED, k_neighbors=max(1, k_neighbors))
    
    if balancing_method == 'adasyn':
        if min_class_size < 2:
            return None
        k_neighbors = min(5, min_class_size - 1)
        return ADASYN(random_state=RANDOM_SEED, n_neighbors=max(1, k_neighbors))
    
    return None


# ============================================================================
# PIPELINE BUILDER
# ============================================================================

def build_microbiome_pipeline(config_dict: Dict[str, Any], model, y: np.ndarray):
    steps = []
    # Scaling
    if config_dict['scaling'] != 'none':
        steps.append(('scaler', MicrobiomeScaler(method=config_dict['scaling'])))
    # Feature selection
    if config_dict['feature_selection'] != 'none':
        steps.append(('feature_selector', MicrobiomeFeatureSelector(
            method=config_dict['feature_selection'],
            n_features=config_dict['n_features']
        )))
    # Balancing
    balancer = get_balancer(config_dict['balancing'], y)
    if balancer is not None:
        steps.append(('balancer', balancer))
    # Classifier
    steps.append(('clf', model))
    return Pipeline(steps=steps)


# ============================================================================
# HYPERPARAMETER PARSER
# ============================================================================

def parse_best_params(params_str: str) -> Dict:
    """
    Parse best_params string from CSV results.
    Example: "{'clf__n_estimators': 100, 'clf__max_depth': 10}"
    """
    if pd.isna(params_str) or params_str == '' or params_str == '{}':
        return {}
    
    try:
        return ast.literal_eval(params_str)
    except Exception as e:
        print(f"      [WARNING] Failed to parse params: {params_str[:100]}... Error: {e}")
        return {}


# ============================================================================
# TRAIN WITH SAVED HYPERPARAMETERS
# ============================================================================

def train_with_saved_params(
    X: np.ndarray,
    y: np.ndarray,
    config_dict: Dict[str, Any],
    model_name: str,
    best_params: Dict
):
    """Train pipeline with SAVED hyperparameters (no search)."""
    print(f"    Training with saved hyperparameters...")
    
    base_model = get_model_instance(model_name)
    if base_model is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    pipeline = build_microbiome_pipeline(config_dict, base_model, y)
    
    if best_params:
        try:
            pipeline.set_params(**best_params)
            print(f"      [OK] Applied {len(best_params)} hyperparameters")
        except Exception as e:
            print(f"      [WARNING] Could not set params: {e}")
    else:
        print(f"      [INFO] No hyperparameters provided, using defaults")
    
    pipeline.fit(X, y)
    print(f"      [OK] Training complete")
    return pipeline


# ============================================================================
# SHAP NORMALIZATION & COMPUTATION
# ============================================================================

def normalize_shap_output(shap_values, n_samples: int, n_features: int, model_type: str) -> np.ndarray:
    """Normalize SHAP output to shape (n_samples, n_features)."""
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]
    
    shap_values = np.asarray(shap_values)
    
    if len(shap_values.shape) == 1:
        raise ValueError(f"SHAP is 1D: {shap_values.shape}")
    
    elif len(shap_values.shape) == 2:
        if shap_values.shape == (n_samples, n_features):
            return shap_values
        if shap_values.shape == (n_features, n_samples):
            return shap_values.T
        return shap_values if shap_values.shape[0] == n_samples else shap_values.T
    
    elif len(shap_values.shape) == 3:
        n_classes = 2
        if shap_values.shape == (n_classes, n_samples, n_features):
            return shap_values[-1, :, :]
        if shap_values.shape == (n_samples, n_classes, n_features):
            return shap_values[:, -1, :]
        if shap_values.shape == (n_samples, n_features, n_classes):
            return shap_values[:, :, -1]
        min_dim = np.argmin(shap_values.shape)
        result = (shap_values[-1, :, :] if min_dim == 0 else
                  (shap_values[:, -1, :] if min_dim == 1 else shap_values[:, :, -1]))
        return result.T if result.shape == (n_features, n_samples) else result
    
    else:
        raise ValueError(f"Unexpected SHAP dimensionality: {shap_values.shape}")


def compute_shap_all_samples(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    original_feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """Compute SHAP values on ALL samples and map back to original features."""
    print(f"    Computing SHAP values for {model_name}...")
    
    try:
        X_transformed = X.copy()
        feature_indices = list(range(len(original_feature_names)))
        
        for step_name, transformer in pipeline.steps[:-1]:
            if step_name == 'feature_selector':
                X_transformed = transformer.transform(X_transformed)
                feature_indices = [feature_indices[i] for i in transformer.selected_features_]
            elif step_name == 'balancer':
                continue  # skip balancing for SHAP
            else:
                X_transformed = transformer.transform(X_transformed)
        
        clf = pipeline.named_steps['clf']
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=1e10, neginf=-1e10)
        n_samples_shap, n_features_shap = X_transformed.shape
        
        if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            explainer = shap.TreeExplainer(clf)
            shap_raw = explainer.shap_values(X_transformed)
            shap_values = normalize_shap_output(shap_raw, n_samples_shap, n_features_shap, model_name)
        else:
            background = shap.kmeans(X_transformed, min(100, n_samples_shap)) if n_samples_shap > 100 else X_transformed
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_raw = explainer.shap_values(X_transformed)
            shap_values = normalize_shap_output(shap_raw, n_samples_shap, n_features_shap, model_name)
        
        mean_abs_shap_selected = np.mean(np.abs(shap_values), axis=0).flatten()
        
        shap_all_features = np.zeros(len(original_feature_names))
        for i, original_idx in enumerate(feature_indices):
            if i < len(mean_abs_shap_selected):
                shap_all_features[original_idx] = float(mean_abs_shap_selected[i])
        
        shap_df = pd.DataFrame({
            'feature': original_feature_names,
            'mean_abs_shap': shap_all_features,
            'selected': [i in feature_indices for i in range(len(original_feature_names))]
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        print(f"      [OK] SHAP: {n_samples_shap} samples, {len(feature_indices)} features")
        return shap_df
    
    except Exception as e:
        print(f"      [WARNING] SHAP failed: {e}")
        # Fallback: feature_importances_ or coef_
        try:
            clf = pipeline.named_steps['clf']
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                coef = clf.coef_
                importances = np.abs(coef[0] if coef.ndim > 1 else coef)
            else:
                raise RuntimeError("No importances/coefs available.")
            
            importances = np.asarray(importances).flatten()
            feature_indices = list(range(len(original_feature_names)))
            for step_name, transformer in pipeline.steps[:-1]:
                if step_name == 'feature_selector':
                    feature_indices = [feature_indices[i] for i in transformer.selected_features_]
            
            shap_all_features = np.zeros(len(original_feature_names))
            for i in range(min(len(importances), len(feature_indices))):
                shap_all_features[feature_indices[i]] = float(importances[i])
            
            return pd.DataFrame({
                'feature': original_feature_names,
                'mean_abs_shap': shap_all_features,
                'selected': [i in feature_indices for i in range(len(original_feature_names))]
            })
        except Exception:
            return pd.DataFrame({
                'feature': original_feature_names,
                'mean_abs_shap': np.zeros(len(original_feature_names)),
                'selected': [False] * len(original_feature_names)
            })


# ============================================================================
# CROSS-VALIDATION HELPER (SINGLE RUN)
# ============================================================================

def run_cv_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    config_dict: Dict[str, Any],
    model_name: str,
    best_params: Dict,
    cv_splits: int,
    random_state: int = RANDOM_SEED
) -> Dict[str, List[float]]:
    """Run CV and return per-fold scores."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for tr_idx, te_idx in skf.split(X, y):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        
        model = get_model_instance(model_name)
        pipeline = build_microbiome_pipeline(config_dict, model, ytr)
        
        if best_params:
            try:
                pipeline.set_params(**best_params)
            except:
                pass
        
        try:
            pipeline.fit(Xtr, ytr)
            ypred = pipeline.predict(Xte)
            try:
                yproba = pipeline.predict_proba(Xte)[:, 1]
            except:
                yproba = ypred.astype(float)
            
            scores['accuracy'].append(accuracy_score(yte, ypred))
            scores['precision'].append(precision_score(yte, ypred, zero_division=0))
            scores['recall'].append(recall_score(yte, ypred, zero_division=0))
            scores['f1'].append(f1_score(yte, ypred, zero_division=0))
            if len(np.unique(yte)) > 1:
                scores['auc'].append(roc_auc_score(yte, yproba))
            else:
                scores['auc'].append(np.nan)
        except:
            continue
    
    return scores


# ============================================================================
# VALIDATION WITH PERMUTATION TEST AND BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def validate_model_with_saved_params(
    X: np.ndarray,
    y: np.ndarray,
    config_dict: Dict[str, Any],
    model_name: str,
    best_params: Dict,
    cv: int = 5,
    n_permutations: int = 100,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Optional[Dict]:
    """
    CV validation using saved hyperparameters with:
    - Permutation test for model significance
    - Bootstrap confidence intervals for metrics
    """
    print(f"    Validating with {cv}-fold CV + permutation test + bootstrap CI...")
    
    try:
        min_class = np.bincount(y).min() if np.bincount(y).size > 1 else 2
        cv_splits = max(2, min(cv, int(min_class)))
        
        # ================================================================
        # STEP 1: OBSERVED CV PERFORMANCE
        # ================================================================
        print(f"      Step 1/3: Computing observed CV performance...")
        observed_scores = run_cv_evaluation(X, y, config_dict, model_name, best_params, cv_splits)
        
        if len(observed_scores['f1']) == 0:
            return None
        
        observed_f1 = np.mean(observed_scores['f1'])
        observed_auc = np.nanmean(observed_scores['auc'])
        observed_accuracy = np.mean(observed_scores['accuracy'])
        
        print(f"        Observed: F1={observed_f1:.3f}, AUC={observed_auc:.3f}, Acc={observed_accuracy:.3f}")
        
        # ================================================================
        # STEP 2: PERMUTATION TEST
        # ================================================================
        print(f"      Step 2/3: Permutation test ({n_permutations} permutations)...")
        rng = np.random.RandomState(RANDOM_SEED)
        
        perm_f1_scores = []
        perm_auc_scores = []
        perm_accuracy_scores = []
        
        for perm_idx in range(n_permutations):
            if (perm_idx + 1) % 20 == 0:
                print(f"        Permutation {perm_idx + 1}/{n_permutations}...")
            
            # Shuffle labels
            y_shuffled = rng.permutation(y)
            
            try:
                perm_scores = run_cv_evaluation(
                    X, y_shuffled, config_dict, model_name, best_params, 
                    cv_splits, random_state=RANDOM_SEED + perm_idx
                )
                
                if len(perm_scores['f1']) > 0:
                    perm_f1_scores.append(np.mean(perm_scores['f1']))
                    perm_auc_scores.append(np.nanmean(perm_scores['auc']))
                    perm_accuracy_scores.append(np.mean(perm_scores['accuracy']))
            except:
                continue
        
        # Calculate p-values
        n_perm_success = len(perm_f1_scores)
        if n_perm_success > 0:
            p_value_f1 = (np.sum(np.array(perm_f1_scores) >= observed_f1) + 1) / (n_perm_success + 1)
            p_value_auc = (np.sum(np.array(perm_auc_scores) >= observed_auc) + 1) / (n_perm_success + 1)
            p_value_accuracy = (np.sum(np.array(perm_accuracy_scores) >= observed_accuracy) + 1) / (n_perm_success + 1)
            
            null_f1_mean = np.mean(perm_f1_scores)
            null_f1_std = np.std(perm_f1_scores)
            null_auc_mean = np.nanmean(perm_auc_scores)
            null_auc_std = np.nanstd(perm_auc_scores)
        else:
            p_value_f1 = p_value_auc = p_value_accuracy = np.nan
            null_f1_mean = null_f1_std = null_auc_mean = null_auc_std = np.nan
        
        print(f"        Permutation p-values: F1={p_value_f1:.4f}, AUC={p_value_auc:.4f}")
        print(f"        Null distribution: F1={null_f1_mean:.3f}±{null_f1_std:.3f}")
        
        # ================================================================
        # STEP 3: BOOTSTRAP CONFIDENCE INTERVALS
        # ================================================================
        print(f"      Step 3/3: Bootstrap CI ({n_bootstrap} iterations, {confidence_level*100:.0f}% CI)...")
        
        n_samples = X.shape[0]
        boot_f1_scores = []
        boot_auc_scores = []
        boot_accuracy_scores = []
        boot_precision_scores = []
        boot_recall_scores = []
        
        for boot_idx in range(n_bootstrap):
            if (boot_idx + 1) % 20 == 0:
                print(f"        Bootstrap {boot_idx + 1}/{n_bootstrap}...")
            
            # Bootstrap sample
            boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_indices]
            y_boot = y[boot_indices]
            
            # Check if bootstrap sample has both classes
            if len(np.unique(y_boot)) < 2:
                continue
            
            try:
                boot_scores = run_cv_evaluation(
                    X_boot, y_boot, config_dict, model_name, best_params,
                    cv_splits, random_state=RANDOM_SEED + n_permutations + boot_idx
                )
                
                if len(boot_scores['f1']) > 0:
                    boot_f1_scores.append(np.mean(boot_scores['f1']))
                    boot_auc_scores.append(np.nanmean(boot_scores['auc']))
                    boot_accuracy_scores.append(np.mean(boot_scores['accuracy']))
                    boot_precision_scores.append(np.mean(boot_scores['precision']))
                    boot_recall_scores.append(np.mean(boot_scores['recall']))
            except:
                continue
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        
        n_boot_success = len(boot_f1_scores)
        if n_boot_success > 10:
            ci_f1 = (np.percentile(boot_f1_scores, lower_pct), np.percentile(boot_f1_scores, upper_pct))
            ci_auc = (np.nanpercentile(boot_auc_scores, lower_pct), np.nanpercentile(boot_auc_scores, upper_pct))
            ci_accuracy = (np.percentile(boot_accuracy_scores, lower_pct), np.percentile(boot_accuracy_scores, upper_pct))
            ci_precision = (np.percentile(boot_precision_scores, lower_pct), np.percentile(boot_precision_scores, upper_pct))
            ci_recall = (np.percentile(boot_recall_scores, lower_pct), np.percentile(boot_recall_scores, upper_pct))
        else:
            ci_f1 = ci_auc = ci_accuracy = ci_precision = ci_recall = (np.nan, np.nan)
        
        print(f"        Bootstrap {confidence_level*100:.0f}% CI: F1=[{ci_f1[0]:.3f}, {ci_f1[1]:.3f}], AUC=[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]")
        
        # ================================================================
        # COMPILE RESULTS
        # ================================================================
        result = {
            # Observed performance
            'accuracy': float(np.mean(observed_scores['accuracy'])),
            'accuracy_std': float(np.std(observed_scores['accuracy'])),
            'precision': float(np.mean(observed_scores['precision'])),
            'precision_std': float(np.std(observed_scores['precision'])),
            'recall': float(np.mean(observed_scores['recall'])),
            'recall_std': float(np.std(observed_scores['recall'])),
            'f1': float(np.mean(observed_scores['f1'])),
            'f1_std': float(np.std(observed_scores['f1'])),
            'auc': float(np.nanmean(observed_scores['auc'])),
            'auc_std': float(np.nanstd(observed_scores['auc'])),
            'n_folds_success': len(observed_scores['f1']),
            
            # Permutation test results
            'permutation_p_value_f1': float(p_value_f1) if not np.isnan(p_value_f1) else None,
            'permutation_p_value_auc': float(p_value_auc) if not np.isnan(p_value_auc) else None,
            'permutation_p_value_accuracy': float(p_value_accuracy) if not np.isnan(p_value_accuracy) else None,
            'null_f1_mean': float(null_f1_mean) if not np.isnan(null_f1_mean) else None,
            'null_f1_std': float(null_f1_std) if not np.isnan(null_f1_std) else None,
            'null_auc_mean': float(null_auc_mean) if not np.isnan(null_auc_mean) else None,
            'null_auc_std': float(null_auc_std) if not np.isnan(null_auc_std) else None,
            'n_permutations_success': n_perm_success,
            'model_significant_f1_0.05': p_value_f1 < 0.05 if not np.isnan(p_value_f1) else None,
            'model_significant_f1_0.01': p_value_f1 < 0.01 if not np.isnan(p_value_f1) else None,
            'model_significant_auc_0.05': p_value_auc < 0.05 if not np.isnan(p_value_auc) else None,
            'model_significant_auc_0.01': p_value_auc < 0.01 if not np.isnan(p_value_auc) else None,
            
            # Bootstrap confidence intervals
            f'ci_{confidence_level*100:.0f}_f1_lower': float(ci_f1[0]) if not np.isnan(ci_f1[0]) else None,
            f'ci_{confidence_level*100:.0f}_f1_upper': float(ci_f1[1]) if not np.isnan(ci_f1[1]) else None,
            f'ci_{confidence_level*100:.0f}_auc_lower': float(ci_auc[0]) if not np.isnan(ci_auc[0]) else None,
            f'ci_{confidence_level*100:.0f}_auc_upper': float(ci_auc[1]) if not np.isnan(ci_auc[1]) else None,
            f'ci_{confidence_level*100:.0f}_accuracy_lower': float(ci_accuracy[0]) if not np.isnan(ci_accuracy[0]) else None,
            f'ci_{confidence_level*100:.0f}_accuracy_upper': float(ci_accuracy[1]) if not np.isnan(ci_accuracy[1]) else None,
            f'ci_{confidence_level*100:.0f}_precision_lower': float(ci_precision[0]) if not np.isnan(ci_precision[0]) else None,
            f'ci_{confidence_level*100:.0f}_precision_upper': float(ci_precision[1]) if not np.isnan(ci_precision[1]) else None,
            f'ci_{confidence_level*100:.0f}_recall_lower': float(ci_recall[0]) if not np.isnan(ci_recall[0]) else None,
            f'ci_{confidence_level*100:.0f}_recall_upper': float(ci_recall[1]) if not np.isnan(ci_recall[1]) else None,
            'n_bootstrap_success': n_boot_success,
            'confidence_level': confidence_level
        }
        
        print(f"\n      [OK] Validation complete:")
        print(f"        CV: F1={result['f1']:.3f}±{result['f1_std']:.3f}, AUC={result['auc']:.3f}±{result['auc_std']:.3f}")
        print(f"        Permutation test: p(F1)={p_value_f1:.4f}, p(AUC)={p_value_auc:.4f}")
        sig_str = "SIGNIFICANT" if result.get('model_significant_f1_0.05') else "NOT significant"
        print(f"        Model performance is {sig_str} (p<0.05)")
        print(f"        {confidence_level*100:.0f}% CI: F1=[{ci_f1[0]:.3f}, {ci_f1[1]:.3f}], AUC=[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]")
        
        return result
    
    except Exception as e:
        print(f"      [WARNING] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# RESULTS LOADER (CSV OR EXCEL)
# ============================================================================

def load_results_file(results_file: str) -> pd.DataFrame:
    if results_file.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(results_file)
    else:
        return pd.read_csv(results_file)


# ============================================================================
# MAIN ANALYSIS - SINGLE CONFIG FOR A SINGLE TASK
# ============================================================================

def analyze_microbiome_results_single_config(
    results_file: str,
    input_folder: str,
    mapping_file: str,
    output_folder: str = "microbiome_shap_single_config",
    validate: bool = False,
    config_name: str = None,
    model_name: str = None,
    n_permutations: int = 100,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
):
    """
    Run SHAP analysis for ONE microbiome task and ONE specified configuration.

    Parameters:
        results_file: path to that task's results.csv (from main pipeline)
        input_folder: folder with original microbiome Excel files (perprojectmsp)
        mapping_file: abbreviation mapping Excel used in main pipeline
        output_folder: where to save SHAP outputs
        validate: if True, run CV validation with permutation test and bootstrap CI
        config_name: EXACT config_name to analyze (required)
        model_name: OPTIONAL model_name to filter by (e.g., 'RandomForest')
        n_permutations: number of label permutations for significance test (default 100)
        n_bootstrap: number of bootstrap iterations for CI (default 100)
        confidence_level: confidence level for CI (default 0.95 = 95%)
    """
    print(f"\n{'='*80}")
    print("MICROBIOME SHAP ANALYSIS - SINGLE CONFIG, SINGLE TASK")
    print(f"{'='*80}")
    print("- Uses results from comprehensive_microbiome_pipeline.py")
    print("- Rebuilds X, y, features from original Excel")
    print("- Trains with SAVED best_params (no GridSearch)")
    print("- Computes SHAP on ALL samples")
    if validate:
        print(f"- Validation includes:")
        print(f"    • Cross-validation performance")
        print(f"    • Permutation test ({n_permutations} permutations)")
        print(f"    • Bootstrap {confidence_level*100:.0f}% CI ({n_bootstrap} iterations)")
    print("- No 'top 20' or threshold selection: ONLY specified config is used")
    print(f"{'='*80}\n")
    
    if config_name is None:
        raise ValueError("You must provide config_name (exact string from results).")
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load results file
    print(f"  Loading results: {results_file}")
    results_df = load_results_file(results_file)
    print(f"  Rows in results file: {len(results_df):,}")
    
    if 'best_params' not in results_df.columns:
        print("  WARNING: 'best_params' column not found; assuming '{}' for all rows")
        results_df['best_params'] = '{}'
    
    # Read project and task info from first row (as written by main pipeline)
    required_cols = ['project', 'task_name']
    for col in required_cols:
        if col not in results_df.columns:
            raise ValueError(f"Column '{col}' not found in results_file. "
                             "Please run main pipeline as provided.")
    
    project_name = str(results_df.iloc[0]['project'])
    task_name = str(results_df.iloc[0]['task_name'])
    print(f"\n  Detected project: {project_name}")
    print(f"  Detected task:    {task_name}")
    
    # Rebuild X, y from original Excel for this task
    project_xlsx = Path(input_folder) / f"{project_name}.xlsx"
    project_xls = Path(input_folder) / f"{project_name}.xls"
    if project_xlsx.exists():
        excel_file = str(project_xlsx)
    elif project_xls.exists():
        excel_file = str(project_xls)
    else:
        raise FileNotFoundError(
            f"Could not find Excel file for project '{project_name}' in {input_folder} "
            f"(tried {project_xlsx} and {project_xls})"
        )
    
    print(f"\n  Loading task data from: {excel_file}")
    mapper = AbbreviationMapper(mapping_file)
    tasks = load_classification_tasks_from_excel(excel_file, mapper, min_samples=1)
    task_match = None
    for t in tasks:
        if t['task_name'] == task_name:
            task_match = t
            break
    
    if task_match is None:
        raise RuntimeError(f"Task '{task_name}' not found in {excel_file}. "
                           "Check that mapping_file and input_folder match the main pipeline.")
    
    X = task_match['X']
    y = task_match['y']
    feature_names = task_match['feature_names']
    print(f"  Task data: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Filter results for specified config (and optional model)
    print("\n  Filtering results for specified configuration:")
    print(f"    config_name == '{config_name}'")
    mask = (results_df['config_name'] == config_name)
    if model_name is not None:
        print(f"    model_name  == '{model_name}'")
        mask = mask & (results_df['model_name'] == model_name)
    
    selected = results_df[mask].copy()
    if selected.empty:
        print("\n  ERROR: No rows in results_file match the specified config/model.")
        print("         Check spelling and that config/model exist in this results file.")
        return
    
    print(f"  Found {len(selected)} matching row(s) for specified config/model.\n")
    selected.to_csv(output_path / 'selected_configurations.csv', index=False)
    print("  Saved selected_configurations.csv\n")
    
    all_shap = []
    all_validation = []
    
    for idx, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"\n  [{idx}/{len(selected)}] {row['config_name']} + {row['model_name']}")
        
        try:
            cfg = parse_microbiome_config(row['config_name'])
            print(f"    Parsed config: {cfg}")
            
            best_params = parse_best_params(row.get('best_params', '{}'))
            if best_params:
                print(f"    Loaded {len(best_params)} hyperparameters")
            else:
                print(f"    No hyperparameters available, using defaults")
            
            # Train
            trained_pipeline = train_with_saved_params(
                X, y, cfg, row['model_name'], best_params
            )
            
            # Optional validation with permutation test and bootstrap CI
            if validate:
                val = validate_model_with_saved_params(
                    X, y, cfg, row['model_name'], best_params, 
                    cv=5,
                    n_permutations=n_permutations,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level
                )
                if val:
                    val['index_in_selected'] = idx
                    val['config_name'] = row['config_name']
                    val['model_name'] = row['model_name']
                    val['original_f1'] = row.get('mean_f1', np.nan)
                    val['original_auc'] = row.get('mean_auc', np.nan)
                    val['project'] = project_name
                    val['task_name'] = task_name
                    all_validation.append(val)
            
            # SHAP (unchanged)
            shap_df = compute_shap_all_samples(
                trained_pipeline, X, y, feature_names, row['model_name']
            )
            
            if len(shap_df) > 0:
                shap_df['index_in_selected'] = idx
                shap_df['config_name'] = row['config_name']
                shap_df['model_name'] = row['model_name']
                shap_df['original_f1'] = row.get('mean_f1', np.nan)
                shap_df['original_auc'] = row.get('mean_auc', np.nan)
                shap_df['project'] = project_name
                shap_df['task_name'] = task_name
                all_shap.append(shap_df)
                
                safe_name = row['config_name'].replace('/', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                shap_file = output_path / f"config_{idx:02d}_{safe_name}_{row['model_name']}_SHAP.csv"
                shap_df.to_csv(shap_file, index=False)
                print(f"    Saved: {shap_file.name}")
        
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate across selected rows (often just one)
    if all_shap:
        combined = pd.concat(all_shap, ignore_index=True)
        combined.to_csv(output_path / 'ALL_SHAP_VALUES.csv', index=False)
        
        summary = combined.groupby('feature').agg({
            'mean_abs_shap': ['mean', 'std', 'min', 'max', 'count'],
            'selected': 'sum'
        }).reset_index()
        summary.columns = ['feature', 'mean_shap', 'std_shap', 'min_shap', 'max_shap', 'n_rows', 'n_selected']
        summary = summary.sort_values('mean_shap', ascending=False)
        summary.to_csv(output_path / 'FEATURE_IMPORTANCE_SUMMARY.csv', index=False)
        
        print(f"\n{'='*80}")
        print("TOP 20 MOST IMPORTANT FEATURES (Across Selected Rows)")
        print(f"{'='*80}")
        print(summary.head(20)[['feature', 'mean_shap', 'n_selected', 'n_rows']].to_string(index=False))
        print(f"{'='*80}\n")
    
    if all_validation:
        val_df = pd.DataFrame(all_validation)
        val_df.to_csv(output_path / 'VALIDATION_PERFORMANCE.csv', index=False)
        
        # Print validation summary with permutation test and CI
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY (with Permutation Test & Bootstrap CI)")
        print(f"{'='*80}")
        for _, v in val_df.iterrows():
            print(f"\nConfig: {v['config_name']} + {v['model_name']}")
            print(f"  Task: {v.get('project', 'N/A')} / {v.get('task_name', 'N/A')}")
            print(f"  Performance: F1={v['f1']:.3f}±{v['f1_std']:.3f}, AUC={v['auc']:.3f}±{v['auc_std']:.3f}")
            
            # Permutation test
            p_f1 = v.get('permutation_p_value_f1')
            p_auc = v.get('permutation_p_value_auc')
            if p_f1 is not None:
                sig_f1 = "✓ SIGNIFICANT" if v.get('model_significant_f1_0.05') else "✗ not significant"
                print(f"  Permutation test: p(F1)={p_f1:.4f} {sig_f1}")
                print(f"                    p(AUC)={p_auc:.4f}")
                print(f"  Null distribution: F1={v.get('null_f1_mean', 0):.3f}±{v.get('null_f1_std', 0):.3f}")
            
            # Confidence intervals
            ci_level = v.get('confidence_level', 0.95)
            ci_f1_lower = v.get(f'ci_{ci_level*100:.0f}_f1_lower')
            ci_f1_upper = v.get(f'ci_{ci_level*100:.0f}_f1_upper')
            ci_auc_lower = v.get(f'ci_{ci_level*100:.0f}_auc_lower')
            ci_auc_upper = v.get(f'ci_{ci_level*100:.0f}_auc_upper')
            if ci_f1_lower is not None:
                print(f"  {ci_level*100:.0f}% CI: F1=[{ci_f1_lower:.3f}, {ci_f1_upper:.3f}]")
                print(f"          AUC=[{ci_auc_lower:.3f}, {ci_auc_upper:.3f}]")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print("MICROBIOME SHAP ANALYSIS COMPLETE (SINGLE CONFIG, SINGLE TASK)")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print("\nGenerated files:")
    print("  ├── selected_configurations.csv")
    if all_shap:
        print("  ├── ALL_SHAP_VALUES.csv")
        print("  ├── FEATURE_IMPORTANCE_SUMMARY.csv")
    if all_validation:
        print("  ├── VALIDATION_PERFORMANCE.csv  (includes permutation p-values & bootstrap CIs)")
    print("  └── config_XX_[config]_[model]_SHAP.csv")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage:
    # 1) Set path to ONE task's results.csv (from main microbiome pipeline)
    # 2) Set config_to_analyze to the exact config_name string in that file
    # 3) Optionally set model_to_analyze to one model (e.g., 'RandomForest')

    results_file_path = r'microbe_results\results.csv'  # ← UPDATE
    input_folder = "input_folder"          # folder with original microbiome Excel files
    mapping_file = "abbreviation_mapping.xlsx"
    output_folder = "shap_microbiome"

    config_to_analyze = "standard_random_forest_199f(10%)_NoBalance"  # e.g., "standard_differential_50f(10%)_smote"
    model_to_analyze = "XGBoost"    # e.g., "RandomForest"

    # Permutation test and bootstrap CI settings (only used when validate=True)
    N_PERMUTATIONS = 1000  # Number of label shuffles for permutation test
    N_BOOTSTRAP = 1000     # Number of bootstrap iterations for CI
    CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

    analyze_microbiome_results_single_config(
        results_file=results_file_path,
        input_folder=input_folder,
        mapping_file=mapping_file,
        output_folder=output_folder,
        validate=True,  # Set to True to run permutation test + bootstrap CI
        config_name=config_to_analyze,
        model_name=model_to_analyze,
        n_permutations=N_PERMUTATIONS,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL
    )