"""
metabolomics_shap_analysis_columnlabels_single_config_v2.py

SHAP ANALYSIS FOR METABOLOMICS PIPELINE (COLUMN-HEADER LABELS, SINGLE CONFIG)
- Compatible with comprehensive_metabolomics_pipeline.py (column-based labels version)
- You provide:
    * results_file: path to all_results.csv (from that pipeline run)
    * data_file: the SAME Excel file used in the main pipeline
    * config_name: EXACT config_name to analyze
    * model_name: (optional) model_name to restrict to

This script will:
    - Rebuild X, y, feature_names from the original data_file
    - Parse config_name → scaling, feature_selection, n_features, balancing
    - Load best_params from results_file
    - Train the pipeline on ALL samples with saved hyperparameters
    - Compute SHAP on ALL samples
    - Validation includes:
        - Cross-validation performance metrics
        - Permutation test for model significance
        - Bootstrap confidence intervals for metrics
    - Save:
        - selected_configurations.csv
        - config_XX_[config]_[model]_SHAP.csv
        - ALL_SHAP_VALUES.csv
        - FEATURE_IMPORTANCE_SUMMARY.csv
        - VALIDATION_PERFORMANCE.csv (with permutation p-value and CIs)
"""

import os
import warnings
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
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

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)


# ============================================================================
# PREPROCESSING TRANSFORMERS (MATCH MAIN PIPELINE)
# ============================================================================

class MetabolomicsScaler(BaseEstimator, TransformerMixin):
    """Metabolomics-specific scaling."""
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


class MetabolomicsFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection - same logic as in main pipeline:
    - differential: Mann-Whitney U + FDR, effect-size weighted
    - random_forest: feature importance
    - mutual_info: mutual information
    - f_classif: ANOVA F-test
    - variance: unsupervised by variance
    """
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
# CONFIG PARSER (MATCH build_metabolomics_configs NAMING)
# ============================================================================

def parse_metabolomics_config(config_name: str) -> Dict[str, Any]:
    """
    Parse config names like:
        'standard_differential_50f(10%)_smote'
        'NoScale_AllFeatures(100%)_NoBalance'
        'robust_f_classif_150f(30%)_undersample'
    Returns dict: scaling, feature_selection, n_features, balancing
    """
    cfg = {
        'scaling': 'none',
        'feature_selection': 'none',
        'n_features': 1000,
        'balancing': 'none'
    }
    
    parts = config_name.split('_')
    
    # Scaling
    if 'NoScale' in parts:
        cfg['scaling'] = 'none'
    elif 'standard' in parts:
        cfg['scaling'] = 'standard'
    elif 'robust' in parts:
        cfg['scaling'] = 'robust'
    elif 'log' in parts:
        cfg['scaling'] = 'log'
    
    # Feature selection
    if 'AllFeatures' in config_name:
        cfg['feature_selection'] = 'none'
    elif 'mutual' in config_name or 'info' in config_name:
        cfg['feature_selection'] = 'mutual_info'
    elif 'classif' in config_name or 'f_classif' in parts:
        cfg['feature_selection'] = 'f_classif'
    elif 'random' in parts and 'forest' in parts:
        cfg['feature_selection'] = 'random_forest'
    elif 'differential' in parts:
        cfg['feature_selection'] = 'differential'
    elif 'variance' in parts:
        cfg['feature_selection'] = 'variance'
    
    # n_features from "XXXf(YY%)"
    for part in parts:
        if 'f(' in part and '%' in part:
            try:
                num_str = part.split('f(')[0]
                cfg['n_features'] = int(num_str)
            except:
                pass
    
    # Balancing
    if 'NoBalance' in parts:
        cfg['balancing'] = 'none'
    elif 'smote' in parts:
        cfg['balancing'] = 'smote'
    elif 'adasyn' in parts:
        cfg['balancing'] = 'adasyn'
    elif 'undersample' in parts:
        cfg['balancing'] = 'undersample'
    
    return cfg


# ============================================================================
# MODELS (MATCH MAIN PIPELINE)
# ============================================================================

def get_model_instance(model_name: str):
    models = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric='logloss',
            n_jobs=1, verbosity=0, use_label_encoder=False
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
# PIPELINE BUILDER (MATCH MAIN PIPELINE; FIXED k_neighbors=5)
# ============================================================================

def build_metabolomics_pipeline(config_dict: Dict[str, Any], model):
    steps = []
    if config_dict['scaling'] != 'none':
        steps.append(('scaler', MetabolomicsScaler(method=config_dict['scaling'])))
    if config_dict['feature_selection'] != 'none':
        steps.append(('feature_selector', MetabolomicsFeatureSelector(
            method=config_dict['feature_selection'],
            n_features=config_dict['n_features']
        )))
    if config_dict['balancing'] == 'smote':
        steps.append(('balancer', SMOTE(random_state=RANDOM_SEED, k_neighbors=5)))
    elif config_dict['balancing'] == 'adasyn':
        steps.append(('balancer', ADASYN(random_state=RANDOM_SEED)))
    elif config_dict['balancing'] == 'undersample':
        steps.append(('balancer', RandomUnderSampler(random_state=RANDOM_SEED)))
    steps.append(('clf', model))
    return Pipeline(steps=steps)


# ============================================================================
# HYPERPARAMETER PARSER
# ============================================================================

def parse_best_params(params_str: str) -> Dict:
    """
    Parse best_params string from all_results.csv.
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
    
    pipeline = build_metabolomics_pipeline(config_dict, base_model)
    
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
    
    if shap_values.ndim == 1:
        raise ValueError(f"SHAP is 1D: {shap_values.shape}")
    
    if shap_values.ndim == 2:
        if shap_values.shape == (n_samples, n_features):
            return shap_values
        if shap_values.shape == (n_features, n_samples):
            return shap_values.T
        return shap_values if shap_values.shape[0] == n_samples else shap_values.T
    
    if shap_values.ndim == 3:
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
    
    raise ValueError(f"Unexpected SHAP dimensionality: {shap_values.shape}")


def compute_shap_all_samples(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    original_feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """Compute SHAP values on ALL samples and map to original features."""
    print(f"    Computing SHAP values for {model_name}...")
    
    try:
        X_transformed = X.copy()
        feature_indices = list(range(len(original_feature_names)))
        
        for step_name, transformer in pipeline.steps[:-1]:
            if step_name == 'feature_selector':
                X_transformed = transformer.transform(X_transformed)
                feature_indices = [feature_indices[i] for i in transformer.selected_features_]
            elif step_name == 'balancer':
                continue
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
        for i, orig_idx in enumerate(feature_indices):
            if i < len(mean_abs_shap_selected):
                shap_all_features[orig_idx] = float(mean_abs_shap_selected[i])
        
        shap_df = pd.DataFrame({
            'feature': original_feature_names,
            'mean_abs_shap': shap_all_features,
            'selected': [i in feature_indices for i in range(len(original_feature_names))]
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        print(f"      [OK] SHAP: {n_samples_shap} samples, {len(feature_indices)} features")
        return shap_df
    
    except Exception as e:
        print(f"      [WARNING] SHAP failed: {e}")
        # Fallback to importances/coefs if available
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
        pipeline = build_metabolomics_pipeline(config_dict, model)
        
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
) -> Dict:
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
# DATA LOADER (COLUMN HEADERS AS LABELS) - COPIED FROM MAIN PIPELINE
# ============================================================================

def extract_class_from_column_name(
    col_name: str,
    cirrhosis_classes: List[str] = ['ACLF', 'AD', 'CC'],
    healthy_classes: List[str] = ['HS']
) -> Optional[str]:
    """
    Extract class label from column header.
    Returns:
        'liver_cirrhosis' for ACLF, AD, CC
        'healthy' for HS
        None if no match
    """
    col_upper = str(col_name).upper().strip()
    
    for cls in cirrhosis_classes:
        pattern = rf'^{cls}[_\-\.\d\s]|^{cls}$'
        if re.match(pattern, col_upper):
            return 'liver_cirrhosis'
    
    for cls in healthy_classes:
        pattern = rf'^{cls}[_\-\.\d\s]|^{cls}$'
        if re.match(pattern, col_upper):
            return 'healthy'
    
    return None


def load_metabolomics_data_from_column_headers(
    data_file: str,
    cirrhosis_classes: List[str] = ['ACLF', 'AD', 'CC'],
    healthy_classes: List[str] = ['HS'],
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series, Dict]:
    """
    Load metabolomics data where column headers contain class annotations.
    (Same logic as in comprehensive_metabolomics_pipeline.py,
     but output_dir is optional and used only for saving auxiliary files.)
    """
    print(f"\n{'='*80}")
    print("LOADING METABOLOMICS DATA (COLUMN HEADERS AS LABELS)")
    print(f"{'='*80}")
    print(f"Data file: {data_file}")
    print(f"\nClass grouping:")
    print(f"  → liver_cirrhosis: {cirrhosis_classes}")
    print(f"  → healthy: {healthy_classes}")
    
    df = pd.read_excel(data_file)
    print(f"\nRaw data shape: {df.shape}")
    print(f"First column (features): '{df.columns[0]}'")
    print(f"Sample columns: {df.shape[1] - 1}")
    
    feature_col = df.columns[0]
    feature_names_raw = df[feature_col].values
    sample_columns = df.columns[1:].tolist()
    data_raw = df.iloc[:, 1:]
    
    print(f"\nSample column examples (first 10):")
    for col in sample_columns[:10]:
        extracted = extract_class_from_column_name(col, cirrhosis_classes, healthy_classes)
        print(f"  '{col}' → {extracted}")
    
    # Filter metadata rows
    numeric_mask = []
    for i in range(df.shape[0]):
        row = df.iloc[i, 1:]
        try:
            num = pd.to_numeric(row, errors='coerce').notna().sum()
            numeric_mask.append(num > (len(row) * 0.5))
        except:
            numeric_mask.append(False)
    numeric_mask = np.array(numeric_mask)
    
    kept = numeric_mask.sum()
    removed = (~numeric_mask).sum()
    if removed > 0:
        print(f"\nFiltered {removed} metadata rows; kept {kept} feature rows")
    
    feature_names = [str(f) for f in feature_names_raw[numeric_mask]]
    data_numeric = data_raw.iloc[numeric_mask, :].apply(pd.to_numeric, errors='coerce')
    
    # Transpose to samples × features
    data_T = data_numeric.T
    data_T.columns = feature_names
    data_T.index = sample_columns
    
    print(f"\nData matrix shape (samples × features): {data_T.shape}")
    
    labels = {}
    unrecognized = []
    class_counts = {'liver_cirrhosis': [], 'healthy': []}
    original_class_counts = {cls: 0 for cls in cirrhosis_classes + healthy_classes}
    
    for col in sample_columns:
        label = extract_class_from_column_name(col, cirrhosis_classes, healthy_classes)
        if label is not None:
            labels[col] = label
            class_counts[label].append(col)
            col_upper = str(col).upper()
            for cls in cirrhosis_classes + healthy_classes:
                if col_upper.startswith(cls):
                    original_class_counts[cls] += 1
                    break
        else:
            unrecognized.append(col)
    
    print(f"\n{'='*80}")
    print("LABEL EXTRACTION RESULTS")
    print(f"{'='*80}")
    print("\nOriginal class distribution:")
    for cls, count in original_class_counts.items():
        group = 'liver_cirrhosis' if cls in cirrhosis_classes else 'healthy'
        print(f"  {cls}: {count} samples → {group}")
    
    print("\nGrouped class distribution:")
    for group, samples in class_counts.items():
        print(f"  {group}: {len(samples)} samples")
    
    if unrecognized:
        print(f"\nWARNING: {len(unrecognized)} unrecognized columns (skipped)")
        for col in unrecognized[:10]:
            print(f"    - '{col}'")
        if len(unrecognized) > 10:
            print(f"    ... and {len(unrecognized) - 10} more")
        if output_dir is not None:
            pd.DataFrame({'Unrecognized_Columns': unrecognized}).to_csv(
                str(Path(output_dir) / "unrecognized_columns.csv"), index=False
            )
    
    recognized_samples = [s for s in sample_columns if s in labels]
    X_df = data_T.loc[recognized_samples]
    y_orig = pd.Series([labels[s] for s in recognized_samples], index=recognized_samples)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_orig)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    miss = int(X_df.isnull().sum().sum())
    print(f"\nHandling missing values... missing count = {miss}")
    if miss > 0:
        print("Imputing with ZERO (fillna(0))")
        X_df = X_df.fillna(0)
    else:
        print("No missing values.")
    
    X = X_df.values.astype(float)
    feature_names_final = X_df.columns.tolist()
    
    print(f"\n{'='*80}")
    print("FINAL DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Samples:  {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Label mapping: {label_mapping}")
    print("Class distribution:")
    for cls, cnt in y_orig.value_counts().items():
        print(f"  {cls}: {cnt} ({cnt/len(y_orig)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    if output_dir is not None:
        sample_mapping = pd.DataFrame({
            'Sample_ID': recognized_samples,
            'Original_Column': recognized_samples,
            'Assigned_Class': [labels[s] for s in recognized_samples],
            'Encoded_Label': y_encoded
        })
        sample_mapping.to_csv(str(Path(output_dir) / "sample_class_mapping_shap.csv"), index=False)
        print(f"✓ Saved sample-class mapping to {output_dir}/sample_class_mapping_shap.csv")
    
    return X, y_encoded, feature_names_final, y_orig, label_mapping


# ============================================================================
# RESULTS LOADER
# ============================================================================

def load_results_file(results_file: str) -> pd.DataFrame:
    if results_file.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(results_file)
    return pd.read_csv(results_file)


# ============================================================================
# MAIN ANALYSIS - SINGLE CONFIG
# ============================================================================

def analyze_metabolomics_results_single_config(
    results_file: str,
    data_file: str,
    output_folder: str = "metabolomics_shap_columnlabels_single_config",
    validate: bool = False,
    config_name: str = None,
    model_name: str = None,
    cirrhosis_classes: List[str] = ['ACLF', 'AD', 'CC'],
    healthy_classes: List[str] = ['HS'],
    n_permutations: int = 100,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
):
    """
    Run SHAP analysis for ONE metabolomics run (column-header labels) and
    ONE specified configuration.

    Parameters:
        results_file: path to all_results.csv (from main pipeline run)
        data_file: the SAME Excel data file used in that run
        output_folder: directory for SHAP outputs
        validate: if True, run CV validation with permutation test and bootstrap CI
        config_name: EXACT config_name (required)
        model_name: OPTIONAL model_name to filter (e.g., 'RandomForest')
        cirrhosis_classes, healthy_classes: label prefixes (must match main run)
        n_permutations: number of label permutations for significance test (default 100)
        n_bootstrap: number of bootstrap iterations for CI (default 100)
        confidence_level: confidence level for CI (default 0.95 = 95%)
    """
    print(f"\n{'='*80}")
    print("METABOLOMICS SHAP ANALYSIS - COLUMN LABELS, SINGLE CONFIG")
    print(f"{'='*80}")
    print("- Uses results from comprehensive_metabolomics_pipeline.py")
    print("- Rebuilds X, y, features from original data_file")
    print("- Trains with SAVED best_params (no GridSearch)")
    print("- Computes SHAP on ALL samples")
    if validate:
        print(f"- Validation includes:")
        print(f"    • Cross-validation performance")
        print(f"    • Permutation test ({n_permutations} permutations)")
        print(f"    • Bootstrap {confidence_level*100:.0f}% CI ({n_bootstrap} iterations)")
    print("- No 'top 20' selection: ONLY specified config is used")
    print(f"{'='*80}\n")
    
    if config_name is None:
        raise ValueError("You must provide config_name (exact string from all_results.csv).")
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load data
    X, y, feature_names, y_orig, label_mapping = load_metabolomics_data_from_column_headers(
        data_file=data_file,
        cirrhosis_classes=cirrhosis_classes,
        healthy_classes=healthy_classes,
        output_dir=str(output_path)
    )
    
    # Load results
    print(f"  Loading results: {results_file}")
    results_df = load_results_file(results_file)
    print(f"  Rows in results file: {len(results_df):,}")
    
    if 'best_params' not in results_df.columns:
        print("  WARNING: 'best_params' column not found; assuming '{}' for all rows")
        results_df['best_params'] = '{}'
    
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
    
    print(f"  Found {len(selected)} matching row(s) for the specified config/model.\n")
    selected.to_csv(output_path / "selected_configurations.csv", index=False)
    print("  Saved selected_configurations.csv\n")
    
    all_shap = []
    all_validation = []
    
    for idx, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"\n  [{idx}/{len(selected)}] {row['config_name']} + {row['model_name']}")
        
        try:
            cfg = parse_metabolomics_config(row['config_name'])
            print(f"    Parsed config: {cfg}")
            
            best_params = parse_best_params(row.get('best_params', '{}'))
            if best_params:
                print(f"    Loaded {len(best_params)} hyperparameters")
            else:
                print(f"    No hyperparameters available, using defaults")
            
            # Train with saved hyperparameters
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
    
    # Aggregate across selected rows
    if all_shap:
        combined = pd.concat(all_shap, ignore_index=True)
        combined.to_csv(output_path / "ALL_SHAP_VALUES.csv", index=False)
        
        summary = combined.groupby('feature').agg({
            'mean_abs_shap': ['mean', 'std', 'min', 'max', 'count'],
            'selected': 'sum'
        }).reset_index()
        summary.columns = ['feature', 'mean_shap', 'std_shap', 'min_shap', 'max_shap', 'n_rows', 'n_selected']
        summary = summary.sort_values('mean_shap', ascending=False)
        summary.to_csv(output_path / "FEATURE_IMPORTANCE_SUMMARY.csv", index=False)
        
        print(f"\n{'='*80}")
        print("TOP 20 MOST IMPORTANT FEATURES (Across Selected Rows)")
        print(f"{'='*80}")
        print(summary.head(20)[['feature', 'mean_shap', 'n_selected', 'n_rows']].to_string(index=False))
        print(f"{'='*80}\n")
    
    if all_validation:
        val_df = pd.DataFrame(all_validation)
        val_df.to_csv(output_path / "VALIDATION_PERFORMANCE.csv", index=False)
        
        # Print validation summary with permutation test and CI
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY (with Permutation Test & Bootstrap CI)")
        print(f"{'='*80}")
        for _, v in val_df.iterrows():
            print(f"\nConfig: {v['config_name']} + {v['model_name']}")
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
    print("METABOLOMICS SHAP ANALYSIS COMPLETE (COLUMN LABELS, SINGLE CONFIG)")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print("\nGenerated files:")
    print("  ├── selected_configurations.csv")
    if all_shap:
        print("  ├── ALL_SHAP_VALUES.csv")
        print("  ├── FEATURE_IMPORTANCE_SUMMARY.csv")
    if all_validation:
        print("  ├── VALIDATION_PERFORMANCE.csv  (includes permutation p-values & bootstrap CIs)")
    print("  ├── sample_class_mapping_shap.csv")
    print("  ├── unrecognized_columns.csv (if any)")
    print("  └── config_XX_[config]_[model]_SHAP.csv")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION (EXAMPLE)
# ============================================================================

if __name__ == "__main__":
    # Example usage:
    # 1) Set results_file_path to the all_results.csv from your main pipeline run
    # 2) Set data_file_path to the SAME Excel file used there
    # 3) Set config_to_analyze to the exact config_name string in all_results.csv
    # 4) Optionally set model_to_analyze to a specific model_name

    results_file_path = "LC_Metabolomics_results/all_results.csv"  # ← UPDATE
    data_file_path = "LC_real_Metabolomics.xlsx"  # ← SAME file used in the main run
    output_folder = "shap_analysis_LC_metabolomics4"

    config_to_analyze = "standard_random_forest_41f(30%)_NoBalance"  # e.g., "standard_differential_50f(10%)_smote"
    model_to_analyze = "XGBoost"   # e.g., "RandomForest"

    CIRRHOSIS_CLASSES = ['ACLF', 'AD', 'CC']
    HEALTHY_CLASSES = ['HS']

    # Permutation test and bootstrap CI settings (only used when validate=True)
    N_PERMUTATIONS = 1000  # Number of label shuffles for permutation test
    N_BOOTSTRAP = 1000     # Number of bootstrap iterations for CI
    CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

    analyze_metabolomics_results_single_config(
        results_file=results_file_path,
        data_file=data_file_path,
        output_folder=output_folder,
        validate=True,  # Set to True to run permutation test + bootstrap CI
        config_name=config_to_analyze,
        model_name=model_to_analyze,
        cirrhosis_classes=CIRRHOSIS_CLASSES,
        healthy_classes=HEALTHY_CLASSES,
        n_permutations=N_PERMUTATIONS,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL
    )