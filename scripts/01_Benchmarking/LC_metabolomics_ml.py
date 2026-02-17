

import os
import warnings
import json
import time
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# imblearn for balancing + pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp

warnings.filterwarnings("ignore")
os.environ['LOKY_MAX_CPU_COUNT'] = str(mp.cpu_count())

# Reproducibility
np.random.seed(42)
import random
random.seed(42)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# PREPROCESSING TRANSFORMERS
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
# CONFIGURATION SYSTEM
# ============================================================================

@dataclass
class MetabolomicsConfig:
    """Metabolomics pipeline configuration - matching microbiome."""
    name: str
    scaling: str
    feature_selection: str
    n_features: int
    balancing: str
    
    def to_dict(self):
        return asdict(self)


def build_metabolomics_configs(max_features: int) -> List[MetabolomicsConfig]:
    """
    Grid with DYNAMIC feature selection based on percentages.
    """
    configs = []
    scalers = ['none', 'standard', 'robust', 'log']
    fs_methods = ['none', 'variance', 'f_classif', 'mutual_info', 'random_forest', 'differential']
    feature_percentages_for_selection = [0.1, 0.3, 0.5]
    balancers = ['none', 'smote', 'undersample']
    
    print(f"\n{'='*80}")
    print(f"BUILDING DYNAMIC CONFIGURATION GRID")
    print(f"{'='*80}")
    print(f"Total features in dataset: {max_features}")
    
    feature_counts_for_selection = []
    for p in feature_percentages_for_selection:
        count = int(max_features * p)
        feature_counts_for_selection.append(max(5, count))
    
    print(f"\nFeature selection strategy:")
    print(f"  - When FS method = 'none': uses ALL features (100% = {max_features})")
    print(f"  - When FS method != 'none': uses subsets")
    for pct, count in zip(feature_percentages_for_selection, feature_counts_for_selection):
        print(f"      {int(pct*100):3d}% → {count:4d} features")
    
    for sc in scalers:
        for fs in fs_methods:
            if fs == 'none':
                n_list = [max_features]
            else:
                n_list = sorted(list(set(feature_counts_for_selection)))
            
            for k in n_list:
                for bal in balancers:
                    if fs == 'none':
                        feat_str = f'AllFeatures(100%)'
                    else:
                        pct = (k / max_features) * 100
                        feat_str = f"{fs}_{k}f({pct:.0f}%)"
                    
                    scale_str = sc if sc != 'none' else 'NoScale'
                    bal_str = bal if bal != 'none' else 'NoBalance'
                    
                    name = f"{scale_str}_{feat_str}_{bal_str}"
                    
                    configs.append(MetabolomicsConfig(
                        name=name, 
                        scaling=sc, 
                        feature_selection=fs, 
                        n_features=k, 
                        balancing=bal
                    ))
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs):,}")
    print(f"{'='*80}\n")
    
    return configs


# ============================================================================
# MODELS + REDUCED HYPERPARAMETER GRIDS
# ============================================================================

def get_models():
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', n_jobs=1, verbosity=0, use_label_encoder=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            random_state=42, verbose=-1, n_jobs=1, force_col_wise=True
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42, max_iter=2000, solver='liblinear'
        ),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
    }
    print(f"Loaded {len(models)} models: {', '.join(models.keys())}")
    return models


def get_search_spaces():
    """REDUCED hyperparameter grids for GridSearchCV."""
    spaces = {
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
    
    print("\nHyperparameter grid sizes:")
    for model_name, params in spaces.items():
        total_combos = 1
        for param_values in params.values():
            total_combos *= len(param_values)
        print(f"  {model_name}: {total_combos} combinations")
    
    return spaces


# ============================================================================
# PIPELINE BUILDER
# ============================================================================

def build_pipeline(config: MetabolomicsConfig, model):
    steps = []
    if config.scaling != 'none':
        steps.append(('scaler', MetabolomicsScaler(method=config.scaling)))
    if config.feature_selection != 'none':
        steps.append(('feature_selector', MetabolomicsFeatureSelector(
            method=config.feature_selection, n_features=config.n_features
        )))
    if config.balancing == 'smote':
        steps.append(('balancer', SMOTE(random_state=42, k_neighbors=5)))
    
    elif config.balancing == 'undersample':
        steps.append(('balancer', RandomUnderSampler(random_state=42)))
    steps.append(('clf', model))
    return Pipeline(steps=steps)


# ============================================================================
# EVALUATOR WITH GRIDSEARCHCV
# ============================================================================

def evaluate_single_combination(
    X: np.ndarray,
    y: np.ndarray,
    config_dict: Dict,
    model_name: str,
    model_params: Dict,
    search_space: Dict,
    outer_cv: int = 5,
    inner_cv: int = 3,
    random_state: int = 42,
    combination_id: str = ""
) -> Dict:
    try:
        config = MetabolomicsConfig(**config_dict)
        model_map = {
            'RandomForest': RandomForestClassifier,
            'XGBoost': xgb.XGBClassifier,
            'LightGBM': lgb.LGBMClassifier,
            'LogisticRegression': LogisticRegression,
            'SVM_RBF': SVC,
            'MLP': MLPClassifier
        }
        base_model = model_map[model_name](**model_params)
        
        outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        scores = {m: [] for m in ['accuracy','precision','recall','f1','auc']}
        
        best_params_list = []
        
        for fold_idx, (tr, te) in enumerate(outer.split(X, y)):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            
            pipe = build_pipeline(config, clone(base_model))
            
            space = search_space.get(model_name, {})
            inner = StratifiedKFold(
                n_splits=max(2, min(inner_cv, np.bincount(ytr).min() if np.bincount(ytr).size>1 else 2)),
                shuffle=True, random_state=random_state
            )
            
            try:
                search = GridSearchCV(
                    pipe, 
                    param_grid=space, 
                    cv=inner,
                    scoring='roc_auc', 
                    n_jobs=1, 
                    error_score='raise'
                )
                search.fit(Xtr, ytr)
                best_pipe = search.best_estimator_
                best_params_list.append(search.best_params_)
            except Exception:
                pipe.fit(Xtr, ytr)
                best_pipe = pipe
                best_params_list.append({})
            
            ypred = best_pipe.predict(Xte)
            try:
                yproba = best_pipe.predict_proba(Xte)[:,1]
            except Exception:
                try:
                    yproba = best_pipe.decision_function(Xte)
                except Exception:
                    yproba = ypred.astype(float)
            
            scores['accuracy'].append(accuracy_score(yte, ypred))
            scores['precision'].append(precision_score(yte, ypred, zero_division=0))
            scores['recall'].append(recall_score(yte, ypred, zero_division=0))
            scores['f1'].append(f1_score(yte, ypred, zero_division=0))
            try:
                if len(np.unique(yte)) > 1:
                    scores['auc'].append(roc_auc_score(yte, yproba))
                else:
                    scores['auc'].append(np.nan)
            except Exception:
                scores['auc'].append(np.nan)
        
        best_params_aggregated = {}
        if best_params_list and any(best_params_list):
            all_params = {}
            for params in best_params_list:
                for key, val in params.items():
                    if key not in all_params:
                        all_params[key] = []
                    all_params[key].append(val)
            
            for key, values in all_params.items():
                best_params_aggregated[key] = max(set(values), key=values.count)
        
        return {
            'config_name': config.name,
            'model_name': model_name,
            'scaling': config.scaling,
            'feature_selection': config.feature_selection,
            'n_features': int(config.n_features),
            'balancing': config.balancing,
            'mean_accuracy': float(np.nanmean(scores['accuracy'])),
            'std_accuracy': float(np.nanstd(scores['accuracy'])),
            'mean_precision': float(np.nanmean(scores['precision'])),
            'std_precision': float(np.nanstd(scores['precision'])),
            'mean_recall': float(np.nanmean(scores['recall'])),
            'std_recall': float(np.nanstd(scores['recall'])),
            'mean_f1': float(np.nanmean(scores['f1'])),
            'std_f1': float(np.nanstd(scores['f1'])),
            'mean_auc': float(np.nanmean(scores['auc'])),
            'std_auc': float(np.nanstd(scores['auc'])),
            'best_params': str(best_params_aggregated),
            'success': True, 
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'config_name': config_dict.get('name','Unknown'),
            'model_name': model_name,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


class JoblibParallelEvaluator:
    """Parallel evaluator - NO TIMEOUT, runs until completion."""
    
    def __init__(self, outer_cv=5, inner_cv=3, random_state=42,
                 n_jobs=-1, verbose=10):
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.random_state = random_state
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.verbose = verbose
        self.results_ = []
        self.skipped_ = []
    
    def evaluate(self, X, y, configs, models, search_spaces):
        print(f"\nParallel eval: {len(configs)} configs × {len(models)} models = {len(configs)*len(models):,}")
        print(f"Workers: {self.n_jobs}")
        print("Using GridSearchCV for reproducibility")
        print("⚠️  NO TIMEOUT - will run until ALL combinations complete")
        
        combos = []
        for cfg in configs:
            for mname, minst in models.items():
                combos.append({
                    'X': X, 'y': y,
                    'config_dict': cfg.to_dict(),
                    'model_name': mname,
                    'model_params': minst.get_params(),
                    'search_space': search_spaces,
                    'outer_cv': self.outer_cv,
                    'inner_cv': self.inner_cv,
                    'random_state': self.random_state,
                    'combination_id': f"{cfg.name}_{mname}"
                })
        
        try:
            with parallel_backend('loky', n_jobs=self.n_jobs):
                results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(evaluate_single_combination)(**c) for c in combos
                )
        except Exception as e:
            print(f"Parallel failed: {e}\nFalling back to sequential...")
            results = [evaluate_single_combination(**c) for c in combos]
        
        ok = []
        for r in results:
            if r.get('success', False):
                ok.append(r)
            else:
                self.skipped_.append(r)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Successful: {len(ok):,}/{len(combos):,}")
        print(f"Failed: {len(self.skipped_):,}/{len(combos):,}")
        print(f"{'='*80}\n")
        
        self.results_ = ok
        return pd.DataFrame(ok) if ok else pd.DataFrame()


# ============================================================================
# DATA LOADER - COLUMN HEADERS AS CLASS LABELS (NO LABEL FILE)
# ============================================================================

def extract_class_from_column_name(col_name: str, 
                                    cirrhosis_classes: List[str] = ['ACLF', 'AD', 'CC'],
                                    healthy_classes: List[str] = ['HS']) -> Optional[str]:
    """
    Extract class label from column header.
    
    Handles formats like:
    - "ACLF_001", "ACLF-sample1", "ACLF.1"
    - "AD_sample", "AD1", "AD_2"
    - "CC_001", "CC-1"
    - "HS_001", "HS1", "HS_sample"
    
    Returns:
        'liver_cirrhosis' for ACLF, AD, CC
        'healthy' for HS
        None if no match
    """
    col_upper = str(col_name).upper().strip()
    
    # Check for cirrhosis classes
    for cls in cirrhosis_classes:
        # Match patterns like: "ACLF", "ACLF_", "ACLF-", "ACLF.", "ACLF1", etc.
        pattern = rf'^{cls}[_\-\.\d\s]|^{cls}$'
        if re.match(pattern, col_upper):
            return 'liver_cirrhosis'
    
    # Check for healthy classes
    for cls in healthy_classes:
        pattern = rf'^{cls}[_\-\.\d\s]|^{cls}$'
        if re.match(pattern, col_upper):
            return 'healthy'
    
    return None


def load_metabolomics_data_from_column_headers(
    data_file: str,
    output_dir: str,
    cirrhosis_classes: List[str] = ['ACLF', 'AD', 'CC'],
    healthy_classes: List[str] = ['HS']
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series, Dict]:
    """
    Load metabolomics data where column headers contain class annotations.
    
    Expected format:
    - First column: feature names (metabolite IDs)
    - Remaining columns: samples with class-annotated headers
      e.g., "ACLF_001", "AD_002", "CC_003", "HS_001"
    
    Args:
        data_file: Path to Excel file
        output_dir: Output directory for logs
        cirrhosis_classes: List of class prefixes to group as "liver_cirrhosis"
        healthy_classes: List of class prefixes to group as "healthy"
    
    Returns:
        X: Feature matrix (samples × features)
        y: Encoded labels
        feature_names: List of feature names
        y_orig: Original string labels
        label_mapping: Dict mapping class names to encoded values
    """
    print(f"\n{'='*80}")
    print(f"LOADING METABOLOMICS DATA (COLUMN HEADERS AS LABELS)")
    print(f"{'='*80}")
    print(f"Data file: {data_file}")
    print(f"\nClass grouping:")
    print(f"  → liver_cirrhosis: {cirrhosis_classes}")
    print(f"  → healthy: {healthy_classes}")
    
    # Load data
    df = pd.read_excel(data_file)
    print(f"\nRaw data shape: {df.shape}")
    print(f"First column (features): '{df.columns[0]}'")
    print(f"Sample columns: {df.shape[1] - 1}")
    
    # Feature names from first column
    feature_col = df.columns[0]
    feature_names_raw = df[feature_col].values
    
    # Sample data (all columns except first)
    sample_columns = df.columns[1:].tolist()
    data_raw = df.iloc[:, 1:]
    
    print(f"\nSample column examples (first 10):")
    for col in sample_columns[:10]:
        extracted = extract_class_from_column_name(col, cirrhosis_classes, healthy_classes)
        print(f"  '{col}' → {extracted}")
    
    # Filter metadata rows (keep only numeric feature rows)
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
    
    # Transpose: samples × features
    data_T = data_numeric.T
    data_T.columns = feature_names
    data_T.index = sample_columns
    
    print(f"\nData matrix shape (samples × features): {data_T.shape}")
    
    # Extract labels from column names
    labels = {}
    unrecognized = []
    class_counts = {'liver_cirrhosis': [], 'healthy': []}
    original_class_counts = {cls: 0 for cls in cirrhosis_classes + healthy_classes}
    
    for col in sample_columns:
        label = extract_class_from_column_name(col, cirrhosis_classes, healthy_classes)
        if label is not None:
            labels[col] = label
            class_counts[label].append(col)
            # Track original class
            col_upper = str(col).upper()
            for cls in cirrhosis_classes + healthy_classes:
                if col_upper.startswith(cls):
                    original_class_counts[cls] += 1
                    break
        else:
            unrecognized.append(col)
    
    print(f"\n{'='*80}")
    print(f"LABEL EXTRACTION RESULTS")
    print(f"{'='*80}")
    print(f"\nOriginal class distribution:")
    for cls, count in original_class_counts.items():
        group = 'liver_cirrhosis' if cls in cirrhosis_classes else 'healthy'
        print(f"  {cls}: {count} samples → {group}")
    
    print(f"\nGrouped class distribution:")
    for group, samples in class_counts.items():
        print(f"  {group}: {len(samples)} samples")
    
    if unrecognized:
        print(f"\n⚠️  WARNING: {len(unrecognized)} unrecognized columns (will be skipped):")
        for col in unrecognized[:10]:
            print(f"    - '{col}'")
        if len(unrecognized) > 10:
            print(f"    ... and {len(unrecognized) - 10} more")
        
        # Save unrecognized for reference
        pd.DataFrame({'Unrecognized_Columns': unrecognized}).to_csv(
            f"{output_dir}/unrecognized_columns.csv", index=False
        )
    
    # Filter to only samples with recognized labels
    recognized_samples = [s for s in sample_columns if s in labels]
    X_df = data_T.loc[recognized_samples]
    y_orig = pd.Series([labels[s] for s in recognized_samples], index=recognized_samples)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_orig)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # Zero imputation
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
    print(f"FINAL DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Samples:  {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Label mapping: {label_mapping}")
    print("Class distribution:")
    for cls, cnt in y_orig.value_counts().items():
        print(f"  {cls}: {cnt} ({cnt/len(y_orig)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    # Save sample-to-class mapping for reference
    sample_mapping = pd.DataFrame({
        'Sample_ID': recognized_samples,
        'Original_Column': recognized_samples,
        'Assigned_Class': [labels[s] for s in recognized_samples],
        'Encoded_Label': y_encoded
    })
    sample_mapping.to_csv(f"{output_dir}/sample_class_mapping.csv", index=False)
    print(f"✓ Saved sample-class mapping to {output_dir}/sample_class_mapping.csv")
    
    return X, y_encoded, feature_names_final, y_orig, label_mapping


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results_df: pd.DataFrame, output_dir: str):
    plot_dir = Path(output_dir)/'plots'
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Top 20 by AUC
    top20 = results_df.nlargest(20, 'mean_auc')
    plt.figure(figsize=(14,10))
    y_pos = np.arange(len(top20))
    plt.barh(y_pos, top20['mean_auc'], xerr=top20['std_auc'], alpha=0.85)
    plt.yticks(y_pos, [f"{r['config_name']}\n{r['model_name']}" for _, r in top20.iterrows()], fontsize=8)
    plt.xlabel('Mean AUC'); plt.title('Top 20 Configurations by AUC', fontweight='bold')
    plt.grid(axis='x', alpha=0.3); plt.tight_layout()
    plt.savefig(plot_dir/'top20_auc.png', dpi=300, bbox_inches='tight'); plt.close()
    
    # Model comparison
    model_stats = results_df.groupby('model_name').agg({'mean_auc':['mean','std'], 'mean_f1':['mean','std']}).round(3)
    fig, axes = plt.subplots(1,2, figsize=(14,6))
    for ax,(metric,title) in zip(axes, [('mean_auc','AUC'), ('mean_f1','F1-Score')]):
        idx = model_stats.index
        means = model_stats[(metric,'mean')]; stds = model_stats[(metric,'std')]
        ax.bar(range(len(idx)), means, yerr=stds, capsize=5, alpha=0.85)
        ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx, rotation=45, ha='right')
        ax.set_ylabel(title); ax.set_title(f'Model Comparison - {title}', fontweight='bold'); ax.grid(axis='y', alpha=0.3)
        for i,(m,s) in enumerate(zip(means, stds)):
            ax.text(i, m, f'{m:.3f}\n±{s:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(plot_dir/'model_comparison.png', dpi=300, bbox_inches='tight'); plt.close()
    
    # Preprocessing comparison
    scaling_stats = results_df.groupby('scaling')['mean_auc'].agg(['mean','std']).round(3)
    fs_stats = results_df.groupby('feature_selection')['mean_auc'].agg(['mean','std']).round(3)
    bal_stats = results_df.groupby('balancing')['mean_auc'].agg(['mean','std']).round(3)
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    for ax, stats_df, title in zip(axes, [scaling_stats, fs_stats, bal_stats],
                                   ['Scaling', 'Feature Selection', 'Balancing']):
        idx = stats_df.index; means = stats_df['mean']; stds = stats_df['std']
        ax.bar(range(len(idx)), means, yerr=stds, capsize=5, alpha=0.85)
        ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx, rotation=45, ha='right')
        ax.set_ylabel('Mean AUC'); ax.set_title(f'{title} Performance', fontweight='bold'); ax.grid(axis='y', alpha=0.3)
        for i,(m,s) in enumerate(zip(means, stds)):
            ax.text(i, m, f'{m:.3f}\n±{s:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig(plot_dir/'preprocessing_comparison.png', dpi=300, bbox_inches='tight'); plt.close()
    print("✓ Plots saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_comprehensive_metabolomics_analysis(
    data_file: str,
    output_folder: str = "metabolomics_results",
    n_jobs: int = -1,
    cirrhosis_classes: List[str] = ['ACLF', 'AD', 'CC'],
    healthy_classes: List[str] = ['HS']
):
    """
    Run comprehensive metabolomics analysis with column-header-based labels.
    
    Args:
        data_file: Path to data file (features × samples, column headers are class-annotated)
        output_folder: Output directory name
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        cirrhosis_classes: List of prefixes for liver cirrhosis class
        healthy_classes: List of prefixes for healthy class
    """
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"METABOLOMICS ANALYSIS - COLUMN HEADERS AS LABELS")
    print(f"{'='*80}")
    print("  ✓ NO SEPARATE LABEL FILE - uses column headers")
    print(f"  ✓ Groups: {cirrhosis_classes} → liver_cirrhosis")
    print(f"  ✓ Groups: {healthy_classes} → healthy")
    print("  ✓ 6 Models (RF, XGBoost, LightGBM, LogReg, SVM, MLP)")
    print("  ✓ 6 FS methods (incl. differential Mann-Whitney + FDR)")
    print("  ✓ 3 Balancing (SMOTE, undersample, none)")
    print("  ✓ Zero imputation, 5-fold CV, inner 3-fold GridSearchCV")
    print("  ✓ SMART feature selection:")
    print("      - 'none' → 100% of features (no selection)")
    print("      - Other methods → 10%, 30%, 50% (actual selection)")
    print("  ✓ NO TIMEOUT - will complete all combinations\n")
    
    out_dir = f"{output_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    
    # Load data using column headers as labels
    X, y, feature_names, y_orig, label_mapping = load_metabolomics_data_from_column_headers(
        data_file=data_file,
        output_dir=out_dir,
        cirrhosis_classes=cirrhosis_classes,
        healthy_classes=healthy_classes
    )
    
    configs = build_metabolomics_configs(max_features=X.shape[1])
    models = get_models()
    spaces = get_search_spaces()
    
    evaluator = JoblibParallelEvaluator(
        outer_cv=5, inner_cv=3, random_state=42,
        n_jobs=n_jobs, verbose=10
    )
    results_df = evaluator.evaluate(X, y, configs, models, spaces)
    if results_df.empty:
        print("No successful runs. Exiting.")
        return
    
    results_df.to_csv(f"{out_dir}/all_results.csv", index=False)
    
    print(f"\n{'='*80}\nTOP 10 RESULTS BY AUC\n{'='*80}")
    top10_df = results_df.nlargest(10, 'mean_auc')[
        ['config_name','model_name','mean_auc','std_auc','mean_f1','best_params']
    ]
    print(top10_df.to_string(index=False))
    
    best_idx = results_df['mean_auc'].idxmax()
    best = results_df.loc[best_idx]
    print(f"\n{'='*80}\nBEST CONFIGURATION\n{'='*80}")
    print(f"Config: {best['config_name']}")
    print(f"Model: {best['model_name']}")
    print(f"AUC: {best['mean_auc']:.4f} ± {best['std_auc']:.4f}")
    print(f"F1:  {best['mean_f1']:.4f} ± {best['std_f1']:.4f}")
    print(f"Best Hyperparameters: {best['best_params']}")
    print(f"{'='*80}\n")
    
    create_visualizations(results_df, out_dir)
    
    runtime_hours = (time.time() - start_time) / 3600
    summary = {
        'settings': {
            'label_source': 'column_headers',
            'class_grouping': {
                'liver_cirrhosis': cirrhosis_classes,
                'healthy': healthy_classes
            },
            'cv_strategy': 'StratifiedKFold with GridSearchCV',
            'outer_folds': 5, 'inner_folds': 3, 'random_state': 42,
            'models': list(models.keys()),
            'feature_selection_methods': ['none','variance','f_classif','mutual_info','random_forest','differential'],
            'feature_selection_strategy': {
                'none': '100% (all features)',
                'others': '10%, 30%, 50% (subsets)'
            },
            'balancing_methods': ['none','smote','undersample'],
            'imputation': 'zero',
            'timeout': 'None (runs to completion)',
            'hyperparameter_tuning': 'GridSearchCV (exhaustive)'
        },
        'data': {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'label_mapping': {str(k): int(v) for k,v in label_mapping.items()},
            'class_distribution': y_orig.value_counts().to_dict()
        },
        'best_result': {
            'config': str(best['config_name']),
            'model': str(best['model_name']),
            'auc': float(best['mean_auc']),
            'auc_std': float(best['std_auc']),
            'f1': float(best['mean_f1']),
            'f1_std': float(best['std_f1']),
            'accuracy': float(best['mean_accuracy']),
            'accuracy_std': float(best['std_accuracy']),
            'best_hyperparameters': str(best['best_params'])
        },
        'runtime_hours': float(runtime_hours),
        'runtime_minutes': float(runtime_hours * 60)
    }
    with open(f"{out_dir}/summary.json",'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save best hyperparameters separately
    best_params_file = f"{out_dir}/best_hyperparameters.txt"
    with open(best_params_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BEST MODEL HYPERPARAMETERS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration: {best['config_name']}\n")
        f.write(f"Model: {best['model_name']}\n")
        f.write(f"AUC: {best['mean_auc']:.4f} ± {best['std_auc']:.4f}\n")
        f.write(f"F1: {best['mean_f1']:.4f} ± {best['std_f1']:.4f}\n\n")
        f.write("Hyperparameters:\n")
        f.write("-"*80 + "\n")
        f.write(f"{best['best_params']}\n")
        f.write("="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total runtime: {runtime_hours:.2f} hours ({runtime_hours*60:.1f} minutes)")
    print(f"Results saved to: {out_dir}")
    print(f"  ├── all_results.csv")
    print(f"  ├── summary.json")
    print(f"  ├── best_hyperparameters.txt")
    print(f"  ├── sample_class_mapping.csv")
    print(f"  └── plots/")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURATION - UPDATE THESE FOR YOUR DATA
    # ==========================================================================
    
    # Your data file (features in first column, samples as columns with class prefixes)
    DATA_FILE = 'LC_Metabolomics.xlsx'
    
    # Output directory
    OUTPUT_DIR = 'LC_Metabolomics_LiverCirrhosis_vs_Healthy'
    
    # Number of parallel workers (use -1 for all CPUs)
    N_JOBS = -1
    
    # Class groupings - MODIFY THESE IF YOUR COLUMN HEADERS USE DIFFERENT PREFIXES
    # Columns starting with these prefixes will be grouped as "liver_cirrhosis"
    CIRRHOSIS_CLASSES = ['ACLF', 'AD', 'CC']
    
    # Columns starting with these prefixes will be grouped as "healthy"
    HEALTHY_CLASSES = ['HS']
    
    # ==========================================================================
    # RUN ANALYSIS
    # ==========================================================================
    
    run_comprehensive_metabolomics_analysis(
        data_file=DATA_FILE,
        output_folder=OUTPUT_DIR,
        n_jobs=N_JOBS,
        cirrhosis_classes=CIRRHOSIS_CLASSES,
        healthy_classes=HEALTHY_CLASSES
    )