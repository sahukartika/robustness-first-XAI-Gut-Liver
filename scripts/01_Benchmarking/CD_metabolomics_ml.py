

import os
import warnings
import json
import time
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
            # log1p with pseudocount inferred from data
            min_nonzero = X[X > 0].min() if np.any(X > 0) else 1e-6
            pseudocount = min_nonzero / 2
            return np.log1p(X + pseudocount)
        elif self.scaler_ is not None:
            return self.scaler_.transform(X)
        else:
            return X


class MetabolomicsFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection - matching microbiome:
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
    
    configs = []
    scalers = ['none', 'standard', 'robust', 'log']
    fs_methods = ['none', 'variance', 'f_classif', 'mutual_info', 'random_forest', 'differential']
    
    # Feature percentages for ACTUAL feature selection (not 100%)
    # 100% is only used when feature_selection='none'
    feature_percentages_for_selection = [0.1, 0.3, 0.5]  # 10%, 30%, 50%
    
    balancers = ['none', 'smote', 'undersample']
    
    print(f"\n{'='*80}")
    print(f"BUILDING DYNAMIC CONFIGURATION GRID")
    print(f"{'='*80}")
    print(f"Total features in dataset: {max_features}")
    
    # Calculate actual feature counts from percentages
    feature_counts_for_selection = []
    for p in feature_percentages_for_selection:
        count = int(max_features * p)
        # Ensure at least 5 features minimum
        feature_counts_for_selection.append(max(5, count))
    
    print(f"\nFeature selection strategy:")
    print(f"  - When FS method = 'none': uses ALL features (100% = {max_features})")
    print(f"  - When FS method != 'none': uses subsets")
    for pct, count in zip(feature_percentages_for_selection, feature_counts_for_selection):
        print(f"      {int(pct*100):3d}% → {count:4d} features")
    
    for sc in scalers:
        for fs in fs_methods:
            if fs == 'none':
                # When no feature selection, use ALL features (100%)
                n_list = [max_features]
            else:
                # When using feature selection methods, use subsets (10%, 30%, 50%)
                # Remove duplicates and sort
                n_list = sorted(list(set(feature_counts_for_selection)))
            
            for k in n_list:
                for bal in balancers:
                    # Create readable name with percentage
                    if fs == 'none':
                        feat_str = f'AllFeatures(100%)'
                    else:
                        # Calculate percentage for this k value
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
    print(f"  - Scaling methods: {len(scalers)}")
    print(f"  - Feature selection:")
    print(f"      * 'none' (100% features): 1 option")
    print(f"      * Other methods (10%, 30%, 50%): 5 methods × 3 percentages")
    print(f"  - Balancing methods: {len(balancers)}")
    print(f"\nBreakdown:")
    print(f"  - Configs with NO feature selection: {len([c for c in configs if c.feature_selection == 'none']):,}")
    print(f"  - Configs WITH feature selection: {len([c for c in configs if c.feature_selection != 'none']):,}")
    print(f"{'='*80}\n")
    
    return configs


# ============================================================================
# MODELS + REDUCED HYPERPARAMETER GRIDS (FOR GRIDSEARCHCV)
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
    """
    REDUCED hyperparameter grids for GridSearchCV.
    Focused on key parameters with reasonable ranges for reproducibility.
    """
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
            'clf__num_leaves': [15, 31],
            'clf__subsample': [0.8, 1.0],
            'clf__colsample_bytree': [0.8, 1.0],
            'clf__reg_alpha': [0, 0.1],
            'clf__reg_lambda': [1, 10],
            'clf__min_child_samples': [2, 5, 10],  # ✅ Added for small samples
            'clf__min_data_in_leaf': [1, 3, 5],    # ✅ Added for small samples
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
    
    # Calculate total combinations per model
    print("\nHyperparameter grid sizes:")
    for model_name, params in spaces.items():
        total_combos = 1
        for param_values in params.values():
            total_combos *= len(param_values)
        print(f"  {model_name}: {total_combos} combinations")
    
    return spaces


# ============================================================================
# PIPELINE BUILDER (WITH BALANCING)
# ============================================================================

def build_pipeline(config: MetabolomicsConfig, model):
    steps = []
    # Scaling
    if config.scaling != 'none':
        steps.append(('scaler', MetabolomicsScaler(method=config.scaling)))
    # Feature selection (only if not 'none')
    if config.feature_selection != 'none':
        steps.append(('feature_selector', MetabolomicsFeatureSelector(
            method=config.feature_selection, n_features=config.n_features
        )))
    # Balancing
    if config.balancing == 'smote':
        steps.append(('balancer', SMOTE(random_state=42, k_neighbors=2)))
    
    elif config.balancing == 'undersample':
        steps.append(('balancer', RandomUnderSampler(random_state=42)))
    # Classifier
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
        
        best_params_list = []  # Store best params from each fold
        
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
                # GridSearchCV instead of RandomizedSearchCV
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
        
        # Aggregate best parameters across folds
        best_params_aggregated = {}
        if best_params_list and any(best_params_list):
            # For each parameter, find most common value across folds
            all_params = {}
            for params in best_params_list:
                for key, val in params.items():
                    if key not in all_params:
                        all_params[key] = []
                    all_params[key].append(val)
            
            for key, values in all_params.items():
                # Most common value
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
            'best_params': str(best_params_aggregated),  # Save best hyperparameters
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
# DATA LOADER (4 FILES) - ZERO IMPUTATION + METADATA FILTER
# ============================================================================

def load_metabolomics_data_4files(
    forward_pos_file: str,
    forward_neg_file: str,
    reverse_pos_file: str,
    reverse_neg_file: str,
    label_file: str,
    output_dir: str
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series, Dict]:
    """
    Load 4 files (fwd_pos, fwd_neg, rev_pos, rev_neg) where:
    - First column: feature names (may include metadata rows like 'retention index')
    - Remaining columns: sample IDs
    Zero imputation and flexible label parsing included.
    """
    print(f"\n{'='*80}\nLOADING METABOLOMICS DATA (4 FILES)\n{'='*80}")
    
    print(f"\n1. Loading forward positive: {forward_pos_file}")
    fwd_pos_df = pd.read_excel(forward_pos_file)
    print(f"   Shape: {fwd_pos_df.shape}")
    
    print(f"2. Loading forward negative: {forward_neg_file}")
    fwd_neg_df = pd.read_excel(forward_neg_file)
    print(f"   Shape: {fwd_neg_df.shape}")
    
    print(f"3. Loading reverse positive: {reverse_pos_file}")
    rev_pos_df = pd.read_excel(reverse_pos_file)
    print(f"   Shape: {rev_pos_df.shape}")
    
    print(f"4. Loading reverse negative: {reverse_neg_file}")
    rev_neg_df = pd.read_excel(reverse_neg_file)
    print(f"   Shape: {rev_neg_df.shape}")
    
    print(f"5. Loading labels: {label_file}")
    labels_df = pd.read_excel(label_file)
    print(f"   Shape: {labels_df.shape}")
    
    # Process with metadata filtering
    def process_phase(df, prefix):
        print(f"\n  Processing {prefix}...")
        print(f"    Original shape: {df.shape}")
        
        feats_raw = df.iloc[:, 0].values
        data_raw = df.iloc[:, 1:]
        
        # Identify metadata rows (non-numeric)
        numeric_mask = []
        for idx in range(len(df)):
            row_data = df.iloc[idx, 1:]
            try:
                numeric_count = pd.to_numeric(row_data, errors='coerce').notna().sum()
                is_numeric_row = numeric_count > (len(row_data) * 0.5)
                numeric_mask.append(is_numeric_row)
            except:
                numeric_mask.append(False)
        
        numeric_mask = np.array(numeric_mask)
        print(f"    Filtered {np.sum(~numeric_mask)} metadata rows; kept {np.sum(numeric_mask)} feature rows")
        
        # Filter to only numeric rows
        features = feats_raw[numeric_mask]
        data_numeric = data_raw.iloc[numeric_mask, :]
        
        # Convert to numeric
        data_numeric = data_numeric.apply(pd.to_numeric, errors='coerce')
        
        # Transpose to samples × features
        data = data_numeric.T
        data.columns = [f"{prefix}_{feat}" for feat in features]
        data.index = df.columns[1:].tolist()
        
        print(f"    Final shape: {data.shape}")
        return data
    
    print("\nProcessing phases...")
    fwd_pos_data = process_phase(fwd_pos_df, "fwd_pos")
    fwd_neg_data = process_phase(fwd_neg_df, "fwd_neg")
    rev_pos_data = process_phase(rev_pos_df, "rev_pos")
    rev_neg_data = process_phase(rev_neg_df, "rev_neg")
    
    # Combine all 4 phases
    print("\nCombining all 4 phases...")
    combined_data = pd.concat([fwd_pos_data, fwd_neg_data, rev_pos_data, rev_neg_data], axis=1)
    print(f"  Combined shape: {combined_data.shape}")
    print(f"  Total features: {combined_data.shape[1]}")
    print(f"    - Forward positive: {fwd_pos_data.shape[1]}")
    print(f"    - Forward negative: {fwd_neg_data.shape[1]}")
    print(f"    - Reverse positive: {rev_pos_data.shape[1]}")
    print(f"    - Reverse negative: {rev_neg_data.shape[1]}")
    
    # FLEXIBLE LABEL LOADING
    print("\nDetecting label file format...")
    labels_dict = {}
    
    if labels_df.shape[1] > 2:
        print("  Format: First row contains labels, columns are sample IDs")
        sample_ids_from_labels = labels_df.columns[1:].tolist()
        label_values = labels_df.iloc[0, 1:].values
        labels_dict = dict(zip(sample_ids_from_labels, label_values))
    elif labels_df.shape[1] == 2:
        print("  Format: Two columns (sample_id, label)")
        labels_df.columns = ['sample_id', 'label']
        labels_dict = dict(zip(labels_df['sample_id'], labels_df['label']))
    else:
        print("  Format: Single column pair")
        labels_dict = dict(zip(labels_df.iloc[:, 0], labels_df.iloc[:, 1]))
    
    print(f"  Extracted {len(labels_dict)} labels")
    
    # Match samples with labels
    print("\nMatching samples with labels...")
    all_samples = combined_data.index.tolist()
    samples_with_labels = [s for s in all_samples if s in labels_dict]
    samples_without_labels = [s for s in all_samples if s not in labels_dict]
    
    if samples_without_labels:
        print(f"WARNING: {len(samples_without_labels)} samples without labels (skipped)")
        pd.DataFrame({'Skipped_Samples': samples_without_labels}).to_csv(
            f"{output_dir}/skipped_samples.csv", index=False
        )
    
    # Filter to samples with labels
    X_df = combined_data.loc[samples_with_labels]
    y_orig = pd.Series([labels_dict[s] for s in samples_with_labels], index=samples_with_labels)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_orig)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # ZERO IMPUTATION
    miss = int(X_df.isnull().sum().sum())
    print(f"\nHandling missing values... missing count = {miss}")
    if miss > 0:
        print("Imputing with ZERO (fillna(0))")
        X_df = X_df.fillna(0)
    else:
        print("No missing values.")
    
    X = X_df.values.astype(float)
    feat_names = X_df.columns.tolist()
    
    print(f"\n{'='*80}\nDATA SUMMARY\n{'='*80}")
    print(f"Samples:  {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Label mapping: {label_mapping}")
    print("Class distribution:")
    for cls, cnt in y_orig.value_counts().items():
        print(f"  {cls}: {cnt}")
    print(f"{'='*80}\n")
    
    return X, y_encoded, feat_names, y_orig, label_mapping


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
    forward_pos_file: str,
    forward_neg_file: str,
    reverse_pos_file: str,
    reverse_neg_file: str,
    label_file: str,
    output_folder: str = "metabolomics_results",
    n_jobs: int = -1
):
    """
    Run comprehensive metabolomics analysis - NO TIMEOUT.
    
    Args:
        forward_pos_file, forward_neg_file, reverse_pos_file, reverse_neg_file: Data files
        label_file: Labels file
        output_folder: Output directory name
        n_jobs: Number of parallel jobs (-1 = all CPUs)
    """
    start_time = time.time()
    
    
    out_dir = f"{output_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    
    X, y, feature_names, y_orig, label_mapping = load_metabolomics_data_4files(
        forward_pos_file, forward_neg_file, reverse_pos_file, reverse_neg_file,
        label_file, out_dir
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
            'label_mapping': {str(k): int(v) for k,v in label_mapping.items()}
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
    
    # Save best hyperparameters separately for easy access
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
    print(f"  ├── all_results.csv (all combinations with hyperparameters)")
    print(f"  ├── summary.json")
    print(f"  ├── best_hyperparameters.txt")
    print(f"  └── plots/")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # FILE PATHS (UPDATE THESE TO MATCH YOUR FILES)
    FORWARD_POS_FILE = 'CD_HILIC POSITIVE ION MODE.xlsx'
    FORWARD_NEG_FILE = 'CD_HILIC NEGATIVE ION MODE.xlsx'
    REVERSE_POS_FILE = 'CD_Reversed phase POSITIVE ION MODE.xlsx'
    REVERSE_NEG_FILE = 'CD_Reversed phase NEGATIVE ION MODE.xlsx'
    LABEL_FILE = 'CD_label.xlsx'
    
    OUTPUT_DIR = 'CD_metabolomics_results'
    N_JOBS = 2  # Number of parallel workers (use -1 for all CPUs)
    
    run_comprehensive_metabolomics_analysis(
        forward_pos_file=FORWARD_POS_FILE,
        forward_neg_file=FORWARD_NEG_FILE,
        reverse_pos_file=REVERSE_POS_FILE,
        reverse_neg_file=REVERSE_NEG_FILE,
        label_file=LABEL_FILE,
        output_folder=OUTPUT_DIR,
        n_jobs=N_JOBS
    )