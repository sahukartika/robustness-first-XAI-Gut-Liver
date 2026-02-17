
from typing import List, Dict, Any, Tuple, Optional, Set
import os
import warnings
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
def load_classification_tasks_from_excel_with_subset(
    excel_file: str,
    mapper: AbbreviationMapper,
    allowed_features: Set[str],
    min_samples: int = 25
) -> List[Dict]:
    """
    Variant of load_classification_tasks_from_excel that restricts to
    a GIVEN SET of feature names (allowed_features), e.g. triple-common
    microbes.

    Only features whose column names are in allowed_features will be used
    for building X (disease vs healthy) for each task.
    """
    print(f"\nLoading (with feature subset) from: {excel_file}")
    print(f"  Allowed features in subset: {len(allowed_features)}")
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
            df = df.T  # samples as rows, features as columns
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

        # Original intersection of features present in both disease & healthy
        common_features = list(set(disease_df.columns) & set(healthy_df.columns))
        if not common_features:
            continue

        # RESTRICT to allowed_features
        common_features_subset = [f for f in common_features if f in allowed_features]
        if len(common_features_subset) == 0:
            print(f"    Skipping task {disease_abbr}_{geo_abbr}_{seq_abbr}: "
                  f"no overlap with allowed_features.")
            continue

        print(f"    Task {disease_abbr}_{geo_abbr}_{seq_abbr}: "
              f"{len(common_features)} common features → "
              f"{len(common_features_subset)} after subset filter")

        disease_df = disease_df[common_features_subset].fillna(0).astype(float)
        healthy_df = healthy_df[common_features_subset].fillna(0).astype(float)
        
        X = np.vstack([disease_df.values, healthy_df.values])
        y = np.array([1] * len(disease_df) + [0] * len(healthy_df))
        
        n_disease = np.sum(y == 1)
        n_healthy = np.sum(y == 0)
        if n_disease < min_samples or n_healthy < min_samples:
            print(f"    Skipping task {disease_abbr}_{geo_abbr}_{seq_abbr}: "
                  f"insufficient samples (D={n_disease}, H={n_healthy}).")
            continue
        
        task = {
            'X': X, 'y': y,
            'feature_names': common_features_subset,
            'task_name': f"{disease_abbr}_{geo_abbr}_{seq_abbr}",
            'disease_abbr': disease_abbr,
            'disease_full': disease_info['disease_full'],
            'geography_abbr': geo_abbr,
            'geography_full': disease_info['geography_full'],
            'sequencer_abbr': seq_abbr,
            'sequencer_full': disease_info['sequencer_full'],
            'n_disease': int(n_disease),
            'n_healthy': int(n_healthy),
            'n_features': int(len(common_features_subset))
        }
        classification_tasks.append(task)
        print(f"    ✅ {task['task_name']}: "
              f"{n_disease}D + {n_healthy}H, {len(common_features_subset)} features (subset)")
    
    return classification_tasks
# ============================================================================
# ABBREVIATION MAPPING (MICROBIOME-SPECIFIC INPUT HANDLING)
# ============================================================================
def load_feature_subset(
    feature_file: str,
    col_candidates: List[str] = None,
    fallback_col_idx: int = 0
) -> Set[str]:
    """
    Load a list of microbe feature names (e.g. triple-common microbes) from
    a CSV/Excel file and return them as a Python set.

    Assumes the file has a column with microbe IDs/names that MATCH the
    column names in your microbiome Excel data (feature columns).

    Parameters
    ----------
    feature_file : str
        Path to file containing feature names (one per row).
    col_candidates : list of str, optional
        Column names to try, e.g. ['feature', 'microbe', 'name'].
    fallback_col_idx : int
        Column index to use if no candidate column names are found.

    Returns
    -------
    set of str
        Allowed feature names (exact strings, stripped).
    """
    if col_candidates is None:
        col_candidates = ["feature", "microbe", "Microbe", "name", "Name"]

    path = Path(feature_file)
    if not path.exists():
        raise FileNotFoundError(f"Feature subset file not found: {feature_file}")

    print(f"\nLoading feature subset from: {feature_file}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    print(f"  Shape: {df.shape}")
    col_to_use = None
    for c in col_candidates:
        if c in df.columns:
            col_to_use = c
            break
    if col_to_use is None:
        col_to_use = df.columns[fallback_col_idx]
        print(f"  No candidate column name found; using column index {fallback_col_idx} = '{col_to_use}'")
    else:
        print(f"  Using column: '{col_to_use}'")

    series = df[col_to_use].astype(str).str.strip()
    series = series[series.notna() & (series != "") & (series.str.lower() != "nan")]
    subset = set(series.tolist())

    print(f"  Loaded {len(subset)} allowed features (subset).")
    return subset
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


# ============================================================================
# PREPROCESSING TRANSFORMERS (IDENTICAL TO METABOLOMICS)
# ============================================================================

class MicrobiomeScaler(BaseEstimator, TransformerMixin):
    """Scaling - IDENTICAL to metabolomics."""
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


class MicrobiomeFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection - IDENTICAL to metabolomics."""
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
# CONFIGURATION SYSTEM (IDENTICAL TO METABOLOMICS)
# ============================================================================

@dataclass
class MicrobiomeConfig:
    """Configuration - IDENTICAL to metabolomics."""
    name: str
    scaling: str
    feature_selection: str
    n_features: int
    balancing: str
    
    def to_dict(self):
        return asdict(self)


def build_microbiome_configs(max_features: int) -> List[MicrobiomeConfig]:
    """
    Grid with DYNAMIC feature selection - IDENTICAL TO METABOLOMICS:
    - Scaling: none, standard, robust, log
    - FS: 
        * 'none' → uses 100% of features (no selection needed)
        * other methods → uses 10%, 30%, 50% of features (actual selection)
    - Balancing: none, smote, adasyn, undersample
    """
    configs = []
    scalers = ['standard']
    fs_methods = ['none']
    
    # Feature percentages for ACTUAL feature selection (not 100%)
    # 100% is only used when feature_selection='none'
    feature_percentages_for_selection = [0.1, 0.3, 0.5]  # 10%, 30%, 50%
    
    balancers = ['none']
    
    print(f"\n{'='*80}")
    print(f"BUILDING DYNAMIC CONFIGURATION GRID")
    print(f"{'='*80}")
    print(f"Total features in dataset: {max_features}")
    
    # Calculate actual feature counts from percentages
    feature_counts_for_selection = []
    for p in feature_percentages_for_selection:
        count = int(max_features * p)
        # Ensure at least 5 features minimum
        feature_counts_for_selection.append(max(1, count))
    
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
                    
                    configs.append(MicrobiomeConfig(
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
# MODELS + HYPERPARAMETERS (IDENTICAL TO METABOLOMICS)
# ============================================================================

def get_models():
    models = {
        
        'XGBoost': xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', n_jobs=1, verbosity=0, use_label_encoder=False
        )
        
    }
    print(f"Loaded {len(models)} models: {', '.join(models.keys())}")
    return models


def get_search_spaces():
    """Hyperparameter grids - IDENTICAL TO METABOLOMICS."""
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
    
    # Calculate total combinations per model
    print("\nHyperparameter grid sizes:")
    for model_name, params in spaces.items():
        total_combos = 1
        for param_values in params.values():
            total_combos *= len(param_values)
        print(f"  {model_name}: {total_combos} combinations")
    
    return spaces


# ============================================================================
# PIPELINE BUILDER (IDENTICAL TO METABOLOMICS)
# ============================================================================

def build_pipeline(config: MicrobiomeConfig, model):
    """Pipeline builder - IDENTICAL TO METABOLOMICS."""
    steps = []
    # Scaling
    if config.scaling != 'none':
        steps.append(('scaler', MicrobiomeScaler(method=config.scaling)))
    # Feature selection (only if not 'none')
    if config.feature_selection != 'none':
        steps.append(('feature_selector', MicrobiomeFeatureSelector(
            method=config.feature_selection, n_features=config.n_features
        )))
    # Balancing
    if config.balancing == 'smote':
        steps.append(('balancer', SMOTE(random_state=42, k_neighbors=5)))
    elif config.balancing == 'adasyn':
        steps.append(('balancer', ADASYN(random_state=42)))
    elif config.balancing == 'undersample':
        steps.append(('balancer', RandomUnderSampler(random_state=42)))
    # Classifier
    steps.append(('clf', model))
    return Pipeline(steps=steps)


# ============================================================================
# EVALUATOR (IDENTICAL TO METABOLOMICS)
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
        config = MicrobiomeConfig(**config_dict)
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
                # GridSearchCV
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
                yproba = best_pipe.predict_proba(Xte)[:, 1]
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
            'config_name': config_dict.get('name', 'Unknown'),
            'model_name': model_name,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


class JoblibParallelEvaluator:
    """Parallel evaluator - IDENTICAL TO METABOLOMICS."""
    
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
    
    def get_best_model(self, metric='mean_auc'):
        if not self.results_:
            return None, None
        valid = [r for r in self.results_ if metric in r and pd.notna(r[metric])]
        if not valid:
            return None, None
        best = max(valid, key=lambda r: r[metric])
        return best['config_name'], best['model_name']


# ============================================================================
# DATA LOADER (ONLY DIFFERENCE: MICROBIOME-SPECIFIC MULTI-TASK INPUT)
# ============================================================================

def load_classification_tasks_from_excel(
    excel_file: str,
    mapper: AbbreviationMapper,
    min_samples: int = 25
) -> List[Dict]:
    """
    Load multi-task classification from Excel - ONLY DIFFERENCE FROM METABOLOMICS.
    
    This is the ONLY different part - everything else is identical to metabolomics!
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
        
        # Zero imputation (SAME AS METABOLOMICS)
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
        print(f"    ✅ {task['task_name']}: {n_disease}D + {n_healthy}H, {len(common_features)} features")
    
    return classification_tasks


# ============================================================================
# VISUALIZATION (IDENTICAL TO METABOLOMICS)
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
# MAIN RUNNER
# ============================================================================

def run_microbiome_analysis_with_feature_subset(
    feature_subset_file: str,
    input_folder: str = "perprojectmsp",
    mapping_file: str = "abbreviation_mapping.xlsx",
    output_folder: str = "microbiome_results_feature_subset",
    min_samples: int = 25,
    n_jobs: int = 10
):
    """
    Run the comprehensive microbiome analysis but RESTRICT all tasks
    to a given subset of microbes (features), e.g. triple-common microbes.

    Args:
        feature_subset_file: Path to CSV/Excel with feature names (one column).
        input_folder: Folder containing per-project MSP Excel files.
        mapping_file: Abbreviation mapping Excel.
        output_folder: Base output dir; a timestamp suffix will be added.
        min_samples: Minimum disease/healthy samples per task.
        n_jobs: Parallel workers (-1 for all CPUs).
    """
    start_time = time.time()
    print(f"\n{'='*80}\nMICROBIOME ANALYSIS USING FEATURE SUBSET\n{'='*80}")
    print(f"Feature subset file: {feature_subset_file}")
    
    allowed_features = load_feature_subset(feature_subset_file)
    if not allowed_features:
        print("❌ No features loaded from subset file; aborting.")
        return
    
    mapper = AbbreviationMapper(mapping_file)
    out_dir = f"{output_folder}_{len(allowed_features)}feat_{int(time.time())}"
    output_path = Path(out_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    project_files = list(Path(input_folder).glob("*.xlsx")) + list(Path(input_folder).glob("*.xls"))
    if not project_files:
        print(f"❌ No Excel files found in {input_folder}")
        return
    
    all_tasks = []
    for project_file in project_files:
        project_name = project_file.stem
        tasks = load_classification_tasks_from_excel_with_subset(
            str(project_file), mapper,
            allowed_features=allowed_features,
            min_samples=min_samples
        )
        for t in tasks:
            t['project'] = project_name
        all_tasks.extend(tasks)
    
    if not all_tasks:
        print("\n❌ No valid tasks found after applying feature subset!")
        return
    
    print(f"\nTOTAL TASKS (after subset): {len(all_tasks)}")
    max_features = min([task['n_features'] for task in all_tasks])
    
    configs = build_microbiome_configs(max_features=max_features)
    models = get_models()
    search_spaces = get_search_spaces()
    
    print(f"\nPipeline statistics (subset):")
    print(f"  Configurations: {len(configs):,}")
    print(f"  Models: {len(models)}")
    print(f"  Combinations per task: {len(configs) * len(models):,}")
    print(f"  TOTAL EVALUATIONS: {len(all_tasks) * len(configs) * len(models):,}")
    
    all_results = []
    evaluator = JoblibParallelEvaluator(
        outer_cv=5, inner_cv=3, random_state=42,
        n_jobs=n_jobs, verbose=5
    )
    
    for idx, task in enumerate(all_tasks, 1):
        print(f"\n{'='*80}")
        print(f"TASK {idx}/{len(all_tasks)}: {task['task_name']} "
              f"({task['n_features']} features from subset)")
        print(f"{'='*80}")
        
        task_start = time.time()
        results_df = evaluator.evaluate(task['X'], task['y'], configs, models, search_spaces)
        task_elapsed = time.time() - task_start
        
        if len(results_df) > 0:
            for key in ['project', 'task_name', 'disease_abbr', 'disease_full',
                        'geography_abbr', 'geography_full', 'sequencer_abbr',
                        'sequencer_full', 'n_disease', 'n_healthy', 'n_features']:
                results_df[key] = task[key]
            
            task_dir = output_path / task['project'] / task['task_name']
            task_dir.mkdir(exist_ok=True, parents=True)
            results_df.to_csv(task_dir / 'results.csv', index=False)
            all_results.append(results_df)
            
            best_config, best_model = evaluator.get_best_model('mean_auc')
            if best_config:
                best = results_df[(results_df['config_name'] == best_config) &
                                  (results_df['model_name'] == best_model)].iloc[0]
                
                best_params_file = task_dir / 'best_hyperparameters.txt'
                with open(best_params_file, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write("BEST MODEL HYPERPARAMETERS (FEATURE SUBSET)\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Configuration: {best_config}\n")
                    f.write(f"Model: {best_model}\n")
                    f.write(f"AUC: {best['mean_auc']:.4f} ± {best['std_auc']:.4f}\n")
                    f.write(f"F1: {best['mean_f1']:.4f} ± {best['std_f1']:.4f}\n\n")
                    f.write("Hyperparameters:\n")
                    f.write("-"*80 + "\n")
                    f.write(f"{best['best_params']}\n")
                    f.write("="*80 + "\n")
                
                print(f"\n🏆 BEST (subset): {best_config} + {best_model}")
                print(f"   AUC: {best['mean_auc']:.3f} ± {best['std_auc']:.3f}")
                print(f"   Best Params: {best['best_params']}")
            
            create_visualizations(results_df, task_dir)
        
        print(f"\n⏱️  Task completed in {task_elapsed/60:.1f} minutes")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_path / 'ALL_RESULTS.csv', index=False)
        
        runtime_hours = (time.time() - start_time) / 3600
        
        print(f"\n{'='*80}\n🎉 ALL TASKS COMPLETED (FEATURE SUBSET)\n{'='*80}")
        print(f"Total results: {len(combined):,}")
        print(f"Total runtime: {runtime_hours:.2f} hours ({runtime_hours*60:.1f} minutes)")
        
        top_auc = combined.nlargest(10, 'mean_auc')[
            ['project', 'task_name', 'disease_full', 'model_name',
             'config_name', 'mean_auc', 'std_auc', 'mean_f1', 'best_params']
        ]
        print(f"\n📊 TOP 10 RESULTS BY AUC (subset):\n")
        print(top_auc.to_string(index=False))
        
        summary = combined.groupby(['config_name', 'model_name']).agg({
            'mean_auc': ['mean', 'std', 'min', 'max'],
            'mean_f1': ['mean', 'std', 'min', 'max']
        }).round(3)
        summary.to_csv(output_path / 'SUMMARY_STATISTICS.csv')
        
        summary_dict = {
            'settings': {
                'cv_strategy': 'StratifiedKFold with GridSearchCV',
                'outer_folds': 5, 'inner_folds': 3, 'random_state': 42,
                'models': list(models.keys()),
                'feature_selection_methods': ['none','variance','f_classif','mutual_info','random_forest','differential'],
                'feature_selection_strategy': {
                    'none': '100% (all subset features)',
                    'others': '10%, 30%, 50% (of subset)'
                },
                'balancing_methods': ['none','smote','adasyn','undersample'],
                'pipeline_order': 'Scaling → FeatureSelection → Balancing → Classifier',
                'timeout': 'None',
                'hyperparameter_tuning': 'GridSearchCV',
                'feature_subset_file': feature_subset_file,
                'n_allowed_features': len(allowed_features)
            },
            'data': {
                'n_tasks': len(all_tasks),
                'total_combinations': len(combined)
            },
            'runtime_hours': float(runtime_hours),
            'runtime_minutes': float(runtime_hours * 60)
        }
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary_dict, f, indent=2)
    else:
        print("\n❌ No results generated")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    input_folder = "msp5"
    mapping_file = "abbreviation_mapping.xlsx"
    triple_common_file = r"feature_comparison_shap_results\across_micro_features\triple_common_micro_features_CD_CRC_LC_with_shap.csv"  # ← UPDATE THIS
    output_folder = "signature_validation"

    available_cpus = mp.cpu_count()
    n_jobs = 1  # or any number you like

    print(f"Available CPUs: {available_cpus}")
    print(f"Using {n_jobs} workers")

    run_microbiome_analysis_with_feature_subset(
        feature_subset_file=triple_common_file,
        input_folder=input_folder,
        mapping_file=mapping_file,
        output_folder=output_folder,
        min_samples=25,
        n_jobs=n_jobs
    )