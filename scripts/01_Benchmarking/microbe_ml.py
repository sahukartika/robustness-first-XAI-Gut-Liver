

import os
import warnings
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Set
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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
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


# ============================================================================
# ABBREVIATION MAPPING (MICROBIOME-SPECIFIC INPUT HANDLING)
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


# ============================================================================
# PREPROCESSING TRANSFORMERS (IDENTICAL TO METABOLOMICS)
# ============================================================================

class MicrobiomeScaler(BaseEstimator, TransformerMixin):
   
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
   
    name: str
    scaling: str
    feature_selection: str
    n_features: int
    balancing: str
    
    def to_dict(self):
        return asdict(self)


def build_microbiome_configs(max_features: int) -> List[MicrobiomeConfig]:
    
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
                    
                    configs.append(MicrobiomeConfig(
                        name=name, 
                        scaling=sc, 
                        feature_selection=fs, 
                        n_features=k, 
                        balancing=bal
                    ))
    
    print(f"\nTotal configurations: {len(configs):,}")
    print(f"{'='*80}\n")
    
    return configs


# ============================================================================
# MODELS + HYPERPARAMETERS (IDENTICAL TO METABOLOMICS)
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
# PIPELINE BUILDER (IDENTICAL TO METABOLOMICS)
# ============================================================================

def build_pipeline(config: MicrobiomeConfig, model):
    
    steps = []
    if config.scaling != 'none':
        steps.append(('scaler', MicrobiomeScaler(method=config.scaling)))
    if config.feature_selection != 'none':
        steps.append(('feature_selector', MicrobiomeFeatureSelector(
            method=config.feature_selection, n_features=config.n_features
        )))
    if config.balancing == 'smote':
        steps.append(('balancer', SMOTE(random_state=42, k_neighbors=5)))
    
    elif config.balancing == 'undersample':
        steps.append(('balancer', RandomUnderSampler(random_state=42)))
    steps.append(('clf', model))
    return Pipeline(steps=steps)


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_classification_tasks_from_excel(
    excel_file: str,
    mapper: AbbreviationMapper,
    min_samples: int = 25
) -> List[Dict]:
    """Load multi-task classification from Excel."""
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
# EXTERNAL VALIDATION DATA LOADER
# ============================================================================

def load_external_validation_data(
    external_folder: str,
    mapper: AbbreviationMapper,
    training_feature_names: List[str],
    disease_abbr: str,
    min_samples: int = 5
) -> Optional[Dict]:
    """
    Load external validation data for a specific disease.
    
    Searches through Excel files in external_folder to find data matching
    the disease abbreviation. Returns data aligned with training features.
    
    Args:
        external_folder: Folder containing external validation Excel files
        mapper: Abbreviation mapper
        training_feature_names: List of feature names from training data (ordered)
        disease_abbr: Disease abbreviation to find
        min_samples: Minimum samples required
        
    Returns:
        Dict with X, y, metadata, or None if not found
    """
    external_path = Path(external_folder)
    if not external_path.exists():
        print(f"    ⚠️  External folder not found: {external_folder}")
        return None
    
    excel_files = list(external_path.glob("*.xlsx")) + list(external_path.glob("*.xls"))
    if not excel_files:
        print(f"    ⚠️  No Excel files in external folder")
        return None
    
    print(f"    Searching for disease '{disease_abbr}' in {len(excel_files)} external files...")
    
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
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)
                    df = df.T  # Transpose: samples as rows
                    sheet_info['data'] = df
                    sheet_info['sheet_name'] = sheet_name
                    
                    if sheet_info['disease_abbr'] == disease_abbr:
                        disease_sheets.append(sheet_info)
                    elif sheet_info['is_healthy']:
                        healthy_sheets.append(sheet_info)
                except Exception as e:
                    continue
            
            if not disease_sheets:
                continue  # Try next file
            
            print(f"      Found in {excel_file.name}: {len(disease_sheets)} disease sheets, {len(healthy_sheets)} healthy sheets")
            
            # Combine all disease samples
            disease_dfs = []
            for sheet_info in disease_sheets:
                df = sheet_info['data'].copy()
                # Align to training features
                available = [f for f in training_feature_names if f in df.columns]
                if len(available) < len(training_feature_names) * 0.5:  # Need at least 50% features
                    print(f"        ⚠️  Sheet {sheet_info['sheet_name']}: only {len(available)}/{len(training_feature_names)} features available")
                    continue
                
                # Create aligned dataframe with all training features (missing = 0)
                aligned_df = pd.DataFrame(0.0, index=df.index, columns=training_feature_names)
                for f in available:
                    aligned_df[f] = df[f].values
                aligned_df = aligned_df.fillna(0).astype(float)
                disease_dfs.append(aligned_df)
            
            if not disease_dfs:
                continue
            
            disease_df = pd.concat(disease_dfs, axis=0)
            X_disease = disease_df.values
            y_disease = np.ones(len(disease_df))
            
            # Try to find matching healthy samples
            X_healthy = None
            y_healthy = None
            has_healthy = False
            
            if healthy_sheets:
                healthy_dfs = []
                for sheet_info in healthy_sheets:
                    df = sheet_info['data'].copy()
                    available = [f for f in training_feature_names if f in df.columns]
                    if len(available) < len(training_feature_names) * 0.5:
                        continue
                    
                    aligned_df = pd.DataFrame(0.0, index=df.index, columns=training_feature_names)
                    for f in available:
                        aligned_df[f] = df[f].values
                    aligned_df = aligned_df.fillna(0).astype(float)
                    healthy_dfs.append(aligned_df)
                
                if healthy_dfs:
                    healthy_df = pd.concat(healthy_dfs, axis=0)
                    X_healthy = healthy_df.values
                    y_healthy = np.zeros(len(healthy_df))
                    has_healthy = True
            
            # Combine
            if has_healthy:
                X = np.vstack([X_disease, X_healthy])
                y = np.concatenate([y_disease, y_healthy])
            else:
                X = X_disease
                y = y_disease
            
            n_disease = int(np.sum(y == 1))
            n_healthy = int(np.sum(y == 0)) if has_healthy else 0
            
            if n_disease < min_samples:
                print(f"        ⚠️  Insufficient disease samples: {n_disease}")
                continue
            
            result = {
                'X': X,
                'y': y,
                'feature_names': training_feature_names,
                'n_disease': n_disease,
                'n_healthy': n_healthy,
                'has_healthy': has_healthy,
                'source_file': str(excel_file.name),
                'disease_abbr': disease_abbr,
                'can_compute_all_metrics': has_healthy and n_healthy >= min_samples
            }
            
            status = "disease+healthy" if has_healthy else "disease only"
            print(f"      ✅ Loaded external data: {n_disease}D + {n_healthy}H ({status})")
            return result
            
        except Exception as e:
            print(f"        Error processing {excel_file.name}: {e}")
            continue
    
    print(f"      ❌ No external validation data found for '{disease_abbr}'")
    return None


def load_all_external_validation(
    external_folder: str,
    mapper: AbbreviationMapper,
    tasks: List[Dict],
    min_samples: int = 5
) -> Dict[str, Dict]:
    """
    Load external validation data for all diseases found in tasks.
    
    Args:
        external_folder: Folder with external validation Excel files
        mapper: Abbreviation mapper
        tasks: List of training tasks
        min_samples: Minimum samples required
        
    Returns:
        Dict mapping disease_abbr to external validation data
    """
    if not external_folder:
        return {}
    
    external_path = Path(external_folder)
    if not external_path.exists():
        print(f"\n⚠️  External validation folder not found: {external_folder}")
        return {}
    
    print(f"\n{'='*80}")
    print("LOADING EXTERNAL VALIDATION DATA")
    print(f"{'='*80}")
    print(f"External folder: {external_folder}")
    
    # Get unique diseases and their feature names
    disease_info = {}
    for task in tasks:
        disease = task['disease_abbr']
        if disease not in disease_info:
            disease_info[disease] = {
                'feature_names': task['feature_names'],
                'disease_full': task['disease_full']
            }
    
    print(f"Diseases to find: {set(disease_info.keys())}")
    
    external_data = {}
    for disease_abbr, info in disease_info.items():
        print(f"\n  Looking for {disease_abbr} ({info['disease_full']})...")
        
        ext_data = load_external_validation_data(
            external_folder=external_folder,
            mapper=mapper,
            training_feature_names=info['feature_names'],
            disease_abbr=disease_abbr,
            min_samples=min_samples
        )
        
        if ext_data is not None:
            external_data[disease_abbr] = ext_data
    
    print(f"\n{'='*80}")
    print(f"EXTERNAL VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Found external data for {len(external_data)}/{len(disease_info)} diseases:")
    for disease, data in external_data.items():
        status = "✓ full metrics" if data['can_compute_all_metrics'] else "⚠️ predictions only"
        print(f"  {disease}: {data['n_disease']}D + {data['n_healthy']}H from {data['source_file']} ({status})")
    
    return external_data


# ============================================================================
# EVALUATOR WITH EXTERNAL VALIDATION
# ============================================================================

def evaluate_single_combination_with_external(
    X: np.ndarray,
    y: np.ndarray,
    config_dict: Dict,
    model_name: str,
    model_params: Dict,
    search_space: Dict,
    external_X: Optional[np.ndarray] = None,
    external_y: Optional[np.ndarray] = None,
    external_has_healthy: bool = False,
    outer_cv: int = 5,
    inner_cv: int = 3,
    random_state: int = 42,
    combination_id: str = ""
) -> Dict:
    """
    Evaluate a single configuration with internal CV and optional external validation.
    """
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
        
        # ===== INTERNAL CROSS-VALIDATION =====
        outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        best_params_list = []
        
        for fold_idx, (tr, te) in enumerate(outer.split(X, y)):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            
            pipe = build_pipeline(config, clone(base_model))
            space = search_space.get(model_name, {})
            inner = StratifiedKFold(
                n_splits=max(2, min(inner_cv, np.bincount(ytr).min() if np.bincount(ytr).size > 1 else 2)),
                shuffle=True, random_state=random_state
            )
            
            try:
                search = GridSearchCV(
                    pipe, param_grid=space, cv=inner,
                    scoring='roc_auc', n_jobs=1, error_score='raise'
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
            except:
                try:
                    yproba = best_pipe.decision_function(Xte)
                except:
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
            except:
                scores['auc'].append(np.nan)
        
        # Aggregate best params
        best_params_aggregated = {}
        if best_params_list and any(best_params_list):
            all_params = {}
            for params in best_params_list:
                for key, val in params.items():
                    if key not in all_params:
                        all_params[key] = []
                    all_params[key].append(val)
            for key, values in all_params.items():
                try:
                    best_params_aggregated[key] = max(set(values), key=values.count)
                except:
                    best_params_aggregated[key] = values[0]
        
        # ===== EXTERNAL VALIDATION =====
        external_results = {
            'external_validated': False,
            'external_accuracy': np.nan,
            'external_precision': np.nan,
            'external_recall': np.nan,
            'external_f1': np.nan,
            'external_auc': np.nan,
            'external_n_disease': 0,
            'external_n_healthy': 0,
            'external_disease_pred_positive_rate': np.nan,
            'external_healthy_pred_positive_rate': np.nan
        }
        
        if external_X is not None and len(external_X) > 0:
            try:
                # Train on ALL training data with best params
                final_model = clone(base_model)
                if best_params_aggregated:
                    clf_params = {k.replace('clf__', ''): v 
                                  for k, v in best_params_aggregated.items() 
                                  if k.startswith('clf__')}
                    try:
                        final_model.set_params(**clf_params)
                    except:
                        pass
                
                final_pipe = build_pipeline(config, final_model)
                final_pipe.fit(X, y)
                
                # Predict on external data
                ext_pred = final_pipe.predict(external_X)
                try:
                    ext_proba = final_pipe.predict_proba(external_X)[:, 1]
                except:
                    try:
                        ext_proba = final_pipe.decision_function(external_X)
                    except:
                        ext_proba = ext_pred.astype(float)
                
                external_results['external_validated'] = True
                external_results['external_n_disease'] = int(np.sum(external_y == 1))
                external_results['external_n_healthy'] = int(np.sum(external_y == 0))
                
                # Prediction rates
                if np.sum(external_y == 1) > 0:
                    disease_mask = external_y == 1
                    external_results['external_disease_pred_positive_rate'] = float(np.mean(ext_pred[disease_mask]))
                
                if external_has_healthy and np.sum(external_y == 0) > 0:
                    healthy_mask = external_y == 0
                    external_results['external_healthy_pred_positive_rate'] = float(np.mean(ext_pred[healthy_mask]))
                    
                    # Full metrics if we have both classes
                    if len(np.unique(external_y)) > 1:
                        external_results['external_accuracy'] = float(accuracy_score(external_y, ext_pred))
                        external_results['external_precision'] = float(precision_score(external_y, ext_pred, zero_division=0))
                        external_results['external_recall'] = float(recall_score(external_y, ext_pred, zero_division=0))
                        external_results['external_f1'] = float(f1_score(external_y, ext_pred, zero_division=0))
                        try:
                            external_results['external_auc'] = float(roc_auc_score(external_y, ext_proba))
                        except:
                            pass
                            
            except Exception as e:
                external_results['external_error'] = str(e)
        
        result = {
            'config_name': config.name,
            'model_name': model_name,
            'scaling': config.scaling,
            'feature_selection': config.feature_selection,
            'n_features': int(config.n_features),
            'balancing': config.balancing,
            # Internal CV metrics
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
        
        # Add external results
        result.update(external_results)
        
        # Calculate generalization gap
        if external_results['external_validated'] and pd.notna(external_results['external_auc']):
            result['generalization_gap'] = result['mean_auc'] - external_results['external_auc']
        else:
            result['generalization_gap'] = np.nan
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'config_name': config_dict.get('name', 'Unknown'),
            'model_name': model_name,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


class JoblibParallelEvaluatorWithExternal:
    """Parallel evaluator with external validation support."""
    
    def __init__(self, outer_cv=5, inner_cv=3, random_state=42, n_jobs=-1, verbose=10):
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.random_state = random_state
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.verbose = verbose
        self.results_ = []
        self.skipped_ = []
    
    def evaluate(self, X, y, configs, models, search_spaces,
                 external_X=None, external_y=None, external_has_healthy=False):
        """Run evaluation with optional external validation."""
        print(f"\nParallel eval: {len(configs)} configs × {len(models)} models = {len(configs)*len(models):,}")
        print(f"Workers: {self.n_jobs}")
        print("Using GridSearchCV for reproducibility")
        
        if external_X is not None:
            ext_status = "disease+healthy" if external_has_healthy else "disease only"
            print(f"External validation: {len(external_X)} samples ({ext_status})")
        else:
            print("External validation: None")
        
        combos = []
        for cfg in configs:
            for mname, minst in models.items():
                combos.append({
                    'X': X, 'y': y,
                    'config_dict': cfg.to_dict(),
                    'model_name': mname,
                    'model_params': minst.get_params(),
                    'search_space': search_spaces,
                    'external_X': external_X,
                    'external_y': external_y,
                    'external_has_healthy': external_has_healthy,
                    'outer_cv': self.outer_cv,
                    'inner_cv': self.inner_cv,
                    'random_state': self.random_state,
                    'combination_id': f"{cfg.name}_{mname}"
                })
        
        try:
            with parallel_backend('loky', n_jobs=self.n_jobs):
                results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(evaluate_single_combination_with_external)(**c) for c in combos
                )
        except Exception as e:
            print(f"Parallel failed: {e}\nFalling back to sequential...")
            results = [evaluate_single_combination_with_external(**c) for c in combos]
        
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
    
    def get_best_by_external(self, metric='external_auc'):
        """Get best model by external validation metric."""
        if not self.results_:
            return None, None
        valid = [r for r in self.results_ 
                 if r.get('external_validated', False) and metric in r and pd.notna(r[metric])]
        if not valid:
            return None, None
        best = max(valid, key=lambda r: r[metric])
        return best['config_name'], best['model_name']
    
    def get_best_by_generalization(self):
        """Get model with smallest generalization gap (and reasonable internal AUC)."""
        if not self.results_:
            return None, None
        valid = [r for r in self.results_ 
                 if r.get('external_validated', False) 
                 and 'generalization_gap' in r 
                 and pd.notna(r['generalization_gap'])
                 and r['mean_auc'] >= 0.6]
        if not valid:
            return None, None
        best = min(valid, key=lambda r: abs(r['generalization_gap']))
        return best['config_name'], best['model_name']


# ============================================================================
# VISUALIZATION WITH EXTERNAL VALIDATION
# ============================================================================

def create_visualizations_with_external(results_df: pd.DataFrame, output_dir: str, has_external: bool = False):
    """Create visualizations including external validation results."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Top 20 by internal AUC
    top20 = results_df.nlargest(20, 'mean_auc')
    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(top20))
    plt.barh(y_pos, top20['mean_auc'], xerr=top20['std_auc'], alpha=0.85)
    plt.yticks(y_pos, [f"{r['config_name']}\n{r['model_name']}" for _, r in top20.iterrows()], fontsize=8)
    plt.xlabel('Mean AUC (Internal CV)')
    plt.title('Top 20 Configurations by Internal AUC', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'top20_internal_auc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # External validation comparison (if available)
    if has_external and 'external_auc' in results_df.columns:
        external_valid = results_df[results_df['external_validated'] == True].copy()
        if len(external_valid) > 0 and external_valid['external_auc'].notna().any():
            # Top 20 by external AUC
            top20_ext = external_valid.nlargest(20, 'external_auc')
            
            plt.figure(figsize=(16, 10))
            y_pos = np.arange(len(top20_ext))
            width = 0.35
            
            plt.barh(y_pos - width/2, top20_ext['mean_auc'], width, label='Internal CV', alpha=0.8)
            plt.barh(y_pos + width/2, top20_ext['external_auc'], width, label='External Validation', alpha=0.8)
            
            plt.yticks(y_pos, [f"{r['config_name']}\n{r['model_name']}" for _, r in top20_ext.iterrows()], fontsize=8)
            plt.xlabel('AUC')
            plt.title('Internal vs External AUC (Top 20 by External)', fontweight='bold')
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / 'internal_vs_external_auc.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Scatter plot: Internal vs External
            plt.figure(figsize=(10, 10))
            valid_for_scatter = external_valid[external_valid['external_auc'].notna()]
            plt.scatter(valid_for_scatter['mean_auc'], valid_for_scatter['external_auc'], 
                       alpha=0.5, c='steelblue')
            plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5, label='Perfect generalization')
            plt.xlabel('Internal CV AUC')
            plt.ylabel('External Validation AUC')
            plt.title('Internal vs External AUC Scatter', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / 'generalization_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Model comparison
    model_stats = results_df.groupby('model_name').agg({
        'mean_auc': ['mean', 'std'], 
        'mean_f1': ['mean', 'std']
    }).round(3)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (metric, title) in zip(axes, [('mean_auc', 'AUC'), ('mean_f1', 'F1-Score')]):
        idx = model_stats.index
        means = model_stats[(metric, 'mean')]
        stds = model_stats[(metric, 'std')]
        ax.bar(range(len(idx)), means, yerr=stds, capsize=5, alpha=0.85)
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(idx, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(f'Model Comparison - {title}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m, f'{m:.3f}\n±{s:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Preprocessing comparison
    scaling_stats = results_df.groupby('scaling')['mean_auc'].agg(['mean', 'std']).round(3)
    fs_stats = results_df.groupby('feature_selection')['mean_auc'].agg(['mean', 'std']).round(3)
    bal_stats = results_df.groupby('balancing')['mean_auc'].agg(['mean', 'std']).round(3)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, stats_df, title in zip(axes, [scaling_stats, fs_stats, bal_stats],
                                   ['Scaling', 'Feature Selection', 'Balancing']):
        idx = stats_df.index
        means = stats_df['mean']
        stds = stats_df['std']
        ax.bar(range(len(idx)), means, yerr=stds, capsize=5, alpha=0.85)
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(idx, rotation=45, ha='right')
        ax.set_ylabel('Mean AUC')
        ax.set_title(f'{title} Performance', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m, f'{m:.3f}\n±{s:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / 'preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Plots saved")


# ============================================================================
# MAIN RUNNER WITH EXTERNAL VALIDATION
# ============================================================================

def run_comprehensive_microbiome_analysis(
    input_folder: str = "perprojectmsp",
    external_folder: str = None,  
    mapping_file: str = "abbreviation_mapping.xlsx",
    output_folder: str = "microbiome_results",
    min_samples: int = 25,
    external_min_samples: int = 5,  
    n_jobs: int = 10
):
    """
    Run comprehensive microbiome analysis with optional external validation.
    
    Args:
        input_folder: Folder containing training data (per-project MSP Excel files)
        external_folder: Folder containing external validation data (one project per disease)
        mapping_file: Abbreviation mapping Excel
        output_folder: Base output directory
        min_samples: Minimum samples per class for training tasks
        external_min_samples: Minimum samples for external validation
        n_jobs: Parallel workers
    """
    start_time = time.time()
    print(f"\n{'='*80}")
    print("MICROBIOME ANALYSIS WITH EXTERNAL VALIDATION")
    print(f"{'='*80}")
    print("  ✓ 6 Models (RF, XGBoost, LightGBM, LogReg, SVM, MLP)")
    print("  ✓ 6 FS methods (incl. differential Mann-Whitney + FDR)")
    print("  ✓ 3 Balancing (SMOTE, undersample, none)")
    print("  ✓ Zero imputation, 5-fold CV, inner 3-fold GridSearchCV")
    print("  ✓ SMART feature selection:")
    print("      - 'none' → 100% of features (no selection)")
    print("      - Other methods → 10%, 30%, 50% (actual selection)")
    print("  ✓ Pipeline order: Scaling → FeatureSelection → Balancing → Classifier")
    if external_folder:
        print(f"  ✓ EXTERNAL VALIDATION from: {external_folder}")
    else:
        print("  ⚠️ No external validation (external_folder not specified)")
    print()
    
    mapper = AbbreviationMapper(mapping_file)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Load training tasks
    project_files = list(Path(input_folder).glob("*.xlsx")) + list(Path(input_folder).glob("*.xls"))
    if not project_files:
        print(f"❌ No Excel files found in {input_folder}")
        return
    
    all_tasks = []
    for project_file in project_files:
        project_name = project_file.stem
        tasks = load_classification_tasks_from_excel(
            str(project_file), mapper, min_samples=min_samples
        )
        for t in tasks:
            t['project'] = project_name
        all_tasks.extend(tasks)
    
    if not all_tasks:
        print("\n❌ No valid tasks found!")
        return
    
    print(f"\n{'='*80}")
    print(f"TRAINING TASKS SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(all_tasks)}")
    
    # Load external validation data
    external_data = {}
    if external_folder:
        external_data = load_all_external_validation(
            external_folder=external_folder,
            mapper=mapper,
            tasks=all_tasks,
            min_samples=external_min_samples
        )
    
    # Build configurations
    max_features = min([task['n_features'] for task in all_tasks])
    configs = build_microbiome_configs(max_features=max_features)
    models = get_models()
    search_spaces = get_search_spaces()
    
    print(f"\n{'='*80}")
    print("PIPELINE STATISTICS")
    print(f"{'='*80}")
    print(f"Configurations: {len(configs):,}")
    print(f"Models: {len(models)}")
    print(f"Combinations per task: {len(configs) * len(models):,}")
    print(f"TOTAL EVALUATIONS: {len(all_tasks) * len(configs) * len(models):,}")
    
    # Run evaluations
    all_results = []
    evaluator = JoblibParallelEvaluatorWithExternal(
        outer_cv=5, inner_cv=3, random_state=42,
        n_jobs=n_jobs, verbose=5
    )
    
    for idx, task in enumerate(all_tasks, 1):
        print(f"\n{'='*80}")
        print(f"TASK {idx}/{len(all_tasks)}: {task['task_name']}")
        print(f"Training: {task['n_disease']}D + {task['n_healthy']}H, {task['n_features']} features")
        print(f"{'='*80}")
        
        # Get external validation data for this disease
        disease_abbr = task['disease_abbr']
        ext_data = external_data.get(disease_abbr, None)
        
        external_X = None
        external_y = None
        external_has_healthy = False
        
        if ext_data is not None:
            # Align external features to training features
            training_features = task['feature_names']
            ext_features = ext_data['feature_names']
            
            if training_features == ext_features:
                external_X = ext_data['X']
                external_y = ext_data['y']
                external_has_healthy = ext_data['has_healthy']
                print(f"External validation: {ext_data['n_disease']}D + {ext_data['n_healthy']}H from {ext_data['source_file']}")
            else:
                print(f"⚠️  Feature mismatch with external data - aligning...")
                # Create aligned external data
                ext_df = pd.DataFrame(ext_data['X'], columns=ext_features)
                aligned_df = pd.DataFrame(0.0, index=range(len(ext_df)), columns=training_features)
                for f in training_features:
                    if f in ext_features:
                        aligned_df[f] = ext_df[f].values
                external_X = aligned_df.values
                external_y = ext_data['y']
                external_has_healthy = ext_data['has_healthy']
                print(f"External validation (aligned): {ext_data['n_disease']}D + {ext_data['n_healthy']}H")
        else:
            print(f"⚠️  No external validation data for {disease_abbr}")
        
        task_start = time.time()
        results_df = evaluator.evaluate(
            task['X'], task['y'], 
            configs, models, search_spaces,
            external_X=external_X,
            external_y=external_y,
            external_has_healthy=external_has_healthy
        )
        task_elapsed = time.time() - task_start
        
        if len(results_df) > 0:
            # Add task metadata
            for key in ['project', 'task_name', 'disease_abbr', 'disease_full',
                        'geography_abbr', 'geography_full', 'sequencer_abbr',
                        'sequencer_full', 'n_disease', 'n_healthy', 'n_features']:
                results_df[key] = task[key]
            
            # Add external data info
            if ext_data is not None:
                results_df['external_source'] = ext_data['source_file']
            else:
                results_df['external_source'] = 'None'
            
            # Save task results
            task_dir = output_path / task['project'] / task['task_name']
            task_dir.mkdir(exist_ok=True, parents=True)
            results_df.to_csv(task_dir / 'results.csv', index=False)
            all_results.append(results_df)
            
            # Report best results
            print(f"\n📊 BEST MODELS:")
            
            # By internal AUC
            best_config, best_model = evaluator.get_best_model('mean_auc')
            if best_config:
                best = results_df[(results_df['config_name'] == best_config) &
                                  (results_df['model_name'] == best_model)].iloc[0]
                
                print(f"  Best by Internal AUC: {best_config} + {best_model}")
                print(f"    Internal AUC: {best['mean_auc']:.3f} ± {best['std_auc']:.3f}")
                print(f"    Internal F1: {best['mean_f1']:.3f} ± {best['std_f1']:.3f}")
                
                if ext_data is not None and best.get('external_validated', False):
                    if pd.notna(best.get('external_auc', np.nan)):
                        print(f"    External AUC: {best['external_auc']:.3f}")
                        print(f"    External F1: {best['external_f1']:.3f}")
                    else:
                        print(f"    External Disease Pred Rate: {best.get('external_disease_pred_positive_rate', np.nan):.3f}")
                
                # Save best hyperparameters
                best_params_file = task_dir / 'best_hyperparameters.txt'
                with open(best_params_file, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write("BEST MODEL RESULTS\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Configuration: {best_config}\n")
                    f.write(f"Model: {best_model}\n\n")
                    f.write("INTERNAL CV METRICS:\n")
                    f.write(f"  AUC: {best['mean_auc']:.4f} ± {best['std_auc']:.4f}\n")
                    f.write(f"  F1: {best['mean_f1']:.4f} ± {best['std_f1']:.4f}\n")
                    f.write(f"  Accuracy: {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}\n")
                    f.write(f"  Precision: {best['mean_precision']:.4f} ± {best['std_precision']:.4f}\n")
                    f.write(f"  Recall: {best['mean_recall']:.4f} ± {best['std_recall']:.4f}\n\n")
                    
                    if ext_data is not None and best.get('external_validated', False):
                        f.write("EXTERNAL VALIDATION METRICS:\n")
                        f.write(f"  Source: {ext_data['source_file']}\n")
                        f.write(f"  Samples: {ext_data['n_disease']}D + {ext_data['n_healthy']}H\n")
                        if pd.notna(best.get('external_auc', np.nan)):
                            f.write(f"  AUC: {best['external_auc']:.4f}\n")
                            f.write(f"  F1: {best['external_f1']:.4f}\n")
                            f.write(f"  Accuracy: {best['external_accuracy']:.4f}\n")
                            f.write(f"  Precision: {best['external_precision']:.4f}\n")
                            f.write(f"  Recall: {best['external_recall']:.4f}\n")
                        f.write(f"  Disease Pred Rate: {best.get('external_disease_pred_positive_rate', np.nan):.4f}\n")
                        if external_has_healthy:
                            f.write(f"  Healthy Pred Rate: {best.get('external_healthy_pred_positive_rate', np.nan):.4f}\n")
                        f.write("\n")
                    
                    f.write("HYPERPARAMETERS:\n")
                    f.write("-"*80 + "\n")
                    f.write(f"{best['best_params']}\n")
                    f.write("="*80 + "\n")
            
            # By external AUC (if available)
            if ext_data is not None and external_has_healthy:
                best_ext_config, best_ext_model = evaluator.get_best_by_external('external_auc')
                if best_ext_config:
                    best_ext = results_df[(results_df['config_name'] == best_ext_config) &
                                         (results_df['model_name'] == best_ext_model)].iloc[0]
                    print(f"\n  Best by External AUC: {best_ext_config} + {best_ext_model}")
                    print(f"    Internal AUC: {best_ext['mean_auc']:.3f}")
                    print(f"    External AUC: {best_ext['external_auc']:.3f}")
                
                # By generalization gap
                best_gen_config, best_gen_model = evaluator.get_best_by_generalization()
                if best_gen_config:
                    best_gen = results_df[(results_df['config_name'] == best_gen_config) &
                                         (results_df['model_name'] == best_gen_model)].iloc[0]
                    print(f"\n  Best Generalization: {best_gen_config} + {best_gen_model}")
                    print(f"    Internal AUC: {best_gen['mean_auc']:.3f}")
                    print(f"    External AUC: {best_gen['external_auc']:.3f}")
                    print(f"    Gap: {best_gen['generalization_gap']:.3f}")
            
            # Create visualizations
            has_external = ext_data is not None and results_df['external_validated'].any()
            create_visualizations_with_external(results_df, task_dir, has_external=has_external)
        
        print(f"\n⏱️  Task completed in {task_elapsed/60:.1f} minutes")
    
    # Combine and save all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_path / 'ALL_RESULTS.csv', index=False)
        
        runtime_hours = (time.time() - start_time) / 3600
        
        print(f"\n{'='*80}")
        print("🎉 ALL TASKS COMPLETED")
        print(f"{'='*80}")
        print(f"Total results: {len(combined):,}")
        print(f"Total runtime: {runtime_hours:.2f} hours ({runtime_hours*60:.1f} minutes)")
        
        # Print top results by internal AUC
        print(f"\n📊 TOP 10 BY INTERNAL AUC:")
        top_internal = combined.nlargest(10, 'mean_auc')[
            ['task_name', 'disease_full', 'model_name', 'config_name', 
             'mean_auc', 'std_auc', 'external_auc', 'external_validated']
        ]
        print(top_internal.to_string(index=False))
        
        # Print top results by external AUC (if available)
        if 'external_auc' in combined.columns:
            external_valid = combined[
                (combined['external_validated'] == True) & 
                (combined['external_auc'].notna())
            ]
            if len(external_valid) > 0:
                print(f"\n📊 TOP 10 BY EXTERNAL AUC:")
                top_external = external_valid.nlargest(10, 'external_auc')[
                    ['task_name', 'disease_full', 'model_name', 'config_name',
                     'mean_auc', 'external_auc', 'external_f1', 'generalization_gap']
                ]
                print(top_external.to_string(index=False))
        
        # Save summary statistics
        summary = combined.groupby(['config_name', 'model_name']).agg({
            'mean_auc': ['mean', 'std', 'min', 'max'],
            'mean_f1': ['mean', 'std', 'min', 'max']
        }).round(3)
        summary.to_csv(output_path / 'SUMMARY_STATISTICS.csv')
        
        # Save summary JSON
        summary_dict = {
            'settings': {
                'cv_strategy': 'StratifiedKFold with GridSearchCV',
                'outer_folds': 5,
                'inner_folds': 3,
                'random_state': 42,
                'models': list(models.keys()),
                'feature_selection_methods': ['none', 'variance', 'f_classif', 'mutual_info', 'random_forest', 'differential'],
                'feature_selection_strategy': {
                    'none': '100% (all features)',
                    'others': '10%, 30%, 50% (subsets)'
                },
                'balancing_methods': ['none', 'smote', 'undersample'],
                'pipeline_order': 'Scaling → FeatureSelection → Balancing → Classifier',
                'external_validation': external_folder is not None,
                'external_folder': external_folder
            },
            'data': {
                'n_training_tasks': len(all_tasks),
                'diseases_with_external': list(external_data.keys()) if external_data else [],
                'total_combinations': len(combined)
            },
            'runtime_hours': float(runtime_hours),
            'runtime_minutes': float(runtime_hours * 60)
        }
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
    else:
        print("\n❌ No results generated")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    input_folder = "msp5"  # Training data folder containing project excel file
    external_folder = "External Validation"  # External validation folder containing project excel file(set to None to skip)
    mapping_file = "abbreviation_mapping.xlsx"
    output_folder = "microbiome_results_with_external"
    
    available_cpus = mp.cpu_count()
    n_jobs = -1
    
    print(f"Available CPUs: {available_cpus}")
    print(f"Using {n_jobs} workers")
    
    run_comprehensive_microbiome_analysis(
        input_folder=input_folder,
        external_folder=external_folder,  # Pass None to skip external validation
        mapping_file=mapping_file,
        output_folder=output_folder,
        min_samples=25,
        external_min_samples=2,
        n_jobs=n_jobs
    )
    
