"""
single_feature_auc_analysis_with_distribution.py

Single-Feature Classification Analysis for ALL Features:
- For each feature in each task, compute its AUC using cross-validated logistic regression
- Save all results to a single CSV
- Generate a single log-scale distribution plot with skewness per disease
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)


# ============================================================================
# SINGLE FEATURE AUC METHODS
# ============================================================================

def compute_auc_direct(feature_values: np.ndarray, y: np.ndarray) -> float:
    try:
        if np.std(feature_values) < 1e-30:
            return np.nan
        mask = np.isfinite(feature_values)
        if mask.sum() < 10:
            return np.nan
        auc = roc_auc_score(y[mask], feature_values[mask])
        return max(auc, 1 - auc)
    except:
        return np.nan


def compute_auc_cv_logistic(
    feature_values: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Tuple[float, float]:
    try:
        X = feature_values.reshape(-1, 1)
        if np.std(feature_values) < 1e-30:
            return np.nan, np.nan
        mask = np.isfinite(feature_values)
        if mask.sum() < 10:
            return np.nan, np.nan
        X_clean = X[mask]
        y_clean = y[mask]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        clf = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
        min_class = min(np.bincount(y_clean))
        actual_cv = min(cv, min_class)
        if actual_cv < 2:
            return np.nan, np.nan
        cv_splitter = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, y_clean, cv=cv_splitter, scoring='roc_auc')
        return float(np.mean(scores)), float(np.std(scores))
    except:
        return np.nan, np.nan


# ============================================================================
# ABBREVIATION MAPPER
# ============================================================================

class AbbreviationMapper:
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

        possible_full = ['Original', 'Full Name', 'Name', 'Full', 'Fullname']
        possible_abbr = ['Abbreviation', 'Abbr', 'Code', 'Short', 'Abbreviations']

        full_col = abbr_col = None
        for col in mapping_df.columns:
            col_str = str(col).strip()
            if col_str in possible_full:
                full_col = col
            if col_str in possible_abbr:
                abbr_col = col

        if full_col is None or abbr_col is None:
            cols = mapping_df.columns.tolist()
            if len(cols) >= 3:
                full_col, abbr_col = cols[1], cols[2]
            elif len(cols) >= 2:
                full_col, abbr_col = cols[0], cols[1]

        if full_col is None or abbr_col is None:
            print("  ERROR: Could not identify mapping columns!")
            return

        for _, row in mapping_df.iterrows():
            try:
                full_name = str(row[full_col]).strip()
                abbr = str(row[abbr_col]).strip()
                if full_name and abbr and full_name != 'nan' and abbr != 'nan':
                    self.abbr_to_full[abbr] = full_name
                    self.full_to_abbr[full_name] = abbr
                    if any(k in full_name.lower() for k in self.healthy_keywords):
                        self.healthy_abbrs.add(abbr)
            except:
                continue
        print(f"  Loaded {len(self.abbr_to_full)} mappings")

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
        disease_abbr, geo_abbr, seq_abbr = self.parse_sheet_name(sheet_name)
        if disease_abbr is None:
            return None
        return {
            'disease_abbr': disease_abbr,
            'disease_full': self.get_full_name(disease_abbr),
            'geography_abbr': geo_abbr,
            'geography_full': self.get_full_name(geo_abbr),
            'sequencer_abbr': seq_abbr,
            'sequencer_full': self.get_full_name(seq_abbr),
            'is_healthy': self.is_healthy(disease_abbr)
        }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_classification_tasks(
    excel_file: str,
    mapper: AbbreviationMapper,
    min_samples: int = 25
) -> List[Dict]:

    print(f"\nLoading from: {excel_file}")

    try:
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names
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
        except:
            continue

    disease_groups = {}
    for sheet_name, sheet_info in parsed_sheets.items():
        if not sheet_info['is_healthy']:
            key = (sheet_info['disease_abbr'],
                   sheet_info['geography_abbr'],
                   sheet_info['sequencer_abbr'])
            disease_groups[key] = sheet_info

    tasks = []
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

        n_disease, n_healthy = np.sum(y == 1), np.sum(y == 0)
        if n_disease < min_samples or n_healthy < min_samples:
            continue

        task = {
            'X': X,
            'y': y,
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
            'n_features': len(common_features)
        }
        tasks.append(task)
        print(f"  ✅ {task['task_name']}: {n_disease}D + {n_healthy}H, {len(common_features)} features")

    return tasks


# ============================================================================
# SINGLE FEATURE ANALYSIS
# ============================================================================

def analyze_single_feature(
    feature_idx: int,
    feature_name: str,
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'direct'
) -> Dict:

    feature_values = X[:, feature_idx]

    if method == 'direct':
        auc = compute_auc_direct(feature_values, y)
        return {
            'feature_name': feature_name,
            'feature_idx': feature_idx,
            'auc': auc,
            'auc_std': np.nan
        }
    else:
        auc_mean, auc_std = compute_auc_cv_logistic(feature_values, y, cv=5)
        return {
            'feature_name': feature_name,
            'feature_idx': feature_idx,
            'auc': auc_mean,
            'auc_std': auc_std
        }


def analyze_task_features_parallel(
    task: Dict,
    method: str = 'direct',
    n_jobs: int = -1
) -> pd.DataFrame:

    X = task['X']
    y = task['y']
    feature_names = task['feature_names']
    n_features = len(feature_names)

    print(f"  Analyzing {n_features} features using '{method}' method...")

    if n_jobs == 1:
        results = []
        for i, fname in enumerate(feature_names):
            res = analyze_single_feature(i, fname, X, y, method)
            results.append(res)
    else:
        n_jobs_actual = mp.cpu_count() if n_jobs == -1 else n_jobs
        results = Parallel(n_jobs=n_jobs_actual, verbose=0)(
            delayed(analyze_single_feature)(i, fname, X, y, method)
            for i, fname in enumerate(feature_names)
        )

    df = pd.DataFrame(results)
    df['task_name'] = task['task_name']
    df['disease'] = task['disease_abbr']
    df['disease_full'] = task['disease_full']
    df['geography'] = task['geography_abbr']
    df['sequencer'] = task['sequencer_abbr']
    df['n_disease'] = task['n_disease']
    df['n_healthy'] = task['n_healthy']
    df['n_total_samples'] = task['n_disease'] + task['n_healthy']

    return df


# ============================================================================
# MAIN ANALYSIS — CSV + DISTRIBUTION PLOT ONLY
# ============================================================================

def run_analysis(
    input_folder: str,
    mapping_file: str,
    output_folder: str,
    method: str = 'cv_logistic',
    min_samples: int = 25,
    n_jobs: int = -1,
    col_wrap: int = 4,
    bins: int = 30
):
    start_time = time.time()

    print("=" * 80)
    print("SINGLE-FEATURE AUC ANALYSIS — CSV + DISTRIBUTION PLOT")
    print("=" * 80)
    print(f"Method: {method}")

    # Setup
    mapper = AbbreviationMapper(mapping_file)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # Find project files
    project_files = list(Path(input_folder).glob("*.xlsx")) + \
                    list(Path(input_folder).glob("*.xls"))

    if not project_files:
        print(f"❌ No Excel files found in {input_folder}")
        return

    print(f"\nFound {len(project_files)} project files")

    # Load all tasks
    all_tasks = []
    for pf in project_files:
        tasks = load_all_classification_tasks(str(pf), mapper, min_samples)
        for t in tasks:
            t['project'] = pf.stem
        all_tasks.extend(tasks)

    if not all_tasks:
        print("❌ No valid tasks found!")
        return

    n_tasks = len(all_tasks)
    total_features = sum(t['n_features'] for t in all_tasks)

    print(f"\nTotal tasks: {n_tasks}")
    print(f"Total feature-task pairs to analyze: {total_features:,}")

    # =========================================================================
    # ANALYZE EACH TASK
    # =========================================================================

    all_results = []

    for idx, task in enumerate(all_tasks, 1):
        print(f"\n[{idx}/{n_tasks}] Task: {task['task_name']} ({task['n_features']} features)")

        task_start = time.time()
        task_df = analyze_task_features_parallel(task, method=method, n_jobs=n_jobs)
        task_df['project'] = task.get('project', 'unknown')
        all_results.append(task_df)

        task_time = time.time() - task_start
        valid_count = task_df['auc'].notna().sum()
        print(f"  Done in {task_time:.1f}s — {valid_count} valid AUCs")

    # Combine and save CSV
    combined = pd.concat(all_results, ignore_index=True)

    csv_path = output_path / 'all_feature_auc_results.csv'
    combined.to_csv(csv_path, index=False)
    print(f"\n✅ CSV saved: {csv_path}")
    print(f"   Rows: {len(combined):,}")

    # =========================================================================
    # DISTRIBUTION PLOT WITH LOG Y-AXIS AND SKEWNESS
    # =========================================================================

    print("\nCreating distribution plot...")

    # Prepare data
    plot_df = combined[['feature_name', 'auc', 'disease']].copy()
    plot_df['auc'] = pd.to_numeric(plot_df['auc'], errors='coerce')
    plot_df = plot_df.dropna(subset=['auc', 'disease'])
    plot_df = plot_df[(plot_df['auc'] >= 0) & (plot_df['auc'] <= 1)]

    # Skewness per disease
    skewness = plot_df.groupby('disease')['auc'].skew()

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    g = sns.displot(
        data=plot_df, x="auc", col="disease", col_wrap=col_wrap,
        bins=bins, kde=False,
        common_bins=True, common_norm=False,
        facet_kws=dict(sharey=False),
        height=3.0, aspect=1.2
    )

    for ax, disease in zip(g.axes.flatten(), g.col_names):
        skew_val = skewness.get(disease, np.nan)
        ax.set_title(f"{disease}\n(skew={skew_val:.2f})")
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        ax.set_ylim(0.8, None)

    g.set_axis_labels("Univariate AUC", "Taxa count (log scale)")
    plt.tight_layout()

    plot_path = output_path / 'AUC_by_disease_hist_logY.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plot saved: {plot_path}")
    print(f"\nSkewness values by disease:\n{skewness}")

    # =========================================================================
    # DONE
    # =========================================================================

    runtime_min = (time.time() - start_time) / 60

    print(f"\n{'=' * 80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Runtime: {runtime_min:.1f} minutes")
    print(f"\nOutput files ({output_path}):")
    print(f"  1. all_feature_auc_results.csv  — all feature AUCs across all tasks")
    print(f"  2. AUC_by_disease_hist_logY.png — log-scale AUC distribution per disease with skewness")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # =========================================================================
    # CONFIGURATION — EDIT THESE VALUES
    # =========================================================================

    INPUT_FOLDER = "input_folder" #folder containing all training project files
    MAPPING_FILE = "abbreviation_mapping.xlsx"
    OUTPUT_FOLDER = "all_features_auc_resultscv_logistics"

    METHOD = 'cv_logistic'   # 'direct' is faster; 'cv_logistic' more robust
    MIN_SAMPLES = 25
    COL_WRAP = 4             # columns in faceted plot
    BINS = 30                # histogram bins

    available_cpus = mp.cpu_count()
    N_JOBS = min(20, available_cpus)

    print(f"Available CPUs: {available_cpus}")
    print(f"Using {N_JOBS} workers")

    # =========================================================================
    # RUN
    # =========================================================================

    run_analysis(
        input_folder=INPUT_FOLDER,
        mapping_file=MAPPING_FILE,
        output_folder=OUTPUT_FOLDER,
        method=METHOD,
        min_samples=MIN_SAMPLES,
        n_jobs=N_JOBS,
        col_wrap=COL_WRAP,
        bins=BINS
    )

    print("\n🎉 Done!")