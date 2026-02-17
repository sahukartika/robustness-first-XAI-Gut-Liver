"""
best_robust_config_simple.py

Find all (model, scaling, feature_selection, balancing) configurations
ranked by BEST worst-case (highest min) performance across all tasks.

Outputs only: all_configs_ranked_by_robustness.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def load_results(results_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Load all result files."""
    data = {}
    print("\nLoading data files...")
    for task, path in results_dict.items():
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            if 'success' in df.columns:
                df = df[df['success'] == True]
            data[task] = df
            print(f"  ✓ {task}: {len(df):,} rows")
        else:
            print(f"  ✗ Missing: {path}")
    return data


def find_best_robust_config(data: Dict[str, pd.DataFrame],
                            metric: str,
                            models: list,
                            require_all_tasks: bool = True) -> pd.DataFrame:
    """
    For each unique (model, scaling, feature_selection, balancing) configuration:
      - Find the BEST AUC for that config in each task
      - Compute the MIN (worst-case) across all tasks
      
    Returns:
        DataFrame sorted by 'min_across_tasks' descending (best first)
    """
    
    tasks = list(data.keys())
    n_tasks = len(tasks)
    
    config_task_best = {}
    
    print(f"\nAnalyzing configurations across {n_tasks} tasks...")
    
    for task_name, df in data.items():
        for model in models:
            mdf = df[df['model_name'] == model]
            if mdf.empty:
                continue
            
            grouped = mdf.groupby(['scaling', 'feature_selection', 'balancing'])
            
            for (scaling, fs, bal), group in grouped:
                config_key = (model, str(scaling), str(fs), str(bal))
                best_auc = group[metric].max()
                
                if config_key not in config_task_best:
                    config_task_best[config_key] = {}
                config_task_best[config_key][task_name] = float(best_auc)
    
    print(f"  Found {len(config_task_best)} unique configurations")
    
    records = []
    
    for config_key, task_values in config_task_best.items():
        model, scaling, fs, bal = config_key
        
        if require_all_tasks and len(task_values) < n_tasks:
            continue
        
        values = np.array(list(task_values.values()))
        
        records.append({
            'model': model,
            'scaling': scaling,
            'feature_selection': fs,
            'balancing': bal,
            'min_across_tasks': float(np.min(values)),
            'max_across_tasks': float(np.max(values)),
            'mean_across_tasks': float(np.mean(values)),
            'std_across_tasks': float(np.std(values)) if len(values) > 1 else 0.0,
            'range': float(np.max(values) - np.min(values)),
            'n_tasks': len(task_values),
        })
    
    if not records:
        print("  ⚠ No configurations found in all tasks!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values('min_across_tasks', ascending=False).reset_index(drop=True)
    
    print(f"  Configurations present in all {n_tasks} tasks: {len(results_df)}")
    
    return results_df


if __name__ == "__main__":
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    
    PRIMARY_METRIC = 'mean_auc'
    OUTPUT_FILE = Path("config_ranking.csv")
    
    RESULTS = {
        'CRC_microbiome': r'E:\NAFLD\microbiome_results_with_externalCRC\phylum_abundance_PRJEB10878\D5_G2_S3\results.csv',
        'CRC_metabolomics': r'E:\NAFLD\CRC_Metabolomics_results3_20251203_001038\all_results.csv',
        'Cirrhosis_microbiome': r'E:\NAFLD\microbiome_results_with_externalLC\phylum_abundance_PRJEB6337\D9_G2_S3\results.csv',
        'Cirrhosis_metabolomics': r'E:\NAFLD\LC_Metabolomics_LiverCirrhosis_vs_Healthy_20251202_130555\all_results.csv',
        'Crohns_microbiome': r'E:\NAFLD\microbiome_results_with_externalCD\phylum_abundance_PRJEB15371\D4_G2_S3\results.csv',
        'Crohns_metabolomics': r'E:\NAFLD\CD_metabolomics_results3lightgbm_20251226_110833\all_results.csv',
    }
    
    MODELS = ['RandomForest', 'XGBoost', 'LightGBM',
              'LogisticRegression', 'SVM_RBF', 'MLP']
    
    # =====================================================================
    # RUN
    # =====================================================================
    
    print("\n" + "="*60)
    print("  RANKING CONFIGS BY BEST WORST-CASE PERFORMANCE")
    print("="*60)
    
    data = load_results(RESULTS)
    if not data:
        print("No data loaded!")
        exit(1)
    
    config_df = find_best_robust_config(
        data=data,
        metric=PRIMARY_METRIC,
        models=MODELS,
        require_all_tasks=True
    )
    
    if config_df.empty:
        print("No configurations found!")
        exit(1)
    
    config_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Saved: {OUTPUT_FILE}")
    print(f"  Total configurations ranked: {len(config_df)}")
    print(f"  Best worst-case AUC: {config_df.iloc[0]['min_across_tasks']:.4f}")
    print("="*60)