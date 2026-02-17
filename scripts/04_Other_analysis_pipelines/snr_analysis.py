"""
dimension_6_task_normalization_rank_aggregation.py

DIMENSION 6: Task-Specific Normalization & Rank Aggregation

Goals:
- Normalize performance within each task so tasks are comparable
- For each preprocessing component (scaling, FS, balancing) and model,
  compute per-task effect sizes (Z-score based)
- Generate task-specific heatmaps AND aggregated cross-task heatmaps
- Compute SNR (Signal-to-Noise Ratio = mean/std) for cross-task consistency
- Generate SNR heatmaps alongside mean Z-score heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
sns.set_style("whitegrid")


# ============================================================================
# CONFIGURATION
# ============================================================================

class NormRankConfig:
    """Configuration for Dimension 6 analysis."""

    ORIGINAL_RESULTS = {
        'CRC_microbiome': r'microbiome_results_CRC\phylum_abundance_PRJEB10878\D5_G2_S3\results.csv',
        'CRC_metabolomics': r'CRC_Metabolomics_results\all_results.csv',
        'Cirrhosis_microbiome': r'microbiome_results_LC\phylum_abundance_PRJEB6337\D9_G2_S3\results.csv',
        'Cirrhosis_metabolomics': r'LC_Metabolomics_results\all_results.csv',
        'Crohns_microbiome': r'microbiome_results_CD\phylum_abundance_PRJEB15371\D4_G2_S3\results.csv',
        'Crohns_metabolomics': r'CD_metabolomics_results\all_results.csv',
    }

    OUTPUT_ROOT = "snr_analysis"

    # Choose which metric to normalize
    PRIMARY_METRIC = 'mean_auc'
    SECONDARY_METRIC = 'mean_f1'

    # Models
    MODELS = ['RandomForest', 'XGBoost', 'LightGBM',
              'LogisticRegression', 'SVM_RBF', 'MLP']

    COMPONENTS = ['scaling', 'feature_selection', 'balancing']

    # Display names
    TASK_DISPLAY_NAMES = {
        'CRC_microbiome': 'CRC\n(Microbiome)',
        'CRC_metabolomics': 'CRC\n(Metabolomics)',
        'Cirrhosis_microbiome': 'Cirrhosis\n(Microbiome)',
        'Cirrhosis_metabolomics': 'Cirrhosis\n(Metabolomics)',
        'Crohns_microbiome': "Crohn's\n(Microbiome)",
        'Crohns_metabolomics': "Crohn's\n(Metabolomics)",
    }

    COMPONENT_DISPLAY_NAMES = {
        'scaling': 'Scaling Method',
        'feature_selection': 'Feature Selection',
        'balancing': 'Class Balancing'
    }


# ============================================================================
# UTILS
# ============================================================================

def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, (tuple, list)):
                key = "_".join(str(x) for x in k)
            else:
                key = str(k)
            new_dict[key] = convert_to_json_serializable(v)
        return new_dict
    if isinstance(obj, list):
        return [convert_to_json_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(x) for x in obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_within_task_normalization(perf: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(perf)
    mean = perf.mean()
    std = perf.std()
    if std > 0:
        z = (perf - mean) / std
    else:
        z = np.zeros_like(perf)

    ranks = perf.argsort().argsort()
    percentile = ranks / (n - 1) * 100 if n > 1 else np.zeros_like(perf)

    pmin, pmax = perf.min(), perf.max()
    if pmax > pmin:
        rel = (perf - pmin) / (pmax - pmin)
    else:
        rel = np.zeros_like(perf)

    return {
        'z': z,
        'percentile': percentile,
        'relative': rel
    }


# ============================================================================
# TASK-LEVEL NORMALIZATION AND EFFECTS
# ============================================================================

def analyze_task_normalization(task_name: str,
                               task_file: str,
                               config: NormRankConfig) -> Dict:
    print(f"\n{'='*80}")
    print(f"ANALYZING: {task_name}")
    print(f"{'='*80}")

    df = pd.read_csv(task_file)
    print(f"  Loaded {len(df):,} configurations")

    if config.PRIMARY_METRIC not in df.columns:
        print(f"  ❌ ERROR: {config.PRIMARY_METRIC} not found in columns")
        return {'task': task_name, 'effects': [], 'normalized_configs': []}

    if 'model_name' not in df.columns:
        print("  ❌ ERROR: 'model_name' column missing")
        return {'task': task_name, 'effects': [], 'normalized_configs': []}

    if 'success' in df.columns:
        df = df[df['success'] == True].copy()
        print(f"  After success filter: {len(df):,} configs")

    for comp in config.COMPONENTS:
        if comp not in df.columns:
            print(f"  ⚠️ Column '{comp}' missing; filling with 'unknown'")
            df[comp] = 'unknown'

    df['performance'] = df[config.PRIMARY_METRIC].astype(float)

    norm_records = []
    effects_records = []

    for model in config.MODELS:
        mdf = df[df['model_name'] == model].copy()
        if len(mdf) == 0:
            continue

        perf = mdf['performance'].values
        norm = compute_within_task_normalization(perf)
        mdf['z_score'] = norm['z']
        mdf['percentile'] = norm['percentile']
        mdf['relative'] = norm['relative']

        for _, row in mdf.iterrows():
            norm_records.append({
                'task': task_name,
                'model': model,
                'config_name': row['config_name'],
                'scaling': row['scaling'],
                'feature_selection': row['feature_selection'],
                'balancing': row['balancing'],
                'performance': row['performance'],
                'z_score': row['z_score'],
                'percentile': row['percentile'],
                'relative': row['relative']
            })

        for comp in config.COMPONENTS:
            comp_stats = (mdf
                          .groupby(comp)['z_score']
                          .agg(['mean', 'std', 'count'])
                          .reset_index()
                          .rename(columns={'mean': 'mean_z', 'std': 'std_z'}))
            if len(comp_stats) == 0:
                continue

            for _, row in comp_stats.iterrows():
                effects_records.append({
                    'task': task_name,
                    'model': model,
                    'component': comp,
                    'method': row[comp],
                    'mean_z': float(row['mean_z']),
                    'std_z': float(row['std_z']) if not np.isnan(row['std_z']) else 0.0,
                    'n_configs': int(row['count'])
                })

    print(f"  ✓ Task-level normalization complete for {task_name}")
    return {
        'task': task_name,
        'effects': effects_records,
        'normalized_configs': norm_records
    }


# ============================================================================
# TASK-SPECIFIC HEATMAP VISUALIZATION
# ============================================================================

def plot_task_specific_heatmaps(task_results: Dict,
                                output_dir: Path,
                                config: NormRankConfig):
    task_name = task_results['task']
    effects = task_results['effects']

    if not effects:
        print(f"  ⚠️ No effects for {task_name}, skipping heatmaps")
        return

    eff_df = pd.DataFrame(effects)

    task_fig_dir = output_dir / task_name
    task_fig_dir.mkdir(parents=True, exist_ok=True)

    for comp in config.COMPONENTS:
        sub = eff_df[eff_df['component'] == comp]
        if len(sub) == 0:
            continue

        pivot = sub.pivot(index='model', columns='method', values='mean_z')

        model_order = [m for m in config.MODELS if m in pivot.index]
        if model_order:
            pivot = pivot.reindex(model_order)

        n_methods = len(pivot.columns)
        n_models = len(pivot.index)
        fig_width = max(8, n_methods * 1.2)
        fig_height = max(5, n_models * 0.8)

        plt.figure(figsize=(fig_width, fig_height))

        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
        vmin = -vmax

        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap='RdBu_r',
            center=0, vmin=vmin, vmax=vmax,
            cbar_kws={'label': 'Mean Z-score'},
            annot_kws={'size': 9}
        )

        comp_display = config.COMPONENT_DISPLAY_NAMES.get(comp, comp.replace('_', ' ').title())

        plt.title(f"{task_name}: {comp_display}\n(Z-score normalized {config.PRIMARY_METRIC})",
                  fontweight='bold', fontsize=12)
        plt.xlabel(comp_display, fontweight='bold')
        plt.ylabel('Model', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        out_file = task_fig_dir / f"heatmap_{comp}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {task_name}/{out_file.name}")


def plot_all_tasks_combined_heatmap(all_task_results: List[Dict],
                                    output_dir: Path,
                                    config: NormRankConfig):
    print(f"\n{'='*80}")
    print("GENERATING COMBINED MULTI-TASK HEATMAPS")
    print(f"{'='*80}\n")

    all_effects = []
    for tres in all_task_results:
        if tres and 'effects' in tres:
            all_effects.extend(tres['effects'])

    if not all_effects:
        print("  ❌ No effects to plot")
        return

    eff_df = pd.DataFrame(all_effects)
    task_names = eff_df['task'].unique().tolist()
    n_tasks = len(task_names)

    for comp in config.COMPONENTS:
        comp_df = eff_df[eff_df['component'] == comp]
        if len(comp_df) == 0:
            continue

        n_cols = 3
        n_rows = (n_tasks + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_tasks > 1 else [axes]

        global_vmax = comp_df['mean_z'].abs().max()
        global_vmin = -global_vmax

        for idx, task_name in enumerate(task_names):
            ax = axes[idx]

            task_comp_df = comp_df[comp_df['task'] == task_name]
            if len(task_comp_df) == 0:
                ax.set_visible(False)
                continue

            pivot = task_comp_df.pivot(index='model', columns='method', values='mean_z')

            model_order = [m for m in config.MODELS if m in pivot.index]
            if model_order:
                pivot = pivot.reindex(model_order)

            sns.heatmap(
                pivot, annot=True, fmt=".2f", cmap='RdBu_r',
                center=0, vmin=global_vmin, vmax=global_vmax,
                cbar=(idx == len(task_names) - 1),
                ax=ax, annot_kws={'size': 8}
            )

            task_display = config.TASK_DISPLAY_NAMES.get(task_name, task_name)
            ax.set_title(task_display, fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.set_ylabel('Model' if idx % n_cols == 0 else '')
            ax.tick_params(axis='x', rotation=45)

        for idx in range(n_tasks, len(axes)):
            axes[idx].set_visible(False)

        comp_display = config.COMPONENT_DISPLAY_NAMES.get(comp, comp.replace('_', ' ').title())
        fig.suptitle(f"{comp_display}: Z-score Normalized Effects Across All Tasks",
                     fontweight='bold', fontsize=14, y=1.02)

        plt.tight_layout()

        out_file = output_dir / f"combined_all_tasks_{comp}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_file.name}")


# ============================================================================
# CROSS-TASK META-ANALYSIS (WITH SNR)
# ============================================================================

def aggregate_cross_task_effects(all_task_results: List[Dict],
                                 config: NormRankConfig) -> pd.DataFrame:
    """
    Combine per-task effects into meta-analytic table.
    Adds SNR column = meta_mean_z / meta_std_z.
    """
    print(f"\n{'='*80}")
    print("CROSS-TASK META-ANALYSIS OF PREPROCESSING EFFECTS")
    print(f"{'='*80}\n")

    all_effects = []
    for tres in all_task_results:
        if not tres or 'effects' not in tres:
            continue
        all_effects.extend(tres['effects'])

    if not all_effects:
        print("  ❌ No effects to aggregate")
        return pd.DataFrame()

    eff_df = pd.DataFrame(all_effects)

    group_cols = ['model', 'component', 'method']
    meta_records = []

    for (model, comp, method), g in eff_df.groupby(group_cols):
        mean_z = g['mean_z'].mean()
        std_z = g['mean_z'].std()
        n_tasks = g['task'].nunique()

        # SNR = mean / std (handle zero or NaN std)
        if pd.notna(std_z) and std_z > 0:
            snr = mean_z / std_z
        else:
            snr = np.nan

        per_task_z = g.set_index('task')['mean_z'].to_dict()

        meta_records.append({
            'model': model,
            'component': comp,
            'method': method,
            'n_tasks': int(n_tasks),
            'meta_mean_z': float(mean_z),
            'meta_std_z': float(std_z) if not np.isnan(std_z) else 0.0,
            'SNR': float(snr) if not np.isnan(snr) else np.nan,
            'per_task_z': per_task_z
        })

    meta_df = pd.DataFrame(meta_records)

    meta_df = meta_df.sort_values(['component', 'model', 'meta_mean_z'],
                                  ascending=[True, True, False])

    print("  Example aggregated results (top 10 rows):")
    display_cols = ['model', 'component', 'method', 'n_tasks', 'meta_mean_z', 'meta_std_z', 'SNR']
    print(meta_df[display_cols].head(10).to_string(index=False))

    return meta_df


# ============================================================================
# CROSS-TASK AGGREGATED HEATMAPS (MEAN Z-SCORE)
# ============================================================================

def plot_meta_heatmaps(meta_df: pd.DataFrame,
                       output_dir: Path,
                       config: NormRankConfig):
    if meta_df.empty:
        return

    print(f"\n{'='*80}")
    print("GENERATING AGGREGATED CROSS-TASK HEATMAPS (MEAN Z-SCORE)")
    print(f"{'='*80}\n")

    for comp in config.COMPONENTS:
        sub = meta_df[meta_df['component'] == comp]
        if len(sub) == 0:
            continue

        pivot = sub.pivot(index='model', columns='method', values='meta_mean_z')

        model_order = [m for m in config.MODELS if m in pivot.index]
        if model_order:
            pivot = pivot.reindex(model_order)

        n_methods = len(pivot.columns)
        n_models = len(pivot.index)
        fig_width = max(10, n_methods * 1.5)
        fig_height = max(6, n_models * 0.9)

        plt.figure(figsize=(fig_width, fig_height))

        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
        vmin = -vmax

        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap='RdBu_r',
            center=0, vmin=vmin, vmax=vmax,
            cbar_kws={'label': 'Meta-mean Z-score (across 6 tasks)'}
        )

        comp_display = config.COMPONENT_DISPLAY_NAMES.get(comp, comp.replace('_', ' ').title())

        plt.title(f"Aggregated Cross-Task Effects: {comp_display}\n"
                  f"(Meta-mean Z-score of {config.PRIMARY_METRIC} across all 6 tasks)",
                  fontweight='bold', fontsize=12)
        plt.xlabel(comp_display, fontweight='bold')
        plt.ylabel('Model', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        out_file = output_dir / f"meta_heatmap_{comp}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_file.name}")


# ============================================================================
# CROSS-TASK SNR HEATMAPS
# ============================================================================

def plot_snr_heatmaps(meta_df: pd.DataFrame,
                      output_dir: Path,
                      config: NormRankConfig):
    """
    For each component, plot model × method heatmap of SNR values
    (mean Z / std Z across tasks). Uses same RdBu_r color gradient.
    """
    if meta_df.empty:
        return

    print(f"\n{'='*80}")
    print("GENERATING AGGREGATED CROSS-TASK HEATMAPS (SNR = MEAN/STD)")
    print(f"{'='*80}\n")

    for comp in config.COMPONENTS:
        sub = meta_df[meta_df['component'] == comp]
        if len(sub) == 0:
            continue

        pivot = sub.pivot(index='model', columns='method', values='SNR')

        model_order = [m for m in config.MODELS if m in pivot.index]
        if model_order:
            pivot = pivot.reindex(model_order)

        n_methods = len(pivot.columns)
        n_models = len(pivot.index)
        fig_width = max(10, n_methods * 1.5)
        fig_height = max(6, n_models * 0.9)

        plt.figure(figsize=(fig_width, fig_height))

        # Symmetric color limits based on SNR range
        finite_vals = pivot.values[np.isfinite(pivot.values)]
        if len(finite_vals) > 0:
            vmax = max(abs(finite_vals.min()), abs(finite_vals.max()))
        else:
            vmax = 1.0
        vmin = -vmax

        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap='RdBu_r',
            center=0, vmin=vmin, vmax=vmax,
            cbar_kws={'label': 'SNR (Mean Z / Std Z across 6 tasks)'}
        )

        comp_display = config.COMPONENT_DISPLAY_NAMES.get(comp, comp.replace('_', ' ').title())

        plt.title(f"Cross-Task SNR: {comp_display}\n"
                  f"(Signal-to-Noise Ratio = Mean Z-score / Std Z-score across all 6 tasks)",
                  fontweight='bold', fontsize=12)
        plt.xlabel(comp_display, fontweight='bold')
        plt.ylabel('Model', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        out_file = output_dir / f"snr_heatmap_{comp}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_file.name}")


# ============================================================================
# HEATMAPS WITH STD ANNOTATIONS
# ============================================================================

def plot_meta_heatmap_with_std(meta_df: pd.DataFrame,
                               output_dir: Path,
                               config: NormRankConfig):
    if meta_df.empty:
        return

    print(f"\n{'='*80}")
    print("GENERATING HEATMAPS WITH CROSS-TASK VARIABILITY")
    print(f"{'='*80}\n")

    for comp in config.COMPONENTS:
        sub = meta_df[meta_df['component'] == comp]
        if len(sub) == 0:
            continue

        pivot_mean = sub.pivot(index='model', columns='method', values='meta_mean_z')
        pivot_std = sub.pivot(index='model', columns='method', values='meta_std_z')

        model_order = [m for m in config.MODELS if m in pivot_mean.index]
        if model_order:
            pivot_mean = pivot_mean.reindex(model_order)
            pivot_std = pivot_std.reindex(model_order)

        annot_df = pivot_mean.copy().astype(str)
        for i in range(len(pivot_mean.index)):
            for j in range(len(pivot_mean.columns)):
                mean_val = pivot_mean.iloc[i, j]
                std_val = pivot_std.iloc[i, j]
                if pd.notna(mean_val):
                    annot_df.iloc[i, j] = f"{mean_val:.2f}\n±{std_val:.2f}"
                else:
                    annot_df.iloc[i, j] = ""

        n_methods = len(pivot_mean.columns)
        n_models = len(pivot_mean.index)
        fig_width = max(12, n_methods * 1.8)
        fig_height = max(7, n_models * 1.0)

        plt.figure(figsize=(fig_width, fig_height))

        vmax = max(abs(pivot_mean.min().min()), abs(pivot_mean.max().max()))
        vmin = -vmax

        sns.heatmap(
            pivot_mean, annot=annot_df, fmt="", cmap='RdBu_r',
            center=0, vmin=vmin, vmax=vmax,
            cbar_kws={'label': 'Meta-mean Z-score'},
            annot_kws={'size': 8}
        )

        comp_display = config.COMPONENT_DISPLAY_NAMES.get(comp, comp.replace('_', ' ').title())

        plt.title(f"Cross-Task Effects with Variability: {comp_display}\n"
                  f"(Mean Z ± Std across 6 tasks)",
                  fontweight='bold', fontsize=12)
        plt.xlabel(comp_display, fontweight='bold')
        plt.ylabel('Model', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        out_file = output_dir / f"meta_heatmap_with_std_{comp}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_file.name}")


# ============================================================================
# SUMMARY TABLE GENERATION
# ============================================================================

def create_per_task_effects_table(all_task_results: List[Dict],
                                  output_dir: Path,
                                  config: NormRankConfig):
    print(f"\n{'='*80}")
    print("CREATING PER-TASK EFFECTS TABLE")
    print(f"{'='*80}\n")

    all_effects = []
    for tres in all_task_results:
        if tres and 'effects' in tres:
            all_effects.extend(tres['effects'])

    if not all_effects:
        return

    eff_df = pd.DataFrame(all_effects)

    for comp in config.COMPONENTS:
        comp_df = eff_df[eff_df['component'] == comp]
        if len(comp_df) == 0:
            continue

        wide_df = comp_df.pivot_table(
            index=['model', 'method'],
            columns='task',
            values='mean_z'
        )

        wide_df['Mean'] = wide_df.mean(axis=1)
        wide_df['Std'] = wide_df.std(axis=1)
        wide_df['SNR'] = wide_df.apply(
            lambda row: row['Mean'] / row['Std'] if row['Std'] > 0 else np.nan,
            axis=1
        )

        wide_df = wide_df.sort_values('Mean', ascending=False)
        wide_df = wide_df.reset_index()

        out_file = output_dir / f"per_task_effects_{comp}.csv"
        wide_df.to_csv(out_file, index=False)
        print(f"  ✓ Saved: {out_file.name}")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_dimension_6_analysis(config: NormRankConfig):
    print(f"\n{'='*80}")
    print("DIMENSION 6: TASK NORMALIZATION & RANK AGGREGATION")
    print(f"{'='*80}")
    print(f"Primary metric: {config.PRIMARY_METRIC}")
    print(f"Secondary metric: {config.SECONDARY_METRIC}")
    print(f"{'='*80}\n")

    output_root = Path(config.OUTPUT_ROOT)
    per_task_dir = output_root / "per_task_results"
    cross_task_dir = output_root / "cross_task_analysis"
    task_figures_dir = output_root / "figures_per_task"
    combined_figures_dir = output_root / "figures_combined"
    meta_figures_dir = output_root / "figures_aggregated"
    snr_figures_dir = output_root / "figures_snr"
    summary_dir = output_root / "summary_reports"

    for d in [per_task_dir, cross_task_dir, task_figures_dir,
              combined_figures_dir, meta_figures_dir, snr_figures_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_root}\n")

    print("Checking data files:")
    for task_name, filepath in config.ORIGINAL_RESULTS.items():
        if Path(filepath).exists():
            print(f"  ✓ {task_name}")
        else:
            print(f"  ❌ {task_name}: {filepath}")
    print()

    all_task_results = []

    # ========================================================================
    # STEP 1: Per-task analysis and task-specific heatmaps
    # ========================================================================

    for task_name, task_file in config.ORIGINAL_RESULTS.items():
        if not Path(task_file).exists():
            print(f"⚠️  Skipping {task_name}: file not found")
            continue

        try:
            tres = analyze_task_normalization(task_name, task_file, config)
            all_task_results.append(tres)

            out_file = per_task_dir / f"{task_name}_norm_effects.json"
            to_save = convert_to_json_serializable(tres)
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(to_save, f, indent=2)
            print(f"  ✓ Saved: {out_file.name}")

            print(f"\n  Generating heatmaps for {task_name}:")
            plot_task_specific_heatmaps(tres, task_figures_dir, config)

        except Exception as e:
            print(f"  ❌ Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    all_task_results = [
        tr for tr in all_task_results
        if isinstance(tr, dict) and tr.get('effects')
    ]

    if not all_task_results:
        print("\n❌ No tasks successfully processed!")
        return

    # ========================================================================
    # STEP 2: Combined multi-task heatmaps
    # ========================================================================

    plot_all_tasks_combined_heatmap(all_task_results, combined_figures_dir, config)

    # ========================================================================
    # STEP 3: Cross-task meta-analysis (with SNR)
    # ========================================================================

    meta_df = aggregate_cross_task_effects(all_task_results, config)

    # Save meta results (with SNR column, without per_task_z)
    meta_save = meta_df.drop(columns=['per_task_z'], errors='ignore')
    meta_file = cross_task_dir / "meta_component_effects.csv"
    meta_save.to_csv(meta_file, index=False)
    print(f"\n✓ Saved: {meta_file.name}")

    # ========================================================================
    # STEP 4: Aggregated cross-task heatmaps (mean Z-score)
    # ========================================================================

    plot_meta_heatmaps(meta_df, meta_figures_dir, config)
    plot_meta_heatmap_with_std(meta_df, meta_figures_dir, config)

    # ========================================================================
    # STEP 5: SNR heatmaps
    # ========================================================================

    plot_snr_heatmaps(meta_df, snr_figures_dir, config)

    # ========================================================================
    # STEP 6: Per-task effects table (wide format, with SNR column)
    # ========================================================================

    create_per_task_effects_table(all_task_results, cross_task_dir, config)

    # ========================================================================
    # STEP 7: Summary report
    # ========================================================================

    summary_file = summary_dir / "dimension_6_executive_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DIMENSION 6: TASK NORMALIZATION & RANK AGGREGATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tasks Analyzed: {len(all_task_results)}\n")
        f.write(f"Primary Metric: {config.PRIMARY_METRIC}\n\n")

        f.write("Tasks:\n")
        for tres in all_task_results:
            f.write(f"  - {tres['task']}\n")

        f.write("\nOutputs Generated:\n")
        f.write(f"  - Per-task JSON results: {per_task_dir}\n")
        f.write(f"  - Task-specific heatmaps: {task_figures_dir}\n")
        f.write(f"  - Combined multi-task heatmaps: {combined_figures_dir}\n")
        f.write(f"  - Aggregated mean Z-score heatmaps: {meta_figures_dir}\n")
        f.write(f"  - SNR heatmaps: {snr_figures_dir}\n")
        f.write(f"  - Meta-analysis CSV (with SNR): {meta_file}\n")
        f.write(f"  - Per-task effects tables (with SNR): {cross_task_dir}\n")

        f.write("\nSNR (Signal-to-Noise Ratio):\n")
        f.write("  SNR = Meta-mean Z-score / Meta-std Z-score\n")
        f.write("  High positive SNR → consistently beneficial across tasks\n")
        f.write("  High negative SNR → consistently harmful across tasks\n")
        f.write("  SNR near zero → inconsistent or no effect\n")

    print(f"\n✓ Saved summary: {summary_file.name}")
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results saved to: {output_root}\n")

    print("Generated outputs:")
    print(f"  📁 {per_task_dir}          — per-task JSON results")
    print(f"  📁 {task_figures_dir}      — per-task heatmaps (Z-score)")
    print(f"  📁 {combined_figures_dir}  — all 6 tasks side-by-side")
    print(f"  📁 {meta_figures_dir}      — aggregated mean Z-score heatmaps")
    print(f"  📁 {snr_figures_dir}       — SNR heatmaps (mean/std)")
    print(f"  📁 {cross_task_dir}        — CSV tables (with SNR column)")
    print(f"  📁 {summary_dir}           — executive summary")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = NormRankConfig()
    run_dimension_6_analysis(config)
    print("\n🎉 Done!")