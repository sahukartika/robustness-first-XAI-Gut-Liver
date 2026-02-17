"""
================================================================================
PATHWAY OVER-REPRESENTATION ANALYSIS (ORA) WITH SHAP VALUES
================================================================================

Description:
    Performs pathway enrichment analysis using Fisher's exact test to identify
    KEGG pathways significantly over-represented in disease-associated microbes.
    
    INCLUDES: Sum of SHAP values per pathway for importance ranking.

Statistical Method:
    - Fisher's exact test (one-sided, greater)
    - Multiple testing correction: Benjamini-Hochberg FDR

Inputs:
    1. SHAP file: Microbe importance scores
    2. Pathway file: Microbe → KEGG pathway mapping
    3. MSP mapping file (optional): MSP → real microbe name

Outputs:
    1. PATHWAY_ORA_RESULTS_ALL.xlsx: All tested pathways with statistics
    2. PATHWAY_ORA_RESULTS_SIGNIFICANT.xlsx: FDR-significant pathways
    3. PATHWAY_ORA_SUMMARY.txt: Analysis summary
    4. Visualization plots

================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_shap_data(
    shap_file: str,
    feature_col_idx: int = 0,
    shap_col_idx: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load and aggregate SHAP values for microbes.
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]]
        - DataFrame with aggregated SHAP values
        - Dictionary mapping microbe_clean -> shap_value
    """
    shap_path = Path(shap_file)
    
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP file not found: {shap_file}")
    
    print(f"Loading SHAP file: {shap_file}")
    
    # Load based on file type
    if shap_path.suffix.lower() in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(shap_path)
        dfs = []
        for sheet in xl.sheet_names:
            df_sheet = xl.parse(sheet, header=0)
            if df_sheet.shape[1] > max(feature_col_idx, shap_col_idx):
                sub = df_sheet.iloc[:, [feature_col_idx, shap_col_idx]].copy()
                sub.columns = ["microbe", "shap_value"]
                dfs.append(sub)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        df = pd.read_csv(shap_path)
        df = df.iloc[:, [feature_col_idx, shap_col_idx]].copy()
        df.columns = ["microbe", "shap_value"]
    
    # Clean and aggregate
    df["microbe"] = df["microbe"].astype(str).str.strip()
    df["microbe_clean"] = df["microbe"].str.lower().str.strip()
    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce")
    
    # Remove invalid entries
    df = df.dropna(subset=["shap_value"])
    df = df[df["microbe_clean"] != "nan"]
    
    # Aggregate by microbe (mean SHAP)
    agg = df.groupby(["microbe", "microbe_clean"]).agg({
        "shap_value": "mean"
    }).reset_index()
    
    agg = agg.sort_values("shap_value", ascending=False).reset_index(drop=True)
    
    # Create lookup dictionary: microbe_clean -> shap_value
    shap_lookup = dict(zip(agg["microbe_clean"], agg["shap_value"]))
    
    print(f"  Loaded {len(agg)} unique microbes")
    print(f"  SHAP range: [{agg['shap_value'].min():.4f}, {agg['shap_value'].max():.4f}]")
    print(f"  SHAP sum: {agg['shap_value'].sum():.4f}")
    
    return agg, shap_lookup


def load_microbe_pathway_mapping(
    pathway_file: str,
    microbe_col_idx: int = 0,
    pathway_col_idx: int = 3,
) -> pd.DataFrame:
    """
    Load microbe → pathway mapping from Excel file.
    """
    path = Path(pathway_file)
    
    if not path.exists():
        raise FileNotFoundError(f"Pathway file not found: {pathway_file}")
    
    print(f"\nLoading pathway mapping: {pathway_file}")
    
    xl = pd.ExcelFile(path)
    print(f"  Sheets found: {xl.sheet_names}")
    
    records = []
    
    for sheet in xl.sheet_names:
        df_sheet = xl.parse(sheet, header=0)
        
        if df_sheet.shape[1] <= max(microbe_col_idx, pathway_col_idx):
            print(f"  WARNING: Sheet '{sheet}' has insufficient columns, skipped")
            continue
        
        for _, row in df_sheet.iterrows():
            microbe = str(row.iloc[microbe_col_idx]).strip()
            pathways_cell = row.iloc[pathway_col_idx]
            
            # Skip invalid microbe entries
            if not microbe or microbe.lower() == "nan":
                continue
            
            # Parse pathway cell (may contain multiple pathways, one per line)
            if pd.isna(pathways_cell):
                continue
            
            if isinstance(pathways_cell, str):
                lines = pathways_cell.splitlines()
            else:
                lines = [str(pathways_cell)]
            
            for line in lines:
                line = line.strip()
                if not line or line.lower() == "nan":
                    continue
                
                # Parse pathway format: "ko00010 - Glycolysis / Gluconeogenesis"
                if " - " in line:
                    pathway_id, pathway_name = line.split(" - ", 1)
                    pathway_id = pathway_id.strip()
                    pathway_name = pathway_name.strip()
                else:
                    pathway_id = ""
                    pathway_name = line
                
                records.append({
                    "microbe": microbe,
                    "microbe_clean": microbe.lower().strip(),
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_name,
                    "pathway_full": line,
                    "source_sheet": sheet,
                })
    
    df = pd.DataFrame(records)
    
    if df.empty:
        raise ValueError("No valid microbe-pathway pairs found!")
    
    # Remove duplicates
    df = df.drop_duplicates(
        subset=["microbe_clean", "pathway_full"]
    ).reset_index(drop=True)
    
    n_microbes = df["microbe_clean"].nunique()
    n_pathways = df["pathway_full"].nunique()
    
    print(f"  Loaded {len(df)} microbe-pathway pairs")
    print(f"  Unique microbes: {n_microbes}")
    print(f"  Unique pathways: {n_pathways}")
    
    return df


def load_msp_to_name_mapping(
    mapping_file: str,
    msp_col_idx: int = 0,
    name_col_idx: int = 1,
) -> Dict[str, str]:
    """
    Load MSP → real microbe name mapping.
    Returns dictionary: lowercase_msp -> real_name
    """
    path = Path(mapping_file)
    
    if not path.exists():
        print(f"WARNING: Mapping file not found: {mapping_file}")
        return {}
    
    print(f"\nLoading MSP name mapping: {mapping_file}")
    
    name_mapping = {}
    
    if path.suffix.lower() in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, header=0)
            if df.shape[1] <= max(msp_col_idx, name_col_idx):
                continue
            for _, row in df.iterrows():
                msp = str(row.iloc[msp_col_idx]).strip()
                name = str(row.iloc[name_col_idx]).strip()
                if msp and name and msp.lower() != "nan" and name.lower() != "nan":
                    name_mapping[msp.lower()] = name
    else:
        df = pd.read_csv(path)
        if df.shape[1] > max(msp_col_idx, name_col_idx):
            for _, row in df.iterrows():
                msp = str(row.iloc[msp_col_idx]).strip()
                name = str(row.iloc[name_col_idx]).strip()
                if msp and name and msp.lower() != "nan" and name.lower() != "nan":
                    name_mapping[msp.lower()] = name
    
    print(f"  Loaded {len(name_mapping)} MSP→name mappings")
    
    return name_mapping


# ==============================================================================
# OVER-REPRESENTATION ANALYSIS (ORA) WITH SHAP VALUES
# ==============================================================================

def select_disease_microbes(
    shap_df: pd.DataFrame,
    method: str = "threshold",
    threshold: float = 0.0,
    top_n: int = None,
    top_percentile: float = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Select disease-associated microbes based on SHAP values.
    """
    if method == "top_n" and top_n is not None:
        selected = shap_df.head(top_n).copy()
        description = f"Top {top_n} microbes by SHAP value"
    
    elif method == "percentile" and top_percentile is not None:
        cutoff = shap_df["shap_value"].quantile(1 - top_percentile)
        selected = shap_df[shap_df["shap_value"] >= cutoff].copy()
        description = f"Top {top_percentile*100:.0f}% microbes (SHAP ≥ {cutoff:.4f})"
    
    else:  # Default: threshold
        selected = shap_df[shap_df["shap_value"] > threshold].copy()
        description = f"Microbes with SHAP > {threshold}"
    
    return selected, description


def perform_ora_with_shap(
    disease_microbes: Set[str],
    background_microbes: Set[str],
    microbe_pathway_df: pd.DataFrame,
    shap_lookup: Dict[str, float],
    name_mapping: Dict[str, str] = None,
    min_pathway_size: int = 2,
    min_disease_hits: int = 1,
) -> pd.DataFrame:
    """
    Perform Over-Representation Analysis using Fisher's exact test.
    INCLUDES: Sum of SHAP values for disease microbes in each pathway.
    
    Parameters:
    -----------
    disease_microbes : Set[str]
        Set of disease-associated microbe names (lowercase)
    background_microbes : Set[str]
        Set of all microbes in background (lowercase)
    microbe_pathway_df : pd.DataFrame
        Microbe-pathway mapping
    shap_lookup : Dict[str, float]
        Dictionary mapping microbe_clean -> shap_value
    name_mapping : Dict[str, str]
        Dictionary mapping microbe_clean -> real_name (optional)
    min_pathway_size : int
        Minimum number of microbes in a pathway to test
    min_disease_hits : int
        Minimum disease microbes in pathway to include in results
    
    Returns:
    --------
    pd.DataFrame
        ORA results with statistics and SHAP sums for each pathway
    """
    print("\n" + "=" * 60)
    print("OVER-REPRESENTATION ANALYSIS (Fisher's Exact Test)")
    print("WITH SHAP VALUE AGGREGATION")
    print("=" * 60)
    
    if name_mapping is None:
        name_mapping = {}
    
    # Ensure we only consider microbes present in both datasets
    disease_in_background = disease_microbes & background_microbes
    non_disease = background_microbes - disease_microbes
    
    n_disease = len(disease_in_background)
    n_non_disease = len(non_disease)
    n_total = len(background_microbes)
    
    print(f"\nMicrobe counts:")
    print(f"  Disease-associated: {n_disease}")
    print(f"  Non-disease (background): {n_non_disease}")
    print(f"  Total in background: {n_total}")
    
    if n_disease == 0:
        raise ValueError("No disease microbes found in background set!")
    
    # Calculate total SHAP of disease microbes
    total_disease_shap = sum(shap_lookup.get(m, 0) for m in disease_in_background)
    print(f"  Total SHAP of disease microbes: {total_disease_shap:.4f}")
    
    # Get all pathways
    all_pathways = microbe_pathway_df["pathway_full"].unique()
    print(f"\nPathways to test: {len(all_pathways)}")
    
    results = []
    pathways_tested = 0
    
    for pathway in all_pathways:
        # Get microbes with this pathway
        pathway_microbes = set(
            microbe_pathway_df[
                microbe_pathway_df["pathway_full"] == pathway
            ]["microbe_clean"].unique()
        )
        
        # Only consider microbes in our background
        pathway_microbes_in_bg = pathway_microbes & background_microbes
        
        # Skip pathways with too few microbes
        if len(pathway_microbes_in_bg) < min_pathway_size:
            continue
        
        # Build contingency table
        a = len(disease_in_background & pathway_microbes_in_bg)
        b = len(non_disease & pathway_microbes_in_bg)
        c = len(disease_in_background - pathway_microbes_in_bg)
        d = len(non_disease - pathway_microbes_in_bg)
        
        # Skip if no disease microbes have this pathway
        if a < min_disease_hits:
            continue
        
        pathways_tested += 1
        
        # Fisher's exact test (one-sided: enrichment)
        contingency_table = [[a, b], [c, d]]
        odds_ratio, pvalue = stats.fisher_exact(
            contingency_table, 
            alternative='greater'
        )
        
        # Get pathway info
        pathway_info = microbe_pathway_df[
            microbe_pathway_df["pathway_full"] == pathway
        ].iloc[0]
        
        # Get disease microbes in this pathway
        disease_microbes_in_pathway = disease_in_background & pathway_microbes_in_bg
        
        # =====================================================================
        # CALCULATE SHAP SUM FOR THIS PATHWAY
        # =====================================================================
        shap_sum = sum(shap_lookup.get(m, 0) for m in disease_microbes_in_pathway)
        shap_mean = shap_sum / len(disease_microbes_in_pathway) if disease_microbes_in_pathway else 0
        shap_max = max((shap_lookup.get(m, 0) for m in disease_microbes_in_pathway), default=0)
        shap_min = min((shap_lookup.get(m, 0) for m in disease_microbes_in_pathway), default=0)
        
        # Percentage of total disease SHAP
        shap_percentage = (shap_sum / total_disease_shap * 100) if total_disease_shap > 0 else 0
        
        # Get microbe details with SHAP values
        microbe_shap_details = []
        for m in sorted(disease_microbes_in_pathway):
            shap_val = shap_lookup.get(m, 0)
            real_name = name_mapping.get(m, m)
            microbe_shap_details.append(f"{real_name} ({shap_val:.4f})")
        
        # Calculate expected count under null hypothesis
        expected = (n_disease * len(pathway_microbes_in_bg)) / n_total
        
        # Fold enrichment
        fold_enrichment = a / expected if expected > 0 else np.inf
        
        results.append({
            "pathway_id": pathway_info["pathway_id"],
            "pathway_name": pathway_info["pathway_name"],
            "pathway_full": pathway,
            
            # Counts
            "disease_in_pathway": a,
            "disease_not_in_pathway": c,
            "background_in_pathway": a + b,
            "background_not_in_pathway": c + d,
            
            # Contingency table values
            "a_disease_pathway": a,
            "b_nondisease_pathway": b,
            "c_disease_nopathway": c,
            "d_nondisease_nopathway": d,
            
            # Statistics
            "expected_count": expected,
            "fold_enrichment": fold_enrichment,
            "odds_ratio": odds_ratio,
            "pvalue": pvalue,
            
            # =====================================================================
            # SHAP VALUES (NEW!)
            # =====================================================================
            "shap_sum": shap_sum,
            "shap_mean": shap_mean,
            "shap_max": shap_max,
            "shap_min": shap_min,
            "shap_percentage": shap_percentage,
            
            # Microbe info
            "disease_microbes_in_pathway": "; ".join(sorted(disease_microbes_in_pathway)),
            "disease_microbes_real_names": "; ".join(sorted(
                name_mapping.get(m, m) for m in disease_microbes_in_pathway
            )),
            "disease_microbes_with_shap": "; ".join(microbe_shap_details),
        })
    
    print(f"Pathways tested: {pathways_tested}")
    
    if not results:
        print("WARNING: No pathways met the testing criteria!")
        return pd.DataFrame()
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction (Benjamini-Hochberg FDR)
    _, fdr_values, _, _ = multipletests(
        results_df["pvalue"], 
        method="fdr_bh"
    )
    results_df["fdr"] = fdr_values
    
    # Also add Bonferroni correction for reference
    results_df["pvalue_bonferroni"] = np.minimum(
        results_df["pvalue"] * len(results_df), 
        1.0
    )
    
    # Sort by p-value
    results_df = results_df.sort_values("pvalue").reset_index(drop=True)
    
    # Add rank
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    
    # Add SHAP-based rank
    results_df["rank_by_shap"] = results_df["shap_sum"].rank(ascending=False).astype(int)
    
    # Summary statistics
    n_sig_005 = (results_df["fdr"] < 0.05).sum()
    n_sig_010 = (results_df["fdr"] < 0.10).sum()
    n_sig_nominal = (results_df["pvalue"] < 0.05).sum()
    
    print(f"\nResults summary:")
    print(f"  Pathways with nominal p < 0.05: {n_sig_nominal}")
    print(f"  Pathways with FDR < 0.10: {n_sig_010}")
    print(f"  Pathways with FDR < 0.05: {n_sig_005}")
    
    # SHAP summary
    print(f"\nSHAP value summary (significant pathways, FDR < 0.05):")
    sig_df = results_df[results_df["fdr"] < 0.05]
    if len(sig_df) > 0:
        print(f"  Total SHAP sum across all significant pathways: {sig_df['shap_sum'].sum():.4f}")
        print(f"  Average SHAP sum per pathway: {sig_df['shap_sum'].mean():.4f}")
        print(f"  Max SHAP sum (single pathway): {sig_df['shap_sum'].max():.4f}")
    
    return results_df


# ==============================================================================
# VISUALIZATION FUNCTIONS WITH SHAP
# ==============================================================================

def plot_bar_chart_with_shap(df: pd.DataFrame, 
                              n_top: int = 20,
                              output_file: str = None):
    """
    Create horizontal bar chart with SHAP color coding
    """
    # Add derived columns if not present
    if 'neg_log10_fdr' not in df.columns:
        df = df.copy()
        df['neg_log10_fdr'] = -np.log10(df['fdr'].clip(lower=1e-100))
    
    # Get top pathways by significance
    plot_df = df.nsmallest(n_top, 'fdr').copy()
    plot_df = plot_df.sort_values('neg_log10_fdr', ascending=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    # =========================================================================
    # Left plot: -log10(FDR) colored by Fold Enrichment
    # =========================================================================
    colors1 = plt.cm.Reds(plot_df['fold_enrichment'] / plot_df['fold_enrichment'].max())
    
    bars1 = ax1.barh(range(len(plot_df)), 
                     plot_df['neg_log10_fdr'],
                     color=colors1,
                     edgecolor='black',
                     linewidth=0.5)
    
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels(plot_df['pathway_name'], fontsize=9)
    ax1.set_xlabel('-log₁₀(FDR)', fontsize=12, fontweight='bold')
    ax1.set_title('Statistical Significance\n(Color = Fold Enrichment)', fontsize=12, fontweight='bold')
    
    # Add significance lines
    ax1.axvline(x=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    ax1.axvline(x=-np.log10(0.01), color='gray', linestyle=':', linewidth=1)
    
    # Colorbar for fold enrichment
    sm1 = plt.cm.ScalarMappable(cmap='Reds', 
                                 norm=plt.Normalize(vmin=plot_df['fold_enrichment'].min(), 
                                                    vmax=plot_df['fold_enrichment'].max()))
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1, shrink=0.5, aspect=20)
    cbar1.set_label('Fold Enrichment', fontsize=10)
    
    # =========================================================================
    # Right plot: SHAP Sum colored by significance
    # =========================================================================
    colors2 = plt.cm.Blues(plot_df['neg_log10_fdr'] / plot_df['neg_log10_fdr'].max())
    
    bars2 = ax2.barh(range(len(plot_df)), 
                     plot_df['shap_sum'],
                     color=colors2,
                     edgecolor='black',
                     linewidth=0.5)
    
    ax2.set_xlabel('Sum of SHAP Values', fontsize=12, fontweight='bold')
    ax2.set_title('SHAP Importance\n(Color = -log₁₀(FDR))', fontsize=12, fontweight='bold')
    
    # Colorbar for significance
    sm2 = plt.cm.ScalarMappable(cmap='Blues', 
                                 norm=plt.Normalize(vmin=plot_df['neg_log10_fdr'].min(), 
                                                    vmax=plot_df['neg_log10_fdr'].max()))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.5, aspect=20)
    cbar2.set_label('-log₁₀(FDR)', fontsize=10)
    
    plt.suptitle(f'Top {n_top} Enriched KEGG Pathways', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_file}")
    
    plt.show()
    
    return fig


def plot_shap_vs_significance(df: pd.DataFrame,
                               fdr_threshold: float = 0.05,
                               output_file: str = None):
    """
    Scatter plot: SHAP sum vs -log10(FDR)
    Helps visualize relationship between statistical significance and biological importance
    """
    plot_df = df.copy()
    plot_df['neg_log10_fdr'] = -np.log10(plot_df['fdr'].clip(lower=1e-100))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by significance
    colors = ['red' if fdr < fdr_threshold else 'gray' for fdr in plot_df['fdr']]
    
    # Size by number of disease microbes
    sizes = plot_df['disease_in_pathway'] * 2
    
    scatter = ax.scatter(
        x=plot_df['shap_sum'],
        y=plot_df['neg_log10_fdr'],
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.3
    )
    
    # Add labels for top pathways
    top_by_shap = plot_df.nlargest(5, 'shap_sum')
    top_by_fdr = plot_df.nsmallest(5, 'fdr')
    top_pathways = pd.concat([top_by_shap, top_by_fdr]).drop_duplicates()
    
    for _, row in top_pathways.iterrows():
        ax.annotate(
            row['pathway_name'][:30],
            xy=(row['shap_sum'], row['neg_log10_fdr']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=7,
            alpha=0.8
        )
    
    # Add significance line
    ax.axhline(y=-np.log10(fdr_threshold), color='blue', linestyle='--', 
               linewidth=1, label=f'FDR = {fdr_threshold}')
    
    # Labels
    ax.set_xlabel('Sum of SHAP Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('-log₁₀(FDR)', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Importance vs Statistical Significance', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='red', label=f'FDR < {fdr_threshold}'),
        Patch(facecolor='gray', label=f'FDR ≥ {fdr_threshold}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=5, label='10 microbes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='50 microbes'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_file}")
    
    plt.show()
    
    return fig


def plot_dot_plot_with_shap(df: pd.DataFrame,
                             n_top: int = 25,
                             output_file: str = None):
    """
    Create dot plot with SHAP coloring
    X-axis: Fold Enrichment
    Y-axis: Pathway name
    Dot size: Number of disease microbes
    Dot color: SHAP sum
    """
    # Add derived columns if not present
    if 'neg_log10_fdr' not in df.columns:
        df = df.copy()
        df['neg_log10_fdr'] = -np.log10(df['fdr'].clip(lower=1e-100))
    
    # Get top pathways
    plot_df = df.nsmallest(n_top, 'fdr').copy()
    plot_df = plot_df.sort_values('fdr', ascending=False)  # Reverse for plot
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot
    scatter = ax.scatter(
        x=plot_df['fold_enrichment'],
        y=range(len(plot_df)),
        s=plot_df['disease_in_pathway'] * 3,  # Scale dot size
        c=plot_df['shap_sum'],
        cmap='YlOrRd',
        edgecolors='black',
        linewidth=0.5,
        alpha=0.8
    )
    
    # Y-axis labels with FDR
    labels = [f"{row['pathway_name']} (FDR={row['fdr']:.1e})" 
              for _, row in plot_df.iterrows()]
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=8)
    
    # Labels and title
    ax.set_xlabel('Fold Enrichment', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_top} Enriched KEGG Pathways\n(Dot size = Microbe count, Color = SHAP sum)', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar for SHAP
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Sum of SHAP Values', fontsize=10)
    
    # Add size legend
    sizes = [10, 50, 100]
    for size in sizes:
        ax.scatter([], [], s=size*3, c='gray', alpha=0.5, 
                   label=f'{size} microbes', edgecolors='black', linewidth=0.5)
    ax.legend(title='Disease microbes', loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_file}")
    
    plt.show()
    
    return fig


def plot_top_pathways_by_shap(df: pd.DataFrame,
                               n_top: int = 20,
                               fdr_threshold: float = 0.05,
                               output_file: str = None):
    """
    Bar chart of top pathways ranked by SHAP sum (filtered by significance)
    """
    # Filter by significance
    sig_df = df[df['fdr'] < fdr_threshold].copy()
    
    if len(sig_df) == 0:
        print(f"No pathways with FDR < {fdr_threshold}")
        return None
    
    # Sort by SHAP sum and get top
    plot_df = sig_df.nlargest(n_top, 'shap_sum').copy()
    plot_df = plot_df.sort_values('shap_sum', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by -log10(FDR)
    plot_df['neg_log10_fdr'] = -np.log10(plot_df['fdr'].clip(lower=1e-100))
    colors = plt.cm.RdYlBu_r(plot_df['neg_log10_fdr'] / plot_df['neg_log10_fdr'].max())
    
    bars = ax.barh(range(len(plot_df)), 
                   plot_df['shap_sum'],
                   color=colors,
                   edgecolor='black',
                   linewidth=0.5)
    
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['pathway_name'], fontsize=9)
    
    ax.set_xlabel('Sum of SHAP Values', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_top} Pathways by SHAP Importance\n(FDR < {fdr_threshold}, Color = -log₁₀(FDR))', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', 
                                norm=plt.Normalize(vmin=plot_df['neg_log10_fdr'].min(), 
                                                   vmax=plot_df['neg_log10_fdr'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('-log₁₀(FDR)', fontsize=10)
    
    # Add SHAP value labels
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.annotate(f'{row["shap_sum"]:.3f}', 
                    xy=(row['shap_sum'] + 0.01, i),
                    fontsize=8, va='center')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_file}")
    
    plt.show()
    
    return fig


# ==============================================================================
# PUBLICATION TABLES
# ==============================================================================

def create_publication_table_with_shap(df: pd.DataFrame,
                                        n_top: int = 20,
                                        output_file: str = None) -> pd.DataFrame:
    """
    Create a publication-ready table with SHAP values
    """
    # Get top pathways
    table_df = df.nsmallest(n_top, 'fdr').copy()
    
    # Format columns for publication
    pub_table = pd.DataFrame({
        'Rank': range(1, len(table_df) + 1),
        'Pathway ID': table_df['pathway_id'],
        'Pathway Name': table_df['pathway_name'],
        'Disease Microbes (n)': table_df['disease_in_pathway'].astype(int),
        'Background (n)': table_df['background_in_pathway'].astype(int),
        'Fold Enrichment': table_df['fold_enrichment'].apply(lambda x: f'{x:.2f}'),
        'Odds Ratio': table_df['odds_ratio'].apply(lambda x: f'{x:.2f}'),
        'P-value': table_df['pvalue'].apply(lambda x: f'{x:.2e}'),
        'FDR': table_df['fdr'].apply(lambda x: f'{x:.2e}'),
        'SHAP Sum': table_df['shap_sum'].apply(lambda x: f'{x:.4f}'),
        'SHAP Mean': table_df['shap_mean'].apply(lambda x: f'{x:.4f}'),
        'SHAP %': table_df['shap_percentage'].apply(lambda x: f'{x:.2f}%'),
        'SHAP Rank': table_df['rank_by_shap'].astype(int),
    })
    
    if output_file:
        pub_table.to_excel(output_file, index=False)
        print(f"Saved: {output_file}")
    
    return pub_table


def create_detailed_results_table(df: pd.DataFrame,
                                   fdr_threshold: float = 0.05,
                                   output_file: str = None) -> pd.DataFrame:
    """
    Create detailed results table with microbe-level SHAP information
    """
    sig_df = df[df['fdr'] < fdr_threshold].copy()
    
    detailed_table = pd.DataFrame({
        'Rank (FDR)': sig_df['rank'],
        'Rank (SHAP)': sig_df['rank_by_shap'],
        'Pathway ID': sig_df['pathway_id'],
        'Pathway Name': sig_df['pathway_name'],
        'Disease Microbes (n)': sig_df['disease_in_pathway'],
        'Total Microbes (n)': sig_df['background_in_pathway'],
        'Expected': sig_df['expected_count'].apply(lambda x: f'{x:.1f}'),
        'Fold Enrichment': sig_df['fold_enrichment'].apply(lambda x: f'{x:.2f}'),
        'Odds Ratio': sig_df['odds_ratio'].apply(lambda x: f'{x:.2f}'),
        'P-value': sig_df['pvalue'].apply(lambda x: f'{x:.2e}'),
        'FDR': sig_df['fdr'].apply(lambda x: f'{x:.2e}'),
        'SHAP Sum': sig_df['shap_sum'].apply(lambda x: f'{x:.4f}'),
        'SHAP Mean': sig_df['shap_mean'].apply(lambda x: f'{x:.4f}'),
        'SHAP Max': sig_df['shap_max'].apply(lambda x: f'{x:.4f}'),
        'SHAP Min': sig_df['shap_min'].apply(lambda x: f'{x:.4f}'),
        'SHAP %': sig_df['shap_percentage'].apply(lambda x: f'{x:.2f}'),
        'Microbes (Real Names)': sig_df['disease_microbes_real_names'],
        'Microbes with SHAP': sig_df['disease_microbes_with_shap'],
    })
    
    if output_file:
        detailed_table.to_excel(output_file, index=False)
        print(f"Saved: {output_file}")
    
    return detailed_table


# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

def print_summary_statistics_with_shap(df: pd.DataFrame):
    """Print comprehensive summary statistics including SHAP"""
    
    print("\n" + "=" * 70)
    print("ORA RESULTS SUMMARY (WITH SHAP VALUES)")
    print("=" * 70)
    
    # Significance counts
    print("\n📊 SIGNIFICANCE SUMMARY:")
    print("-" * 40)
    print(f"  Total pathways tested:       {len(df)}")
    print(f"  Significant (FDR < 0.05):    {(df['fdr'] < 0.05).sum()}")
    print(f"  Significant (FDR < 0.01):    {(df['fdr'] < 0.01).sum()}")
    print(f"  Significant (FDR < 0.001):   {(df['fdr'] < 0.001).sum()}")
    
    sig_df = df[df['fdr'] < 0.05]
    
    if len(sig_df) > 0:
        # SHAP statistics
        print("\n📈 SHAP VALUE SUMMARY (Significant pathways):")
        print("-" * 40)
        print(f"  Total SHAP sum:    {sig_df['shap_sum'].sum():.4f}")
        print(f"  Mean SHAP per pathway: {sig_df['shap_sum'].mean():.4f}")
        print(f"  Median SHAP per pathway: {sig_df['shap_sum'].median():.4f}")
        print(f"  Max SHAP (single pathway): {sig_df['shap_sum'].max():.4f}")
        
        print("\n🏆 TOP PATHWAYS BY SHAP SUM:")
        print("-" * 40)
        top_shap = sig_df.nlargest(5, 'shap_sum')
        for _, row in top_shap.iterrows():
            print(f"  {row['pathway_name'][:40]:<40} SHAP={row['shap_sum']:.4f}")
        
        print("\n🏆 TOP PATHWAYS BY FOLD ENRICHMENT:")
        print("-" * 40)
        top_fe = sig_df.nlargest(5, 'fold_enrichment')
        for _, row in top_fe.iterrows():
            print(f"  {row['pathway_name'][:40]:<40} FE={row['fold_enrichment']:.2f}×")
        
        print("\n🏆 TOP PATHWAYS BY SIGNIFICANCE:")
        print("-" * 40)
        top_sig = sig_df.nsmallest(5, 'fdr')
        for _, row in top_sig.iterrows():
            print(f"  {row['pathway_name'][:40]:<40} FDR={row['fdr']:.2e}")
        
        # Correlation between SHAP and significance
        from scipy.stats import spearmanr
        corr, p = spearmanr(sig_df['shap_sum'], -np.log10(sig_df['fdr']))
        print(f"\n📊 SHAP vs Significance Correlation:")
        print("-" * 40)
        print(f"  Spearman correlation: {corr:.3f} (p={p:.3e})")


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def run_pathway_ora_with_shap(
    shap_file: str,
    pathway_file: str,
    msp_mapping_file: str = None,
    
    # Column indices
    shap_feature_col: int = 0,
    shap_value_col: int = 1,
    pathway_microbe_col: int = 0,
    pathway_pathway_col: int = 3,
    
    # Disease microbe selection
    selection_method: str = "threshold",
    shap_threshold: float = 0.0,
    top_n: int = None,
    top_percentile: float = None,
    
    # ORA parameters
    min_pathway_size: int = 2,
    min_disease_hits: int = 1,
    
    # Significance threshold
    fdr_threshold: float = 0.1,
    
    # Output
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Run complete ORA pipeline with SHAP value tracking.
    """
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(shap_file).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "=" * 70)
    print("PATHWAY OVER-REPRESENTATION ANALYSIS WITH SHAP VALUES")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 1: Loading Data")
    print("-" * 50)
    
    shap_df, shap_lookup = load_shap_data(
        shap_file=shap_file,
        feature_col_idx=shap_feature_col,
        shap_col_idx=shap_value_col,
    )
    
    pathway_df = load_microbe_pathway_mapping(
        pathway_file=pathway_file,
        microbe_col_idx=pathway_microbe_col,
        pathway_col_idx=pathway_pathway_col,
    )
    
    name_mapping = {}
    if msp_mapping_file:
        name_mapping = load_msp_to_name_mapping(msp_mapping_file)
    
    # -------------------------------------------------------------------------
    # Step 2: Select disease-associated microbes
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 2: Selecting Disease-Associated Microbes")
    print("-" * 50)
    
    disease_df, selection_desc = select_disease_microbes(
        shap_df=shap_df,
        method=selection_method,
        threshold=shap_threshold,
        top_n=top_n,
        top_percentile=top_percentile,
    )
    
    print(f"Selection method: {selection_desc}")
    print(f"Disease-associated microbes selected: {len(disease_df)}")
    print(f"Total SHAP of selected microbes: {disease_df['shap_value'].sum():.4f}")
    
    # Get microbe sets
    disease_microbes = set(disease_df["microbe_clean"].unique())
    background_microbes = set(pathway_df["microbe_clean"].unique())
    
    # Check overlap
    overlap = disease_microbes & background_microbes
    
    print(f"\nOverlap with pathway database:")
    print(f"  Disease microbes in pathway DB: {len(overlap)}")
    print(f"  Disease microbes NOT in pathway DB: {len(disease_microbes - background_microbes)}")
    
    if len(overlap) == 0:
        raise ValueError("No overlap between disease microbes and pathway database!")
    
    # -------------------------------------------------------------------------
    # Step 3: Perform ORA with SHAP
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 3: Over-Representation Analysis")
    print("-" * 50)
    
    results_df = perform_ora_with_shap(
        disease_microbes=overlap,
        background_microbes=background_microbes,
        microbe_pathway_df=pathway_df,
        shap_lookup=shap_lookup,
        name_mapping=name_mapping,
        min_pathway_size=min_pathway_size,
        min_disease_hits=min_disease_hits,
    )
    
    if results_df.empty:
        print("\nNo pathways to analyze!")
        return results_df
    
    # -------------------------------------------------------------------------
    # Step 4: Print summary
    # -------------------------------------------------------------------------
    print_summary_statistics_with_shap(results_df)
    
    # -------------------------------------------------------------------------
    # Step 5: Save results
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 4: Saving Results")
    print("-" * 50)
    
    # All results
    all_file = output_dir / "PATHWAY_ORA_RESULTS_ALL.xlsx"
    results_df.to_excel(all_file, index=False)
    print(f"All results: {all_file}")
    
    # Significant results
    sig_df = results_df[results_df["fdr"] < fdr_threshold].copy()
    sig_file = output_dir / f"PATHWAY_ORA_SIGNIFICANT_FDR{fdr_threshold}.xlsx"
    sig_df.to_excel(sig_file, index=False)
    print(f"Significant (FDR < {fdr_threshold}): {sig_file} ({len(sig_df)} pathways)")
    
    # Publication table
    pub_table = create_publication_table_with_shap(
        results_df, n_top=20,
        output_file=str(output_dir / "PATHWAY_ORA_PUBLICATION_TABLE.xlsx")
    )
    
    # Detailed table
    detailed_table = create_detailed_results_table(
        results_df, fdr_threshold=fdr_threshold,
        output_file=str(output_dir / "PATHWAY_ORA_DETAILED_RESULTS.xlsx")
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Create visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 5: Creating Visualizations")
    print("-" * 50)
    
    # Bar chart with SHAP
    plot_bar_chart_with_shap(
        results_df, n_top=20,
        output_file=str(output_dir / "ORA_barplot_significance_and_shap.png")
    )
    
    # SHAP vs Significance scatter
    plot_shap_vs_significance(
        results_df, fdr_threshold=fdr_threshold,
        output_file=str(output_dir / "ORA_shap_vs_significance.png")
    )
    
    # Dot plot with SHAP coloring
    plot_dot_plot_with_shap(
        results_df, n_top=25,
        output_file=str(output_dir / "ORA_dotplot_shap_colored.png")
    )
    
    # Top pathways by SHAP
    plot_top_pathways_by_shap(
        results_df, n_top=20, fdr_threshold=fdr_threshold,
        output_file=str(output_dir / "ORA_top_pathways_by_shap.png")
    )
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  - PATHWAY_ORA_RESULTS_ALL.xlsx")
    print(f"  - PATHWAY_ORA_SIGNIFICANT_FDR{fdr_threshold}.xlsx")
    print(f"  - PATHWAY_ORA_PUBLICATION_TABLE.xlsx")
    print(f"  - PATHWAY_ORA_DETAILED_RESULTS.xlsx")
    print(f"  - ORA_barplot_significance_and_shap.png")
    print(f"  - ORA_shap_vs_significance.png")
    print(f"  - ORA_dotplot_shap_colored.png")
    print(f"  - ORA_top_pathways_by_shap.png")
    
    return results_df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    
    # Input files
    SHAP_FILE = "ALL_SHAP_VALUES.csv"
    PATHWAY_FILE = "msp_kegg_pathways.xlsx"
    MSP_MAPPING_FILE = "mspmap.xlsx"  # Set to None if not available
    
    # Column indices (0-based)
    SHAP_FEATURE_COL = 0
    SHAP_VALUE_COL = 1
    PATHWAY_MICROBE_COL = 0
    PATHWAY_PATHWAY_COL = 3
    
    # Disease microbe selection
    SELECTION_METHOD = "threshold"  # "threshold", "top_n", "percentile"
    SHAP_THRESHOLD = 0.0
    TOP_N = None
    TOP_PERCENTILE = None
    
    # ORA parameters
    MIN_PATHWAY_SIZE = 1
    MIN_DISEASE_HITS = 1
    
    # Significance threshold
    FDR_THRESHOLD = 0.05
    
    # Output directory
    OUTPUT_DIR ="pathways"  # None = same as SHAP file
    
    # ==========================================================================
    # Run analysis
    # ==========================================================================
    
    results = run_pathway_ora_with_shap(
        shap_file=SHAP_FILE,
        pathway_file=PATHWAY_FILE,
        msp_mapping_file=MSP_MAPPING_FILE,
        
        shap_feature_col=SHAP_FEATURE_COL,
        shap_value_col=SHAP_VALUE_COL,
        pathway_microbe_col=PATHWAY_MICROBE_COL,
        pathway_pathway_col=PATHWAY_PATHWAY_COL,
        
        selection_method=SELECTION_METHOD,
        shap_threshold=SHAP_THRESHOLD,
        top_n=TOP_N,
        top_percentile=TOP_PERCENTILE,
        
        min_pathway_size=MIN_PATHWAY_SIZE,
        min_disease_hits=MIN_DISEASE_HITS,
        
        fdr_threshold=FDR_THRESHOLD,
        
        output_dir=OUTPUT_DIR,
    )