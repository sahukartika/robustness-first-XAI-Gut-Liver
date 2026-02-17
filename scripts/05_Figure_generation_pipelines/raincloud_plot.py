import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL FONT SETTINGS - MAXIMUM BOLD (SET BEFORE EVERYTHING)
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Black', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'black'
plt.rcParams['axes.labelweight'] = 'black'
plt.rcParams['axes.titleweight'] = 'black'
plt.rcParams['figure.titleweight'] = 'black'

# =============================================================================
# >>>>>>>>>>>>>>>>  FONT & OUTPUT CONTROL  <<<<<<<<<<<<<<<<
# =============================================================================
FONT_WEIGHT = 900              # 700=bold, 900=black (MAXIMUM)
TITLE_SIZE = 36                # Match reference
AXIS_LABEL_SIZE = 26           # Match reference
TICK_LABEL_SIZE = 26           # Match reference
OUTPUT_DPI = 1200
FIGSIZE = (14, 10)             # Slightly larger for bigger fonts
# =============================================================================


# =============================================================================
# 1) LOAD DATA (MATCHES YOUR CSV FORMAT)
# =============================================================================

def load_results(results_dict: dict) -> dict:
    data = {}
    print("\nLoading data files...")
    for task, path in results_dict.items():
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)

            # keep only successful runs
            if 'success' in df.columns:
                df = df[df['success'].astype(str).str.upper() == 'TRUE']

            data[task] = df
            print(f"  ✓ {task}: {len(df):,} rows")
        else:
            print(f"  ✗ Missing: {path}")
    return data


# =============================================================================
# 2) LONG FORMAT DATAFRAME FOR ALL LEVELS
# =============================================================================

def prepare_long_dataframe(data: dict, metric: str) -> pd.DataFrame:

    records = []

    for task, df in data.items():
        for _, r in df.iterrows():

            records.append({
                "Task": task,
                "Scaler": str(r.get("scaling", "None")),
                "FeatureSelector": str(r.get("feature_selection", "None")),
                "Balancer": str(r.get("balancing", "None")),
                "Model": str(r.get("model_name", "Unknown")),
                "AUC": float(r[metric])
            })

    comp_df = pd.DataFrame(records)

    comp_df = comp_df.replace({"nan": "None", "": "None", np.nan: "None"})

    print("\nFinal long dataframe shape:", comp_df.shape)
    return comp_df


# =============================================================================
# 3) UNIVERSAL HALF-VIOLIN RAINCLOUD FUNCTION (WITH MAXIMUM BOLD)
# =============================================================================

def half_violin_raincloud(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    out_path: Path,
    figsize=FIGSIZE,
    dpi=OUTPUT_DPI,
    # ===== FONT SETTINGS =====
    title_size=TITLE_SIZE,
    axis_label_size=AXIS_LABEL_SIZE,
    tick_label_size=TICK_LABEL_SIZE,
    font_weight=FONT_WEIGHT
):

    df[group_col] = df[group_col].astype(str)
    levels = sorted(df[group_col].unique())
    positions = np.arange(len(levels))

    # ========================================
    # RE-APPLY FONT SETTINGS (SAFETY)
    # ========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial Black', 'Arial', 'Helvetica']
    plt.rcParams['font.weight'] = 'black'
    plt.rcParams['axes.labelweight'] = 'black'
    plt.rcParams['axes.titleweight'] = 'black'

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.tab10

    for i, lvl in enumerate(levels):

        data = df[df[group_col] == lvl][value_col].dropna().values
        if len(data) == 0:
            continue

        color = cmap(i / max(1, len(levels)-1))

        # ---- FULL VIOLIN FIRST ----
        parts = ax.violinplot(
            [data],
            positions=[i],
            widths=0.6,
            showmeans=False,
            showextrema=False,
            showmedians=False
        )

        # ---- CUT TO TRUE HALF VIOLIN ----
        for pc in parts['bodies']:
            verts = pc.get_paths()[0].vertices
            mid = np.mean(verts[:, 0])
            verts[verts[:, 0] < mid, 0] = mid
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # ---- BOX (same colour, BLACK median) ----
        bp = ax.boxplot(
            [data],
            positions=[i - 0.15],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2.5),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5)
        )

        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
            patch.set_linewidth(1.5)

        # ---- BLACK MEAN DOT ----
        ax.scatter(
            i, np.mean(data),
            color='black',
            s=100,
            zorder=5
        )

        # ---- JITTERED POINTS ----
        jitter = np.random.uniform(-0.08, 0.08, len(data))
        ax.scatter(
            np.full_like(data, i - 0.3) + jitter,
            data,
            alpha=0.35,
            s=12,
            color='black'
        )

    # ========================================
    # X-AXIS TICKS - MAXIMUM BOLD
    # ========================================
    ax.set_xticks(positions)
    ax.set_xticklabels(levels, rotation=45, ha='right')
    
    # Explicit loop for X-tick labels
    for label in ax.get_xticklabels():
        label.set_fontsize(tick_label_size)
        label.set_fontweight(font_weight)
        label.set_fontfamily('Arial')

    # ========================================
    # Y-AXIS TICKS - MAXIMUM BOLD
    # ========================================
    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size)
        label.set_fontweight(font_weight)
        label.set_fontfamily('Arial')

    # ========================================
    # AXIS LABELS - MAXIMUM BOLD
    # ========================================
    ax.set_ylabel(
        "AUC", 
        fontsize=axis_label_size, 
        fontweight=font_weight,
        fontfamily='Arial'
    )
    
    ax.set_xlabel(
        group_col, 
        fontsize=axis_label_size, 
        fontweight=font_weight,
        fontfamily='Arial'
    )

    ax.set_ylim(0, 1.05)
    
    # Reference line
    ax.axhline(0.5, linestyle='--', color='gray', alpha=0.5, linewidth=2)

    # ========================================
    # TITLE - MAXIMUM BOLD
    # ========================================
    ax.set_title(
        title, 
        fontsize=title_size, 
        fontweight=font_weight,
        fontfamily='Arial'
    )

    # ========================================
    # THICKER SPINES
    # ========================================
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✓ Saved: {out_path}")


# =============================================================================
# 4) MAIN PIPELINE — GENERATES ALL PLOTS YOU ASKED FOR
# =============================================================================

if __name__ == "__main__":

    METRIC = "mean_auc"
    OUTPUT_DIR = Path("Raincloud plot")

    RESULTS = {
        'CRC_microbiome': r'microbiome_results_CRC\phylum_abundance_PRJEB10878\D5_G2_S3\results.csv',
        'CRC_metabolomics': r'CRC_Metabolomics_results\all_results.csv',
        'LC_microbiome': r'microbiome_results_LC\phylum_abundance_PRJEB6337\D9_G2_S3\results.csv',
        'LC_metabolomics': r'LC_Metabolomics_results\all_results.csv',
        'CD_microbiome': r'microbiome_results_CD\phylum_abundance_PRJEB15371\D4_G2_S3\results.csv',
        'CD_metabolomics': r'CD_metabolomics_results\all_results.csv',
    }

    print("\n===== LOADING DATA =====")
    data = load_results(RESULTS)

    print("\n===== BUILDING LONG DATAFRAME =====")
    comp_df = prepare_long_dataframe(data, METRIC)

    print("\n===== GENERATING HALF-VIOLIN PLOTS =====")

    # ---- PREPROCESSING LEVEL ----
    half_violin_raincloud(
        comp_df, "Scaler", "AUC",
        "AUC Across Scalers",
        OUTPUT_DIR / "raincloud_scaler.tiff"
    )

    half_violin_raincloud(
        comp_df, "FeatureSelector", "AUC",
        "AUC Across Feature Selection Methods",
        OUTPUT_DIR / "raincloud_feature_selection.tiff"
    )

    half_violin_raincloud(
        comp_df, "Balancer", "AUC",
        "AUC Across Balancing Methods",
        OUTPUT_DIR / "raincloud_balancer.tiff"
    )

    # ---- MODEL LEVEL ----
    half_violin_raincloud(
        comp_df, "Model", "AUC",
        "AUC Across Models",
        OUTPUT_DIR / "raincloud_model.tiff"
    )

    # ---- TASK LEVEL (YOUR REQUEST) ----
    half_violin_raincloud(
        comp_df, "Task", "AUC",
        "AUC Across Tasks",
        OUTPUT_DIR / "raincloud_tasks.tiff"
    )

    print("\n✅ ALL HALF-VIOLIN TIFF PLOTS SAVED IN:", OUTPUT_DIR.absolute())