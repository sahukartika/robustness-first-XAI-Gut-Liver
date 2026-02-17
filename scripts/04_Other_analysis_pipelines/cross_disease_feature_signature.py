"""
compare_nonzero_shap_features_across_diseases.py

Feature-level comparison across diseases, SEPARATELY for:
  - Microbiome (microbial features)
  - Metabolomics (metabolite features)

UPDATED: Now saves SHAP values for each disease when saving common features.

IMPORTANT:
  - This script DOES NOT try to find common features between microbes and metabolites
    within a disease. It only compares features ACROSS DISEASES within each omics.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import from_contents, UpSet


# ==========================================================
# Feature cleaning
# ==========================================================

def clean_micro_feature(raw: str) -> str:
    """
    Clean a MICROBIOME feature name:
      - strip whitespace
      - lowercase
    """
    return str(raw).strip().lower()


def clean_meta_feature(raw: str) -> str:
    """
    Clean a METABOLOMICS feature name:
      - strip whitespace
      - remove LC-MS prefixes (fwd_pos_, fwd_neg_, rev_pos_, rev_neg_, etc.)
      - remove suffix tokens like '_pos', '_neg', ' pos', ' neg', etc.
      - lowercase
    """
    s = str(raw).strip()
    if not s:
        return ""

    s_lower = s.lower()

    # Remove typical LC-MS prefixes
    prefix_list = [
        "fwd_pos_", "fwd_neg_", "rev_pos_", "rev_neg_",
        "fwd_", "rev_", "pos_", "neg_"
    ]
    for p in prefix_list:
        if s_lower.startswith(p):
            s = s[len(p):]
            s_lower = s.lower()
            break

    # Remove suffix tokens at the end (underscore-separated)
    suffix_tokens = {
        "pos", "neg", "positive", "negative", "rev", "reverse",
        "fwd", "forward", "esi+", "esi-", "mode", "ion"
    }

    parts = s.split("_")
    if len(parts) > 1 and parts[-1].lower() in suffix_tokens:
        s = "_".join(parts[:-1])

    # Space-separated suffix
    parts_space = s.split(" ")
    if len(parts_space) > 1 and parts_space[-1].lower() in suffix_tokens:
        s = " ".join(parts_space[:-1])

    return s.strip().lower()


# ==========================================================
# Load feature sets from SHAP files (with SHAP values)
# ==========================================================

def load_feature_dict_from_shap(
    shap_file: str,
    feature_col: str = "feature",
    shap_col: str = "mean_abs_shap",
    shap_threshold: float = 0.0,
    omics_type: str = "micro",  # "micro" or "meta"
) -> Tuple[Set[str], Dict[str, float], Dict[str, str]]:
    """
    Load features with SHAP > shap_threshold from a SHAP file
    (CSV or Excel), clean them, and return:
      - A set of cleaned feature IDs
      - A dict mapping cleaned feature -> SHAP value
      - A dict mapping cleaned feature -> original feature name
    """
    path = Path(shap_file)
    if not path.exists():
        raise FileNotFoundError(f"SHAP file not found: {shap_file}")

    print(f"\nLoading SHAP features from: {shap_file} (omics_type={omics_type})")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if feature_col not in df.columns or shap_col not in df.columns:
        raise ValueError(
            f"SHAP file must contain columns '{feature_col}' and '{shap_col}'. "
            f"Columns found: {list(df.columns)}"
        )

    print(f"  Shape: {df.shape}")
    df = df[[feature_col, shap_col]].copy()
    df[shap_col] = pd.to_numeric(df[shap_col], errors="coerce")
    df = df[df[shap_col] > shap_threshold]

    # Clean feature names
    if omics_type == "meta":
        df["feature_clean"] = df[feature_col].astype(str).apply(clean_meta_feature)
    else:
        df["feature_clean"] = df[feature_col].astype(str).apply(clean_micro_feature)

    df = df[df["feature_clean"] != ""]

    # If duplicate cleaned features, take the max SHAP value
    if len(df) > 0:
        df_grouped = df.groupby("feature_clean").agg({
            shap_col: "max",
            feature_col: "first"  # Keep first original name
        }).reset_index()

        feature_set = set(df_grouped["feature_clean"].tolist())
        shap_dict = dict(zip(df_grouped["feature_clean"], df_grouped[shap_col]))
        original_name_dict = dict(zip(df_grouped["feature_clean"], df_grouped[feature_col]))
    else:
        feature_set = set()
        shap_dict = {}
        original_name_dict = {}

    print(f"  Non-zero SHAP features (after cleaning): {len(feature_set)}")
    return feature_set, shap_dict, original_name_dict


# ==========================================================
# Plot helpers
# ==========================================================

def plot_venn2(
    set1: Set[str],
    set2: Set[str],
    labels: List[str],
    title: str,
    out_file: Path,
):
    plt.figure(figsize=(6, 6))
    v = venn2([set1, set2], set_labels=labels)

    # Increase font sizes
    if v.set_labels:
        for text in v.set_labels:
            if text:
                text.set_fontsize(16)
    if v.subset_labels:
        for text in v.subset_labels:
            if text:
                text.set_fontsize(14)

    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved Venn2 to: {out_file}")

def plot_venn3(
    setA: Set[str],
    setB: Set[str],
    setC: Set[str],
    labels: List[str],
    title: str,
    out_file: Path,
):
    plt.figure(figsize=(7, 7))
    v = venn3([setA, setB, setC], set_labels=labels)

    # Increase font sizes
    if v.set_labels:
        for text in v.set_labels:
            if text:
                text.set_fontsize(16)
    if v.subset_labels:
        for text in v.subset_labels:
            if text:
                text.set_fontsize(14)

    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved Venn3 to: {out_file}")


def plot_upset(
    set_dict: Dict[str, Set[str]],
    title: str,
    out_file: Path,
):
    # Filter out empty sets
    contents = {k: v for k, v in set_dict.items() if len(v) > 0}
    
    if len(contents) < 2:
        print(f"  Skipping UpSet plot (need at least 2 non-empty sets): {out_file}")
        return
        
    upset_data = from_contents(contents)

    plt.figure(figsize=(8, 6))
    UpSet(upset_data, show_counts=True).plot()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved UpSet plot to: {out_file}")


# ==========================================================
# Helper to create DataFrame with SHAP values from multiple diseases
# ==========================================================

def create_common_features_df(
    common_features: Set[str],
    disease_shap_dicts: Dict[str, Dict[str, float]],
    disease_original_dicts: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """
    Create a DataFrame with common features and their SHAP values from each disease.
    
    Args:
        common_features: Set of common feature names (cleaned)
        disease_shap_dicts: Dict mapping disease -> {feature_clean -> shap_value}
        disease_original_dicts: Dict mapping disease -> {feature_clean -> original_name}
    
    Returns:
        DataFrame with columns: feature, original_name_*, shap_*, mean_shap_across_diseases
    """
    diseases = list(disease_shap_dicts.keys())
    
    # Handle empty common features
    if len(common_features) == 0:
        # Return empty DataFrame with expected columns
        cols = ["feature"]
        cols += [f"original_name_{dis}" for dis in diseases]
        cols += [f"shap_{dis}" for dis in diseases]
        cols.append("mean_shap_across_diseases")
        return pd.DataFrame(columns=cols)
    
    rows = []
    for feat in sorted(common_features):
        row = {"feature": feat}
        
        shap_values = []
        for dis in diseases:
            # Get SHAP value
            shap_val = disease_shap_dicts[dis].get(feat, np.nan)
            row[f"shap_{dis}"] = shap_val
            if not np.isnan(shap_val):
                shap_values.append(shap_val)
            
            # Get original feature name
            orig_name = disease_original_dicts[dis].get(feat, "")
            row[f"original_name_{dis}"] = orig_name
        
        # Calculate mean SHAP
        row["mean_shap_across_diseases"] = np.mean(shap_values) if shap_values else np.nan
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns: feature, original names, SHAP values, mean
    col_order = ["feature"]
    col_order += [f"original_name_{dis}" for dis in diseases]
    col_order += [f"shap_{dis}" for dis in diseases]
    col_order.append("mean_shap_across_diseases")
    
    df = df[col_order]
    df = df.sort_values("mean_shap_across_diseases", ascending=False)
    
    return df


def create_single_disease_features_df(
    features: Set[str],
    shap_dict: Dict[str, float],
    original_dict: Dict[str, str],
    disease_name: str,
) -> pd.DataFrame:
    """
    Create a DataFrame with features and their SHAP values for a single disease.
    """
    if len(features) == 0:
        return pd.DataFrame(columns=[
            "feature", 
            f"original_name_{disease_name}", 
            f"shap_{disease_name}"
        ])
    
    rows = []
    for feat in sorted(features):
        row = {
            "feature": feat,
            f"original_name_{disease_name}": original_dict.get(feat, ""),
            f"shap_{disease_name}": shap_dict.get(feat, np.nan),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(f"shap_{disease_name}", ascending=False)
    return df


# ==========================================================
# Across-disease analysis (for a given omics level)
# ==========================================================

def across_disease_analysis(
    level_sets: Dict[str, Set[str]],
    level_shap_dicts: Dict[str, Dict[str, float]],
    level_original_dicts: Dict[str, Dict[str, str]],
    level_name: str,            # "micro" or "meta"
    output_root_path: Path,
):
    """
    level_sets : dict
        Keys = disease names; values = sets of features at this omics level.
    level_shap_dicts : dict
        Keys = disease names; values = dict mapping feature -> SHAP value.
    level_original_dicts : dict
        Keys = disease names; values = dict mapping feature -> original name.
    level_name : str
        'micro' or 'meta'.
    """
    diseases = sorted(level_sets.keys())
    if not diseases:
        return

    level_dir = output_root_path / f"across_{level_name}_features"
    level_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n=== Across-diseases feature analysis: {level_name} level ===")
    
    # Save per-disease feature sets with SHAP values
    for dis in diseases:
        print(f"  {dis}: {len(level_sets[dis])} {level_name}-level features")
        
        df_single = create_single_disease_features_df(
            features=level_sets[dis],
            shap_dict=level_shap_dicts[dis],
            original_dict=level_original_dicts[dis],
            disease_name=dis,
        )
        df_single.to_csv(
            level_dir / f"{dis}_{level_name}_features_with_shap.csv",
            index=False
        )

    # UpSet across all diseases
    if len(diseases) >= 2:
        upset_file = level_dir / f"upset_{level_name}_features_all_diseases.png"
        plot_upset(
            set_dict=level_sets,
            title=f"{level_name.capitalize()} features across diseases",
            out_file=upset_file,
        )

    # If exactly 3 diseases: 3-way Venn + triple intersection with SHAP values
    if len(diseases) == 3:
        d1, d2, d3 = diseases
        set1, set2, set3 = level_sets[d1], level_sets[d2], level_sets[d3]

        venn3_file = level_dir / f"venn3_{level_name}_features_{d1}_{d2}_{d3}.png"
        plot_venn3(
            setA=set1, setB=set2, setC=set3,
            labels=[d1, d2, d3],
            title=f"{level_name.capitalize()} features across {d1}, {d2}, {d3}",
            out_file=venn3_file,
        )

        triple_intersection = set1 & set2 & set3
        
        # Create DataFrame with SHAP values from all three diseases
        df_triple = create_common_features_df(
            common_features=triple_intersection,
            disease_shap_dicts={d: level_shap_dicts[d] for d in [d1, d2, d3]},
            disease_original_dicts={d: level_original_dicts[d] for d in [d1, d2, d3]},
        )
        df_triple.to_csv(
            level_dir / f"triple_common_{level_name}_features_{d1}_{d2}_{d3}_with_shap.csv",
            index=False
        )
        
        print(f"  Triple intersection ({d1} ∩ {d2} ∩ {d3}) at {level_name} level: "
              f"{len(triple_intersection)} features")

    # Pairwise intersections & Venns with SHAP values
    if len(diseases) >= 2:
        for i in range(len(diseases)):
            for j in range(i + 1, len(diseases)):
                a, b = diseases[i], diseases[j]
                set_a, set_b = level_sets[a], level_sets[b]
                inter_ab = set_a & set_b

                print(f"  Pair ({a}, {b}) at {level_name} level:")
                print(f"    |{a}| = {len(set_a)}, |{b}| = {len(set_b)}, "
                      f"|{a} ∩ {b}| = {len(inter_ab)}")

                # Create DataFrame with SHAP values from both diseases
                df_pair = create_common_features_df(
                    common_features=inter_ab,
                    disease_shap_dicts={d: level_shap_dicts[d] for d in [a, b]},
                    disease_original_dicts={d: level_original_dicts[d] for d in [a, b]},
                )
                df_pair.to_csv(
                    level_dir / f"pair_common_{level_name}_features_{a}_{b}_with_shap.csv",
                    index=False
                )

                venn2_file = level_dir / f"venn2_{level_name}_features_{a}_{b}.png"
                plot_venn2(
                    set1=set_a,
                    set2=set_b,
                    labels=[a, b],
                    title=f"{level_name.capitalize()} features across {a} & {b}",
                    out_file=venn2_file,
                )

    # Save all features across all diseases with their SHAP values (union)
    all_features_union = set()
    for dis in diseases:
        all_features_union.update(level_sets[dis])
    
    if len(all_features_union) == 0:
        print(f"  No features found for {level_name} level across any disease.")
        return
    
    # Create comprehensive DataFrame with all features and their SHAP values
    rows_all = []
    for feat in sorted(all_features_union):
        row = {"feature": feat}
        
        # Track which diseases have this feature
        present_in = []
        shap_values_present = []
        
        for dis in diseases:
            if feat in level_sets[dis]:
                present_in.append(dis)
                shap_val = level_shap_dicts[dis].get(feat, np.nan)
                row[f"shap_{dis}"] = shap_val
                row[f"original_name_{dis}"] = level_original_dicts[dis].get(feat, "")
                if not np.isnan(shap_val):
                    shap_values_present.append(shap_val)
            else:
                row[f"shap_{dis}"] = np.nan
                row[f"original_name_{dis}"] = ""
        
        row["present_in_diseases"] = ";".join(present_in)
        row["num_diseases"] = len(present_in)
        row["mean_shap_across_present_diseases"] = (
            np.mean(shap_values_present) if shap_values_present else np.nan
        )
        rows_all.append(row)
    
    df_all = pd.DataFrame(rows_all)
    
    # Reorder columns
    col_order = ["feature", "present_in_diseases", "num_diseases"]
    col_order += [f"original_name_{dis}" for dis in diseases]
    col_order += [f"shap_{dis}" for dis in diseases]
    col_order.append("mean_shap_across_present_diseases")
    df_all = df_all[col_order]
    
    df_all = df_all.sort_values(
        ["num_diseases", "mean_shap_across_present_diseases"],
        ascending=[False, False]
    )
    
    df_all.to_csv(
        level_dir / f"all_{level_name}_features_union_with_shap.csv",
        index=False
    )
    print(f"  Saved comprehensive union table with {len(df_all)} features")


# ==========================================================
# Main comparison logic
# ==========================================================

def compare_features(
    disease_config: Dict[str, Dict[str, str]],
    output_root: str = "feature_comparison_shap_results",
    feature_col: str = "feature",
    shap_col: str = "mean_abs_shap",
    shap_threshold: float = 0.0,
):
    """
    disease_config : dict
        Example:
          {
            "CD": {
              "micro": "shap_CD_microbiome/ALL_SHAP_VALUES.csv",
              "meta":  "shap_analysis_CD_metabolomics_single_config/ALL_SHAP_VALUES.csv",
            },
            ...
          }
    """
    output_root_path = Path(output_root)
    output_root_path.mkdir(exist_ok=True, parents=True)

    diseases = list(disease_config.keys())
    print(f"Diseases: {diseases}")

    # Store sets, SHAP dicts, and original name dicts for each disease/omics
    micro_sets: Dict[str, Set[str]] = {}
    micro_shap_dicts: Dict[str, Dict[str, float]] = {}
    micro_original_dicts: Dict[str, Dict[str, str]] = {}
    
    meta_sets: Dict[str, Set[str]] = {}
    meta_shap_dicts: Dict[str, Dict[str, float]] = {}
    meta_original_dicts: Dict[str, Dict[str, str]] = {}

    # Per-disease: record micro & meta feature sets with SHAP values
    for dis in diseases:
        print(f"\n=== Processing disease: {dis} (feature-level) ===")
        out_dir_dis = output_root_path / dis
        out_dir_dis.mkdir(exist_ok=True, parents=True)

        micro_file = disease_config[dis]["micro"]
        meta_file = disease_config[dis]["meta"]

        # Load microbiome features with SHAP values
        micro_set, micro_shap, micro_orig = load_feature_dict_from_shap(
            micro_file, feature_col=feature_col, shap_col=shap_col,
            shap_threshold=shap_threshold, omics_type="micro"
        )
        
        # Load metabolomics features with SHAP values
        meta_set, meta_shap, meta_orig = load_feature_dict_from_shap(
            meta_file, feature_col=feature_col, shap_col=shap_col,
            shap_threshold=shap_threshold, omics_type="meta"
        )

        micro_sets[dis] = micro_set
        micro_shap_dicts[dis] = micro_shap
        micro_original_dicts[dis] = micro_orig
        
        meta_sets[dis] = meta_set
        meta_shap_dicts[dis] = meta_shap
        meta_original_dicts[dis] = meta_orig

        # Save per-disease feature sets with SHAP values
        df_micro = create_single_disease_features_df(
            features=micro_set,
            shap_dict=micro_shap,
            original_dict=micro_orig,
            disease_name=dis,
        )
        df_micro.to_csv(out_dir_dis / "micro_features_with_shap.csv", index=False)
        
        df_meta = create_single_disease_features_df(
            features=meta_set,
            shap_dict=meta_shap,
            original_dict=meta_orig,
            disease_name=dis,
        )
        df_meta.to_csv(out_dir_dis / "meta_features_with_shap.csv", index=False)

        print(f"  {dis}:")
        print(f"    Microbiome features (non-zero SHAP, cleaned):  {len(micro_set)}")
        print(f"    Metabolome features (non-zero SHAP, cleaned):  {len(meta_set)}")

    # Across-disease analyses (MICROBIOME)
    across_disease_analysis(
        level_sets=micro_sets,
        level_shap_dicts=micro_shap_dicts,
        level_original_dicts=micro_original_dicts,
        level_name="micro",
        output_root_path=output_root_path,
    )

    # Across-disease analyses (METABOLOMICS)
    across_disease_analysis(
        level_sets=meta_sets,
        level_shap_dicts=meta_shap_dicts,
        level_original_dicts=meta_original_dicts,
        level_name="meta",
        output_root_path=output_root_path,
    )

    print(f"\nAll feature-level results saved under: {output_root_path}")


# ==========================================================
# MAIN (use your paths)
# ==========================================================

if __name__ == "__main__":
    DISEASE_CONFIG = {
        "CD": {
            "micro": "shap_CD_microbiome4/ALL_SHAP_VALUES.csv",
            "meta":  "shap_analysis_CD_metabolomics4/ALL_SHAP_VALUES.csv",
        },
        "LC": {
            "micro": "shap_LC_microbiome4/ALL_SHAP_VALUES.csv",
            "meta":  "shap_analysis_LC_metabolomics4/ALL_SHAP_VALUES.csv",
        },
        "CRC": {
            "micro": "shap_CRC_microbiome4/ALL_SHAP_VALUES.csv",
            "meta":  "shap_analysis_CRC_metabolomics4/ALL_SHAP_VALUES.csv",
        },
    }

    OUTPUT_ROOT = "feature_comparison_shap_results"

    FEATURE_COL = "feature"
    SHAP_COL = "mean_abs_shap"
    SHAP_THRESHOLD = 0.0

    compare_features(
        disease_config=DISEASE_CONFIG,
        output_root=OUTPUT_ROOT,
        feature_col=FEATURE_COL,
        shap_col=SHAP_COL,
        shap_threshold=SHAP_THRESHOLD,
    )