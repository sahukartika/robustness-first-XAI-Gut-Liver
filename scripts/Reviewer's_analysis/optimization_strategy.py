"""
compare_optimization_strategies.py  (Reviewer 1 R1.3 | Reviewer 2 R2.1)

Compares pipeline-selection strategies on the same 1,152-config benchmark.
For each (model, scaling, feature_selection, balancing) configuration,
the best AUC across the three feature-retention percentages (10/30/50%)
is taken per task — identical to the original maximin selection procedure.

Strategies compared:
  1. Maximin               : maximise min AUC across all tasks   [paper approach]
  2. Mean-AUC              : maximise mean AUC across all tasks
  3. Median-AUC            : maximise median AUC across all tasks
  4. Pareto-Optimal        : minimise max regret + mean regret jointly (Pareto front)
  5. Per-task best         : oracle upper bound (not a single pipeline)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
import openpyxl

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS: Dict[str, str] = {
    "CRC_microbiome":   r"E:\NAFLD\minimize nafld\microbiome_results_with_externalCRC\phylum_abundance_PRJEB10878\D5_G2_S3\results.csv",
    "CRC_metabolomics": r"E:\NAFLD\minimize nafld\CRC_Metabolomics_results3_20251203_001038\all_results.csv",
    "LC_microbiome":    r"E:\NAFLD\minimize nafld\microbiome_results_with_externalLC\phylum_abundance_PRJEB6337\D9_G2_S3\results.csv",
    "LC_metabolomics":  r"E:\NAFLD\minimize nafld\LC_Metabolomics_LiverCirrhosis_vs_Healthy_20251202_130555\all_results.csv",
    "CD_microbiome":    r"E:\NAFLD\minimize nafld\microbiome_results_with_externalCD\phylum_abundance_PRJEB15371\D4_G2_S3\results.csv",
    "CD_metabolomics":  r"E:\NAFLD\minimize nafld\CD_metabolomics_results3lightgbm_20251226_110833\all_results.csv",
}

METRIC  = "mean_auc"
MODELS  = ["RandomForest", "XGBoost", "LightGBM",
           "LogisticRegression", "SVM_RBF", "MLP"]

OUTPUT_DIR = Path("optimization_strategy_comparison")

# ============================================================================
# HELPER: compute all performance metrics for a selected config row
# ============================================================================

def config_metrics(row: pd.Series, tasks: List[str]) -> Dict[str, float]:
    """
    Given a selected-config row (pd.Series) and the task list,
    return a dict with min / mean / median / max / std / range AUC
    computed from that config's per-task AUC values.
    """
    vals = np.array([float(row[t]) for t in tasks if t in row.index])
    return {
        "min_AUC":    float(np.nanmin(vals)),
        "mean_AUC":   float(np.nanmean(vals)),
        "median_AUC": float(np.nanmedian(vals)),
        "max_AUC":    float(np.nanmax(vals)),
        "std_AUC":    float(np.nanstd(vals)),
        "range_AUC":  float(np.nanmax(vals) - np.nanmin(vals)),
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(results_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    print("\nLoading results files …")
    for task, path in results_dict.items():
        p = Path(path)
        if not p.exists():
            print(f"  ✗ MISSING: {path}")
            continue
        df = pd.read_csv(p)
        if "success" in df.columns:
            df = df[df["success"] == True].copy()
        if "model_name" in df.columns and "model" not in df.columns:
            df = df.rename(columns={"model_name": "model"})
        data[task] = df
        print(f"  ✓ {task}: {len(df):,} rows")
    return data


# ============================================================================
# COLLAPSE ACROSS N_FEATURES
# ============================================================================

def build_collapsed_table(
    data: Dict[str, pd.DataFrame],
    metric: str,
    models: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For each (model, scaling, feature_selection, balancing) config and each
    task, take max(metric) across n_features retention variants (10/30/50%).
    """
    tasks = list(data.keys())
    config_task: Dict[Tuple, Dict[str, float]] = {}

    for task, df in data.items():
        for model in models:
            mdf = df[df["model"] == model]
            if mdf.empty:
                continue
            for (sc, fs, bal), grp in mdf.groupby(
                ["scaling", "feature_selection", "balancing"]
            ):
                key = (str(model), str(sc), str(fs), str(bal))
                best_auc = float(grp[metric].max())
                config_task.setdefault(key, {})[task] = best_auc

    records = []
    n_excluded = 0
    for (model, sc, fs, bal), task_vals in config_task.items():
        if len(task_vals) < len(tasks):
            n_excluded += 1
            continue
        row = {
            "model":             model,
            "scaling":           sc,
            "feature_selection": fs,
            "balancing":         bal,
        }
        row.update(task_vals)
        records.append(row)

    df_wide = pd.DataFrame(records)
    print(f"\n  Configs present in all {len(tasks)} tasks : {len(df_wide):,}")
    print(f"  Configs excluded (missing tasks)          : {n_excluded:,}")
    return df_wide, tasks


# ============================================================================
# STRATEGIES
# ============================================================================

def select_maximin(df_wide: pd.DataFrame, tasks: List[str]) -> pd.Series:
    """c* = argmax_c  min_t AUC(c,t).  Tiebreak: highest max AUC."""
    df = df_wide.copy()
    df["_min"] = df[tasks].min(axis=1)
    df["_max"] = df[tasks].max(axis=1)
    df = df.sort_values(["_min", "_max"], ascending=[False, False])
    return df.iloc[0]


def select_mean(df_wide: pd.DataFrame, tasks: List[str]) -> pd.Series:
    """c* = argmax_c  mean_t AUC(c,t)."""
    df = df_wide.copy()
    df["_mean"] = df[tasks].mean(axis=1)
    return df.loc[df["_mean"].idxmax()]


def select_median(df_wide: pd.DataFrame, tasks: List[str]) -> pd.Series:
    """
    c* = argmax_c  median_t AUC(c,t).
    Tiebreak: highest mean AUC among configs tied on median.
    """
    df = df_wide.copy()
    df["_median"] = df[tasks].median(axis=1)
    df["_mean"]   = df[tasks].mean(axis=1)
    df = df.sort_values(["_median", "_mean"], ascending=[False, False])
    return df.iloc[0]


def select_pareto_optimal(df_wide: pd.DataFrame, tasks: List[str]) -> pd.Series:
    """
    Multi-objective (Pareto-optimal) selection.
    Objectives (both to be minimised):
      (A) max_regret  = max_t  [ best_AUC(t) - AUC(c,t) ]
      (B) mean_regret = mean_t [ best_AUC(t) - AUC(c,t) ]
    """
    df = df_wide.copy()
    task_best = {t: float(df[t].max()) for t in tasks}

    regret_cols = []
    for t in tasks:
        col = f"_regret_{t}"
        df[col] = task_best[t] - df[t]
        regret_cols.append(col)

    df["_max_regret"]  = df[regret_cols].max(axis=1)
    df["_mean_regret"] = df[regret_cols].mean(axis=1)

    max_r  = df["_max_regret"].values
    mean_r = df["_mean_regret"].values
    n      = len(df)

    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (max_r[j] <= max_r[i] and mean_r[j] <= mean_r[i] and
                    (max_r[j] < max_r[i] or mean_r[j] < mean_r[i])):
                is_dominated[i] = True
                break

    pareto_df = df[~is_dominated].copy()
    pareto_df = pareto_df.sort_values(
        ["_max_regret", "_mean_regret"], ascending=[True, True]
    )
    return pareto_df.iloc[0]


def select_per_task_best(df_wide: pd.DataFrame, tasks: List[str]) -> Dict[str, float]:
    """Oracle: independently picks the best config per task — not a single pipeline."""
    return {t: float(df_wide[t].max()) for t in tasks}


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def build_strategy_table(
    strategies: Dict[str, Dict[str, float]],
    tasks: List[str],
) -> pd.DataFrame:
    """
    Rows = strategies.
    Columns = per-task AUC values + min / mean / median / max / std / range.
    """
    records = []
    for strategy, task_aucs in strategies.items():
        vals = [task_aucs.get(t, np.nan) for t in tasks]
        row  = {"strategy": strategy}
        row.update(dict(zip(tasks, vals)))
        row["min_AUC"]    = float(np.nanmin(vals))
        row["mean_AUC"]   = float(np.nanmean(vals))
        row["median_AUC"] = float(np.nanmedian(vals))
        row["max_AUC"]    = float(np.nanmax(vals))
        row["std_AUC"]    = float(np.nanstd(vals))
        row["range_AUC"]  = float(np.nanmax(vals) - np.nanmin(vals))
        records.append(row)
    return pd.DataFrame(records)


def build_config_comparison(
    named_rows: Dict[str, Optional[pd.Series]],
    tasks: List[str],
) -> pd.DataFrame:
    """
    One row per strategy: which pipeline was selected and its full
    performance profile (min / mean / median / max / std AUC).
    """
    records = []
    for strategy, row in named_rows.items():
        if row is None:
            records.append({
                "strategy":          strategy,
                "model":             "N/A",
                "scaling":           "N/A",
                "feature_selection": "N/A",
                "balancing":         "N/A",
                "min_AUC":           np.nan,
                "mean_AUC":          np.nan,
                "median_AUC":        np.nan,
                "max_AUC":           np.nan,
                "std_AUC":           np.nan,
                "range_AUC":         np.nan,
            })
            continue

        m = config_metrics(row, tasks)
        records.append({
            "strategy":          strategy,
            "model":             row.get("model", ""),
            "scaling":           row.get("scaling", ""),
            "feature_selection": row.get("feature_selection", ""),
            "balancing":         row.get("balancing", ""),
            **m,
        })
    return pd.DataFrame(records)


# ============================================================================
# EXCEL EXPORT  — full performance table for all selected members
# ============================================================================

def _border(style: str = "thin") -> Border:
    s = Side(style=style)
    return Border(left=s, right=s, top=s, bottom=s)


def _hex_fill(hex_color: str) -> PatternFill:
    return PatternFill(
        start_color=hex_color, end_color=hex_color, fill_type="solid"
    )


STRATEGY_HEX: Dict[str, str] = {
    "Maximin (paper)":             "2ECC71",   # green
    "Mean-AUC":                    "3498DB",   # blue
    "Median-AUC":                  "9B59B6",   # purple
    "Pareto-Optimal (Multi-Obj.)": "E67E22",   # orange
    "Per-task best (oracle)":      "95A5A6",   # grey
}

# Font colour that contrasts with each background
STRATEGY_FONT_HEX: Dict[str, str] = {
    "Maximin (paper)":             "FFFFFF",
    "Mean-AUC":                    "FFFFFF",
    "Median-AUC":                  "FFFFFF",
    "Pareto-Optimal (Multi-Obj.)": "FFFFFF",
    "Per-task best (oracle)":      "FFFFFF",
}


def write_excel_report(
    strat_df:  pd.DataFrame,
    config_df: pd.DataFrame,
    df_wide:   pd.DataFrame,
    tasks:     List[str],
    named_rows: Dict[str, Optional[pd.Series]],
    out_path:  Path,
):
    """
    Write a multi-sheet Excel workbook:

    Sheet 1 — "Strategy Overview"
        Summary stats (min/mean/median/max/std/range) for every strategy.
        Colour-coded rows, conditional formatting on numeric cells.

    Sheet 2 — "Per-Task AUC"
        Each strategy × task AUC matrix, with colour-scale formatting.

    Sheet 3 — "Selected Configs"
        Which pipeline each strategy selected + its full performance profile.

    Sheet 4 — "All Configs (ranked)"
        Every configuration evaluated, ranked by min AUC (maximin), with
        the five selected configs highlighted in their strategy colour.
        Includes all tasks + summary statistics.
    """
    wb = Workbook()

    # ── shared style constants ──────────────────────────────────────────────
    HEADER_FONT    = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
    BODY_FONT      = Font(name="Calibri", size=10)
    BOLD_BODY_FONT = Font(name="Calibri", size=10, bold=True)
    CENTER_ALIGN   = Alignment(horizontal="center", vertical="center",
                               wrap_text=True)
    LEFT_ALIGN     = Alignment(horizontal="left",   vertical="center")
    THIN_BORDER    = _border("thin")

    DARK_HEADER_FILL  = _hex_fill("2C3E50")   # dark slate for section headers
    ALT_ROW_FILL      = _hex_fill("F2F2F2")   # light grey for alternate rows
    ORACLE_ITALIC     = Font(name="Calibri", size=10, italic=True,
                             color="555555")

    # ── helper: auto-width ──────────────────────────────────────────────────
    def auto_width(ws, min_w: int = 10, max_w: int = 40):
        for col_cells in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col_cells[0].column)
            for cell in col_cells:
                try:
                    max_len = max(max_len, len(str(cell.value or "")))
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(
                max_w, max(min_w, max_len + 2)
            )

    # ── helper: write a styled header row ───────────────────────────────────
    def write_header(ws, row_idx: int, headers: List[str],
                     fill_hex: str = "2C3E50"):
        fill = _hex_fill(fill_hex)
        for col_idx, h in enumerate(headers, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=h)
            cell.font      = HEADER_FONT
            cell.fill      = fill
            cell.alignment = CENTER_ALIGN
            cell.border    = THIN_BORDER
        ws.row_dimensions[row_idx].height = 36

    # ── helper: apply colour-scale conditional formatting ───────────────────
    def color_scale(ws, min_row: int, max_row: int,
                    min_col: int, max_col: int):
        rng = (
            f"{get_column_letter(min_col)}{min_row}:"
            f"{get_column_letter(max_col)}{max_row}"
        )
        rule = ColorScaleRule(
            start_type="min",  start_color="F8696B",
            mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max",    end_color="63BE7B",
        )
        ws.conditional_formatting.add(rng, rule)

    # ========================================================================
    # SHEET 1 — Strategy Overview
    # ========================================================================
    ws1 = wb.active
    ws1.title = "Strategy Overview"

    summary_cols = ["min_AUC", "mean_AUC", "median_AUC",
                    "max_AUC", "std_AUC", "range_AUC"]
    headers_s1 = ["Strategy"] + [c.replace("_", " ").title() for c in summary_cols]

    # title banner
    ws1.merge_cells(f"A1:{get_column_letter(len(headers_s1))}1")
    title_cell = ws1["A1"]
    title_cell.value     = "Pipeline-Selection Strategy Comparison — Summary Statistics"
    title_cell.font      = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
    title_cell.fill      = _hex_fill("1A252F")
    title_cell.alignment = CENTER_ALIGN
    ws1.row_dimensions[1].height = 30

    write_header(ws1, 2, headers_s1)

    data_start_row = 3
    for r_idx, (_, row) in enumerate(strat_df.iterrows()):
        strat = row["strategy"]
        excel_row = data_start_row + r_idx
        is_oracle = "oracle" in strat.lower()

        # strategy label cell
        s_cell = ws1.cell(row=excel_row, column=1, value=strat)
        s_cell.fill      = _hex_fill(STRATEGY_HEX.get(strat, "BDC3C7"))
        s_cell.font      = Font(
            name="Calibri", bold=True, size=10,
            color=STRATEGY_FONT_HEX.get(strat, "000000"),
            italic=is_oracle,
        )
        s_cell.alignment = LEFT_ALIGN
        s_cell.border    = THIN_BORDER

        for c_idx, col in enumerate(summary_cols, start=2):
            val  = row.get(col, np.nan)
            cell = ws1.cell(
                row=excel_row, column=c_idx,
                value=round(float(val), 4) if not np.isnan(val) else "",
            )
            cell.font      = ORACLE_ITALIC if is_oracle else BODY_FONT
            cell.alignment = CENTER_ALIGN
            cell.border    = THIN_BORDER
            # alternate row shading (skip if oracle)
            if not is_oracle and r_idx % 2 == 1:
                cell.fill = ALT_ROW_FILL

        ws1.row_dimensions[excel_row].height = 20

    # colour-scale on numeric block (exclude std/range — lower = better confusion)
    last_data_row = data_start_row + len(strat_df) - 1
    color_scale(ws1, data_start_row, last_data_row, 2, 5)  # min..max cols

    # footnote
    fn_row = last_data_row + 2
    ws1.merge_cells(f"A{fn_row}:{get_column_letter(len(headers_s1))}{fn_row}")
    fn_cell = ws1[f"A{fn_row}"]
    fn_cell.value = (
        "Green shading = rank-1 for that metric.  "
        "Italic grey row = Per-task best (oracle; not a single pipeline).  "
        "AUC = area under the ROC curve (nested cross-validation, "
        "best across feature-retention variants per config)."
    )
    fn_cell.font      = Font(name="Calibri", size=8, italic=True, color="666666")
    fn_cell.alignment = Alignment(wrap_text=True)

    auto_width(ws1)

    # ========================================================================
    # SHEET 2 — Per-Task AUC
    # ========================================================================
    ws2 = wb.create_sheet("Per-Task AUC")

    ws2.merge_cells(f"A1:{get_column_letter(1 + len(tasks))}1")
    t2 = ws2["A1"]
    t2.value     = "Per-Task AUC of Selected Pipeline by Strategy"
    t2.font      = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
    t2.fill      = _hex_fill("1A252F")
    t2.alignment = CENTER_ALIGN
    ws2.row_dimensions[1].height = 30

    headers_s2 = ["Strategy"] + list(tasks)
    write_header(ws2, 2, headers_s2)

    data_start_s2 = 3
    for r_idx, (_, row) in enumerate(strat_df.iterrows()):
        strat     = row["strategy"]
        excel_row = data_start_s2 + r_idx
        is_oracle = "oracle" in strat.lower()

        s_cell = ws2.cell(row=excel_row, column=1, value=strat)
        s_cell.fill      = _hex_fill(STRATEGY_HEX.get(strat, "BDC3C7"))
        s_cell.font      = Font(
            name="Calibri", bold=True, size=10,
            color=STRATEGY_FONT_HEX.get(strat, "000000"),
            italic=is_oracle,
        )
        s_cell.alignment = LEFT_ALIGN
        s_cell.border    = THIN_BORDER

        for c_idx, task in enumerate(tasks, start=2):
            val  = row.get(task, np.nan)
            cell = ws2.cell(
                row=excel_row, column=c_idx,
                value=round(float(val), 4) if not np.isnan(val) else "",
            )
            cell.font      = ORACLE_ITALIC if is_oracle else BODY_FONT
            cell.alignment = CENTER_ALIGN
            cell.border    = THIN_BORDER
            if not is_oracle and r_idx % 2 == 1:
                cell.fill = ALT_ROW_FILL

        ws2.row_dimensions[excel_row].height = 20

    last_s2 = data_start_s2 + len(strat_df) - 1
    color_scale(ws2, data_start_s2, last_s2, 2, 1 + len(tasks))

    # footnote
    fn2 = last_s2 + 2
    ws2.merge_cells(f"A{fn2}:{get_column_letter(1 + len(tasks))}{fn2}")
    fn2_cell = ws2[f"A{fn2}"]
    fn2_cell.value = (
        "Colour scale: red = low AUC (≈ 0.5), yellow = medium, green = high (≈ 1.0).  "
        "Each cell shows the best AUC across feature-retention variants (10 / 30 / 50 %)."
    )
    fn2_cell.font      = Font(name="Calibri", size=8, italic=True, color="666666")
    fn2_cell.alignment = Alignment(wrap_text=True)

    auto_width(ws2)

    # ========================================================================
    # SHEET 3 — Selected Configs
    # ========================================================================
    ws3 = wb.create_sheet("Selected Configs")

    pipeline_cols = ["model", "scaling", "feature_selection", "balancing"]
    stat_cols     = ["min_AUC", "mean_AUC", "median_AUC",
                     "max_AUC", "std_AUC", "range_AUC"]
    headers_s3    = (
        ["Strategy"] +
        [c.replace("_", " ").title() for c in pipeline_cols] +
        [c.replace("_", " ").title() for c in stat_cols]
    )

    ws3.merge_cells(f"A1:{get_column_letter(len(headers_s3))}1")
    t3 = ws3["A1"]
    t3.value     = "Selected Pipeline Configuration per Strategy"
    t3.font      = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
    t3.fill      = _hex_fill("1A252F")
    t3.alignment = CENTER_ALIGN
    ws3.row_dimensions[1].height = 30

    write_header(ws3, 2, headers_s3)

    data_start_s3 = 3
    for r_idx, (_, row) in enumerate(config_df.iterrows()):
        strat     = row["strategy"]
        excel_row = data_start_s3 + r_idx
        is_oracle = "oracle" in strat.lower()

        s_cell = ws3.cell(row=excel_row, column=1, value=strat)
        s_cell.fill      = _hex_fill(STRATEGY_HEX.get(strat, "BDC3C7"))
        s_cell.font      = Font(
            name="Calibri", bold=True, size=10,
            color=STRATEGY_FONT_HEX.get(strat, "000000"),
            italic=is_oracle,
        )
        s_cell.alignment = LEFT_ALIGN
        s_cell.border    = THIN_BORDER

        for c_idx, col in enumerate(pipeline_cols + stat_cols, start=2):
            val = row.get(col, "")
            if col in stat_cols:
                try:
                    val = round(float(val), 4) if val != "" and not pd.isna(val) else ""
                except Exception:
                    val = ""
            cell = ws3.cell(row=excel_row, column=c_idx, value=val)
            cell.font      = ORACLE_ITALIC if is_oracle else BODY_FONT
            cell.alignment = CENTER_ALIGN if col in stat_cols else LEFT_ALIGN
            cell.border    = THIN_BORDER
            if not is_oracle and r_idx % 2 == 1:
                cell.fill = ALT_ROW_FILL

        ws3.row_dimensions[excel_row].height = 20

    last_s3 = data_start_s3 + len(config_df) - 1
    stat_start_col = 1 + len(pipeline_cols) + 1
    color_scale(ws3, data_start_s3, last_s3,
                stat_start_col, stat_start_col + 3)  # min..max

    auto_width(ws3)

    # ========================================================================
    # SHEET 4 — All Configs (ranked by min AUC)
    # ========================================================================
    ws4 = wb.create_sheet("All Configs (ranked)")

    # rebuild full ranking table
    df_rank = df_wide.copy()
    df_rank["min_AUC"]    = df_rank[tasks].min(axis=1)
    df_rank["mean_AUC"]   = df_rank[tasks].mean(axis=1)
    df_rank["median_AUC"] = df_rank[tasks].median(axis=1)
    df_rank["max_AUC"]    = df_rank[tasks].max(axis=1)
    df_rank["std_AUC"]    = df_rank[tasks].std(axis=1)
    df_rank["range_AUC"]  = df_rank["max_AUC"] - df_rank["min_AUC"]
    df_rank = df_rank.sort_values("min_AUC", ascending=False).reset_index(drop=True)
    df_rank.insert(0, "rank", df_rank.index + 1)

    # build a lookup: (model, scaling, fs, bal) → strategy labels
    def make_key(r):
        return (
            str(r.get("model", "")),
            str(r.get("scaling", "")),
            str(r.get("feature_selection", "")),
            str(r.get("balancing", "")),
        )

    selected_keys: Dict[Tuple, str] = {}
    for strat, s_row in named_rows.items():
        if s_row is not None:
            k = make_key(s_row)
            if k in selected_keys:
                selected_keys[k] += f" | {strat}"
            else:
                selected_keys[k] = strat

    display_cols = (
        ["rank", "model", "scaling", "feature_selection", "balancing"] +
        list(tasks) +
        ["min_AUC", "mean_AUC", "median_AUC", "max_AUC", "std_AUC", "range_AUC"]
    )
    headers_s4 = [c.replace("_", " ").title() for c in display_cols] + ["Selected By"]

    ws4.merge_cells(f"A1:{get_column_letter(len(headers_s4))}1")
    t4 = ws4["A1"]
    t4.value = (
        f"All {len(df_rank):,} Configurations — Ranked by Min AUC (maximin criterion)"
    )
    t4.font      = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
    t4.fill      = _hex_fill("1A252F")
    t4.alignment = CENTER_ALIGN
    ws4.row_dimensions[1].height = 30

    write_header(ws4, 2, headers_s4)

    data_start_s4 = 3
    for r_idx, (_, row) in enumerate(df_rank.iterrows()):
        excel_row = data_start_s4 + r_idx
        rkey      = make_key(row)
        strat_hit = selected_keys.get(rkey, "")

        for c_idx, col in enumerate(display_cols, start=1):
            val = row.get(col, "")
            if col in tasks or col in ["min_AUC", "mean_AUC", "median_AUC",
                                        "max_AUC", "std_AUC", "range_AUC"]:
                try:
                    val = round(float(val), 4)
                except Exception:
                    pass
            cell = ws4.cell(row=excel_row, column=c_idx, value=val)
            cell.border    = THIN_BORDER
            cell.alignment = CENTER_ALIGN if c_idx > 5 else LEFT_ALIGN

            if strat_hit:
                # highlight the selected config rows
                primary_strat = strat_hit.split(" | ")[0]
                cell.fill = _hex_fill(
                    STRATEGY_HEX.get(primary_strat, "F9E79F") + "60"  # 60 = ~38% opacity
                    if len(STRATEGY_HEX.get(primary_strat, "F9E79F")) == 6
                    else "F9E79F"
                )
                cell.font = BOLD_BODY_FONT
            elif r_idx % 2 == 1:
                cell.fill = ALT_ROW_FILL
                cell.font = BODY_FONT
            else:
                cell.font = BODY_FONT

        # "Selected By" column
        sel_cell = ws4.cell(
            row=excel_row,
            column=len(display_cols) + 1,
            value=strat_hit,
        )
        sel_cell.border    = THIN_BORDER
        sel_cell.alignment = LEFT_ALIGN
        if strat_hit:
            primary_strat = strat_hit.split(" | ")[0]
            sel_cell.fill = _hex_fill(STRATEGY_HEX.get(primary_strat, "F9E79F"))
            sel_cell.font = Font(
                name="Calibri", bold=True, size=10,
                color=STRATEGY_FONT_HEX.get(primary_strat, "000000"),
            )
        else:
            sel_cell.font = BODY_FONT
            if r_idx % 2 == 1:
                sel_cell.fill = ALT_ROW_FILL

        ws4.row_dimensions[excel_row].height = 18

    # colour-scale on task AUC columns
    task_start_col = display_cols.index(tasks[0]) + 1
    task_end_col   = task_start_col + len(tasks) - 1
    last_s4        = data_start_s4 + len(df_rank) - 1
    color_scale(ws4, data_start_s4, last_s4, task_start_col, task_end_col)

    # freeze top 2 rows + rank+pipeline columns
    ws4.freeze_panes = ws4.cell(row=3, column=6)

    auto_width(ws4, max_w=22)

    wb.save(out_path)
    print(f"  ✓ Excel workbook → {out_path}")


# ============================================================================
# CONSOLE PRINTER
# ============================================================================

def print_strategy_summary(
    named_rows: Dict[str, Optional[pd.Series]],
    tasks: List[str],
):
    header = (
        f"\n  {'Strategy':<35} {'Model':<20} "
        f"{'Min':>7} {'Mean':>7} {'Median':>7} {'Max':>7} {'Std':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for strategy, row in named_rows.items():
        if row is None:
            print(f"  {strategy:<35} {'N/A':<20} "
                  f"{'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7}")
            continue
        m = config_metrics(row, tasks)
        model_label = str(row.get("model", ""))[:19]
        print(
            f"  {strategy:<35} {model_label:<20} "
            f"{m['min_AUC']:>7.4f} {m['mean_AUC']:>7.4f} "
            f"{m['median_AUC']:>7.4f} {m['max_AUC']:>7.4f} {m['std_AUC']:>7.4f}"
        )


# ============================================================================
# VISUALISATIONS
# ============================================================================

STRATEGY_COLORS = {
    "Maximin (paper)":             "#2ecc71",
    "Mean-AUC":                    "#3498db",
    "Median-AUC":                  "#9b59b6",
    "Pareto-Optimal (Multi-Obj.)": "#e67e22",
    "Per-task best (oracle)":      "#95a5a6",
}


def plot_heatmap(strat_df: pd.DataFrame, tasks: List[str], out: Path):
    mat = strat_df.set_index("strategy")[tasks].astype(float)
    fig, ax = plt.subplots(
        figsize=(max(10, len(tasks) * 1.6), max(4, len(mat) * 0.9))
    )
    sns.heatmap(
        mat, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.5, vmax=1.0, linewidths=0.5,
        annot_kws={"size": 10}, ax=ax,
    )
    ax.set_title(
        "Per-task AUC by Pipeline-Selection Strategy\n"
        "(best AUC across feature-retention variants per config)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Strategy", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmap → {out}")


def plot_bargroup(strat_df: pd.DataFrame, tasks: List[str], out: Path):
    n_tasks = len(tasks)
    n_strat = len(strat_df)
    width   = 0.8 / n_strat
    x       = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(max(12, n_tasks * 1.8), 6))
    for i, (_, row) in enumerate(strat_df.iterrows()):
        strat  = row["strategy"]
        vals   = [float(row.get(t, np.nan)) for t in tasks]
        offset = (i - n_strat / 2 + 0.5) * width
        color  = STRATEGY_COLORS.get(strat, f"C{i}")
        bars   = ax.bar(
            x + offset, vals, width,
            label=strat, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC (nested CV, best across feature-retention variants)", fontsize=10)
    ax.set_ylim(0.5, 1.08)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title(
        "Per-task AUC of Selected Pipeline by Strategy\n"
        "Green = Maximin (paper approach)",
        fontsize=12,
    )
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Bar chart → {out}")


def plot_summary_comparison(strat_df: pd.DataFrame, out: Path):
    """Four-panel horizontal bar chart: min | mean | median | std."""
    panels = [
        ("min_AUC",    True,  "Worst-case AUC\n(min across tasks)\n[Maximin optimises ↑]",    "higher"),
        ("mean_AUC",   True,  "Average AUC\n(mean across tasks)\n[Mean-AUC optimises ↑]",     "higher"),
        ("median_AUC", True,  "Median AUC\n(median across tasks)\n[Median-AUC optimises ↑]",  "higher"),
        ("std_AUC",    False, "Cross-task Std AUC\n(std across tasks)\n[lower = more consistent ↓]", "lower"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, max(4, len(strat_df) * 0.8)))

    for ax, (col, ascending_barh, title, direction) in zip(axes, panels):
        df_sorted   = strat_df.sort_values(col, ascending=ascending_barh).copy()
        strategies  = df_sorted["strategy"].tolist()
        y           = np.arange(len(strategies))
        vals        = df_sorted[col].values
        colors      = [STRATEGY_COLORS.get(s, "#95a5a6") for s in strategies]

        bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white")

        best_strategy = df_sorted.iloc[-1]["strategy"] if ascending_barh \
                        else df_sorted.iloc[0]["strategy"]

        for bar, v, strat in zip(bars, vals, strategies):
            if not np.isnan(v):
                fw = "bold" if strat == best_strategy else "normal"
                ax.text(
                    v + (0.001 if col != "std_AUC" else 0.0005),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}",
                    va="center", fontsize=8, fontweight=fw,
                )

        ax.set_yticks(y)
        ax.set_yticklabels(strategies, fontsize=8)
        ax.set_xlabel("AUC", fontsize=9)
        ax.set_title(title, fontsize=9, pad=8)
        ax.grid(axis="x", alpha=0.3)

        if col != "std_AUC":
            ax.set_xlim(0.5, 1.06)
        else:
            ax.set_xlim(0.0, max(vals) * 1.4 + 0.01)

    plt.suptitle(
        "Strategy Comparison: Worst-case | Average | Median | Consistency\n"
        "(best AUC across feature-retention variants per config × task)\n"
        "Bold value = rank-1 strategy for that panel's criterion",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Summary comparison (4-panel) → {out}")


def plot_radar(strat_df: pd.DataFrame, tasks: List[str], out: Path):
    N      = len(tasks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for _, row in strat_df.iterrows():
        strat = row["strategy"]
        vals  = ([float(row.get(t, np.nan)) for t in tasks] +
                 [float(row.get(tasks[0], np.nan))])
        color = STRATEGY_COLORS.get(strat, "grey")
        lw    = 3 if "Maximin" in strat else 1.5
        ls    = "dashed" if "oracle" in strat.lower() else "solid"
        ax.plot(angles, vals, linewidth=lw, linestyle=ls,
                label=strat, color=color)
        ax.fill(angles, vals, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=7)
    ax.set_title(
        "Strategy comparison — per-task AUC\n(bold green = Maximin paper approach)",
        fontsize=12, pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Radar chart → {out}")


# ============================================================================
# PIPELINE AGREEMENT CHECK
# ============================================================================

def check_pipeline_agreement(
    named_rows: Dict[str, Optional[pd.Series]],
    reference_strategy: str = "Maximin (paper)",
) -> Tuple[Dict, Optional[Tuple]]:
    def get_cfg(row):
        if row is None:
            return None
        return (
            str(row.get("model", "")),
            str(row.get("scaling", "")),
            str(row.get("feature_selection", "")),
            str(row.get("balancing", "")),
        )

    ref_cfg    = get_cfg(named_rows.get(reference_strategy))
    agreements = {}
    for strat, row in named_rows.items():
        if strat == reference_strategy:
            continue
        cfg = get_cfg(row)
        agreements[strat] = {
            "same_as_maximin": cfg == ref_cfg,
            "config":          cfg,
        }
    return agreements, ref_cfg


# ============================================================================
# REBUTTAL TEXT
# ============================================================================

def write_rebuttal_text(
    strat_df:   pd.DataFrame,
    config_df:  pd.DataFrame,
    agreements: Dict,
    ref_cfg:    Optional[Tuple],
    tasks:      List[str],
    out:        Path,
):
    def get_row(keyword: str) -> Optional[pd.Series]:
        mask = strat_df["strategy"].str.contains(keyword, case=False, na=False)
        return strat_df[mask].iloc[0] if mask.any() else None

    maximin_row  = get_row("Maximin")
    mean_row     = get_row("Mean-AUC")
    median_row   = get_row("Median-AUC")
    pareto_row   = get_row("Pareto")
    oracle_row   = get_row("oracle")

    same_strats = [s for s, i in agreements.items() if i["same_as_maximin"]]
    diff_strats = [s for s, i in agreements.items() if not i["same_as_maximin"]]

    with open(out, "w", encoding="utf-8") as f:
        f.write("Suggested rebuttal text (Reviewer 1 R1.3 / Reviewer 2 R2.1):\n\n")
        f.write(
            "We compared the maximin criterion against four alternative pipeline-selection "
            "strategies operating on the same collapsed configuration table "
            "(best AUC across the three feature-retention percentages per configuration "
            "per task, identical to the original selection procedure). "
            "Results are reported in Supplementary Table S2 and visualised in "
            "Supplementary Figure SX.\n\n"
        )

        f.write(
            "Strategy performance summary "
            "(min / mean / median / max / std AUC across all six tasks):\n"
        )
        f.write(
            f"  {'Strategy':<35} "
            f"{'Min':>7} {'Mean':>7} {'Median':>7} {'Max':>7} {'Std':>7}\n"
        )
        f.write("  " + "-" * 72 + "\n")

        for label, row in [
            ("Maximin (paper)",             maximin_row),
            ("Mean-AUC",                    mean_row),
            ("Median-AUC",                  median_row),
            ("Pareto-Optimal (Multi-Obj.)", pareto_row),
            ("Per-task best (oracle)",      oracle_row),
        ]:
            if row is not None:
                suffix = "  ← upper bound; not a single pipeline" \
                         if "oracle" in label.lower() else ""
                f.write(
                    f"  {label:<35} "
                    f"{row['min_AUC']:>7.4f} {row['mean_AUC']:>7.4f} "
                    f"{row['median_AUC']:>7.4f} {row['max_AUC']:>7.4f} "
                    f"{row['std_AUC']:>7.4f}{suffix}\n"
                )
        f.write("\n")

        f.write(
            "The median AUC is reported alongside min, mean, max, and std to give a "
            "complete picture of each strategy's performance distribution across tasks.\n\n"
        )

        if maximin_row is not None:
            f.write(
                f"The maximin criterion achieved the highest minimum cross-task AUC "
                f"({maximin_row['min_AUC']:.4f}), confirming its advantage for worst-case "
                f"robustness. Its median AUC was {maximin_row['median_AUC']:.4f}. "
            )

        if median_row is not None:
            f.write(
                f"The Median-AUC strategy selected the config with the highest median "
                f"cross-task AUC ({median_row['median_AUC']:.4f}), with a min AUC of "
                f"{median_row['min_AUC']:.4f}. "
            )
            gap = (maximin_row["min_AUC"] - median_row["min_AUC"]
                   if maximin_row is not None else None)
            if gap is not None and gap > 0.005:
                f.write(
                    f"Its worst-case AUC was {gap:.4f} lower than the maximin floor, "
                    "illustrating the trade-off between median optimisation and "
                    "worst-case robustness. "
                )
            elif gap is not None:
                f.write(
                    "Its worst-case AUC was within 0.005 of the maximin floor, "
                    "indicating negligible practical difference in robustness. "
                )
        f.write("\n\n")

        if same_strats:
            f.write(
                f"The strategies {', '.join(same_strats)} selected the identical pipeline "
                "as maximin, confirming robustness to criterion choice in these cases. "
            )
        if diff_strats:
            f.write(
                f"The strategies {', '.join(diff_strats)} selected different pipelines, "
                "reflecting genuine criterion-dependent trade-offs. "
            )
        f.write("\n\n")

        if pareto_row is not None:
            f.write(
                "The Pareto-Optimal (Multi-Objective) strategy selects configs that are "
                "non-dominated on both worst-case and mean regret simultaneously. It "
                f"achieved min AUC = {pareto_row['min_AUC']:.4f}, "
                f"mean AUC = {pareto_row['mean_AUC']:.4f}, "
                f"median AUC = {pareto_row['median_AUC']:.4f}. "
            )
            if maximin_row is not None:
                if pareto_row["min_AUC"] >= maximin_row["min_AUC"] - 1e-4:
                    f.write(
                        "This matches the maximin robustness floor, confirming that "
                        "multi-objective optimisation does not sacrifice worst-case "
                        "protection. "
                    )
                else:
                    f.write(
                        f"The small gap in min AUC relative to maximin "
                        f"({maximin_row['min_AUC'] - pareto_row['min_AUC']:.4f}) "
                        "reflects the inherent trade-off in jointly optimising "
                        "worst-case and average objectives. "
                    )
            f.write("\n\n")

        f.write(
            "Pipeline-independent evidence for the robustness of the species-level "
            "findings is provided by Mann-Whitney U tests on raw relative abundances "
            "(FDR-corrected p < 0.05 for CAG:41 depletion in all three diseases, "
            "Supplementary Table S4).\n"
        )

    print(f"  ✓ Rebuttal text → {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print("=" * 72)
    print("  OPTIMIZATION STRATEGY COMPARISON")
    print("  Strategies: Maximin | Mean-AUC | Median-AUC | Pareto | Oracle")
    print("  Metrics   : min / mean / median / max / std AUC")
    print("=" * 72)

    data = load_results(RESULTS)
    if len(data) < 2:
        raise RuntimeError("Need at least 2 tasks loaded.")

    df_wide, tasks = build_collapsed_table(data, METRIC, MODELS)
    if df_wide.empty:
        raise RuntimeError("No configurations appear in all tasks.")

    # ── Apply strategies ────────────────────────────────────────────────────
    print("\nApplying selection strategies …")
    row_maximin = select_maximin(df_wide, tasks)
    row_mean    = select_mean(df_wide, tasks)
    row_median  = select_median(df_wide, tasks)
    row_pareto  = select_pareto_optimal(df_wide, tasks)
    oracle_vals = select_per_task_best(df_wide, tasks)

    named_rows: Dict[str, Optional[pd.Series]] = {
        "Maximin (paper)":             row_maximin,
        "Mean-AUC":                    row_mean,
        "Median-AUC":                  row_median,
        "Pareto-Optimal (Multi-Obj.)": row_pareto,
    }

    # ── Console ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SELECTED CONFIG PERFORMANCE")
    print("=" * 72)
    print_strategy_summary(named_rows, tasks)

    # ── Collect per-task AUCs ───────────────────────────────────────────────
    def to_dict(row, tasks):
        return {t: float(row[t]) for t in tasks if t in row.index}

    strategies: Dict[str, Dict[str, float]] = {
        "Maximin (paper)":             to_dict(row_maximin, tasks),
        "Mean-AUC":                    to_dict(row_mean, tasks),
        "Median-AUC":                  to_dict(row_median, tasks),
        "Pareto-Optimal (Multi-Obj.)": to_dict(row_pareto, tasks),
        "Per-task best (oracle)":      oracle_vals,
    }

    # ── Build tables ────────────────────────────────────────────────────────
    strat_df  = build_strategy_table(strategies, tasks)
    config_df = build_config_comparison(named_rows, tasks)

    summary_cols = ["strategy", "min_AUC", "mean_AUC", "median_AUC",
                    "max_AUC", "std_AUC", "range_AUC"]

    strat_df.to_csv( OUTPUT_DIR / "per_task_auc_per_strategy.csv",    index=False)
    config_df.to_csv(OUTPUT_DIR / "selected_config_per_strategy.csv", index=False)
    strat_df[summary_cols].to_csv(
        OUTPUT_DIR / "strategy_summary_stats.csv", index=False
    )

    print("\n  STRATEGY SUMMARY:")
    print(strat_df[summary_cols].to_string(index=False))
    print("\n  SELECTED CONFIG PER STRATEGY:")
    print(config_df.to_string(index=False))

    # ── Agreement check ─────────────────────────────────────────────────────
    agreements, ref_cfg = check_pipeline_agreement(named_rows, "Maximin (paper)")
    print("\n  Pipeline agreement with Maximin:")
    for strat, info in agreements.items():
        marker = "✓ SAME" if info["same_as_maximin"] else "✗ DIFFERENT"
        print(f"    {strat:40s}: {marker}  {info['config']}")

    # ── Excel report ────────────────────────────────────────────────────────
    print("\nGenerating Excel workbook …")
    write_excel_report(
        strat_df  = strat_df,
        config_df = config_df,
        df_wide   = df_wide,
        tasks     = tasks,
        named_rows= named_rows,
        out_path  = OUTPUT_DIR / "strategy_comparison_full_report.xlsx",
    )

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_heatmap(strat_df, tasks,
                 OUTPUT_DIR / "strategy_comparison_heatmap.png")
    plot_bargroup(strat_df, tasks,
                  OUTPUT_DIR / "strategy_comparison_barplot.png")
    plot_summary_comparison(strat_df,
                            OUTPUT_DIR / "strategy_summary_4panel.png")
    plot_radar(strat_df, tasks,
               OUTPUT_DIR / "strategy_comparison_radar.png")

    # ── Rebuttal text ───────────────────────────────────────────────────────
    write_rebuttal_text(
        strat_df, config_df, agreements, ref_cfg, tasks,
        OUTPUT_DIR / "rebuttal_strategy_text.txt",
    )

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print(f"  ├── strategy_comparison_full_report.xlsx   (4-sheet workbook)")
    print(f"  │     Sheet 1 — Strategy Overview          (summary stats, colour-coded)")
    print(f"  │     Sheet 2 — Per-Task AUC               (task × strategy matrix)")
    print(f"  │     Sheet 3 — Selected Configs           (pipeline + performance)")
    print(f"  │     Sheet 4 — All Configs (ranked)       (all configs, top-5 highlighted)")
    print(f"  ├── selected_config_per_strategy.csv")
    print(f"  ├── per_task_auc_per_strategy.csv")
    print(f"  ├── strategy_summary_stats.csv")
    print(f"  ├── strategy_comparison_heatmap.png")
    print(f"  ├── strategy_comparison_barplot.png")
    print(f"  ├── strategy_summary_4panel.png")
    print(f"  ├── strategy_comparison_radar.png")
    print(f"  └── rebuttal_strategy_text.txt")
    print("=" * 72)


if __name__ == "__main__":
    main()