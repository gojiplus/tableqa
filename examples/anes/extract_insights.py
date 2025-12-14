#!/usr/bin/env python3
"""
ANES Insight Extraction

Extracts statistical insights from the ANES dataset using the tableqa framework.
This script demonstrates how to:
1. Load and prepare ANES data
2. Run comprehensive statistical analyses (univariate and bivariate)
3. Generate natural language insights
4. Create visualizations
5. Export results for downstream use

Usage:
    python 02_extract_insights.py \
        --data-zip ../data/raw/anes_timeseries_cdf_csv_20220916.csv.zip \
        --metadata ../data/anes_metadata.csv \
        --output-dir ../output

Dependencies:
    tableqa package with all analysis modules
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.schema import Variable, VariableType

# Configure logging
from statqa.utils.logging import setup_logging
from statqa.visualization.plots import PlotFactory


logger = setup_logging(__name__, level="INFO")


def load_data_from_zip(zip_path: str, pattern: str = r"(?i)\.csv$") -> pd.DataFrame:
    """
    Load CSV data from a ZIP file.

    Args:
        zip_path: Path to ZIP file containing CSV data
        pattern: Regex pattern to match CSV files

    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from ZIP: {zip_path}")
    dfs = []

    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            # Skip macOS metadata files
            if member.startswith("__MACOSX/") or not re.search(pattern, member):
                continue

            logger.info(f"  → Reading {member}")
            with z.open(member) as f:
                dfs.append(pd.read_csv(f, low_memory=False))

    if not dfs:
        raise FileNotFoundError(f"No CSV matched pattern {pattern!r} in {zip_path}")

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    logger.info(f"Loaded data: {len(df):,} rows, {len(df.columns)} columns")

    return df


def load_anes_metadata(meta_path: str) -> tuple[dict, dict, dict]:
    """
    Load ANES metadata from CSV.

    Args:
        meta_path: Path to ANES metadata CSV

    Returns:
        Tuple of (label_map, missing_map, valid_map) dictionaries
    """
    logger.info(f"Loading metadata from: {meta_path}")
    meta = pd.read_csv(meta_path)

    label_map = {}
    missing_map = {}
    valid_map = {}

    for _, row in meta.iterrows():
        var = row["varname"]
        label_map[var] = str(row.get("label", "") or var)

        # Parse valid values and missing codes
        valid = str(row.get("valid_values", "") or "")
        missing = {int(m) for m in re.findall(r"Missing\s+(\d+)", valid)}
        missing_map[var] = missing

        # Parse value coding
        codes = {
            int(m.group(1)): m.group(2).strip() for m in re.finditer(r"(\d+)\.\s*([^\n;]+)", valid)
        }
        valid_map[var] = codes

    logger.info(f"Loaded metadata for {len(label_map)} variables")
    return label_map, missing_map, valid_map


def profile_variables(
    df: pd.DataFrame, labels: dict, missing_map: dict, output_dir: str
) -> pd.DataFrame:
    """
    Create a profile of all variables in the dataset.

    Args:
        df: Input DataFrame
        labels: Variable label mapping
        missing_map: Missing value mapping
        output_dir: Output directory for profile

    Returns:
        DataFrame with variable profiles
    """
    logger.info("Profiling variables...")
    profiles = []
    n = len(df)

    for var in df.columns:
        series = df[var]
        miss_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        profiles.append(
            {
                "varname": var,
                "label": labels.get(var, var),
                "dtype": str(series.dtype),
                "unique": unique_count,
                "missing_pct": round(miss_count / n * 100, 1),
            }
        )

    profile_df = pd.DataFrame(profiles)
    profile_path = Path(output_dir) / "variable_profile.csv"
    profile_df.to_csv(profile_path, index=False)
    logger.info(f"✓ Saved variable profile to {profile_path}")

    return profile_df


def infer_variable_types(profile_df: pd.DataFrame) -> tuple[dict, set]:
    """
    Infer variable types from profile data.

    Args:
        profile_df: DataFrame with variable profiles

    Returns:
        Tuple of (type_map, skip_set)
    """
    types = {}
    skip = set()

    for _, row in profile_df.iterrows():
        var = row["varname"]
        unique_count = int(row["unique"])
        dtype = row["dtype"]

        # Skip variables with only one unique value
        if unique_count <= 1:
            skip.add(var)
            continue

        # Infer type based on dtype
        if dtype.startswith(("int", "float")):
            types[var] = "numeric"
        else:
            types[var] = "categorical"

    logger.info(
        f"Inferred types for {len(types)} variables; "
        f"skipping {len(skip)} single-level variables"
    )
    return types, skip


def run_univariate_analysis(
    df: pd.DataFrame,
    var: str,
    vtype: str,
    labels: dict,
    missing_map: dict,
    valid_map: dict,
    output_dir: str,
) -> dict | None:
    """
    Run univariate analysis for a single variable using tableqa.

    Args:
        df: Input DataFrame
        var: Variable name
        vtype: Variable type ('numeric' or 'categorical')
        labels: Label mapping
        missing_map: Missing value mapping
        valid_map: Valid value mapping
        output_dir: Output directory

    Returns:
        Dictionary with insight and figure path, or None
    """
    pretty_name = labels.get(var, var)

    # Clean missing values
    series = df[var].replace(dict.fromkeys(missing_map.get(var, []), np.nan))

    fig_path = None

    if vtype == "numeric":
        # Convert to numeric
        series = pd.to_numeric(series, errors="coerce")
        miss_count = int(series.isna().sum())
        data = series.dropna()

        if data.empty:
            return None

        # Use tableqa's UnivariateAnalyzer
        analyzer = UnivariateAnalyzer()
        var_meta = Variable(name=var, label=pretty_name, type=VariableType.NUMERIC_CONTINUOUS)

        try:
            result = analyzer.analyze(data, var_meta)

            # Generate visualization
            plotter = PlotFactory(style="seaborn", figsize=(6, 4))
            fig_path = Path(output_dir) / f"univariate_{var}.png"
            plotter.plot_univariate(data, var_meta, output_path=str(fig_path))

            # Format insight
            formatter = InsightFormatter()
            insight_text = formatter.format_univariate(result)
            insight_text += f" (N={len(data)}, dropped {miss_count} missing. No weights applied.)"

        except Exception as e:
            logger.warning(f"Error analyzing {var}: {e}")
            return None

    else:  # categorical
        miss_count = int(series.isna().sum())
        counts = series.dropna().value_counts()
        total = int(counts.sum())

        if total == 0:
            return None

        pct = (counts / total * 100).round(1)

        # Create visualization
        plotter = PlotFactory(style="seaborn", figsize=(6, 4))
        fig_path = Path(output_dir) / f"univariate_{var}.png"

        # Simple bar plot for categorical
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        pct.plot.bar(ax=ax)
        ax.set_title(f"Distribution of {pretty_name} ({var})")
        ax.set_ylabel("Percentage (%)")
        fig.text(
            0.5,
            0.01,
            f"N={total}, dropped {miss_count} missing; no weights",
            ha="center",
            fontsize=8,
        )
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # Format insight
        top_value = pct.idxmax()
        top_desc = valid_map.get(var, {}).get(top_value, "")
        insight_text = (
            f"**{pretty_name}** ({var}): Most common value is {top_value} "
            f"('{top_desc}') at {pct.max():.1f}%. "
            f"N={total}, dropped {miss_count} missing. (No weights applied.)"
        )

    return {"vars": [var], "insight": insight_text, "figure": str(fig_path) if fig_path else None}


def run_bivariate_analysis(
    df: pd.DataFrame,
    x: str,
    y: str,
    types: dict,
    labels: dict,
    missing_map: dict,
    valid_map: dict,
    output_dir: str,
) -> dict | None:
    """
    Run bivariate analysis between two variables using tableqa.

    Args:
        df: Input DataFrame
        x: First variable name
        y: Second variable name
        types: Variable type mapping
        labels: Label mapping
        missing_map: Missing value mapping
        valid_map: Valid value mapping
        output_dir: Output directory

    Returns:
        Dictionary with insight and figure path, or None
    """
    label_x = labels.get(x, x)
    label_y = labels.get(y, y)

    # Skip weight variables
    if "weight" in label_x.lower() or "weight" in label_y.lower():
        return None

    # Prepare data
    sub = df[[x, y]].copy()
    sub[x] = pd.to_numeric(
        sub[x].replace(dict.fromkeys(missing_map.get(x, []), np.nan)), errors="coerce"
    )
    sub[y] = pd.to_numeric(
        sub[y].replace(dict.fromkeys(missing_map.get(y, []), np.nan)), errors="coerce"
    )

    if types.get(x) == "numeric" and types.get(y) == "numeric":
        # Numeric x Numeric: Correlation
        data = sub.dropna()
        if data.shape[0] < 10:
            return None

        from scipy.stats import pearsonr

        r, p = pearsonr(data[x], data[y])

        insight_text = (
            f"Correlation **{label_x}** ↔ **{label_y}**: r={r:.2f} "
            f"(N={len(data)}), p={p:.3f}. (No weights applied.)"
        )
        return {"vars": [x, y], "insight": insight_text, "figure": None}

    elif types.get(x) == "categorical" and types.get(y) == "numeric":
        # Categorical x Numeric: Group means
        data = sub.dropna()
        grp = data.groupby(x)[y].mean().dropna()

        if grp.size < 2:
            return None

        # Create visualization
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        grp.plot.bar(ax=ax)
        ax.set_title(f"Mean {label_y} by {label_x}")
        fig.text(0.5, 0.01, "Dropped missing; no weights", ha="center", fontsize=8)

        fig_path = Path(output_dir) / f"bivariate_{x}_{y}.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # Format insight
        mapping = {str(k): round(v, 2) for k, v in grp.items()}
        insight_text = (
            f"Mean **{label_y}** by **{label_x}**: {mapping}. "
            f"Dropped missing; no weights applied."
        )

        return {"vars": [x, y], "insight": insight_text, "figure": str(fig_path)}

    return None


def main():
    """Main entry point for ANES insight extraction."""
    parser = argparse.ArgumentParser(
        description="Extract statistical insights from ANES data using tableqa framework"
    )
    parser.add_argument("--data-zip", required=True, help="Path to ANES data ZIP file")
    parser.add_argument("--metadata", required=True, help="Path to ANES metadata CSV")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for insights and visualizations"
    )
    parser.add_argument(
        "--max-vars",
        type=int,
        default=50,
        help="Maximum number of variables for bivariate analysis (default: 50)",
    )
    parser.add_argument(
        "--skip-bivariate",
        action="store_true",
        help="Skip bivariate analysis (only run univariate)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("Step 1: Loading ANES data")
    logger.info("=" * 60)
    df = load_data_from_zip(args.data_zip)

    # Step 2: Load metadata
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Loading metadata")
    logger.info("=" * 60)
    labels, missing_map, valid_map = load_anes_metadata(args.metadata)

    # Step 3: Profile variables
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Profiling variables")
    logger.info("=" * 60)
    profile_df = profile_variables(df, labels, missing_map, str(output_dir))

    # Step 4: Infer types
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Inferring variable types")
    logger.info("=" * 60)
    types, skip = infer_variable_types(profile_df)

    # Step 5: Univariate analysis
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Running univariate analyses")
    logger.info("=" * 60)
    insights = []

    for var, vtype in tqdm(types.items(), desc="Univariate", unit="var"):
        if var in skip:
            continue

        insight = run_univariate_analysis(
            df, var, vtype, labels, missing_map, valid_map, str(output_dir)
        )
        if insight:
            insights.append(insight)

    logger.info(f"✓ Completed {len(insights)} univariate analyses")

    # Step 6: Bivariate analysis (optional)
    if not args.skip_bivariate:
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Running bivariate analyses")
        logger.info("=" * 60)

        vars_subset = [v for v in types if v not in skip][: args.max_vars]
        bivariate_count = 0

        for i in tqdm(range(len(vars_subset)), desc="Bivariate", unit="pair"):
            for j in range(i + 1, len(vars_subset)):
                insight = run_bivariate_analysis(
                    df,
                    vars_subset[i],
                    vars_subset[j],
                    types,
                    labels,
                    missing_map,
                    valid_map,
                    str(output_dir),
                )
                if insight:
                    insights.append(insight)
                    bivariate_count += 1

        logger.info(f"✓ Completed {bivariate_count} bivariate analyses")

    # Step 7: Save results
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Saving results")
    logger.info("=" * 60)

    output_file = output_dir / "insights.json"
    with open(output_file, "w") as f:
        json.dump(insights, f, indent=2)

    logger.info(f"✓ Saved {len(insights)} total insights to {output_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Insight extraction complete!")
    logger.info("=" * 60)
    logger.info(f"  Total insights: {len(insights)}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Insights JSON: {output_file}")
    logger.info(f"  Variable profile: {output_dir / 'variable_profile.csv'}")


if __name__ == "__main__":
    main()
