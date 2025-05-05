#!/usr/bin/env python3
"""
insight_extractor.py

Step 2: Insight Extraction for ANES dataset, with variable profiling

Features:
  0. Variable profiling: automatic summary of each column to guide later steps
  1. Descriptive analyses:
     - Univariate: mean, median, std for numeric; frequency & percentage for categorical; density for high-cardinality
     - Bivariate: Pearson correlations (num×num), group means (cat×num); skip weight variables or single-level
  2. Interpretation: clear natural-language insights with metadata labels, missing info, and no-weight disclaimers
  3. Visualization: consistent small (6×4) matplotlib charts saved within output-dir
  4. Reporting: JSON linking each textual insight to its figure (if any)

Usage:
  python3 insight_extractor.py \
    --data-zip anes_data.zip \
    --metadata anes_metadata.csv \
    --templates question_templates.txt \
    --output-dir insights_output

Dependencies:
  pandas, numpy, matplotlib, scipy, tqdm
"""
import argparse
import os
import zipfile
import json
import re
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm.auto import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_data_from_zip(zip_path, pattern=r"(?i)\.csv$"):
    logging.info(f"Loading data from ZIP {zip_path}")
    dfs = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.startswith('__MACOSX/') or not re.search(pattern, member):
                continue
            logging.info(f"  → reading {member}")
            with z.open(member) as f:
                dfs.append(pd.read_csv(f, low_memory=False))
    if not dfs:
        raise FileNotFoundError(f"No CSV matched {pattern!r} in {zip_path}")
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    logging.info(f"Loaded data: {len(df)} rows, {len(df.columns)} cols")
    return df


def load_metadata(meta_path):
    meta = pd.read_csv(meta_path)
    label_map, missing_map, valid_map = {}, {}, {}
    for _, row in meta.iterrows():
        var = row['varname']
        label_map[var] = str(row.get('label','') or var)
        valid = str(row.get('valid_values','') or '')
        missing = {int(m) for m in re.findall(r"Missing\s+(\d+)", valid)}
        missing_map[var] = missing
        codes = {int(m.group(1)): m.group(2).strip()
                 for m in re.finditer(r"(\d+)\.\s*([^\n;]+)", valid)}
        valid_map[var] = codes
    return label_map, missing_map, valid_map


def profile_variables(df, labels, missing_map, output_dir):
    prof = []
    n = len(df)
    for var in df.columns:
        series = df[var]
        miss = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        prof.append({
            'varname': var,
            'label': labels.get(var,var),
            'dtype': str(series.dtype),
            'unique': unique,
            'missing_pct': round(miss/n*100,1)
        })
    prof_df = pd.DataFrame(prof)
    path = os.path.join(output_dir, 'variable_profile.csv')
    prof_df.to_csv(path, index=False)
    logging.info(f"Saved variable profile to {path}")
    return prof_df


def infer_variable_types(profile_df):
    types, skip = {}, set()
    for _, row in profile_df.iterrows():
        var = row['varname']
        uniq = int(row['unique'])
        dt = row['dtype']
        if uniq <= 1:
            skip.add(var)
            continue
        if dt.startswith(('int','float')):
            types[var] = 'numeric'
        else:
            types[var] = 'categorical'
    logging.info(f"Inferred types for {len(types)} vars; skipping {len(skip)} single-level vars")
    return types, skip


def univariate_analysis(df, var, vtype, labels, missing_map, valid_map, output_dir):
    pretty = labels.get(var,var)
    ser = df[var].replace({m:np.nan for m in missing_map.get(var,[])})
    fig_path = None
    if vtype=='numeric':
        ser = pd.to_numeric(ser, errors='coerce')
        miss = int(ser.isna().sum())
        data = ser.dropna()
        if data.empty:
            return None
        mean_, med, std = data.mean(), data.median(), data.std()
        uniq = data.nunique()
        fig, ax = plt.subplots(figsize=(6,4), dpi=80)
        if uniq > 50:
            ax.hist(data, bins=20, density=True, alpha=0.6)
            ax.set_ylabel('Density')
        else:
            weights = np.ones_like(data)/len(data)*100
            ax.hist(data, bins=min(uniq,20), weights=weights)
            ax.set_ylabel('Percentage (%)')
        ax.set_title(f"Distribution of {pretty} ({var})")
        fig.text(0.5,0.01,f"N={len(data)}, dropped {miss} missing; no weights",ha='center',fontsize=8)
        fig_path = os.path.join(output_dir,f"univariate_{var}.png")
        fig.savefig(fig_path,bbox_inches='tight'); plt.close(fig)
        txt = (f"**{pretty}** ({var}): mean={mean_:.2f}, median={med:.2f}, std={std:.2f}. "
               f"N={len(data)}, dropped {miss} missing. (No weights)")
    else:
        miss = int(ser.isna().sum())
        counts = ser.dropna().value_counts()
        total = int(counts.sum())
        pct = (counts/total*100).round(1)
        fig, ax = plt.subplots(figsize=(6,4), dpi=80)
        pct.plot.bar(ax=ax)
        ax.set_title(f"Distribution of {pretty} ({var})")
        ax.set_ylabel('Percentage (%)')
        fig.text(0.5,0.01,f"N={total}, dropped {miss} missing; no weights",ha='center',fontsize=8)
        fig_path = os.path.join(output_dir,f"univariate_{var}.png")
        fig.savefig(fig_path,bbox_inches='tight'); plt.close(fig)
        top = pct.idxmax()
        desc = valid_map.get(var,{}).get(top, '')
        txt = (f"**{pretty}** ({var}): top value={top} ('{desc}') at {pct.max():.1f}%. "
               f"N={total}, dropped {miss} missing. (No weights)")
    return {'vars':[var],'insight':txt,'figure':fig_path}


def bivariate_analysis(df, x, y, types, labels, missing_map, valid_map, output_dir):
    lx, ly = labels.get(x,x), labels.get(y,y)
    if 'weight' in lx.lower() or 'weight' in ly.lower():
        return None
    sub = df[[x,y]].copy()
    sub[x] = pd.to_numeric(sub[x].replace({m:np.nan for m in missing_map.get(x,[])}),errors='coerce')
    sub[y] = pd.to_numeric(sub[y].replace({m:np.nan for m in missing_map.get(y,[])}),errors='coerce')
    if types.get(x)=='numeric' and types.get(y)=='numeric':
        data = sub.dropna()
        if data.shape[0]<10:
            return None
        r,p = pearsonr(data[x], data[y])
        txt = f"Correlation **{lx}**↔**{ly}**: r={r:.2f} (N={len(data)}), p={p:.3f}. (No weights)"
        return {'vars':[x,y],'insight':txt,'figure':None}
    if types.get(x)=='categorical' and types.get(y)=='numeric':
        data = sub.dropna()
        grp = data.groupby(x)[y].mean().dropna()
        if grp.size<2:
            return None
        fig, ax = plt.subplots(figsize=(6,4), dpi=80)
        grp.plot.bar(ax=ax)
        ax.set_title(f"Mean {labels.get(y,y)} by {labels.get(x,x)}")
        fig.text(0.5,0.01,f"Dropped missing; no weights",ha='center',fontsize=8)
        fp = os.path.join(output_dir, f"bivariate_{x}_{y}.png")
        fig.savefig(fp, bbox_inches='tight'); plt.close(fig)
        mapping = {str(k): round(v,2) for k,v in grp.items()}
        txt = (f"Mean **{labels.get(y,y)}** by **{labels.get(x,x)}**: {mapping}. "
               f"Dropped missing; no weights.")
        return {'vars':[x,y],'insight':txt,'figure':fp}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-zip', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--templates', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Starting insight extraction pipeline")

    df = load_data_from_zip(args.data_zip)
    labels, missing_map, valid_map = load_metadata(args.metadata)
    prof_df = profile_variables(df, labels, missing_map, args.output_dir)
    types, skip = infer_variable_types(prof_df)

    insights = []
    for var, vt in tqdm(types.items(), desc="Univariate", unit="var"):
        if var in skip:
            continue
        ins = univariate_analysis(df, var, vt, labels, missing_map, valid_map, args.output_dir)
        if ins:
            insights.append(ins)

    vars50 = [v for v in types if v not in skip][:50]
    for i in tqdm(range(len(vars50)), desc="Bivariate", unit="pair"):
        for j in range(i+1, len(vars50)):
            ins = bivariate_analysis(df, vars50[i], vars50[j], types, labels, missing_map, valid_map, args.output_dir)
            if ins:
                insights.append(ins)

    out = os.path.join(args.output_dir, 'insights.json')
    with open(out, 'w') as f:
        json.dump(insights, f, indent=2)
    logging.info(f"Saved {len(insights)} insights to {out}")
    logging.info("Insight extraction complete")

if __name__ == '__main__':
    main()
