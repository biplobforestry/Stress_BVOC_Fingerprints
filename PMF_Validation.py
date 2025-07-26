# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:25:41 2025
@author: b.dey
email: biplobforestry@gmail.com
version: 1.5
project: Multi-stress interaction effects on BVOC emission fingerprints from oak and beech:
         A cross-investigation using Machine Learning and Positive Matrix Factorization
"""
# PMF Validation Script for BVOC Data
# -----------------------------------
# This script validates PMF factor solutions by comparing them with raw VOC signals.
# It uses PMF output files (factor profiles and time series) and overlays them with 
# the original input matrix ("datawave") to evaluate factor-wise reconstruction and 
# unexplained variance.
#
# Raw file: Contains 'datawave', factor 'profile', and factor 'time series' sheets.
# Ion file: Contains the list of target ions to evaluate (m/z and compound name).
# Output: PNG visualizations and a summary Excel file with explained/unexplained ratios.


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import tkinter as tk
from matplotlib import cm
root = tk.Tk()
root.withdraw()
plt.ioff()


raw_file = ".../Oak_SoFi_validation.xlsx"  
ion_file = ".../Validation_compounds.xlsx"
output_dir = ".../Factor_validation" #change
summary_file = os.path.join(output_dir, "F7_run_44.xlsx") #change


df_raw = pd.read_excel(raw_file, sheet_name='datawave')
df_profile = pd.read_excel(raw_file, sheet_name='F7_run_44_Prof') #change
df_time = pd.read_excel(raw_file, sheet_name='F7_run_44_TS') #change
df_ions = pd.read_excel(ion_file)


df_time['TimeDate'] = pd.to_datetime(df_time['TimeDate'])
df_raw['UTC'] = pd.to_datetime(df_raw['UTCTime'])


factor_cols = [col for col in df_time.columns if col.startswith('Factor_')]


os.makedirs(output_dir, exist_ok=True)


summary_rows = []
not_found = []


for idx, row in df_ions.iterrows():
    ion = row['ion']
    name = row['Name']

    if ion not in df_profile['Name'].values or ion not in df_raw.columns:
        not_found.append(ion)
        continue


    target_row = df_profile[df_profile['Name'] == ion].iloc[0]


    for factor in factor_cols:
        df_time[f"{ion}_{factor}"] = df_time[factor] * target_row[factor]


    factor_totals = {factor: df_time[f"{ion}_{factor}"].sum() for factor in factor_cols}
    total_pmf = sum(factor_totals.values())
    total_raw = df_raw[ion].sum()
    unexplained = max(total_raw - total_pmf, 0)


    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.Set2.colors


    cumulative = pd.Series(0, index=df_time.index)
    for i, factor in enumerate(factor_cols):
        y = df_time[f"{ion}_{factor}"]
        stacked = cumulative + y
        ax.fill_between(df_time['TimeDate'], cumulative, stacked,
                        color=colors[i % len(colors)], label=factor, alpha=0.9)
        cumulative = stacked


    ax.plot(df_raw['UTC'], df_raw[ion], label='Raw Signal', color='black',
            linestyle='--', linewidth=1.5, alpha=0.9)

    ax.set_ylabel(f"{ion} signal [ppt]")
    ax.set_xlabel("Time")
    ax.set_title(name)
    ax.legend(loc='upper left', frameon=False)
    ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)


    fractions = list(factor_totals.values()) + [unexplained]
    labels = factor_cols + ['Unexplained']
    pie_colors = list(colors[:len(factor_cols)]) + ['lightgrey']
    percent_labels = [
        f"{l}: {v / total_raw * 100:.1f}%" if v / total_raw * 100 >= 10 else ""
        for l, v in zip(labels, fractions)
    ]

    inset_ax = ax.inset_axes([0.1, 0.45, 0.2, 0.2], facecolor='none')
    inset_ax.pie(fractions, labels=percent_labels, colors=pie_colors,
                 startangle=90, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 8})
    inset_ax.set_aspect('equal')


    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ion}.png"), dpi=300)
    plt.close(fig)


    dominant_factor = max(factor_totals, key=factor_totals.get)
    dominant_percent = factor_totals[dominant_factor] / total_raw * 100
    summary_rows.append({
        'ion': ion,
        'Name': name,
        'dominating factor': dominant_factor,
        'percentage': round(dominant_percent, 2),
        'unexplained': round(unexplained / total_raw * 100, 2)
    })


df_summary = pd.DataFrame(summary_rows)
df_summary.to_excel(summary_file, index=False)

if not_found:
    print("Compounds not found in datasets:")
    for nf in not_found:
        print(f" - {nf}")
else:
    print("all compounds saved successfully.")
