import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# =====================
# CONFIG
# =====================
input_csv = sys.argv[1]

residue_groups = ['G', 'A', 'V', 'L', 'M', 'I', 'F', 'Y', 'W',
                  'S', 'T', 'C', 'N', 'Q', 'P', 'H', 'K', 'R', 'D', 'E']

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(input_csv)

if not {'name', 'sequence', 'pred_controlPSI'}.issubset(df.columns):
    raise ValueError("CSV must contain: name, sequence, pred_controlPSI")

# Split WT and mutants
wt_df = df[df['name'].str.endswith('_WT')].copy()
mut_df = df[~df['name'].str.endswith('_WT')].copy()

if len(wt_df) == 0:
    raise ValueError("No WT rows found (expected names ending with '_WT')")

# Base peptide names
wt_df['base'] = wt_df['name'].str.replace('_WT', '', regex=False)

# =====================
# PROCESS EACH PEPTIDE
# =====================
for base_name in wt_df['base'].unique():

    print(f"Processing {base_name}...")

    wt_row = wt_df[wt_df['base'] == base_name].iloc[0]
    wt_seq = wt_row['sequence']
    wt_val = wt_row['pred_controlPSI']

    # Subset mutants
    sub_mut = mut_df[mut_df['name'].str.startswith(base_name + "_")]

    delta_list = []

    for _, row in sub_mut.iterrows():

        # Parse mutation (e.g., 19P)
        try:
            mut_tag = row['name'].rsplit('_', 1)[1]
            mut_pos = int(mut_tag[:-1]) - 1
            mut_aa = mut_tag[-1]
        except:
            continue  # skip malformed rows

        if mut_pos >= len(wt_seq):
            continue

        delta = row['pred_controlPSI'] - wt_val

        # If same as WT residue → force 0
        if wt_seq[mut_pos] == mut_aa:
            delta = 0.0

        delta_list.append((mut_pos, wt_seq[mut_pos], mut_aa, delta))

    if len(delta_list) == 0:
        print(f"Skipping {base_name}: no valid mutants found")
        continue

    heat_df = pd.DataFrame(delta_list, columns=['pos','WT_res','mutAA','delta'])

    # Pivot
    heatmap_data = heat_df.pivot_table(
        index='mutAA',
        columns='pos',
        values='delta',
        fill_value=0
    )

    # Sort axes
    heatmap_data = heatmap_data.reindex(residue_groups)
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

    # =====================
    # PLOT
    # =====================
    plt.figure(figsize=(10,6))

    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        center=0,
        annot=False,
        square=True,
        cbar_kws={'label': 'Δ controlPSI'}
    )

    ax = plt.gca()

    # X labels = WT sequence
    wt_sequence = list(wt_seq)
    ax.set_xticks(np.arange(len(wt_sequence)) + 0.5)
    ax.set_xticklabels(wt_sequence, rotation=0, fontsize=10)

    plt.title(f"{base_name}: Saturation Mutagenesis Δ controlPSI")
    plt.xlabel("WT residue")
    plt.ylabel("Mutated amino acid")

    plt.tight_layout()

    out_file = f"{base_name}_heatmap.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved {out_file}")

print("All done.")