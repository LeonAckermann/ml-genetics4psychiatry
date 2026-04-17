"""Plot R2 scores by p-threshold for each illness."""

import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Collect results
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for filepath in glob.glob('results/*/*.json'):
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Extract info from filename
        folder = Path(filepath).parent.name
        # Format: {model}_{illness}_p{p_value}_{distribution}
        parts = folder.split('_')

        # Parse: extract illness and p_clump
        illness = None
        p_clump = None

        for i, part in enumerate(parts):
            if part.startswith('p'):
                # Extract p-value
                p_str = part[1:]  # Remove 'p' prefix
                p_clump = float(p_str)
                # illness is one part before p_clump
                illness = parts[i-1]
                break

        if illness is None or p_clump is None:
            continue

        # Get model name (everything before illness)
        model_idx = next((i for i, p in enumerate(parts) if p == illness), -1)
        if model_idx > 0:
            model = '_'.join(parts[:model_idx])
        else:
            continue

        # Skip linear regression (poor performance)
        if model == 'linear_regression':
            continue

        # Extract R2 scores
        if 'hpo' in data:
            hpo_data = data['hpo']
            scores = hpo_data.get('outer_scores') or hpo_data.get('cv_scores')
            if scores:
                for score in scores:
                    results[illness][p_clump][model].append(score)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Create plots
illnesses = sorted(results.keys())
print(f"Found results for illnesses: {illnesses}")

for illness in illnesses:
    p_values = sorted(results[illness].keys())

    # Prepare data for boxplot
    data_to_plot = []
    labels = []

    for p in p_values:
        models = sorted(results[illness][p].keys())
        for model in models:
            scores = results[illness][p][model]
            data_to_plot.append(scores)
            labels.append(f"{model}\n(p={p})")

    if not data_to_plot:
        print(f"No data for {illness}")
        continue

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Flatten data for seaborn
    plot_data = []
    for p in p_values:
        models = sorted(results[illness][p].keys())
        for model in models:
            scores = results[illness][p][model]
            for score in scores:
                plot_data.append({
                    'p_clump': p,
                    'model': model,
                    'r2': score
                })

    df = pd.DataFrame(plot_data)

    # Create boxplot
    sns.boxplot(
        data=df,
        x='p_clump',
        y='r2',
        hue='model',
        ax=ax,
        palette='Set2'
    )

    ax.set_xlabel('P-value Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{illness}: Model Performance across P-value Thresholds', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save figure
    output_path = f'results/{illness}_r2_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

print("Done!")
