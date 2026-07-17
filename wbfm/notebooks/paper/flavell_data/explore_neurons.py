#!/usr/bin/env python3
"""Explore neuron time series - generates static HTML for quick comparison."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from load_neurons import load_neurons
from wbfm.utils.external.utils_pandas import combine_columns_with_suffix


def load_data():
    """Load and preprocess data."""
    df = load_neurons()
    
    dfs = []
    for dataset_name, group in df.groupby('dataset_name'):
        group = group.dropna(axis=1, how='all')
        group = combine_columns_with_suffix(group)
        dfs.append(group)
    df = pd.concat(dfs, ignore_index=True)
    
    return df


def compute_pairwise_correlations(df, neurons):
    """Compute pairwise correlations for all neuron pairs across all datasets."""
    results = []
    
    for ds in df['dataset_name'].unique():
        group = df[df['dataset_name'] == ds]
        
        for i, n1 in enumerate(neurons):
            for n2 in neurons[i+1:]:
                if n1 in group.columns and n2 in group.columns:
                    v1 = group[n1].values
                    v2 = group[n2].values
                    mask = ~np.isnan(v1) & ~np.isnan(v2)
                    if mask.sum() > 10:
                        corr = np.corrcoef(v1[mask], v2[mask])[0, 1]
                        if not np.isnan(corr):
                            results.append({
                                'dataset': ds,
                                'neuron1': n1,
                                'neuron2': n2,
                                'correlation': corr
                            })
    
    return pd.DataFrame(results)


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {df['dataset_name'].nunique()} datasets")
    
    # Get labeled neurons
    neuron_cols = [c for c in df.columns if not c.startswith('neuron_') and c not in ['dataset_name', 'local_time', 'velocity', 'angular_velocity', 'head_curvature', 'body_curvature', 'pumping']]
    
    # Top neurons to compare (based on earlier analysis)
    neurons = ['AVA', 'AVE', 'RIM', 'URX', 'RMDD', 'BAG', 'RIV', 'RMDV', 'SMDV', 'RME']
    neurons = [n for n in neurons if n in neuron_cols]
    
    print(f"\nComputing correlations between: {neurons}")
    corr_df = compute_pairwise_correlations(df, neurons)
    
    # Create correlation matrix
    corr_matrix = pd.DataFrame(index=neurons, columns=neurons, dtype=float)
    for i, n1 in enumerate(neurons):
        for j, n2 in enumerate(neurons):
            if i == j:
                corr_matrix.loc[n1, n2] = 1.0
            else:
                subset = corr_df[((corr_df['neuron1'] == n1) & (corr_df['neuron2'] == n2)) |
                                ((corr_df['neuron1'] == n2) & (corr_df['neuron2'] == n1))]
                if len(subset) > 0:
                    corr_matrix.loc[n1, n2] = subset['correlation'].mean()
    
    print("\n=== Correlation Matrix (mean across datasets) ===")
    print(corr_matrix.round(3))
    
    # Save correlation matrix to HTML heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=neurons,
        y=neurons,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        showscale=True
    ))
    
    fig.update_layout(
        title='Mean Pairwise Correlation Across Datasets',
        xaxis_title='Neuron',
        yaxis_title='Neuron',
        height=600,
        width=700
    )
    
    fig.write_html('neuron_correlation_heatmap.html')
    print("\nSaved correlation heatmap to neuron_correlation_heatmap.html")
    
    # Now create time series plots for specific datasets
    datasets_to_plot = ['atanas_kim_2023-2023-01-23-01', 'atanas_kim_2023-2022-06-14-01', 'atanas_kim_2023-2022-06-28-01']
    
    for ds in datasets_to_plot:
        group = df[df['dataset_name'] == ds]
        
        # Check which neurons exist
        available = [n for n in neurons if n in group.columns]
        
        fig = go.Figure()
        
        t = group['local_time'].values
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, neuron in enumerate(available):
            y = group[neuron].values
            valid = ~np.isnan(y)
            fig.add_trace(go.Scatter(x=t[valid], y=y[valid], mode='lines', name=neuron, line=dict(color=colors[i % len(colors)])))
        
        # Calculate correlations
        corrs = []
        for i, n1 in enumerate(available):
            for n2 in available[i+1:]:
                v1 = group[n1].values
                v2 = group[n2].values
                mask = ~np.isnan(v1) & ~np.isnan(v2)
                if mask.sum() > 10:
                    c = np.corrcoef(v1[mask], v2[mask])[0, 1]
                    corrs.append(f"{n1}-{n2}: {c:.3f}")
        
        title = f'{ds}<br><sup>Correlations: {", ".join(corrs[:5])}...</sup>'
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Activity', height=400, width=900)
        
        fig.write_html(f'neuron_timeseries_{ds}.html')
        print(f"Saved {ds} time series to neuron_timeseries_{ds}.html")


if __name__ == "__main__":
    main()
