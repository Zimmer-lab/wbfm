#!/usr/bin/env python3
"""Interactive neuron visualizer widget for Jupyter using Plotly."""

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import interact, Dropdown, VBox
from IPython.display import display

try:
    from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
    HAS_WBFM = True
except ImportError:
    HAS_WBFM = False


def load_neuron_data():
    """Load neuron data from hierarchical modeling data source."""
    if not HAS_WBFM:
        raise ImportError("wbfm package not available")
    
    data_dir = get_hierarchical_modeling_dir()
    h5_path = f"{data_dir}/data.h5"
    return pd.read_hdf(h5_path, "df_with_missing")


def get_neuron_columns(df):
    """Get list of neuron column names (exclude metadata and behavior)."""
    behavior_keywords = ['curvature', 'speed', 'ventral', 'dorsal', 'head_', 'body_', 
                        'worm_', 'summed', 'signed', 'pca_']
    
    neuron_cols = []
    for c in df.columns:
        if c in ['dataset_name', 'local_time']:
            continue
        # Exclude behavior-related columns
        if any(kw in c.lower() for kw in behavior_keywords):
            continue
        # Include labeled neurons or neuron_XXX format
        if not c.startswith('neuron_') or c.startswith('neuron_0'):
            neuron_cols.append(c)
        elif c.startswith('neuron_'):
            neuron_cols.append(c)
    
    # Also include specific labeled neurons
    neuron_cols = [c for c in df.columns if c not in ['dataset_name', 'local_time'] 
                   and not any(kw in c.lower() for kw in behavior_keywords)]
    
    return sorted(neuron_cols)


def get_datasets(df):
    """Get list of dataset names."""
    return sorted(df['dataset_name'].unique())


def create_neuron_widget():
    """Create an interactive widget for visualizing neurons."""
    print("Loading data...")
    df = load_neuron_data()
    
    datasets = get_datasets(df)
    neurons = get_neuron_columns(df)
    
    print(f"Loaded {len(datasets)} datasets, {len(neurons)} neuron columns")
    
    # Create dropdowns
    dataset_dropdown = Dropdown(
        options=datasets,
        value=datasets[0],
        description='Dataset:',
        ensure_option=False
    )
    
    neuron_dropdown = Dropdown(
        options=neurons,
        value=neurons[0] if neurons else None,
        description='Neuron:',
        ensure_option=False
    )
    
    # Create output area
    output = go.FigureWidget()
    
    def update_plot(change=None):
        ds_name = dataset_dropdown.value
        neuron_name = neuron_dropdown.value
        
        if neuron_name is None:
            return
        
        group = df[df['dataset_name'] == ds_name]
        
        with output.batch_update():
            output.data = []
            output.layout = {}
            
            # Get neuron data
            y = group[neuron_name].values.astype(float)
            x = group['local_time'].values
            
            # Handle NaN for visualization
            valid_mask = ~pd.isna(y)
            
            output.add_trace(go.Scatter(
                x=x[valid_mask],
                y=y[valid_mask],
                mode='lines',
                name=neuron_name,
                line=dict(color='blue', width=1)
            ))
            
            # Add NaN markers
            nan_mask = pd.isna(y)
            if nan_mask.sum() > 0:
                output.add_trace(go.Sc=x[nan_mask],
                    y=[atter(
                    xy[valid_mask].min() if valid_mask.any() else 0] * nan_mask.sum(),
                    mode='markers',
                    name='NaN',
                    marker=dict(color='red', size=4, symbol='x')
                ))
            
            output.layout.title = f"{neuron_name} in {ds_name}"
            output.layout.xaxis.title = "Time"
            output.layout.yaxis.title = "Activity"
            output.layout.height = 400
            output.layout.width = 800
    
    # Connect callbacks
    dataset_dropdown.observe(update_plot, names='value')
    neuron_dropdown.observe(update_plot, names='value')
    
    # Initial plot
    update_plot()
    
    # Display
    display(VBox([dataset_dropdown, neuron_dropdown, output]))
    
    return dataset_dropdown, neuron_dropdown, output


def quick_plot(dataset_name, neuron_name):
    """Quick plotting function for inline use."""
    df = load_neuron_data()
    group = df[df['dataset_name'] == dataset_name]
    
    y = group[neuron_name].values.astype(float)
    x = group['local_time'].values
    
    valid_mask = ~pd.isna(y)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x[valid_mask],
        y=y[valid_mask],
        mode='lines',
        name=neuron_name,
        line=dict(color='blue', width=1)
    ))
    
    # Add NaN markers
    nan_mask = pd.isna(y)
    if nan_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=x[nan_mask],
            y=[y[valid_mask].min() if valid_mask.any() else 0] * nan_mask.sum(),
            mode='markers',
            name='NaN',
            marker=dict(color='red', size=4, symbol='x')
        ))
    
    fig.update_layout(
        title=f"{neuron_name} in {dataset_name}",
        xaxis_title="Time",
        yaxis_title="Activity",
        height=400,
        width=800
    )
    
    fig.show()


if __name__ == "__main__":
    create_neuron_widget()
