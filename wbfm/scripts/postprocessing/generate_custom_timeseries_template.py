#!/usr/bin/env python
"""
Generate example custom timeseries CSV files for the WBFM trace explorer and dashboard.

Creates the ``behavior/custom_timeseries/`` folder inside a project directory and
writes a sample CSV that demonstrates the required format.

Usage
-----
    python generate_custom_timeseries_template.py -p /path/to/project/project_config.yaml -n 1000

The script will create::

    /path/to/project/behavior/custom_timeseries/example_sine.csv

You can then replace or add your own CSV files following the same format.

CSV format
----------
Each file must have exactly two columns::

    frame,value
    0,1.23
    1,1.45
    2,1.67

- ``frame``: integer frame index (does not need to match neural trace frames; resampling is automatic)
- ``value``: numeric measurement at that frame
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_example_timeseries(project_path: str, n_frames: int):
    """Generate an example custom timeseries CSV file in the project's behavior folder."""
    project_path = Path(project_path)
    if project_path.is_file():
        project_dir = project_path.parent
    else:
        project_dir = project_path

    output_dir = project_dir / 'behavior' / 'custom_timeseries'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a sine wave with noise as example data
    frames = np.arange(n_frames)
    values = np.sin(frames * 2 * np.pi / n_frames * 3) + np.random.normal(0, 0.1, n_frames)

    df = pd.DataFrame({'frame': frames, 'value': np.round(values, 4)})
    output_file = output_dir / 'example_sine.csv'
    df.to_csv(output_file, index=False)

    print(f"Created: {output_file}")
    print(f"  {n_frames} frames, columns: ['frame', 'value']")
    print()
    print("To add your own data:")
    print(f"  1. Create CSV files in {output_dir}/")
    print("  2. Each CSV must have columns 'frame' and 'value' (numeric)")
    print("  3. The filename (without .csv) becomes the name in the GUI dropdown")
    print("  4. Frame counts do not need to match neural traces (resampling is automatic)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate example custom timeseries CSV for WBFM GUI integration')
    parser.add_argument('--project_path', '-p', required=True,
                        help='Path to project_config.yaml or project folder')
    parser.add_argument('--n_frames', '-n', type=int, required=True,
                        help='Number of frames in the example timeseries')
    args = parser.parse_args()

    generate_example_timeseries(args.project_path, args.n_frames)
