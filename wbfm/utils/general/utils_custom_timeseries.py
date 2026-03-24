"""
Utilities for loading and resampling user-defined custom timeseries CSV files.

Custom timeseries are placed in ``<project>/behavior/custom_timeseries/`` as CSV files
with exactly two columns: ``frame`` and ``value``. They are resampled to match the neural
trace frame count and can then be correlated with neural activity in the GUIs.
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import interpolate

logger = logging.getLogger(__name__)


def get_custom_timeseries_path(project_dir: Path) -> Path:
    """Return the standard path for the custom timeseries folder."""
    return project_dir / 'behavior' / 'custom_timeseries'


def load_custom_timeseries_csvs(custom_timeseries_path: Path) -> Dict[str, pd.Series]:
    """
    Load and validate CSV files from a custom_timeseries folder.

    Each CSV must have exactly two columns named ``frame`` and ``value`` (case-sensitive,
    both numeric). The filename stem becomes the timeseries name. Files starting with
    ``._`` (macOS resource forks) are skipped.

    Parameters
    ----------
    custom_timeseries_path : Path
        Path to ``behavior/custom_timeseries/`` folder.

    Returns
    -------
    dict
        ``{filename_stem: pd.Series}`` indexed by frame, or empty dict if nothing valid.
    """
    if not custom_timeseries_path.exists():
        return {}

    csv_files = [f for f in custom_timeseries_path.glob("*.csv") if not f.name.startswith("._")]
    if not csv_files:
        return {}

    csv_data: Dict[str, pd.Series] = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()

            if list(df.columns) != ['frame', 'value']:
                logger.warning("Skipping %s: expected columns ['frame', 'value'], got %s",
                               csv_file.name, list(df.columns))
                continue

            if not pd.api.types.is_numeric_dtype(df['frame']) or not pd.api.types.is_numeric_dtype(df['value']):
                logger.warning("Skipping %s: non-numeric data", csv_file.name)
                continue

            csv_data[csv_file.stem] = df.set_index('frame')['value']
            logger.info("Loaded custom timeseries '%s' (%d frames)", csv_file.stem, len(df))

        except Exception as e:
            logger.warning("Could not load %s: %s", csv_file.name, e)

    return csv_data


def resample_timeseries_to_target_length(
    csv_data: Dict[str, pd.Series],
    target_length: int,
) -> pd.DataFrame:
    """
    Resample each timeseries individually to *target_length* frames using linear interpolation.

    Parameters
    ----------
    csv_data : dict
        ``{name: pd.Series}`` from :func:`load_custom_timeseries_csvs`.
    target_length : int
        Number of frames in the neural traces (master timeline).

    Returns
    -------
    pd.DataFrame
        Columns are timeseries names, index is ``0 .. target_length - 1``.
    """
    if not csv_data:
        return pd.DataFrame()

    processed: Dict[str, np.ndarray] = {}
    for name, series in csv_data.items():
        n = len(series)
        if n == target_length:
            processed[name] = series.values
        else:
            logger.info("Resampling '%s' from %d to %d frames", name, n, target_length)
            original_indices = np.linspace(0, target_length - 1, n)
            new_indices = np.arange(target_length)
            f = interpolate.interp1d(original_indices, series.values,
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
            processed[name] = f(new_indices)

    df = pd.DataFrame(processed, index=range(target_length))
    logger.info("Processed %d custom timeseries, aligned to %d frames", len(processed), target_length)
    return df
