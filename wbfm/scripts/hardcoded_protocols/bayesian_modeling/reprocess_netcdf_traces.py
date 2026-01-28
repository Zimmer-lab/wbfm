#!/usr/bin/env python
"""
Helper script to batch reprocess NetCDF trace files, dropping large variables.

This script loops through a directory and loads/rewrites any .nc files,
removing large deterministic variables to reduce file size.

Usage:
    python reprocess_netcdf_traces.py --input-dir /path/to/traces [--output-dir /path/to/output] [--backup]
"""

import argparse
import os
from pathlib import Path
from tqdm.auto import tqdm
import logging
import arviz as az
import shutil

from wbfm.utils.external.utils_pymc import drop_large_variables_from_idata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def reprocess_nc_files(input_dir, output_dir=None, backup=False, large_vars_to_drop=None,
                       DEBUG=False):
    """
    Reprocess all .nc files in a directory, dropping large variables.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing .nc files to process
    output_dir : str or Path, optional
        Output directory. If None, overwrites files in input_dir
    backup : bool, optional
        If True, create .bak backups of original files (default: False)
    large_vars_to_drop : list of str, optional
        Variable names to drop. Default: ['curvature_term', 'mu', 'sigmoid_term', 'pca_term', 'y']
    
    Returns
    -------
    dict with keys:
        - 'processed': number of files successfully processed
        - 'failed': number of files that failed
        - 'skipped': number of files skipped (already cleaned)
        - 'failed_files': list of filenames that failed
    """
    if large_vars_to_drop is None:
        large_vars_to_drop = ['curvature_term', 'mu', 'sigmoid_term', 'pca_term', 'y']
    
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .nc files
    nc_files = sorted(input_dir.glob('*.nc'))
    
    if not nc_files:
        logger.warning(f"No .nc files found in {input_dir}")
        return {'processed': 0, 'failed': 0, 'skipped': 0, 'failed_files': []}
    
    logger.info(f"Found {len(nc_files)} .nc files in {input_dir}")
    
    results = {
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'failed_files': []
    }
    
    for nc_file in tqdm(nc_files, desc="Processing .nc files"):
        try:
            logger.info(f"Processing {nc_file.name}...")
            
            # Load the trace
            idata = az.from_netcdf(nc_file)
            
            # Create backup if requested
            if backup:
                backup_file = nc_file.with_suffix('.nc.bak')
                shutil.copy2(nc_file, backup_file)
                logger.info(f"  Created backup at {backup_file.name}")
            
            # Drop variables
            idata_cleaned = drop_large_variables_from_idata(idata, large_vars_to_drop, verbose=DEBUG)
            
            # Save to output directory
            output_file = output_dir / nc_file.name
            az.to_netcdf(idata_cleaned, str(output_file))
            logger.info(f"  Saved to {output_file.name}")
            
            results['processed'] += 1
            
        except Exception as e:
            logger.error(f"  Failed to process {nc_file.name}: {e}")
            # Also print the line number
            if DEBUG:
                import traceback
                traceback.print_exc()
            results['failed'] += 1
            results['failed_files'].append(nc_file.name)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch reprocess NetCDF trace files, dropping large variables'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing .nc files to process'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: overwrite input files)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create .bak backups of original files'
    )
    parser.add_argument(
        '--vars-to-drop',
        type=str,
        nargs='+',
        default=None,
        help='Variable names to drop (default: curvature_term mu sigmoid_term pca_term y)'
    )

    # Add debug flag
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed error output'
    )
    
    args = parser.parse_args()
    
    # Run the reprocessing
    results = reprocess_nc_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backup=args.backup,
        large_vars_to_drop=args.vars_to_drop,
        DEBUG=args.debug
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed: {results['processed']}")
    print(f"Skipped:   {results['skipped']}")
    print(f"Failed:    {results['failed']}")
    
    if results['failed_files']:
        print("\nFailed files:")
        for fname in results['failed_files']:
            print(f"  - {fname}")
    
    print("="*60)


if __name__ == '__main__':
    main()
