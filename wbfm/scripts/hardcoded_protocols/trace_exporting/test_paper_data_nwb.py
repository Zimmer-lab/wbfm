import argparse
from pathlib import Path

DEFAULT_NWB_ROOT = Path('/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/nwb')


def check_expected_fields(tester, has_video_or_images, nwb_path=None):
    required_fields = {
        'has_calcium_traces': True,
        'has_neuropal': False,
        'has_behavior_video': False,
        'has_behavior_time_series': True,
    }
    # Legacy pre-revision exports may omit centroids entirely
    is_pre_revision = nwb_path is not None and 'pre_revision' in str(nwb_path)
    if not is_pre_revision:
        required_fields['has_centroids'] = True
    if has_video_or_images:
        required_fields['has_calcium_imaging'] = True

    mismatched_fields = [
        f'{field_name} (expected {expected_value}, found {getattr(tester, field_name)})'
        for field_name, expected_value in required_fields.items()
        if getattr(tester, field_name) != expected_value
    ]
    if mismatched_fields:
        raise ValueError(f'Unexpected NWB fields: {", ".join(mismatched_fields)}')


def main():
    parser = argparse.ArgumentParser(
        description='Load and inspect each NWB file exported from the paper datasets.'
    )
    parser.add_argument(
        '--nwb_dir',
        type=Path,
        help='Directory containing exported NWB files. Defaults to the paper export directory.',
    )
    parser.add_argument(
        '--include_image_data',
        action='store_true',
        help='Test files in the with_images export directory instead of no_images.',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Check NWB structure without reading data payloads.',
    )
    args = parser.parse_args()

    if args.nwb_dir is None:
        export_type = 'with_images' if args.include_image_data else 'no_images'
        nwb_dir = DEFAULT_NWB_ROOT / export_type
    else:
        nwb_dir = args.nwb_dir.expanduser()

    nwb_files = sorted(nwb_dir.rglob('*.nwb'))
    if not nwb_files:
        print(f'No NWB files found in {nwb_dir}')
        return 1

    from wbfm.utils.nwb.test_nwb import TestNWB

    failures = []
    for nwb_file in nwb_files:
        print(f'\nTesting {nwb_file}')
        try:
            tester = TestNWB(str(nwb_file), fast=args.fast)
        except Exception as error:
            print(f'Failed to load {nwb_file}: {error}')
            failures.append(nwb_file)
            continue

        try:
            check_expected_fields(
                tester,
                has_video_or_images=args.include_image_data,
                nwb_path=nwb_file,
            )
        except ValueError as error:
            print(f'Validation failed for {nwb_file}: {error}')
            failures.append(nwb_file)

    print(f'\nTested {len(nwb_files)} NWB file(s); {len(failures)} failed.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())