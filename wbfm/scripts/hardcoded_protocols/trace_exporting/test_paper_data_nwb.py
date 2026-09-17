import argparse
from pathlib import Path

DEFAULT_NWB_ROOT = Path('/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/nwb')


def check_expected_fields(tester, expect_calcium_imaging):
    required_fields = {
        'has_calcium_traces': True,
        'has_centroids': True,
        'has_segmentation_ids': True,
        'has_neuropal': False,
    }
    if expect_calcium_imaging:
        required_fields['has_calcium_imaging'] = True

    missing_fields = [
        field_name
        for field_name, expected_value in required_fields.items()
        if getattr(tester, field_name) != expected_value
    ]
    if missing_fields:
        raise ValueError(f'Unexpected NWB fields: {", ".join(missing_fields)}')


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
            tester = TestNWB(str(nwb_file))
            check_expected_fields(tester, expect_calcium_imaging=args.include_image_data)
        except Exception as error:
            print(f'Failed to load {nwb_file}: {error}')
            failures.append(nwb_file)

    print(f'\nTested {len(nwb_files)} NWB file(s); {len(failures)} failed.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())