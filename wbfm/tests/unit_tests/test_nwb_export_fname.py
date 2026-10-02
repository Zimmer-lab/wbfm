import os
from pathlib import Path
from types import SimpleNamespace

from wbfm.utils.general.utils_filenames import get_sequential_filename
from wbfm.utils.nwb.utils_nwb_export import get_nwb_export_fname


def _dummy_project(shortened_name):
    return SimpleNamespace(shortened_name=shortened_name)


def test_export_fname_without_image_data(tmp_path):
    project = _dummy_project('ZIM2319_GFP_worm3-2022-12-10')
    folder = str(tmp_path)

    expected = os.path.join(folder, 'ZIM2319_GFP_worm3-2022-12-10_no_image_data.nwb')
    assert get_nwb_export_fname(project, folder, include_image_data=False) == expected


def test_export_fname_with_image_data(tmp_path):
    project = _dummy_project('ZIM2319_GFP_worm3-2022-12-10')
    folder = str(tmp_path)

    expected = os.path.join(folder, 'ZIM2319_GFP_worm3-2022-12-10.nwb')
    assert get_nwb_export_fname(project, folder, include_image_data=True) == expected


def test_export_fname_is_the_sequential_filename_base(tmp_path):
    """The skip check must target the same path the writer starts from.

    If these drift apart, already-exported projects get re-exported and silently
    duplicated under a -1/-2/... suffix instead of being skipped.
    """
    project = _dummy_project('ZIM2319_GFP_worm3-2022-12-10')
    folder = str(tmp_path)
    include_image_data = False

    export_fname = get_nwb_export_fname(project, folder, include_image_data)

    # Nothing on disk yet: the writer should use the base name as-is.
    assert get_sequential_filename(export_fname, verbose=0) == export_fname
    Path(export_fname).touch()
    assert Path(export_fname).exists()

    # Base name now exists, so an existence check on it is what makes a re-run
    # skip rather than produce another suffixed copy.
    assert os.path.exists(export_fname)
    # ...while the writer would still have made a suffixed copy had it run.
    assert get_sequential_filename(export_fname, verbose=0) != export_fname


def test_export_fname_matches_skip_check_for_date_suffix_names(tmp_path):
    """Names ending in -<int> (dates) must not be mistaken for a -N suffix."""
    project = _dummy_project('ZIM2319_GFP_worm5-2022-12-10')
    folder = str(tmp_path)

    export_fname = get_nwb_export_fname(project, folder, include_image_data=False)
    assert get_sequential_filename(export_fname, verbose=0) == export_fname

    Path(export_fname).touch()
    suffixed = get_sequential_filename(export_fname, verbose=0)
    assert os.path.basename(suffixed) == 'ZIM2319_GFP_worm5-2022-12-10_no_image_data-1.nwb'