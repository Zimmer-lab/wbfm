import os
from pathlib import Path
from types import SimpleNamespace

import os
import sys

from wbfm.utils.general.utils_filenames import get_sequential_filename
from wbfm.utils.nwb.utils_nwb_export import get_nwb_export_fname, get_nwb_export_fname_from_parts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts',
                                'hardcoded_protocols', 'trace_exporting'))
from export_paper_data_as_nwb import filter_existing_tasks, read_taskfile_entry, write_taskfile


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


def test_export_fname_from_parts_matches_object_version(tmp_path):
    """The string-only helper must agree with the project-object version."""
    project = _dummy_project('ZIM2319_GFP_worm3-2022-12-10')
    folder = str(tmp_path)

    for include_image_data in (False, True):
        assert get_nwb_export_fname_from_parts(
            project.shortened_name, folder, include_image_data
        ) == get_nwb_export_fname(project, folder, include_image_data)


def test_filter_existing_tasks_matches_submitter_and_worker(tmp_path):
    """The submitter and the --only_index worker must see the same list.

    Regression test: the submitter used to filter while the worker indexed
    the unfiltered list, so array tasks skipped (already-done datasets) while
    the actually-missing ones never ran.
    """
    folder = str(tmp_path)
    names = [f'worm{i}-2022-12-10' for i in range(5)]
    tasks = [('gfp', n, f'/fake/{n}/project_config.yaml', folder) for n in names]
    # worm1 and worm3 already exported
    for n in ('worm1-2022-12-10', 'worm3-2022-12-10'):
        Path(os.path.join(folder, f'{n}_no_image_data.nwb')).touch()

    remaining, skipped = filter_existing_tasks(tasks, include_image_data=False, verbose=False)

    assert [n for _, n, _, _ in remaining] == ['worm0-2022-12-10', 'worm2-2022-12-10',
                                               'worm4-2022-12-10']
    assert sorted(skipped) == ['worm1-2022-12-10', 'worm3-2022-12-10']
    # Index stability: worker --only_index i must resolve to the same dataset
    # the submitter counted at position i.
    assert remaining[0][1] == 'worm0-2022-12-10'
    assert remaining[2][1] == 'worm4-2022-12-10'


def test_taskfile_roundtrip_pins_index_to_dataset(tmp_path):
    """Array indexes must resolve via the frozen taskfile, not re-filtering.

    Regression test: workers used to re-enumerate + re-filter at start, so
    late-starting workers saw shrunken lists -> tail tasks no-op'd on wrong
    indexes while missing datasets never ran (and two tasks once raced on
    the same dataset, producing a -1 suffixed duplicate).
    """
    folder = str(tmp_path)
    tasks = [('gfp', f'worm{i}-2022-12-10', f'/fake/{i}.yaml', folder) for i in range(3)]
    taskfile = str(tmp_path / 'tasks.txt')
    write_taskfile(taskfile, tasks)

    assert read_taskfile_entry(taskfile, 0) == ('gfp', 'worm0-2022-12-10', '/fake/0.yaml')
    assert read_taskfile_entry(taskfile, 2) == ('gfp', 'worm2-2022-12-10', '/fake/2.yaml')
    assert read_taskfile_entry(taskfile, 3) is None
    assert read_taskfile_entry(taskfile, -1) is None


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