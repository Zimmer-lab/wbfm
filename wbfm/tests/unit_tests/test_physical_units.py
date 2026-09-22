import pytest

from wbfm.utils.external.custom_errors import IncompleteConfigFileError
from wbfm.utils.external.utils_yaml import load_config
from wbfm.pipeline.project_initialization import _validate_new_project_config
from wbfm.utils.projects.physical_units import PhysicalUnitConversion
from wbfm.utils.projects.project_config_classes import ModularProjectConfig


class DummyProjectConfig:
    def __init__(self, exposure_time):
        self.config = {
            'physical_units': {
                'exposure_time': exposure_time,
            },
        }
        self.logger = type('Logger', (), {'debug': lambda *args, **kwargs: None})()

    def get_num_slices_robust(self):
        return 22


@pytest.mark.parametrize('exposure_time', [None])
def test_missing_exposure_time_raises_incomplete_config_error(exposure_time):
    with pytest.raises(IncompleteConfigFileError, match='exposure_time'):
        PhysicalUnitConversion.load_from_config(DummyProjectConfig(exposure_time))


@pytest.mark.parametrize('exposure_time', [None, ''])
def test_new_project_rejects_missing_exposure_time(exposure_time):
    with pytest.raises(IncompleteConfigFileError, match='exposure_time'):
        _validate_new_project_config({'physical_units': {'exposure_time': exposure_time}})


def test_new_project_rejects_missing_raw_data_config():
    class MissingRawDataConfig:
        def get_remote_raw_data_config_filename(self):
            raise FileNotFoundError

    config = {'physical_units': {'exposure_time': 12}}
    with pytest.raises(IncompleteConfigFileError, match='raw data config'):
        _validate_new_project_config(config, MissingRawDataConfig())


def _make_data_and_project_folders(tmp_path, raw_data_config_contents='exposure_time: 12\n',
                                   project_exposure_time='null'):
    """Mimic the on-disk layout used by build_project_structure_from_config"""
    data_folder = tmp_path / 'data'
    channel_folder = data_folder / 'worm1_Ch0'
    channel_folder.mkdir(parents=True)
    raw_data_config_fname = data_folder / 'config.yaml'
    raw_data_config_fname.write_text(raw_data_config_contents)

    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    project_cfg_fname = project_dir / 'project_config.yaml'
    project_cfg_fname.write_text(f"red_fname: {channel_folder}\n"
                                 f"green_fname: {data_folder / 'worm1_Ch1'}\n"
                                 "physical_units:\n"
                                 f"  exposure_time: {project_exposure_time}\n")
    return channel_folder, raw_data_config_fname, project_dir, project_cfg_fname


def test_new_project_fills_exposure_time_from_raw_data_config(tmp_path):
    _, raw_data_config_fname, project_dir, project_cfg_fname = \
        _make_data_and_project_folders(tmp_path)
    project_config = ModularProjectConfig(str(project_dir))
    config = {'physical_units': {'exposure_time': None}, 'project_dir': str(project_dir)}

    returned_fname = _validate_new_project_config(config, project_config)

    assert returned_fname == str(raw_data_config_fname)
    assert config['physical_units']['exposure_time'] == 12
    assert load_config(project_cfg_fname)['physical_units']['exposure_time'] == 12


def test_new_project_does_not_overwrite_given_exposure_time(tmp_path):
    _, _, project_dir, project_cfg_fname = _make_data_and_project_folders(tmp_path,
                                                                          project_exposure_time='5')
    project_config = ModularProjectConfig(str(project_dir))
    config = {'physical_units': {'exposure_time': 5}, 'project_dir': str(project_dir)}

    _validate_new_project_config(config, project_config)

    assert config['physical_units']['exposure_time'] == 5
    assert load_config(project_cfg_fname)['physical_units']['exposure_time'] == 5


def test_new_project_raises_if_raw_data_config_has_no_exposure_time(tmp_path):
    _, _, project_dir, _ = _make_data_and_project_folders(tmp_path,
                                                          raw_data_config_contents='num_z_planes: 22\n')
    project_config = ModularProjectConfig(str(project_dir))
    config = {'physical_units': {'exposure_time': None}, 'project_dir': str(project_dir)}

    with pytest.raises(IncompleteConfigFileError, match='exposure_time'):
        _validate_new_project_config(config, project_config)
