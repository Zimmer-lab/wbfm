import pytest

from wbfm.utils.external.custom_errors import IncompleteConfigFileError
from wbfm.pipeline.project_initialization import _validate_new_project_config
from wbfm.utils.projects.physical_units import PhysicalUnitConversion


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