import pytest

from wbfm.utils.external.custom_errors import IncompleteConfigFileError
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