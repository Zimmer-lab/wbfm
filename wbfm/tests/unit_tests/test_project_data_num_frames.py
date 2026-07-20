from wbfm.utils.projects.finished_project_data import ProjectData


class DummyProjectConfig:
    has_valid_self_path = True

    def __init__(self):
        self.logger = type('Logger', (), {'debug': lambda *args, **kwargs: None, 'warning': lambda *args, **kwargs: None})()

    def get_tracking_config(self):
        return None

    def get_training_config(self):
        return None

    def get_preprocessing_class(self):
        return None

    @property
    def num_frames(self):
        raise AttributeError

    def get_num_frames_robust(self):
        raise FileNotFoundError


def test_num_frames_falls_back_to_behavior_video_without_recursing():
    project_data = ProjectData(project_dir='.', project_config=DummyProjectConfig())
    project_data.red_data = None
    project_data.project_config = DummyProjectConfig()

    class DummyBehaviorVideo:
        shape = (123,)

    class DummyPhysicalUnitConversion:
        frames_per_volume = 3

    class DummyPosture:
        def __init__(self):
            self.raw_behavior_video = DummyBehaviorVideo()

    project_data.physical_unit_conversion = DummyPhysicalUnitConversion()
    project_data.worm_posture_class = DummyPosture()

    num_frames = project_data.num_frames

    assert num_frames == 41
