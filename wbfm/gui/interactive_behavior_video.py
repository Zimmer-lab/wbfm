import argparse
from wbfm.gui.utils.napari_trace_explorer import napari_behavior_explorer_from_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build behavior GUI with a project')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    parser.add_argument('--behavior_fps', '-b', action='store_true',
                        help='If set, use the behavior fps for the behavior kymograph')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    behavior_fps = args.behavior_fps
    DEBUG = args.DEBUG

    print("Starting behavior explorer GUI, may take up to a minute to load...")

    napari_behavior_explorer_from_config(project_path, fluorescence_fps=not behavior_fps, DEBUG=DEBUG)
