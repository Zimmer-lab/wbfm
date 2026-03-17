import argparse

from wbfm.gui.utils.utils_gui import build_gui_for_grid_plots
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.gui.utils.utils_gui_matplot import ClickableGridPlot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI for exploring folders of grid plots')
    parser.add_argument('--parent_folder', '-p', default=None,
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    parent_folder = args.parent_folder
    DEBUG = args.DEBUG

    build_gui_for_grid_plots(parent_folder, DEBUG)
