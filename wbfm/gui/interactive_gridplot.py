import argparse
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.gui.utils.utils_gui_matplot import ClickableGridPlot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build interactive gridplot')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    project_path = args.project_path
    DEBUG = args.DEBUG

    print("Making grid plot, may take ~30 seconds")

    proj_dat = ProjectData.load_final_project_data_from_config(project_path, verbose=0)
    proj_dat.verbose = 0

    grid = ClickableGridPlot(proj_dat)
