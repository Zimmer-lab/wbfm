import argparse
import logging

from wbfm.gui.utils.napari_trace_explorer import napari_trace_explorer_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    # Boolean args
    parser.add_argument('--force_tracklets_to_be_sparse', action="store_true")
    parser.add_argument('--load_tracklets', action="store_true")
    parser.add_argument('--DEBUG', default=False, help='')

    args = parser.parse_args()

    project_path = args.project_path
    force_tracklets_to_be_sparse = args.force_tracklets_to_be_sparse
    load_tracklets = args.load_tracklets
    if not force_tracklets_to_be_sparse and load_tracklets:
        logging.warning("Tracklets will not be forced to be sparse. This may cause interactivity to crash.")
    DEBUG = args.DEBUG

    print(f"Starting trace explorer GUI with options:"
          f" force_tracklets_to_be_sparse={force_tracklets_to_be_sparse}, "
          f"load_tracklets={load_tracklets}, "
          f"may take a couple minutes to load...")

    napari_trace_explorer_from_config(project_path, load_tracklets=load_tracklets,
                                      force_tracklets_to_be_sparse=force_tracklets_to_be_sparse,
                                      DEBUG=DEBUG)
