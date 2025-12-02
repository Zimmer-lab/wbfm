import logging
import os

import numpy as np
from skimage import filters
from skimage.measure import regionprops
from skimage.draw import rectangle
import pickle
import dask.array as da
from tqdm.auto import tqdm


def get_bounding_box_via_gaussian_blurring(vol, raw_thresh=18, sigma=8):
    """

    Does a maximum intensity projection, blurs the image, finds a threshold, then takes the overall bounding box

    The threshold doesn't need to return a single connected object

    If no objects are found (e.g. a tracking error), this returns None

    Parameters
    ----------
    vol - 3d image data (greyscale)
    raw_thresh - noise threshold; to apply to raw data
    sigma - gaussian filter parameter

    Returns
    -------
    bbox - 2d bounding box for the non-trivial portion of the image

    """

    img_projection = np.max(vol, axis=0)
    img_projection[img_projection < raw_thresh] = 0

    img_filter = filters.gaussian(img_projection, sigma=sigma)

    thresh = filters.threshold_mean(img_filter)
    img_binary = (img_filter > thresh).astype(int)

    props = regionprops(img_binary)
    try:
        bbox = props[0].bbox
    except IndexError:
        # No objects were found
        bbox = None

    return bbox


def bbox2mask(bbox, img_shape):
    mask = np.zeros(img_shape, dtype=bool)
    rr, cc = rectangle(start=bbox[0:2], end=bbox[2:], shape=mask.shape)
    mask[rr, cc] = True

    return mask


def bbox2ind(bbox):
    """
    I always forget this order...

    returns: row_ind, col_ind
    """
    return np.arange(bbox[0], bbox[2]), np.arange(bbox[1], bbox[3])


## Optional main traces function
def calculate_bounding_boxes_from_cfg_and_save(cfg, bbox_fname, red_not_green=True):
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data(cfg, allow_hybrid_loading=True)

    video_4d = project_data.red_data if red_not_green else project_data.green_data
    if video_4d is None:
        video_4d = cfg.get_preprocessing_class().open_raw_data_as_4d_dask(red_not_green=red_not_green)
    all_bboxes = calculate_bounding_boxes_full_video(video_4d)

    try:
        with open(bbox_fname, 'wb') as f:
            pickle.dump(all_bboxes, f)
    except PermissionError:
        logging.warning(f"Permission error when saving bounding boxes at {bbox_fname}"
                        f"If this folder is in another project, this may not be a problem")

    return all_bboxes


def calculate_bounding_boxes_full_video(video_4d):

    f = get_bounding_box_via_gaussian_blurring
    all_bboxes = [f(vol) for vol in tqdm(video_4d)]

    return all_bboxes


def generate_legacy_bbox_fname(project_dir):
    bounding_box_fname = os.path.join(project_dir, '1-segmentation', 'bounding_boxes.pickle')
    return bounding_box_fname
