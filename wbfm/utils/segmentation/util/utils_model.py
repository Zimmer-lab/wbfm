"""
StarDist functions for segmentation
"""
import logging

import numpy as np
import stardist.models
from skimage.segmentation import find_boundaries
from stardist.models import StarDist3D, StarDist2D
import os
from csbdeep.utils import Path, normalize
from wbfm.utils.external.custom_errors import IncompleteConfigFileError
from wbfm.utils.general.hardcoded_paths import load_hardcoded_neural_network_paths
from wbfm.utils.general.utils_filenames import is_absolute_in_any_os


def get_stardist_model(model_name: str = 'students_and_lukas_3d_zarr',
                       folder: str = None, verbose: int = 0) -> stardist.models.StarDist3D:
    """
    Fetches the wanted StarDist model for segmenting images.
    Add new StarDist models as an alias below (incl. sd_options)

    Parameters
    ----------
    model_name : str
        Name of the wanted model. Valid models shortcuts (most modern=students_and_lukas_3d_zarr):
        ['versatile', 'lukas', 'lukas_3d_zarr', 'students_and_lukas_3d_zarr', 'lukas_3d_zarr_25',
                  'charlie', 'charlie_3d', 'charlie_3d_party']
    folder : str
        Path in which the stardist models are saved
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    model : StarDist model
        model object of the StarDist model. Can be directly used for segmenting

    """

    if verbose >= 1:
        print(f'Getting Stardist model: {model_name}')

    # First check if a full path was given
    if is_absolute_in_any_os(model_name):
        folder = os.path.dirname(model_name)
        model_name = os.path.basename(model_name)

    # all self-trained StarDist models reside in that folder. 'nt' for windows, when working locally
    if folder is None:
        try:
            # First, try to load the model using the wbfm installed config file
            path_dict = load_hardcoded_neural_network_paths()
            folder = path_dict['segmentation_paths']['model_parent_folder']
            _model_name = path_dict['segmentation_paths']['model_name']
            if _model_name != model_name:
                logging.warning(f'Model name from config file ({_model_name}) does not match the requested '
                                f'model name ({model_name})! Using requested model name.')
        except IncompleteConfigFileError:
            pass

    # Deprecated: use hardcoded paths
    if folder is None:
        logging.warning("Using hardcoded paths for stardist models! This is deprecated and should be avoided!")
        if os.name == 'nt':
            # folder = Path(r'P:/neurobiology/zimmer/wbfm/TrainedStardist')
            folder = Path(r'Z:/neurobiology/zimmer/wbfm/TrainedStardist')
        else:
            folder = Path('/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedStardist')
            folder_local = Path('/home/charles/Current_work/repos/segmentation/segmentation/notebooks/models')

    # available models' aliases
    sd_options = ['versatile', 'lukas', 'lukas_3d_zarr', 'students_and_lukas_3d_zarr', 'lukas_3d_zarr_25',
                  'charlie', 'charlie_3d', 'charlie_3d_party']

    # create aliases for each model_name
    model_name = model_name.lower()
    if model_name == 'versatile':
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    elif model_name == 'demo_3d':
        model = StarDist3D.from_pretrained('3D_demo')
    elif model_name == 'lukas':
        model = StarDist2D(None, name='stardistNiklas', basedir=folder)
    elif model_name == 'charlie':
        raise NotImplementedError
        # model = StarDist2D(None, name='stardistCharlie', basedir=folder)
    elif model_name == 'charlie_3d':
        raise NotImplementedError
        # model = StarDist3D(None, name='Charlie100-3d', basedir=folder)
    elif model_name == 'lukas_3d_zarr':
        model = StarDist3D(None, name='Lukas3d_zarr', basedir=folder)
    elif model_name == 'students_and_lukas_3d_zarr':
        model = StarDist3D(None, name='Students_and_Lukas_3d_zarr', basedir=folder)
    elif model_name == 'lukas_3d_zarr_25':
        model = StarDist3D(None, name='Lukas3d_zarr_25percentile', basedir=folder)
    elif model_name == 'lukas_3d_zarr_local':
        model = StarDist3D(None, name='Lukas3d_zarr_local', basedir=folder_local)
    elif model_name == 'charlie_3d_party':
        raise NotImplementedError
        # model = StarDist3D(None, name='Charlie100-3d-party', basedir=folder)
    else:
        raise NameError(f'No StarDist model found using {model_name}! Current models are {sd_options}')

    return model


def segment_with_stardist_2d(vol: np.ndarray,
                             model=None,
                             zero_out_borders=False,
                             verbose=0) -> np.ndarray:
    """
    Segments slices of a 3D numpy array (input) and outputs their masks.
    Best model (so far) is Lukas' self-trained 2D model
    Parameters
    ----------
    vol : 3D numpy array
        Original image array
    model : StarDist2D model object
        Object of a Stardist model, which will be used for prediction
    zero_out_borders : bool
        Whether to calculate the object borders (which may be touching) and zero them out (to prevent touching)
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    segmented_masks : 3D numpy array
        2D segmentations of slices concatenated to a 3D array. Each slice has unique values within
        a slice, but will be duplicated across slices (needs to be stitched in next step)!
    """

    if model is None:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')

    if verbose >= 1:
        print(f'Start of 2D segmentation.')

    # initialize output dimensions and other variables
    z = len(vol)
    # xy = vol.shape[1:]
    segmented_masks = np.zeros_like(vol)    # '*' = tuple unpacking
    if zero_out_borders:
        boundary = np.zeros_like(segmented_masks, dtype='bool')
    # segmented_masks = np.zeros((z, *xy))    # '*' = tuple unpacking
    axis_norm = (0, 1)
    # n_channel = 1

    # iterate over images to run stardist on single images
    for idx, plane in enumerate(vol):
        img = plane

        # normalizing images (stardist function)
        img = normalize(img, 1, 99.8, axis=axis_norm)

        # run the prediction
        labels, details = model.predict_instances(img, show_tile_progress=False)

        # save labels in 3D array for output
        segmented_masks[idx] = labels

        if verbose >= 2:
            print(f"Found {len(np.unique(labels))} neurons on slice {idx}/{z}")

        if zero_out_borders:
            # Postprocess to add separation between labels
            # From: watershed.py in 3DeeCellTracker
            labels_bd = find_boundaries(labels, connectivity=2, mode='outer', background=0)

            boundary[idx, :, :] = labels_bd

            # save labels in 3D array for output
            segmented_masks[idx] = labels

    if zero_out_borders:
        segmented_masks[boundary == 1] = 0

    return segmented_masks


def segment_with_stardist_3d(volume: np.array, model: stardist.models.StarDist3D, verbose: object = 0) -> np.ndarray:
    """
    Segments a 3D volume using stardists 3D-segmentation.
    For now, only one self-trained 3D model is available.

    Parameters
    ----------
    volume : 3D numpy array (zxy)
        3D array of volume to segment (should have a bounding box already applied)
    model : StarDist3D object
        StarDist3D model to be used for segmentation; default = Charlies first trained 3D model
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    labels : 3D numpy array
        3D array with segmented masks. Each mask should have a unique ID/value.
    """

    if verbose >= 1:
        print('Start of 3D segmentation')

    # initialize variables
    axis_norm = (0, 1, 2)
    n_channel = 1

    # normalizing images (stardist function)
    img = normalize(volume, 1, 99.8, axis=axis_norm)

    # run the prediction
    labels, details = model.predict_instances(img, show_tile_progress=False)

    return labels
