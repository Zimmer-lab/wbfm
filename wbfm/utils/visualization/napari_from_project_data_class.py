import logging
import os
from dataclasses import dataclass
from typing import Union, List
import numpy as np
import pandas as pd
import zarr
from tqdm.auto import tqdm
import dask.array as da

from wbfm.gui.utils.utils_gui import change_viewer_time_point
from wbfm.utils.external.custom_errors import MissingAnalysisError, NoBehaviorAnnotationsError
from wbfm.utils.neuron_matching.class_frame_pair import FramePair
import napari

from wbfm.utils.general.utils_filenames import get_sequential_filename
from wbfm.utils.visualization.utils_napari import napari_labels_from_traces_dataframe, NapariPropertyHeatMapper, \
    napari_tracks_from_match_list, napari_labels_from_frames


@dataclass
class NapariLayerInitializer:

    @staticmethod
    def napari_of_single_match(project_data0,
                               time_pair=None,
                               which_matches='final_matches',
                               match_object: FramePair = None,
                               rigidly_align_volumetric_images=False,
                               min_confidence=0.0,
                               project_data1=None) -> napari.Viewer:
        """
        Creates a gui to visualize the pairs between two time points

        Can either pass the time points (time_pair) or the FramePair object direction (match_object), but not both

        Parameters
        ----------
        project_data
        time_pair
        which_matches
        match_object
        rigidly_align_volumetric_images
        min_confidence
        project_data1: optional second project, if the Frames come from different videos

        Returns
        -------

        """
        assert time_pair is None or match_object is None, "Cannot pass both time and pair object"
        # Setup
        if time_pair is None:
            time_pair = (match_object.frame0.frame_ind, match_object.frame1.frame_ind)
        if np.isscalar(time_pair):
            time_pair = (time_pair, time_pair + 1)
        if match_object is None:
            match_object: FramePair = project_data0.raw_matches[time_pair]
        if project_data1 is None:
            project_data1 = project_data0

        # Get data and optionally preprocess
        t0, t1 = time_pair
        dat0, dat1 = project_data0.red_data[t0, ...], project_data1.red_data[t1, ...]
        seg0, seg1 = project_data0.raw_segmentation[t0, ...], project_data1.raw_segmentation[t1, ...]
        match_object.load_raw_data(dat0, dat1)
        if rigidly_align_volumetric_images:
            # Ensure that both point cloud and data have rotations
            match_object.preprocess_data(force_rotation=True)
            # Load the rotated versions
            n0_zxy = match_object.pts0_preprocessed  # May be rotated
            dat0 = match_object.dat0_preprocessed
        else:
            # Keep the non-rotated versions
            n0_zxy = match_object.pts0

        n1_zxy = match_object.pts1
        raw_red_data = np.stack([dat0, dat1])
        raw_seg_data = np.stack([seg0, seg1])
        # Scale to physical units... not working with tracks
        z_to_xy_ratio = 1
        # z_to_xy_ratio = project_data0.physical_unit_conversion.z_to_xy_ratio
        # n0_zxy[0, :] = z_to_xy_ratio * n0_zxy[0, :]
        # n1_zxy[0, :] = z_to_xy_ratio * n1_zxy[0, :]

        v = napari.view_image(raw_red_data, ndisplay=3, scale=(1.0, z_to_xy_ratio, 1.0, 1.0))
        v.add_labels(raw_seg_data, scale=(1.0, z_to_xy_ratio, 1.0, 1.0), visible=False)

        # This should not remember the original time point (should place on t=0)
        df = project_data0.final_tracks.loc[[t0], :].set_index(pd.Index([0]))
        options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
        options['name'] = 'n0_final_id'
        options['n_dimensional'] = True
        v.add_points(**options)

        # This should not remember the original time point (should place on t=0)
        df = project_data1.final_tracks.loc[[t1], :].set_index(pd.Index([0]))
        options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
        options['name'] = 'n1_final_id'
        options['text']['color'] = 'green'
        options['n_dimensional'] = True
        options['symbol'] = 'x'
        v.add_points(**options)
        # v.add_points(n0_zxy, size=3, face_color='green', symbol='x', n_dimensional=True)
        # v.add_points(n1_zxy, size=3, face_color='blue', symbol='o', n_dimensional=True)

        if not isinstance(which_matches, list):
            which_matches = [which_matches]
        for these_matches in which_matches:
            list_of_matches = getattr(match_object, these_matches)
            list_of_matches = [m for m in list_of_matches if -1 not in m]
            if min_confidence > 0.0:
                list_of_matches = [m for m in list_of_matches if m[2] > min_confidence]
            if len(list_of_matches) > 0:
                all_tracks_list = napari_tracks_from_match_list(list_of_matches, n0_zxy, n1_zxy)
                v.add_tracks(all_tracks_list, head_length=2, name=these_matches)
            else:
                print(f"No valid matches found for {these_matches}")

        # Add text overlay; temporarily change the neuron locations on the frame
        original_zxy = match_object.frame0.neuron_locs
        match_object.frame0.neuron_locs = n0_zxy
        frames = {0: match_object.frame0, 1: match_object.frame1}
        options = napari_labels_from_frames(frames, num_frames=2, to_flip_zxy=False)
        options['name'] = "Neuron ID in list"
        v.add_points(**options)
        match_object.frame0.neuron_locs = original_zxy

        return v

    @staticmethod
    def add_layers_to_viewer(project_data, viewer=None, which_layers: Union[str, List[str], List[tuple]] = 'all',
                             to_remove_flyback=False, check_if_layers_already_exist=False,
                             dask_for_segmentation=True, force_all_visible=False,
                             gt_neuron_name_dict=None, heatmap_kwargs=None,
                             error_if_missing_layers=True, layer_opt=None):
        if heatmap_kwargs is None:
            heatmap_kwargs = {}
        if viewer is None:
            viewer = napari.Viewer(ndisplay=3)
        if layer_opt is None:
            layer_opt = {}

        basic_valid_layers = ['Red data', 'Green data', 'Raw segmentation',
                              'Colored segmentation', 'Neuron IDs', 'Manual IDs', 'Intermediate global IDs']
        if which_layers == 'all':
            which_layers = basic_valid_layers
        if check_if_layers_already_exist:
            # NOTE: only works if the layer names are the same as these convenience names
            new_layers = set(which_layers) - set([layer.name for layer in viewer.layers])
            which_layers = list(new_layers)

        project_data.logger.info(f"Finished loading data, trying to add following layers: {which_layers}")
        layers_actually_added = []
        xy_pixels = project_data.physical_unit_conversion.zimmer_fluroscence_um_per_pixel_xy
        z_pixels = project_data.physical_unit_conversion.zimmer_um_per_pixel_z
        z_to_xy_ratio = z_pixels / xy_pixels
        scale = (1.0, z_to_xy_ratio, 1.0, 1.0)
        if to_remove_flyback:
            raise NotImplementedError
            # clipping_list = [{'position': [2*z_to_xy_ratio, 0, 0], 'normal': [1, 0, 0], 'enabled': True}]
        else:
            clipping_list = []

        # Raw data (useful if the preprocessing doesn't work)
        if 'Raw red data' in which_layers:
            layer_name = 'Raw red data'
            p = project_data.project_config.get_preprocessing_class()
            dat = p.open_raw_data_as_4d_dask(red_not_green=True)
            viewer.add_image(dat, name=layer_name, opacity=0.5, colormap='PiYG',
                             scale=scale, experimental_clipping_planes=clipping_list)
            layers_actually_added.append(layer_name)
        if 'Raw green data' in which_layers:
            layer_name = 'Raw green data'
            p = project_data.project_config.get_preprocessing_class()
            dat = p.open_raw_data_as_4d_dask(red_not_green=False)
            viewer.add_image(dat, name=layer_name, opacity=0.5, colormap='green',
                             scale=scale, experimental_clipping_planes=clipping_list)
            layers_actually_added.append(layer_name)

        # Normal (processed) data
        if 'Red data' in which_layers:
            layer_name = 'Red data'
            contrast_high = NapariLayerInitializer._get_contrast_limits(project_data, red_not_green=True)
            viewer.add_image(project_data.red_data, name=layer_name, opacity=0.5, colormap='PiYG',
                             contrast_limits=[0, contrast_high],
                             scale=scale,
                             experimental_clipping_planes=clipping_list)
            layers_actually_added.append(layer_name)
        if 'Green data' in which_layers:
            layer_name = 'Green data'
            contrast_high = NapariLayerInitializer._get_contrast_limits(project_data, red_not_green=False)
            viewer.add_image(project_data.green_data, name=layer_name, opacity=0.5, colormap='green',
                             visible=force_all_visible,
                             contrast_limits=[0, contrast_high],
                             scale=scale,
                             experimental_clipping_planes=clipping_list)
            layers_actually_added.append(layer_name)
        if 'Raw segmentation' in which_layers:
            if project_data.raw_segmentation is not None:
                layer_name = 'Raw segmentation'
                viewer.add_labels(project_data.raw_segmentation, name=layer_name,
                                  scale=scale, opacity=0.8, visible=force_all_visible,
                                  rendering='translucent')
                layers_actually_added.append(layer_name)
                # The rendering cannot be initialized to translucent_no_depth, so we do it here
                viewer.layers[layer_name].blending = 'translucent_no_depth'
        if 'Colored segmentation' in which_layers:
            layer_name = 'Colored segmentation'
            if project_data.segmentation is None:
                project_data.logger.warning("Colored segmentation requested but not available, skipping")
            else:
                viewer.add_labels(project_data.segmentation, name=layer_name,
                                  scale=scale, opacity=0.4, visible=force_all_visible)
                layers_actually_added.append(layer_name)
                viewer.layers[layer_name].blending = 'translucent_no_depth'

        # Text overlay with automatic names
        if 'Neuron IDs' in which_layers:
            df = project_data.red_traces
            try:
                options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
                options['visible'] = force_all_visible
                viewer.add_points(**options)
                layers_actually_added.append(options['name'])
            except (KeyError, AttributeError):
                # Some nwb files may not have xyz information
                project_data.logger.warning("Could not add neuron IDs; no xyz information available")

        # Text overlay with manual IDs
        if 'Manual IDs' in which_layers:
            # This has the same information as 'GT IDs' but displays the automatic names by default
            df = project_data.red_traces
            if gt_neuron_name_dict is None:
                gt_neuron_name_dict = project_data.neuron_name_to_manual_id_mapping(confidence_threshold=0,
                                                                                    remove_unnamed_neurons=True,
                                                                                    remove_duplicates=False)

            try:
                options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio,
                                                              neuron_name_dict=gt_neuron_name_dict,
                                                              automatic_label_by_default=False)
                options['visible'] = force_all_visible
                options['name'] = 'Manual IDs'
                viewer.add_points(**options)
                layers_actually_added.append(options['name'])
            except (KeyError, AttributeError):
                # Some nwb files may not have xyz information
                project_data.logger.warning("Could not add neuron IDs; no xyz information available")

        if 'GT IDs' in which_layers:
            # Not added by default!
            df = project_data.final_tracks
            if gt_neuron_name_dict is None:
                neurons_that_are_finished = project_data.finished_neuron_names()
                gt_neuron_name_dict = {name: f"GT_{name.split('_')[1]}" for name in neurons_that_are_finished}
            options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio,
                                                          neuron_name_dict=gt_neuron_name_dict)
            options['name'] = 'GT IDs'
            options['text']['color'] = 'red'
            options['visible'] = force_all_visible
            viewer.add_points(**options)
            layers_actually_added.append(options['name'])

        if 'Intermediate global IDs' in which_layers and project_data.intermediate_global_tracks is not None:
            df = project_data.intermediate_global_tracks
            try:
                options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio, label_using_column_name=True)
                options['name'] = 'Intermediate global IDs'
                options['text']['color'] = 'green'
                options['visible'] = force_all_visible
                viewer.add_points(**options)
                layers_actually_added.append(options['name'])
            except (KeyError, AttributeError):
                project_data.logger.warning("Could not add Intermediate global IDs; no xyz information available")

        if 'Neuropal' in which_layers and project_data.neuropal_manager.data is not None:
            z_np = project_data.physical_unit_conversion.zimmer_um_per_pixel_z_neuropal
            layer_names = ['Red(mNeptune2.5)', 'White(TagRFP)', 'Green(CyOFP1)', 'Blue(mTagBFP2)']
            colormaps = ['red', 'gray', 'green', 'blue']
            for i, (name, cmap) in enumerate(zip(layer_names, colormaps)):
                dat = np.array(project_data.neuropal_manager.data[i])
                viewer.add_image(dat, name=name, colormap=cmap, contrast_limits=[dat.min(), dat.max()],
                                 visible=False, blending='additive',
                                 scale=(z_np/xy_pixels, 1.0, 1.0))
            layers_actually_added.append('Neuropal')

        if 'Neuropal segmentation' in which_layers and project_data.neuropal_manager.segmentation is not None:
            layer_name = 'Neuropal segmentation'
            z_np = project_data.physical_unit_conversion.zimmer_um_per_pixel_z_neuropal
            viewer.add_labels(project_data.neuropal_manager.segmentation, name=layer_name, visible=False,
                              scale=(z_np/xy_pixels, 1.0, 1.0), opacity=1.0)
            _layer = viewer.layers[layer_name]
            _layer.blending = 'translucent_no_depth'
            _layer.rendering = 'translucent'
            # Use the rgb colors from the mean intensity of the neuropal data
            df = project_data.neuropal_manager.segmentation_metadata.get_all_neuron_metadata_for_single_time(0,
                                                                                                     as_dataframe=True,
                                                                                                     use_mean_intensity=True)
            # Napari expects a dict with the label as key and an rgba tuple as value
            # The dataframe is multi-indexed, so we need to convert it to a dictionary
            # channel = 1 is white, which we ignore
            rgb_columns = ['mean_intensity_0', 'mean_intensity_2', 'mean_intensity_3']
            rename_column = ['raw_segmentation_id']
            collapse_df = df.loc[:, df.columns.get_level_values(1).isin(rgb_columns)]
            rename_values = df.loc[:, (slice(None), rename_column)].iloc[0].droplevel(1)
            # Collapse selected columns into lists
            collapsed = collapse_df.groupby(level=0, axis=1).agg(lambda x: list(x.values.tolist()))
            collapsed.columns = [rename_values[neuron] for neuron in collapsed.columns]
            # These are in raw pixel values, so we need to normalize them to 0 to 1
            prop_dict = {k: np.squeeze(v) for k, v in collapsed.to_dict(orient='list').items()}
            df_prop = pd.DataFrame(prop_dict)
            df_prop = df_prop.subtract(df_prop.min(axis=1), axis=0)
            # Also normalize to be 0 to 1, relative to all objects
            df_prop = df_prop.divide(df_prop.max(axis=1), axis=0)
            # Also normalize based on the total intensity across colors per object, to make the colormap work
            df_prop = df_prop.divide(df_prop.sum(axis=0), axis=1)
            prop_dict = {k: np.array(tuple(v) + (1.0, )) for k, v in df_prop.to_dict(orient='list').items()}
            _layer.color = prop_dict
            _layer.color_mode = 'direct'
            layers_actually_added.append('Neuropal segmentation')

        if 'Neuropal Ids' in which_layers and project_data.neuropal_manager.segmentation is not None:
            z_np = project_data.physical_unit_conversion.zimmer_um_per_pixel_z_neuropal

            df = project_data.neuropal_manager.segmentation_metadata.get_all_neuron_metadata_for_single_time(0,
                                                                                                             as_dataframe=True)
            try:
                np_neuron_name_dict = project_data.neuron_name_to_manual_id_mapping(confidence_threshold=0,
                                                                                    remove_unnamed_neurons=True,
                                                                                    remove_duplicates=False,
                                                                                    neuropal_subproject=True)
                options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_np/xy_pixels,
                                                              neuron_name_dict=np_neuron_name_dict,
                                                              automatic_label_by_default=False,
                                                              include_time=False,
                                                              label_using_column_name=True)
                options['visible'] = force_all_visible
                options['name'] = 'Neuropal IDs'
                # options['text']['color'] = 'red'
                viewer.add_points(**options)
                layers_actually_added.append(options['name'])
            except KeyError:
                # Some nwb files may not have xyz information
                project_data.logger.warning("Could not add neuron IDs; no xyz information available")

        # Special layers from the heatmapper class
        for layer_tuple in which_layers:
            if not isinstance(layer_tuple, tuple):
                continue
            test_neuron = project_data.neuron_names[0]
            num_frames = project_data.red_traces[test_neuron].shape[0]
            if project_data.worm_posture_class.has_full_kymograph:
                try:
                    curvature = project_data.worm_posture_class.curvature(fluorescence_fps=True).iloc[0:num_frames]
                except NoBehaviorAnnotationsError:
                    curvature = None
            else:
                curvature = None
            heat_mapper = NapariPropertyHeatMapper(project_data.red_traces, project_data.green_traces,
                                                   curvature_fluorescence_fps=curvature,
                                                   names=project_data.neuron_names)

            if 'heatmap' not in layer_tuple:
                logging.warning(f"Skipping tuple: {layer_tuple}")
                continue
            else:
                method_name = layer_tuple[1]
                layers_actually_added.append(layer_tuple)
                if len(layer_tuple) > 2:
                    layer_name = layer_tuple[2]
                else:
                    layer_name = method_name

            if heatmap_kwargs.get('t', None) is not None:
                seg = project_data.segmentation[heatmap_kwargs['t']]
                del heatmap_kwargs['t']
            else:
                seg = project_data.segmentation

            prop_dict = getattr(heat_mapper, method_name)(**heatmap_kwargs)
            # Note: this layer must be visible for the prop_dict to work correctly
            _layer_opt = dict(name=layer_name, scale=(z_to_xy_ratio, 1.0, 1.0),
                              opacity=0.4, visible=True, rendering='translucent')
            _layer_opt.update(layer_opt)
            _layer = viewer.add_labels(seg, **_layer_opt)
            _layer.blending = 'translucent_no_depth'
            _layer.color = prop_dict
            _layer.color_mode = 'direct'

        project_data.logger.debug(f"Finished adding layers {which_layers}")
        missed_layers = list(set(which_layers) - set(layers_actually_added))
        if len(missed_layers) > 0:
            message = f"Did not add unknown layers: {missed_layers}; " \
                      f"did you mean one of {basic_valid_layers}?"
            if error_if_missing_layers:
                raise MissingAnalysisError(message)
            else:
                project_data.logger.warning(message)

        return viewer

    @staticmethod
    def _get_contrast_limits(project_data, red_not_green=True):
        data = project_data.red_data if red_not_green else project_data.green_data
        if isinstance(data, zarr.Array):
            return 2 * np.max(data[0] + 1)
        elif isinstance(data, da.Array):
            return 2 * np.max(data[0].compute() + 1)


def take_screenshot_using_project(project_data, additional_layers: List[list], base_layers=None, t_target=None,
                                  close_afterwards=False,
                                  **kwargs):
    """
    Example:
    additional_layers = [[('heatmap', 'count_nonnan')],
                         [('heatmap', 'std_of_green')],
                         [('heatmap', 'std_of_red')],
                         [('heatmap', 'max_of_green')],
                         [('heatmap', 'max_of_red')],
                         [('heatmap', 'max_of_ratio')],
                         [('heatmap', 'std_of_ratio')]]
    base_layers = ['Red data', 'Neuron IDs']

    take_screenshot_using_project(project_data, base_layers=base_layers, additional_layers=additional_layers)


    See NapariLayerInitializer().add_layers_to_viewer for valid layer names

    Parameters
    ----------
    project_data
    additional_layers
    base_layers
    t_target
    kwargs

    Returns
    -------

    """
    if t_target is None:
        tracking_cfg = project_data.project_config.get_tracking_config()
        t_target = tracking_cfg.config['final_3d_tracks']['template_time_point']
    if base_layers is None:
        base_layers = ['Red data']

    viewer = NapariLayerInitializer().add_layers_to_viewer(project_data, which_layers=base_layers,
                                                           force_all_visible=True, **kwargs)
    change_viewer_time_point(viewer, t_target=t_target)
    for layer in tqdm(additional_layers):
        if not isinstance(layer, list):
            layer = [layer]
        NapariLayerInitializer().add_layers_to_viewer(project_data, viewer=viewer, which_layers=layer,
                                                      force_all_visible=True, **kwargs)

        # For the output name, assume I'm only adding one layer type over the base layer
        output_folder = project_data.project_config.get_visualization_config(True).absolute_subfolder
        layer_name = layer[0]
        if isinstance(layer_name, tuple):
            layer_name = layer_name[1]
        fname = os.path.join(output_folder, f'{layer_name}_t{t_target}.png')
        fname = get_sequential_filename(fname)
        viewer.screenshot(path=fname)

        if len(additional_layers) > 1:
            viewer.layers.remove(layer_name)

    if close_afterwards:
        viewer.close()
