import time
from pathlib import Path
from typing import Union
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
import numpy as np
import skimage.measure
import zarr
from imutils import MicroscopeDataReader
from matplotlib import animation, pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tifffile import tifffile
from tqdm.auto import tqdm
from matplotlib.colors import ListedColormap
from wbfm.utils.general.video_and_data_conversion.video_conversion_utils import write_numpy_as_avi
from wbfm.utils.projects.finished_project_data import ProjectData, plot_pca_projection_3d_from_project
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_plotly import rgba2float
from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, extend_short_states


def save_video_of_neuron_trace(project_data: ProjectData, neuron_name, t0=0, t1=None, fps=7,
                               line_cmap_vec: np.array=None,
                               to_save=True, shade_kwargs=None):
    """
    Save a video of the trace for a single neuron.

    Default fps is 7, which is twice the real-time speed of the video.

    Note that t0 and t1 are in frames, not seconds.

    Parameters
    ----------
    project_data
    neuron_name
    t0
    t1
    line_cmap_vec

    Returns
    -------

    """

    # Get the data for this neuron
    if shade_kwargs is None:
        shade_kwargs = {}
    df_traces = project_data.calc_default_traces()
    y = df_traces[neuron_name]

    # Get the time vector
    if t1 is None:
        t1 = project_data.num_frames

    t = project_data.x_for_plots[t0:t1]
    y = y[t0:t1]
    if line_cmap_vec is not None:
        line_cmap_vec = line_cmap_vec[t0:t1]

    # Create the filename for the video
    vis_cfg = project_data.project_config.get_visualization_config()
    fname = f"{neuron_name}_trace_{t0}-{t1}.mp4"
    fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

    # Plot the initial trace
    fig, ax = plt.subplots(dpi=200)
    ax.set_xlabel(project_data.x_label_for_plots)
    ax.set_ylabel("Fluorescence (dR/R0)")
    ax.set_title(f"Trace for {neuron_name}")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(y.min(), y.max())
    ax.plot(t, y, lw=2)

    project_data.shade_axis_using_behavior(**shade_kwargs)

    # Set up an animation to plot the trace with a vertical line moving across time
    line = ax.axvline(x=t0, color='k', lw=2)

    # def init():
    #     # line.set_data([], [])
    #     return line,

    def animate(i):
        x = t[i]
        # y = df_traces[neuron_name][x]
        line.set_xdata(x)
        if line_cmap_vec is not None:
            line.set_color(line_cmap_vec[i])
        return line,

    # Save the animation as a video
    if to_save:
        anim = animation.FuncAnimation(fig, animate, frames=t1-t0, interval=20, blit=True)
        anim.save(fname, fps=fps, extra_args=['-vcodec', 'libx264'])
        plt.close(fig)
    print(f"Saved video of trace for neuron {neuron_name} to {fname}")

    return fname


def save_video_of_behavior(project_data: ProjectData, t0=0, t1=None, fps=7,
                           to_save=True):
    """
    Save a video of the behavior, i.e. IR video

    Parameters
    ----------
    project_data
    t0
    t1

    Returns
    -------

    """

    if t1 is None:
        t1 = project_data.num_frames

    video_fname = project_data.worm_posture_class.behavior_video_btf_fname()
    # Note that this reads the entire behavioral video into memory if it wasn't chunked
    store = tifffile.imread(video_fname, aszarr=True)
    video_zarr = zarr.open(store, mode='r')
    # Subset video to be the same fps as the fluorescence
    video_zarr = video_zarr[::project_data.physical_unit_conversion.frames_per_volume, :, :]

    if to_save:
        vis_cfg = project_data.project_config.get_visualization_config()
        fname = f"behavior_{t0}-{t1}.avi"
        fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

        write_numpy_as_avi(video_zarr[t0:t1, ...], fname, fps=fps)
        print(f"Saved video of behavior to {fname}")


def save_video_of_heatmap_with_behavior(project_path: Union[str, Path], output_fname=None):
    """
    Save a video of the heatmap (bottom half) with the behavior (on top)

    Units will not be correct unless the project_config.yaml exposure_time is set correctly

    Parameters
    ----------
    project_path

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_path, verbose=0)

    if output_fname is None:
        output_fname = project_data.project_config.get_visualization_config().resolve_relative_path("heatmap_with_behavior.mp4", prepend_subfolder=True)

    # Get raw data
    df_traces = project_data.calc_default_traces(use_paper_options=True)

    behavior_parent_folder, behavior_raw_folder, behavior_output_folder, \
        background_img, background_video, btf_file = project_data.project_config.get_folders_for_behavior_pipeline()
    video = MicroscopeDataReader(btf_file, as_raw_tiff=True)
    video_array = video.dask_array.squeeze()

    frames_per_volume = project_data.physical_unit_conversion.frames_per_volume
    volumes_per_second = project_data.physical_unit_conversion.volumes_per_second
    um_per_pixel = project_data.physical_unit_conversion.zimmer_behavior_um_per_pixel_xy

    # Sort traces by pc1
    pc1_weights, _ = project_data.calc_pca_modes(n_components=1, return_pca_weights=True, use_paper_options=True)
    heatmap_data = df_traces.T.reindex(pc1_weights.sort_values(by=0, ascending=False).index)

    # Get video properties
    fps = volumes_per_second
    frame_count, width, height = video_array.shape

    # Prepare VideoWriter to save the output
    output_size = (width, height * 2)  # Double the height to accommodate the heatmap
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_fname, fourcc, fps, output_size)

    plot_kwargs = dict()
    plot_kwargs['vmin'] = np.quantile(df_traces.values, 0.1)
    plot_kwargs['vmax'] = np.quantile(df_traces.values, 0.99)

    # Initialize heatmap and line
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    heatmap = ax.imshow(heatmap_data, cmap='jet', interpolation='nearest', aspect='auto',
                        extent=[0, np.max(heatmap_data.T.index), 0, height], **plot_kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons")
    ax.set_yticks([])
    vertical_line = ax.axvline(x=0, color='white', linewidth=2)
    canvas = FigureCanvas(fig)
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(r'$\Delta R/R50$')

    plt.tight_layout()

    # Normalize the 2D array to the range [0, 255] for grayscale
    def normalize_to_grayscale(data):
        norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        return norm_data.astype(np.uint8)

    # Convert the normalized grayscale data to 3D RGB format
    def convert_to_rgb(data):
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

    # Define scale bar parameters
    scale_length_um = 100
    scale_length_px = int(scale_length_um / um_per_pixel)  # Convert mm to pixels
    scale_bar_start = (10, height - 50)  # Starting position (x, y)
    scale_bar_end = (10 + scale_length_px, height - 50)  # Ending position (x, y)

    # Loop through each frame in the video
    num_frames = heatmap_data.shape[1]
    # num_frames = 2
    subsample_rate = 1
    for frame_idx in tqdm(range(0, num_frames, subsample_rate)):
        # Get the video frame from dask array
        frame_idx_behavior = frame_idx * frames_per_volume
        grayscale_frame = normalize_to_grayscale(video_array[frame_idx_behavior].compute()).T
        rgb_frame = convert_to_rgb(grayscale_frame)

        # Update the vertical line to track the current time
        # Units should be same as the 'extent' of the ax.imshow command, which is the x axis of the heatmap
        line_position = heatmap_data.T.index[frame_idx]
        vertical_line.set_xdata([line_position])

        # Render the updated figure to a numpy array
        canvas.draw()
        heatmap_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_image = heatmap_image.reshape(canvas.get_width_height()[::-1] + (3,))
        # Convert heatmap from RGB to BGR to align with OpenCV's format
        heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR)

        # Combine the video frame and heatmap, converting to width/height like opencv expects
        combined_frame = np.vstack((rgb_frame, heatmap_image))

        # Add text label and line for scale bar
        cv2.line(combined_frame, scale_bar_start, scale_bar_end, (255, 255, 255), 2)
        cv2.putText(combined_frame, f'{scale_length_um} um', (10 + scale_length_px // 2 - 20, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Add timestamp annotation
        timestamp = f"Time: {frame_idx / fps:.2f} s"
        cv2.putText(combined_frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the combined frame to the output video
        output_video.write(combined_frame.astype(np.uint8))

    # Release everything when done
    output_video.release()


def save_video_of_trace_overlay_with_behavior(project_path: Union[str, Path], t_range=None,
                                              t_segmentation=10, neurons=None,
                                              output_fname=None, DEBUG=False):
    """
    Save a video of a segmentation max projection colored by neuron activity (bottom half) with the behavior (on top)

    Example:
    save_video_of_trace_overlay_with_behavior(project_data, t_segmentation=None, t_range=[930, 970],
                                          neurons=['VB02', 'DB01', 'RMDDR', 'SIADR', 'SMDDR', 'SIAVR', 'RMED', 'RMEV', 'SMDVR'],
                                          output_fname='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/alternative_ideas/test.mp4',
                                          DEBUG=False)

    Parameters
    ----------
    project_path
    t_segmentation - Time point of segmentation to use; if None, then plots a movie of the segmentation
    neurons - List of neuron names to plot; if None, then plots VB02 and DB01
    output_fname
    DEBUG


    Returns
    -------

    """
    if neurons is None:
        neurons = ['VB02', 'DB01']
    project_data = ProjectData.load_final_project_data_from_config(project_path, verbose=0)

    if output_fname is None:
        output_fname = project_data.project_config.get_visualization_config().resolve_relative_path("heatmap_with_behavior.mp4")

    # Get raw data
    df_traces = project_data.calc_default_traces(use_paper_options=True)

    behavior_parent_folder, behavior_raw_folder, behavior_output_folder, \
        background_img, background_video, btf_file = project_data.project_config.get_folders_for_behavior_pipeline()
    video = MicroscopeDataReader(btf_file, as_raw_tiff=True)
    video_array = video.dask_array.squeeze()

    frames_per_volume = project_data.physical_unit_conversion.frames_per_volume
    volumes_per_second = project_data.physical_unit_conversion.volumes_per_second
    um_per_pixel = project_data.physical_unit_conversion.zimmer_behavior_um_per_pixel_xy

    # Sort traces by pc1
    pc1_weights, _ = project_data.calc_pca_modes(n_components=1, return_pca_weights=True, use_paper_options=True)
    heatmap_data = df_traces.T.reindex(pc1_weights.sort_values(by=0, ascending=False).index)


    # Get the ids of the neurons in the segmentation
    df_raw_red = project_data.red_traces
    name_mapping = project_data.neuron_name_to_manual_id_mapping(confidence_threshold=0, flip_names_and_ids=True)
    neuron_ids = [name_mapping[neuron] for neuron in neurons]
    neuron_seg_labels = [df_raw_red.loc[0, (neuron_id, 'label')].astype(int) for neuron_id in neuron_ids]

    # Make sure the neurons are found at that time point

    # Get max projection of segmentation, which will be colored by neuron activity
    if t_segmentation is not None:
        raw_seg_at_time = project_data.segmentation[t_segmentation]
        # Add an offset to the labels we want, so that other neurons don't block them
        raw_seg_at_time = np.where(np.isin(raw_seg_at_time, neuron_seg_labels), raw_seg_at_time + 1000, 0)
        raw_seg_at_time = np.max(raw_seg_at_time, axis=0)
        # Remove the offset and convert to uint8
        raw_seg_at_time = np.where(raw_seg_at_time > 1000, raw_seg_at_time - 1000, 0).astype(np.uint8)
        # Get bounding box for the segmentation
        bbox = skimage.measure.regionprops((raw_seg_at_time > 0).astype(np.uint8))[0].bbox
        # Crop the segmentation to the bounding box
        raw_seg_at_time = raw_seg_at_time[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        if not all(label in raw_seg_at_time for label in neuron_seg_labels):
            raise ValueError(f"Neurons {neurons} not found in the segmentation at time {t_segmentation}")

    # Get activity for each neuron, normalized across time and converted to 8-bit
    all_neuron_activity = np.array([df_traces[neuron].values/df_traces[neuron].abs().max() for neuron in neurons])
    # Sigmoid to make the activity more visible
    all_neuron_activity = (1 / (1 + np.exp(-10*all_neuron_activity)))
    all_neuron_activity = cv2.normalize(all_neuron_activity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # New: change the range to be 128-255
    # all_neuron_activity = (all_neuron_activity/2 + 128).astype(np.uint8)

    # Get video properties
    fps = volumes_per_second
    frame_count, width, height = video_array.shape

    # Prepare VideoWriter to save the output
    output_size = (width, height * 2)  # Double the height to accommodate the heatmap
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_fname, fourcc, fps, output_size)

    plot_kwargs = dict()
    plot_kwargs['vmin'] = 0
    plot_kwargs['vmax'] = 255

    # Initialize overlay
    # fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    #
    # # ax.set_xlabel("Time (s)")
    # # ax.set_ylabel("Neurons")
    # ax.set_yticks([])
    # # vertical_line = ax.axvline(x=0, color='white', linewidth=2)
    # canvas = FigureCanvas(fig)
    # cbar = fig.colorbar(heatmap, ax=ax)
    # cbar.set_label(r'$\Delta R/R50$')

    plt.tight_layout()

    # Normalize the 2D array to the range [0, 255] for grayscale
    def normalize_to_grayscale(data):
        norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        return norm_data.astype(np.uint8)

    # Convert the normalized grayscale data to 3D RGB format
    def convert_to_rgb(data):
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

    # Define scale bar parameters
    scale_length_um = 100  # 1mm
    scale_length_px = int(scale_length_um / um_per_pixel)  # Convert mm to pixels
    scale_bar_start = (10, height - 50)  # Starting position (x, y)
    scale_bar_end = (10 + scale_length_px, height - 50)  # Ending position (x, y)

    # Loop through each frame in the video
    num_frames = heatmap_data.shape[1]
    # num_frames = 2
    subsample_rate = 1
    if t_range is not None:
        t0, t1 = t_range
    else:
        t0, t1 = 0, num_frames
    for frame_idx in tqdm(range(t0, t1, subsample_rate)):
        # Get the video frame from dask array
        frame_idx_behavior = frame_idx * frames_per_volume
        grayscale_frame = normalize_to_grayscale(video_array[frame_idx_behavior].compute()).T
        rgb_frame = convert_to_rgb(grayscale_frame)

        # Update the colors based on neuron activity
        # Get the activity for each neuron at this time point
        neuron_activity = all_neuron_activity[:, frame_idx]
        # Build 2d LUT: these neurons will be colored, others will be 255 (white); background stays 0 (black)
        neuron_lut = np.full((256, 1), 128, dtype=np.uint8)
        neuron_lut[0] = 0
        for i, label in enumerate(neuron_seg_labels):
            neuron_lut[label] = neuron_activity[i]
            if DEBUG:
                print(f"Neuron {neurons[i]} at label {label} has activity {neuron_activity[i]}")
        # Apply the LUT to the max projection of the segmentation
        if t_segmentation is None:
            # New: change segmentation
            raw_seg_at_time = project_data.segmentation[frame_idx].astype(np.uint16)
            # Add an offset to the labels we want, so that other neurons don't block them
            target_pixels = (1000 * np.isin(raw_seg_at_time, neuron_seg_labels)).astype(np.uint16)
            raw_seg_at_time += target_pixels
            raw_seg_at_time = np.max(raw_seg_at_time, axis=0)
            # Remove the offset and convert to uint8
            raw_seg_at_time -= np.max(target_pixels, axis=0)
            raw_seg_at_time = raw_seg_at_time.astype(np.uint8)
            # raw_seg_at_time = np.max(project_data.segmentation[frame_idx], axis=0).astype(np.uint8)

        heatmap_data = cv2.LUT(raw_seg_at_time, neuron_lut)
        # Flip vertically
        heatmap_data = cv2.flip(heatmap_data, 0)
        if DEBUG:
            print(np.unique(heatmap_data))
        # Actually plot
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        heatmap = ax.imshow(heatmap_data, cmap='RdBu_r',
                            interpolation='nearest', aspect='auto',
                            extent=[0, width, 0, height], **plot_kwargs)
        ax.set_yticks([])
        ax.set_xticks([])
        # Add colorbar
        # cbar = fig.colorbar(heatmap, ax=ax)
        canvas = FigureCanvas(fig)

        # Render the updated figure to a numpy array
        canvas.draw()
        heatmap_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_image = heatmap_image.reshape(canvas.get_width_height()[::-1] + (3,))
        # Convert heatmap from RGB to BGR to align with OpenCV's format
        heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR)

        # Combine the video frame and heatmap, converting to width/height like opencv expects
        combined_frame = np.vstack((rgb_frame, heatmap_image))

        # Add text label and line for scale bar
        cv2.line(combined_frame, scale_bar_start, scale_bar_end, (255, 255, 255), 2)
        cv2.putText(combined_frame, f'{scale_length_um} um', (10 + scale_length_px // 2 - 20, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Add timestamp annotation
        timestamp = f"Time: {frame_idx / fps:.2f} s"
        cv2.putText(combined_frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the combined frame to the output video
        output_video.write(combined_frame.astype(np.uint8))

        # Close the figure to prevent memory leaks
        if not DEBUG:
            plt.close(fig)

        if DEBUG and frame_idx > 200:
            break

    # Release everything when done
    output_video.release()


def save_video_of_pca_plot_with_behavior(project_path: Union[str, Path], plot_3d=False, output_fname=None, t_max=None):
    """
    Save a video of a 3d or 2d pca plot with moving dot for the current time (bottom half) with the behavior (on top)

    Units will not be correct unless the project_config.yaml exposure_time is set correctly

    Parameters
    ----------
    project_path

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_path, verbose=0)

    if output_fname is None:
        output_fname = project_data.project_config.get_visualization_config().resolve_relative_path("pca3d_with_behavior.mp4", prepend_subfolder=True)

    # Get raw data
    behavior_parent_folder, behavior_raw_folder, behavior_output_folder, \
        background_img, background_video, btf_file = project_data.project_config.get_folders_for_behavior_pipeline()
    video = MicroscopeDataReader(btf_file, as_raw_tiff=True)
    video_array = video.dask_array.squeeze()

    frames_per_volume = project_data.physical_unit_conversion.frames_per_volume
    volumes_per_second = project_data.physical_unit_conversion.volumes_per_second
    um_per_pixel = project_data.physical_unit_conversion.zimmer_behavior_um_per_pixel_xy

    # Get pca data to plot
    n_components = 3 if plot_3d else 2
    pca_modes, _ = project_data.calc_pca_modes(n_components=n_components, use_paper_options=True)

    # Get video properties
    fps = volumes_per_second
    frame_count, width, height = video_array.shape

    # Prepare VideoWriter to save the output
    output_size = (width, height * 2)  # Double the height to accommodate the heatmap
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_fname, fourcc, fps, output_size)

    # Initialize behavior-colored pca plot, and black dot for time
    plt.ion()
    fig_opt = dict(figsize=(width / 100, height / 100), dpi=100)
    fig, ax, pca_proj = plot_pca_projection_3d_from_project(project_data, fig_opt=fig_opt,
                                                            include_time_series_subplot=False)

    # Update-able point: https://stackoverflow.com/questions/61326186/how-to-animate-multiple-dots-moving-along-the-circumference-of-a-circle-in-pytho
    (time_dot,) = ax.plot(*pca_proj.iloc[0, :n_components+1], marker="o", color='black', linewidth=2)
    canvas = FigureCanvas(fig)

    plt.tight_layout()

    # Normalize the 2D array to the range [0, 255] for grayscale
    def normalize_to_grayscale(data):
        norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        return norm_data.astype(np.uint8)

    # Convert the normalized grayscale data to 3D RGB format
    def convert_to_rgb(data):
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

    # Define scale bar parameters
    scale_length_um = 100  # 1mm
    scale_length_px = int(scale_length_um / um_per_pixel)  # Convert mm to pixels
    scale_bar_start = (10, height - 50)  # Starting position (x, y)
    scale_bar_end = (10 + scale_length_px, height - 50)  # Ending position (x, y)

    # Loop through each frame in the video
    num_frames = pca_proj.shape[0] if t_max is None else t_max
    subsample_rate = 1
    for frame_idx in tqdm(range(0, num_frames, subsample_rate)):
        # Get the video frame from dask array
        frame_idx_behavior = frame_idx * frames_per_volume
        grayscale_frame = normalize_to_grayscale(video_array[frame_idx_behavior].compute()).T
        rgb_frame = convert_to_rgb(grayscale_frame)

        # Update the dot line to track the current time
        dot_position = list(pca_proj.iloc[frame_idx, :n_components+1])
        time_dot.set_data_3d([dot_position[0]], [dot_position[1]], [dot_position[2]])

        # Render the updated figure to a numpy array
        fig.canvas.draw()
        matplotlib_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        matplotlib_image = matplotlib_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # canvas.draw()
        # matplotlib_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        # matplotlib_image = matplotlib_image.reshape(canvas.get_width_height()[::-1] + (3,))
        # Convert heatmap from RGB to BGR to align with OpenCV's format
        matplotlib_image = cv2.cvtColor(matplotlib_image, cv2.COLOR_RGB2BGR)

        # Combine the video frame and heatmap, converting to width/height like opencv expects
        combined_frame = np.vstack((rgb_frame, matplotlib_image))

        # Add text label and line for scale bar
        cv2.line(combined_frame, scale_bar_start, scale_bar_end, (255, 255, 255), 2)
        cv2.putText(combined_frame, f'{scale_length_um} um', (10 + scale_length_px // 2 - 20, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Add timestamp annotation
        timestamp = f"Time: {frame_idx / fps:.2f} s"
        cv2.putText(combined_frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the combined frame to the output video
        output_video.write(combined_frame.astype(np.uint8))

    # Release everything when done
    output_video.release()


def save_video_of_heatmap_and_pca_with_behavior(project_path: Union[str, Path], output_fname=None,
                                                include_slowing=False, DEBUG=False):
    """
    Save a video of the heatmap (bottom half, stretched) and pca phase plot (top left) with behavior (top right)
    Also: an ethogram under the heatmap

    Units will not be correct unless the project_config.yaml exposure_time is set correctly

    See also: save_video_of_heatmap_with_behavior, save_video_of_pca_plot_with_behavior

    Example usage:
        fname = "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28"
        save_video_of_heatmap_and_pca_with_behavior(fname)

    Parameters
    ----------
    project_path

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_path, verbose=0)

    if output_fname is None:
        output_fname = project_data.project_config.get_visualization_config().resolve_relative_path("heatmap_and_pca_with_behavior.mp4", prepend_subfolder=True)

    # Get raw data (heatmap)
    df_traces = project_data.calc_default_traces(use_paper_options=True, interpolate_nan=True)

    behavior_parent_folder, behavior_raw_folder, behavior_output_folder, \
        background_img, background_video, btf_file = project_data.project_config.get_folders_for_behavior_pipeline()
    video = MicroscopeDataReader(btf_file, as_raw_tiff=True)
    video_array = video.dask_array.squeeze()

    frames_per_volume = project_data.physical_unit_conversion.frames_per_volume
    volumes_per_second = project_data.physical_unit_conversion.volumes_per_second
    um_per_pixel = project_data.physical_unit_conversion.zimmer_behavior_um_per_pixel_xy

    # Sort traces by pc1
    pc1_weights, _ = project_data.calc_pca_modes(n_components=1, return_pca_weights=True, use_paper_options=True)
    heatmap_data = df_traces.T.reindex(pc1_weights.sort_values(by=0, ascending=False).index)

    # Get behavior time series as integers, with custom colormap
    beh_vec_raw = project_data.worm_posture_class.beh_annotation(fluorescence_fps=True, include_slowing=include_slowing,
                                                                 include_collision=False, include_head_cast=False)
    beh_vec = beh_vec_raw.copy()
    beh_vec = BehaviorCodes.convert_to_simple_states_vector(beh_vec)
    # Extend short states (only turns)
    beh_to_extend = [BehaviorCodes.VENTRAL_TURN, BehaviorCodes.DORSAL_TURN, BehaviorCodes.PAUSE]
    # if include_slowing:
    #     beh_to_extend.append(BehaviorCodes.SLOWING)
    for behavior_code in beh_to_extend:
        binary_behavior = BehaviorCodes.vector_equality(beh_vec_raw, behavior_code)
        starts, ends = get_contiguous_blocks_from_column(binary_behavior, already_boolean=True)
        starts, ends = extend_short_states(starts, ends, len(beh_vec_raw), state_length_minimum=10)
        for s, e in zip(starts, ends):
            beh_vec.iloc[s:e] = behavior_code

    sorted_beh_codes = sorted(set(beh_vec.values))
    beh_vec = [b.value for b in beh_vec.values]
    # Sort and convert these integers to sequential values to match up with the colormap
    beh_vec = [sorted(set(beh_vec)).index(b) for b in beh_vec]
    beh_vec = np.array(beh_vec).reshape(1, -1)

    colormap_dict = BehaviorCodes.ethogram_cmap(use_plotly_style_strings=True)
    # sorted_keys = sorted(list(colormap_dict.keys()))
    ethogram_cmap = [rgba2float(colormap_dict[k]) for k in sorted_beh_codes if colormap_dict[k] is not None]
    ethogram_cmap = ListedColormap(ethogram_cmap)

    # Get video properties
    fps = 4 * volumes_per_second
    frame_count, width, height = video_array.shape
    ethogram_height = int(height / 10)

    # Prepare VideoWriter to save the output
    output_size = (width * 2, height * 2 + ethogram_height)  # Accommodate the wide heatmap and pca plot, and ethogram
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_fname, fourcc, fps, output_size)
    print(f"Saving video to {output_fname} with size {output_size}")

    plot_kwargs = dict()
    plot_kwargs['vmin'] = -0.25
    # plot_kwargs['vmin'] = 2*np.quantile(df_traces.values, 0.1)
    plot_kwargs['vmax'] = 0.75
    # plot_kwargs['vmax'] = 2*np.quantile(df_traces.values, 0.95)

    # Initialize heatmap + ethogram as subplots, and line
    fig, ax = plt.subplots(2, 1, figsize=(2 * width / 100, (height + ethogram_height) / 100), dpi=100,
                           gridspec_kw={'height_ratios': [ethogram_height, height]})
    fig.set_tight_layout(True)
    heatmap = ax[1].imshow(heatmap_data, cmap='jet', interpolation='nearest', aspect='auto',
                           extent=[0, np.max(heatmap_data.T.index), 0, height],
                           **plot_kwargs)
    # ax.set_xlabel("Time (s)")
    ax[1].set_xlabel("Time (s)", fontsize=18)
    ax[0].xaxis.set_tick_params(labelsize=18)
    ax[1].set_ylabel("Neurons", fontsize=18)
    ax[1].set_yticks([])
    vertical_line = ax[1].axvline(x=0, color='white', linewidth=4)
    # canvas_heatmap = FigureCanvas(fig)
    ax[1].set_title("Heatmap of neuronal time series", fontsize=18)


    # Initialize ethogram and line
    # fig, ax = plt.subplots(figsize=(2 * width / 100, height / 100 / 10), dpi=100)
    # fig.set_tight_layout(True)
    ethogram = ax[0].imshow(beh_vec, cmap=ethogram_cmap, interpolation='nearest', aspect='auto',
                            extent=[0, np.max(heatmap_data.T.index), 0, ethogram_height])
    ax[0].set_xlabel("")
    plt.xticks(fontsize=18)
    ax[0].set_yticks([])
    vertical_line_ethogram = ax[0].axvline(x=0, color='white', linewidth=4)

    # Ensure the x-axes are aligned by manually adding the colorbar to a specified divided axis
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(heatmap, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    # cbar = fig.colorbar(heatmap, ax=ax[0])
    cbar.set_label(r'$\Delta R/R50$', fontsize=18)
    # Do the same to the ethogram, with no colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # Make this extra axis invisible
    cax.axis('off')

    ax[1].set_xlim(ax[0].get_xlim())
    plt.tight_layout()

    canvas_ethogram_and_heatmap = FigureCanvas(fig)

    # Initialize behavior-colored pca plot, and black dot for time
    plt.ion()
    fig_opt = dict(figsize=(width / 100, height / 100), dpi=100)
    fig, ax, pca_proj = plot_pca_projection_3d_from_project(project_data, fig_opt=fig_opt,
                                                            include_time_series_subplot=False,
                                                            include_slowing=include_slowing)
    plt.legend()
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.zaxis.label.set_size(18)
    fig.set_tight_layout(True)
    ax.set_title("Phase plot of PCA modes", fontsize=24)

    # Update-able point: https://stackoverflow.com/questions/61326186/how-to-animate-multiple-dots-moving-along-the-circumference-of-a-circle-in-pytho
    dot_position = list(pca_proj.iloc[0, :4])
    (time_dot,) = ax.plot(dot_position[0], dot_position[1], dot_position[2], marker="o", color='black',
                          markersize=10)
    canvas_pca = FigureCanvas(fig)

    plt.tight_layout()

    # Normalize the 2D array to the range [0, 255] for grayscale
    def normalize_to_grayscale(data):
        norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        return norm_data.astype(np.uint8)

    # Convert the normalized grayscale data to 3D RGB format
    def convert_to_rgb(data):
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

    # Define scale bar parameters
    scale_length_um = 100  # 1mm
    scale_length_px = int(scale_length_um / um_per_pixel)  # Convert mm to pixels
    scale_bar_start = (10, height - 50)  # Starting position (x, y)
    scale_bar_end = (10 + scale_length_px, height - 50)  # Ending position (x, y)

    # Loop through each frame in the video
    if not DEBUG:
        num_frames = heatmap_data.shape[1]
        subsample_rate = 1
    else:
        num_frames = 10
        subsample_rate = 1

    for frame_idx in tqdm(range(0, num_frames, subsample_rate)):
        # Get the video frame from dask array
        frame_idx_behavior = frame_idx * frames_per_volume
        grayscale_frame = normalize_to_grayscale(video_array[frame_idx_behavior].compute()).T
        rgb_frame = convert_to_rgb(grayscale_frame)

        # Update the dot line to track the current time
        dot_position = list(pca_proj.iloc[frame_idx, :4])
        time_dot.set_data_3d([dot_position[0]], [dot_position[1]], [dot_position[2]])

        # Render the updated figure to a numpy array
        canvas_pca.draw()
        matplotlib_image = np.frombuffer(canvas_pca.tostring_rgb(), dtype=np.uint8)
        matplotlib_image = matplotlib_image.reshape(canvas_pca.get_width_height()[::-1] + (3,))
        # Convert heatmap from RGB to BGR to align with OpenCV's format
        matplotlib_image = cv2.cvtColor(matplotlib_image, cv2.COLOR_RGB2BGR)

        # Update the vertical line to track the current time
        # Units should be same as the 'extent' of the ax.imshow command, which is the x axis of the heatmap
        line_position = heatmap_data.T.index[frame_idx]
        vertical_line.set_xdata([line_position])
        vertical_line_ethogram.set_xdata([line_position])

        # Render the updated figures to a numpy array
        # canvas_heatmap.draw()
        # heatmap_image = np.frombuffer(canvas_heatmap.tostring_rgb(), dtype=np.uint8)
        # heatmap_image = heatmap_image.reshape(canvas_heatmap.get_width_height()[::-1] + (3,))
        # # Convert heatmap from RGB to BGR to align with OpenCV's format
        # heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR)
        #
        # canvas_ethogram.draw()
        # ethogram_image = np.frombuffer(canvas_ethogram.tostring_rgb(), dtype=np.uint8)
        # ethogram_image = ethogram_image.reshape(canvas_ethogram.get_width_height()[::-1] + (3,))
        # ethogram_image = cv2.cvtColor(ethogram_image, cv2.COLOR_RGB2BGR)

        canvas_ethogram_and_heatmap.draw()
        ethogram_and_heatmap_image = np.frombuffer(canvas_ethogram_and_heatmap.tostring_rgb(), dtype=np.uint8)
        ethogram_and_heatmap_image = ethogram_and_heatmap_image.reshape(canvas_ethogram_and_heatmap.get_width_height()[::-1] + (3,))
        # Convert heatmap from RGB to BGR to align with OpenCV's format
        ethogram_and_heatmap_image = cv2.cvtColor(ethogram_and_heatmap_image, cv2.COLOR_RGB2BGR)

        # Combine the video frame and heatmap, converting to width/height like opencv expects
        combined_frame = np.hstack((rgb_frame, matplotlib_image))
        combined_frame = np.vstack((combined_frame, ethogram_and_heatmap_image))
        # print(combined_frame.shape)

        # Add text label and line for scale bar
        cv2.line(combined_frame, scale_bar_start, scale_bar_end, (255, 255, 255), 2)
        cv2.putText(combined_frame, f'{scale_length_um} um', (10 + scale_length_px // 2 - 20, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Add timestamp annotation
        timestamp = f"Time: {frame_idx / fps:.2f} s"
        cv2.putText(combined_frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the combined frame to the output video
        output_video.write(combined_frame.astype(np.uint8))

    # Release everything when done
    output_video.release()


def add_scale_bar(fig, img_shape, um_per_pixel = 0.5, scale_length_um = 100, color="white", size_text=14, DEBUG=False):
    """Note that img_shape is (height, width) and can be a cropped range"""
    scale_length_px = int(scale_length_um / um_per_pixel)  # convert to pixels

    # Define where to place the scale bar (e.g. bottom right)
    bar_height = img_shape[0] - 10   # 10 px from bottom
    bar_start = img_shape[1] - scale_length_px - 10  # 10 px from right
    bar_end = img_shape[1] - 10
    if DEBUG:
        print(f"Adding scale bar of {scale_length_um} um ({scale_length_px} px) from ({bar_start}, {bar_height}) to ({bar_end}, {bar_height})")

    # Add a white line for the scale bar
    fig.add_shape(
        type="line",
        x0=bar_start, y0=bar_height,
        x1=bar_end, y1=bar_height,
        line=dict(color=color if not DEBUG else "red", width=3)
    )

    # Add text label (e.g. "100 µm")
    fig.add_annotation(
        x=(bar_start + bar_end) / 2,
        y=bar_height - 35,  # a bit above the line
        text=f"{scale_length_um} µm",
        showarrow=False,
        font=dict(color=color, size=size_text)
    )

    return fig 
