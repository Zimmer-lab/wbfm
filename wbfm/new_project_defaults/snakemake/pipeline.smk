import logging
import os

from ruamel.yaml import YAML
from wbfm.utils.external.custom_errors import NoBehaviorDataError, RawDataFormatError
from wbfm.utils.projects.finished_project_data import ProjectData
import snakemake

from wbfm.utils.general.hardcoded_paths import load_hardcoded_neural_network_paths


configfile: "snakemake_config.yaml"

# Determine the project folder (the parent of the folder containing the Snakefile)
# NOTE: this is an undocumented feature, and may not work for other versions (this is 7.32)
project_dir = os.path.dirname(snakemake.workflow.workflow.basedir)
logging.info("Detected project folder: ", project_dir)
project_cfg_fname = os.path.join(project_dir,"project_config.yaml")

if not snakemake.__version__.startswith("7.32"):
    logging.warning(f"Note: this pipeline is only tested on snakemake version 7.32.X, but found {snakemake.__version__}")

# Load the folders needed for the behavioral part of the pipeline
project_data = ProjectData.load_final_project_data(project_cfg_fname, allow_hybrid_loading=True)
project_config = project_data.project_config
output_visualization_directory = project_config.get_visualization_config().absolute_subfolder

try:
    raw_data_dir, raw_data_subfolder, output_behavior_dir, background_img, background_video, behavior_btf = \
        project_config.get_folders_for_behavior_pipeline()
except (NoBehaviorDataError, RawDataFormatError, FileNotFoundError) as e:
    # Note: these strings can't be empty, otherwise snakemake can have weird issues
    logging.warning(f"No behavior data found, behavior will not run. Only 'traces' can be processed. "
                    f"Error message: {e}")
    raw_data_dir = "NOTFOUND_raw_data_dir"
    output_behavior_dir = "NOTFOUND_output_behavior_dir"
    background_img = "NOTFOUND_background_img"
    background_video = "NOTFOUND_background_video"
    behavior_btf = "NOTFOUND_behavior_btf"
    raw_data_subfolder = "NOTFOUND_raw_data_subfolder"

# Also get the raw data config file (if it exists)
try:
    raw_data_config_fname = project_config.get_raw_data_config().absolute_self_path
    if raw_data_config_fname is None:
        raise FileNotFoundError
    with open(raw_data_config_fname, 'r') as f:
        worm_config = YAML().load(f)
except FileNotFoundError:
    raw_data_config_fname = "NOTFOUND_raw_data_config_fname"
    worm_config = dict(ventral='NOTFOUND')

# Additionally update the paths used for the behavior pipeline (note that this needs to be loaded even if behavior is not run)
hardcoded_paths = load_hardcoded_neural_network_paths()
# Update the config with the hardcoded paths, but keep the original config if any matching keys are found
_config = hardcoded_paths["behavior_paths"]
_config.update(config)
config = _config
print(f"Loaded snakemake config file with parameters: {config}")


def _run_helper(script_name, project_path, **kwargs):
    """Runs a script with a given name that can't be imported directly (e.g. because it starts with a number)"""
    import importlib
    print("Running script: ", script_name)
    _module = importlib.import_module(f"wbfm.scripts.{script_name}")
    config_updates = dict(project_path=project_path)
    config_updates.update(kwargs)
    _module.ex.run(config_updates=config_updates)


def _cleanup_helper(output_path):
    """Uses the snakemake defined temporary function to clean up intermediate files, based on a flag"""
    if config['delete_intermediate_files']:
        return temporary(output_path)
    else:
        return output_path

# See this for branching function: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#snakefiles-branch-function
# Note that branch was only added in version 8+ of snakemake, which requires python 3.9 for 8.0 and 3.11 for others :(

# Alternate: function to split the branches
# Note that this is needed instead of ruleorder because the output files are different
# https://stackoverflow.com/questions/40510347/can-snakemake-avoid-ambiguity-when-two-different-rule-paths-can-generate-a-given
def _choose_tracker():
    if config.get('use_barlow_tracker', False):
        return os.path.join(project_dir, "3-tracking/barlow_tracker/df_barlow_tracks.h5")
    else:
        return os.path.join(project_dir, "3-tracking/postprocessing/combined_3d_tracks.h5")

# For skipping some steps, use ruleorder
if project_data.check_segmentation():
    print("Detected completed segmentation; allowing rules that skip segmentation")
    ruleorder: alt_build_frame_objects > build_frame_objects
    ruleorder: alt_barlow_embedding > barlow_embedding
else:
    ruleorder: build_frame_objects > alt_build_frame_objects
    ruleorder: barlow_embedding > alt_barlow_embedding

if project_data.check_preprocessed_data():
    print("Detected completed preprocessing; allowing rules that skip preprocessing")
    ruleorder: alt_preprocessing > preprocessing
    # Further check that metadata exists
    if project_data.check_segmentation_metadata():
        ruleorder: alt_segmentation > segmentation
    else:
        ruleorder: segmentation > alt_segmentation
else:
    ruleorder: preprocessing > alt_preprocessing
    ruleorder: segmentation > alt_segmentation

#
# Snakemake for overall targets (either with or without behavior)
#

# By default, wbfm projects will run only traces
# This is important for immobilized worms, which don't have behavior
rule traces:
    input:
        traces=os.path.join(project_dir, "4-traces/green_traces.h5")

# Many projects will also want to run behavior
rule traces_and_behavior:
    input:
        traces=os.path.join(project_dir, "4-traces/green_traces.h5"),
        #trace_summary=os.path.join(output_visualization_directory, "heatmap_with_behavior.mp4"),
        beh_figure=f"{output_behavior_dir}/behavioral_summary_figure.pdf",
        beh_hilbert=f"{output_behavior_dir}/hilbert_inst_amplitude.csv"


rule behavior:
    input:
        beh_figure= f"{output_behavior_dir}/behavioral_summary_figure.pdf",
        beh_hilbert=f"{output_behavior_dir}/hilbert_inst_amplitude.csv"

#
# Snakemake for traces
#

rule preprocessing:
    input:
        cfg=project_cfg_fname
    output:
        os.path.join(project_dir, "dat/bounding_boxes.pickle")
    run:
        try:
            shell("ml p7zip")  # Needed on the cluster as of May 2025
        except:
            # Then we are running locally, so ignore
            pass
        _run_helper("0b-preprocess_working_copy_of_data", str(input.cfg))

# Already have the preprocessed data, but need to make sure that the metadata files are present
rule alt_preprocessing:
    input:
        cfg=project_cfg_fname
    output:
        os.path.join(project_dir, "dat/bounding_boxes.pickle")
    run:
        _run_helper("pipeline_alternate.0+build_bounding_boxes", str(input.cfg))

#
# Segmentation
#
rule segmentation:
    input:
        cfg=project_cfg_fname,
        files=os.path.join(project_dir, "dat/bounding_boxes.pickle")
    output:
        metadata=os.path.join(project_dir, "1-segmentation/metadata.pickle"),
        masks=directory(os.path.join(project_dir, "1-segmentation/masks.zarr"))
    threads: 56
    run:
        _run_helper("1-segment_video", str(input.cfg))

# No input version, i.e. remote preprocessing
rule alt_segmentation:
    input: cfg=project_cfg_fname
    output:
        metadata=os.path.join(project_dir, "1-segmentation/metadata.pickle"),
        masks=directory(os.path.join(project_dir, "1-segmentation/masks.zarr"))
    threads: 56
    run:
        _run_helper("1-segment_video", str(input.cfg))
    

#
# Tracklets
#
rule build_frame_objects:
    input:
        cfg=project_cfg_fname,
        masks=os.path.join(project_dir, "1-segmentation/masks.zarr"),
        metadata=os.path.join(project_dir, "1-segmentation/metadata.pickle")
    output:
        os.path.join(project_dir, "2-training_data/raw/frame_dat.pickle")
    threads: 56
    run:
        _run_helper("2a-build_frame_objects", str(input.cfg))


rule match_frame_pairs:
    input:
        cfg=project_cfg_fname,
        frames=os.path.join(project_dir, "2-training_data/raw/frame_dat.pickle")
    output:
        matches=os.path.join(project_dir, "2-training_data/raw/match_dat.pickle")
    threads: 56
    run:
        _run_helper("2b-match_adjacent_volumes", str(input.cfg))


rule postprocess_matches_to_tracklets:
    input:
        cfg=project_cfg_fname,
        frames=os.path.join(project_dir, "2-training_data/raw/frame_dat.pickle"),
        matches=os.path.join(project_dir, "2-training_data/raw/match_dat.pickle")
    output:
        tracklets=os.path.join(project_dir, "2-training_data/all_tracklets.pickle"),
        clust_df_dat=os.path.join(project_dir, "2-training_data/raw/clust_df_dat.pickle"),
    threads: 8
    run:
        _run_helper("2c-postprocess_matches_to_tracklets", str(input.cfg))


# No input version, e.g. from nwb or remote segmentation
rule alt_build_frame_objects:
    input: cfg=project_cfg_fname
    output:
        os.path.join(project_dir, "2-training_data/raw/frame_dat.pickle")
    threads: 56
    run:
        _run_helper("2a-build_frame_objects", str(input.cfg))

#
# Tracking
#
rule tracking:
    input:
        cfg=project_cfg_fname,
        frames=os.path.join(project_dir, "2-training_data/raw/frame_dat.pickle"),
    output:
        tracks_global=os.path.join(project_dir, "3-tracking/postprocessing/df_tracks_postprocessed.h5"),
    threads: 48
    run:
        _run_helper("3a-track_time_independent", str(input.cfg))

rule combine_tracking_and_tracklets:
    input:
        cfg=project_cfg_fname,
        tracks_global=os.path.join(project_dir, "3-tracking/postprocessing/df_tracks_postprocessed.h5"),
        tracklets=os.path.join(project_dir, "2-training_data/all_tracklets.pickle"),
    output:
        tracks_combined=os.path.join(project_dir, "3-tracking/postprocessing/combined_3d_tracks.h5"),
        tracks_metadata=os.path.join(project_dir, "3-tracking/global2tracklet.pickle"),
    threads: 8
    run:
        _run_helper("3b-match_tracklets_and_tracks_using_neuron_initialization", str(input.cfg))

# Alternate tracker that doesn't need tracklets

rule barlow_embedding:
    input:
        cfg=project_cfg_fname,
        metadata=os.path.join(project_dir, "1-segmentation/metadata.pickle"),
    output:
        embedding=os.path.join(project_dir, "3-tracking/barlow_tracker/worm_tracker_barlow.pickle"),
    threads: 48
    run:
        _run_helper("pipeline_alternate.3-embed_using_barlow", str(input.cfg),
            model_fname=config["barlow_model_path"])

rule barlow_tracking:
    input:
        cfg=project_cfg_fname,
        embedding=os.path.join(project_dir, "3-tracking/barlow_tracker/worm_tracker_barlow.pickle"),
    output:
        tracks_global=os.path.join(project_dir, "3-tracking/barlow_tracker/df_barlow_tracks.h5"),
    threads: 48
    run:
        _run_helper("pipeline_alternate.3-track_using_barlow", str(input.cfg),
            model_fname=config["barlow_model_path"])

# No input version, e.g. from nwb or remote segmentation
rule alt_barlow_embedding:
    input: cfg=project_cfg_fname
    output:
        embedding=os.path.join(project_dir, "3-tracking/barlow_tracker/worm_tracker_barlow.pickle"),
    threads: 48
    run:
        _run_helper("pipeline_alternate.3-embed_using_barlow", str(input.cfg),
            model_fname=config["barlow_model_path"])

#
# Traces
#
rule extract_full_traces:
    input:
        cfg=project_cfg_fname,
        tracks_combined=_choose_tracker(),
    output:
        os.path.join(project_dir, "4-traces/all_matches.pickle"),
        os.path.join(project_dir, "4-traces/red_traces.h5"),
        os.path.join(project_dir, "4-traces/green_traces.h5"),
        masks=os.path.join(project_dir, "4-traces/reindexed_masks.zarr.zip")
    threads: 56
    run:
        try:
            shell("ml p7zip")  # Needed on the cluster as of May 2025
        except:
            # Then we are running locally, so ignore
            pass
        _run_helper("4-make_final_traces", str(input.cfg))


# TODO: FINISH
# rule make_grid_plots_with_behavior:
#     input:
#         cfg=project_cfg_fname,
#         masks=os.path.join(project_dir, "4-traces/reindexed_masks.zarr.zip")
#     output:
#         os.path.join(project_dir, "4-traces/all_matches.pickle"),
#         os.path.join(project_dir, "4-traces/red_traces.h5"),
#         os.path.join(project_dir, "4-traces/green_traces.h5"),
#     threads: 56
#     run:
#         _run_helper("make_default_summary_plots_using_config", str(input.cfg))


#
# Behavioral analysis (kymographs)
#

rule z_project_background:
    input:
        background_video = background_video
    output:
        # New: put the background image in the output folder, and make it temporary
        background_img = _cleanup_helper(background_img)
    run:
        from imutils.src.imfunctions import stack_z_projection

        stack_z_projection(
            str(input.background_video),
            str(output.background_img),
            'mean',
            'uint8',
            0,
        )

rule subtract_background:
    input:
        ndtiff_subfolder = behavior_btf if os.path.exists(behavior_btf) else raw_data_subfolder,
        background_img = background_img
    params:
        do_inverse = config["do_inverse"]
    output:
        background_subtracted_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            "stack_subtract_background",
            '-i', str(input.ndtiff_subfolder),
            '-o', str(output.background_subtracted_img),
            '-bg', str(input.background_img),
            '-invert', str(params.do_inverse),
        ])

rule normalize_img:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted.btf"
    params:
        alpha = config["alpha"],
        beta = config["beta"]
    output:
        normalised_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            "stack_normalise",
            '-i', str(input.input_img),
            '-o', str(output.normalised_img),
            '-a', str(params.alpha),
            '-b', str(params.beta),
        ])

rule worm_unet:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.btf"
    params:
        weights_path = config["main_unet_model"],
    output:
        worm_unet_prediction = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            "unet_segmentation_stack",
            '-i', str(input.input_img),
            '-o', str(output.worm_unet_prediction),
            '-w', str(params.weights_path),
        ])

rule sam2_segment:
    input:
        ndtiff_subfolder = behavior_btf if os.path.exists(behavior_btf) else raw_data_subfolder,
        dlc_csv=f"{output_behavior_dir}/raw_stack_dlc.csv"
    output:
        output_file=_cleanup_helper(f"{output_behavior_dir}/raw_stack_mask.btf"),
    params:
        column_names=["pharynx"],
        model_path=config["sam2_model"],
        sam2_conda_env_name=config["sam2_conda_env_name"],
        batch_size=300
    shell:
        """
        # I started getting an error with the xml_catalog_files_libxml2 variable, so check if it is set
        if [ -z "${{xml_catalog_files_libxml2:-}}" ]; then
            export xml_catalog_files_libxml2=""
        fi

        # Enable CuDNN backend for faster attention
        export TORCH_CUDNN_SDPA_ENABLED=1

        module load cuda-toolkit/12.9.0

        # Activate the environment and the correct cuda
        source /lisc/app/conda/miniforge3/bin/activate {params.sam2_conda_env_name}

        # Run the script directly without temp directory overhead
        python -c "from SAM2_snakemake_scripts.sam2_video_processing_miscroscope_data_loader import main; main(['-tiff_path', '{input.ndtiff_subfolder}', '-output_file_path', '{output.output_file}', '-DLC_csv_file_path', '{input.dlc_csv}', '-column_names', '{params.column_names}', '-SAM2_path', '{params.model_path}', '--batch_size', '{params.batch_size}', '--device', '${{CUDA_VISIBLE_DEVICES:-0}}'])"
        """

rule binarize:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented.btf"
    params:
        threshold = config["threshold"],
        max_value = config["max_value"]
    output:
        binary_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            "stack_make_binary",
            '-i', str(input.input_img),
            '-o', str(output.binary_img),
            '-th', str(params.threshold),
            '-max_val', str(params.max_value),
        ])

rule coil_unet:
    input:
        binary_input_img = f"{output_behavior_dir}/raw_stack_mask.btf",  # From the SAM2 segmentation
        raw_input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.btf"  # Does not need to match the other segmentation; needs to match the training of the coil unet
    params:
        weights_path= config["coiled_shape_unet_model"]
    output:
        coil_unet_prediction = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            "unet_segmentation_contours_with_children",
            '-bi', str(input.binary_input_img),
            '-ri', str(input.raw_input_img),
            '-o', str(output.coil_unet_prediction),
            '-w', str(params.weights_path),
        ])

rule binarize_coil:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented.btf"
    params:
        threshold = config["coil_threshold"], # 240
        max_value = config["coil_new_value"] # 255
    output:
        binary_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented_mask.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            "stack_make_binary",
            '-i', str(input.input_img),
            '-o', str(output.binary_img),
            '-th', str(params.threshold),
            '-max_val', str(params.max_value),
        ])

rule tiff2avi:
    input:
        input_img = behavior_btf if os.path.exists(behavior_btf) else raw_data_subfolder
    params:
        fourcc = config["fourcc"], #"0",
        fps = config["fps"] # "167"
    output:
        avi = _cleanup_helper(f"{output_behavior_dir}/raw_stack.avi")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            "tiff2avi",
            '-i', str(input.input_img),
            '-o',str(output.avi),
            '-fourcc', str(params.fourcc),
            '-fps', str(params.fps),
        ])

rule dlc_analyze_videos:
    input:
        # Will save the output in the same folder as the input by default
        input_avi = f"{output_behavior_dir}/raw_stack.avi"
    params:
        dlc_model_configfile_path = config["head_tail_dlc_project"],
        dlc_conda_env = config["dlc_conda_env_name_only_dlc"]
    output:
        hdf5_file = f"{output_behavior_dir}/raw_stack_dlc.h5",
        csv_file = f"{output_behavior_dir}/raw_stack_dlc.csv"
    shell:
        """
        # I started getting an error with the xml_catalog_files_libxml2 variable, so check if it is set
        if [ -z "${{xml_catalog_files_libxml2:-}}" ]; then
            #echo "Warning: xml_catalog_files_libxml2 is not set, setting it to /lisc/app/conda/miniforge3/etc/xml/catalog"
            export xml_catalog_files_libxml2=""
        fi 
        
        source /lisc/app/conda/miniforge3/bin/activate {params.dlc_conda_env}
        module load cuda-toolkit/12.9.0
        # Also rename the output file to the expected name
        # We don't actually know the name without querying deeplabcut, so just rename it
        python -c "import deeplabcut, os; fname = deeplabcut.analyze_videos('{params.dlc_model_configfile_path}', '{input.input_avi}', videotype='avi', gputouse=${{CUDA_VISIBLE_DEVICES:-0}}, save_as_csv=True); print('Produced raw files with name: ' + fname); os.rename(f'{output_behavior_dir}/raw_stack'+fname+'.h5', '{output_behavior_dir}/raw_stack_dlc.h5'); os.rename(f'{output_behavior_dir}/raw_stack'+fname+'.csv', '{output_behavior_dir}/raw_stack_dlc.csv')"
        """

rule create_centerline:
    input:
        input_binary_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented_mask.btf",  # From the coil unet, not directly from the SAM2 segmentation
        hdf5_file = f"{output_behavior_dir}/raw_stack_dlc.h5"

    params:
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
        number_of_neighbours = "1",
        nose = config['nose'],  # Should actually be the nose
        tail = config['tail'],
        num_splines = config['num_splines'],
        fill_with_DLC = "1"
    output :
        output_skel_X = f"{output_behavior_dir}/skeleton_skeleton_X_coords.csv",
        output_skel_Y = f"{output_behavior_dir}/skeleton_skeleton_Y_coords.csv",
        output_spline_K = f"{output_behavior_dir}/skeleton_spline_K.csv",
        output_spline_X = f"{output_behavior_dir}/skeleton_spline_X_coords.csv",
        output_spline_Y = f"{output_behavior_dir}/skeleton_spline_Y_coords.csv",
        corrected_head = f"{output_behavior_dir}/skeleton_corrected_head_coords.csv",
        corrected_tail = f"{output_behavior_dir}/skeleton_corrected_tail_coords.csv"
    run:
        from centerline_behavior_annotation.centerline.dev import head_and_tail

        head_and_tail.main([
            '-i', str(input.input_binary_img),
            '-h5', str(input.hdf5_file),
            '-o', str(params.output_path),
            '-nose', str(params.nose),
            '-tail', str(params.tail),
            '-num_splines', str(params.num_splines),
            '-n', str(params.number_of_neighbours),
            '-dlc', str(params.fill_with_DLC),
        ])


# Benjamin-style rule that directly reads the config file
rule invert_curvature_sign:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed.csv"
    params:
        ventral = worm_config['ventral'],
    output:
        spline_K_signed = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed.csv"
    run:
        from centerline_behavior_annotation.curvature.src import invert_curvature_sign

        # Call the invert_curvature_sign function with the correct parameters
        invert_curvature_sign.main_benjamin([
            '--spline_K_path', str(input.spline_K),
            '--ventral', str(params.ventral),
            '--output_file_path', str(output.spline_K_signed)
        ])

rule average_kymogram:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K_signed.csv"
    params:
        #rolling_mean_type =,
        window = config['averaging_window']
    output:
        spline_K_avg = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv"
    run:
        import pandas as pd
        df=pd.read_csv(input.spline_K, index_col=None, header=None)
        df=df.rolling(window=params.window, center=True, min_periods=1).mean()
        df.to_csv(output.spline_K_avg, header=None, index=None)
        print('end of python script')

rule average_xy_coords:
    input:
        spline_X= f"{output_behavior_dir}/skeleton_spline_X_coords.csv",
        spline_Y= f"{output_behavior_dir}/skeleton_spline_Y_coords.csv",
    params:
        #rolling_mean_type =,
        window = config['averaging_window']
    output:
        spline_X_avg= f"{output_behavior_dir}/skeleton_spline_X_coords_avg.csv",
        spline_Y_avg= f"{output_behavior_dir}/skeleton_spline_Y_coords_avg.csv",
    run:
        import pandas as pd

        df=pd.read_csv(input.spline_X, index_col=None, header=None)
        df=df.rolling(window=params.window, center=True, min_periods=1).mean()
        df.to_csv(output.spline_X_avg, header=None, index=None)

        df=pd.read_csv(input.spline_Y, index_col=None, header=None)
        df=df.rolling(window=params.window, center=True, min_periods=1).mean()
        df.to_csv(output.spline_Y_avg, header=None, index=None)

rule hilbert_transform_on_kymogram:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed.csv",
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
    params:
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
        fs = config["sampling_frequency"],
        window = config["hilbert_averaging_window"]
    output:
        # wont be created because they the outputs do not have the {sample} root
        hilbert_regenerated_carrier = f"{output_behavior_dir}/hilbert_regenerated_carrier.csv",
        hilbert_inst_freq = f"{output_behavior_dir}/hilbert_inst_freq.csv",
        hilbert_inst_phase = f"{output_behavior_dir}/hilbert_inst_phase.csv",
        hilbert_inst_amplitude = f"{output_behavior_dir}/hilbert_inst_amplitude.csv"

    #This $DIR only goes one time up
    run:
        from centerline_behavior_annotation.behavior_analysis.src import hilbert_transform

        hilbert_transform.main([
            '-i', str(params.output_path),
            '-kp', str(input.spline_K),
            '-fs', str(params.fs),
            '-w', str(params.window),
        ])

rule fast_fourier_transform:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed",
    params:
        # project_folder
        sampling_frequency=config["sampling_frequency"],
        window = config["fft_averaging_window"],
        output_path = f"{output_behavior_dir}/",  # Ulises' functions expect the final slash

    output:
        y_axis_file = f"{output_behavior_dir}/fft_y_axis.csv", #not correct ?
        xf_file = f"{output_behavior_dir}/fft_xf.csv"
    #This $DIR only goes one time up
    run:
        from centerline_behavior_annotation.centerline.dev import fourier_functions

        fourier_functions.main([
            '-i', str(params.output_path),
            '-kp', str(input.spline_K),
            '-fps', str(params.sampling_frequency),
            '-w', str(params.window),
        ])

rule reformat_skeleton_files:
    input:
        spline_K= f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed", #should have signed spline_K
        spline_X= f"{output_behavior_dir}/skeleton_spline_X_coords_equi_dist_segment.csv",
        spline_Y= f"{output_behavior_dir}/skeleton_spline_Y_coords_equi_dist_segment.csv",
        #spline_list = ["{sample}_spline_K.csv", "{sample}_spline_X_coords.csv", "{sample}_spline_Y_coords.csv",]

    output:
        merged_spline_file = f"{output_behavior_dir}/skeleton_merged_spline_data_avg.csv",

    run:
        from centerline_behavior_annotation.centerline.dev import reformat_skeleton_files

        reformat_skeleton_files.main([
            '-i_K', str(input.spline_K),
            '-i_X', str(input.spline_X),
            '-i_Y', str(input.spline_Y),
            '-o', str(output.merged_spline_file),
        ])

rule annotate_behaviour:
    input:
        curvature_file = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed.csv"

    params:
        pca_model_path = config["pca_model"],
        initial_segment = config["initial_segment"],
        final_segment = config["final_segment"],
        window = config["window"]
    output:
        principal_components = f"{output_behavior_dir}/principal_components.csv",
        behaviour_annotation = f"{output_behavior_dir}/beh_annotation.csv"
    run:
        from centerline_behavior_annotation.curvature.src import annotate_reversals_snakemake

        annotate_reversals_snakemake.main([
            '-i', str(input.curvature_file),
            '-pca', str(params.pca_model_path),
            '-i_s', str(params.initial_segment),
            '-f_s', str(params.final_segment),
            '-win', str(params.window),
            '-o_pc', str(output.principal_components),
            '-o_bh', str(output.behaviour_annotation),
        ])

rule annotate_turns:
    input:
        #principal_components = f"{output_behavior_dir}/principal_components.csv"
        spline_K  = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed.csv"
    params:
        output_path = f"{output_behavior_dir}/",  # Ulises' functions expect the final slash
        threshold = config["turn_threshold"],
        initial_segment = config["turn_initial_segment"],
        final_segment = config["turn_final_segment"],
        avg_window = config["turn_avg_window"]
    output:
        turns_annotation = f"{output_behavior_dir}/turns_annotation.csv"

    run:
        from centerline_behavior_annotation.curvature.src import annotate_turns_snakemake

        annotate_turns_snakemake.main([
            '-input', str(input.spline_K),
            '-t', str(params.threshold),
            '-i_s', str(params.initial_segment),
            '-f_s', str(params.final_segment),
            '-avg_window', str(params.avg_window),
            '-bh', str(output.turns_annotation),
        ])

rule self_touch:
    input:
        binary_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask.btf"
    params:
        external_area = [7000, 20000],
        internal_area = [100, 2000],
    output:
        self_touch = f"{output_behavior_dir}/self_touch.csv"
    run:
        from imutils.src.imfunctions import stack_self_touch
        df = stack_self_touch(input.binary_img, params.external_area, params.internal_area)
        df.to_csv(output.self_touch)


rule calculate_parameters:
    #So far it only calculates speed
    input:
        curvature_file = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed.csv" #This is used as a parameter because it is only used to find the main dir
    output:
        speed_file = f"{output_behavior_dir}/raw_worm_speed.csv" # This is never produced, so this will always run
    params:
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
    run:
        from centerline_behavior_annotation.behavior_analysis.src import calculate_parameters

        calculate_parameters.main([
            '-i', str(params.output_path),
            '-r', str(raw_data_dir),
        ])


rule save_signed_speed:
    input:
        raw_speed_file = f"{output_behavior_dir}/raw_worm_speed.csv",
        behaviour_annotation= f"{output_behavior_dir}/beh_annotation.csv"
    output:
        signed_speed_file = f"{output_behavior_dir}/signed_worm_speed.csv" # This is never produced, so this will always run
    run:
        import pandas as pd
        raw_speed_df=pd.read_csv(input.raw_speed_file)
        ethogram_df = pd.read_csv(input.behaviour_annotation)
        signed_speed_df = pd.DataFrame()
        signed_speed_df['Raw Speed Signed (mm/s)'] = raw_speed_df['Raw Speed (mm/s)'] * ethogram_df['0'] * -1 # to invert because fwd is -1 in the ethogram
        signed_speed_df.to_csv(output.signed_speed_file)
        print("If the ethogram had a running average and had less values at the start and end, so will the signed speed")


rule make_behaviour_figure:
    input:
        curvature_file = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed_signed.csv",
        pc_file = f"{output_behavior_dir}/principal_components.csv",
        beh_annotation_file = f"{output_behavior_dir}/beh_annotation.csv",
        speed_file = f"{output_behavior_dir}/signed_worm_speed.csv",
        turns_annotation = f"{output_behavior_dir}/turns_annotation.csv"
    output:
        figure = f"{output_behavior_dir}/behavioral_summary_figure.pdf" #This is never produced, so it will always run
    params:
        output_path = f"{output_behavior_dir}/" # Ulises' functions expect the final slash
    run:
        from centerline_behavior_annotation.behavior_analysis.src import make_figure_2

        make_figure_2.main([
            '-i', str(params.output_path),
            '-r', str(raw_data_dir),
            '-k', str(input.curvature_file),
            '-pcs', str(input.pc_file),
            '-beh', str(input.beh_annotation_file),
            '-speed', str(input.speed_file),
            '-turns', str(input.turns_annotation)
        ])


rule process_skeleton_curvature:
    input:
        skeleton_x = f"{output_behavior_dir}/skeleton_spline_X_coords.csv",
        skeleton_y = f"{output_behavior_dir}/skeleton_spline_Y_coords.csv"
    output:
        output_x = f"{output_behavior_dir}/skeleton_spline_X_coords_equi_dist_segment.csv",
        output_y = f"{output_behavior_dir}/skeleton_spline_Y_coords_equi_dist_segment.csv",
        output_curvature = f"{output_behavior_dir}/skeleton_spline_K_equi_dist_segment.csv",
        output_smoothed_curvature = f"{output_behavior_dir}/skeleton_spline_K__equi_dist_segment_2D_smoothed.csv"
    params:
        spacing = config['relative_spacing'],
        num_sampled_points = config['num_sampled_points'],
        smoothing = config['smoothing'],
        time_sigma = config['time_sigma'],
        spatial_sigma = config['spatial_sigma'],
        max_columns = config['max_columns'],
    run:
        import sys
        from centerline_behavior_annotation.centerline.dev import centerline_equi_distance_2d_smoothing

        centerline_equi_distance_2d_smoothing.main([
            '--skeleton_x', str(input.skeleton_x),
            '--skeleton_y', str(input.skeleton_y),
            '--relative_spacing', str(params.spacing),
            '--num_sampled_points', str(params.num_sampled_points),
            '--smoothing', str(params.smoothing),
            '--time_sigma', str(params.time_sigma),
            '--spatial_sigma', str(params.spatial_sigma),
            '--max_columns', str(params.max_columns),
            '--output_x', str(output.output_x),
            '--output_y', str(output.output_y),
            '--output_curvature', str(output.output_curvature),
            '--output_smoothed_curvature', str(output.output_smoothed_curvature)
        ])



##
## Functions that use both traces and behavior (mostly summary videos/figures)
##

# Does not use behavior annotation, just the raw video
rule make_heatmap_with_behavior_video:
    input:
        cfg=project_cfg_fname,
        traces=os.path.join(project_dir, "4-traces/green_traces.h5"),
        behavior_btf=behavior_btf if os.path.exists(behavior_btf) else raw_data_subfolder,
    output:
        figure=os.path.join(output_visualization_directory, "heatmap_with_behavior.mp4")
    run:
        try:
            _run_helper("visualization.4+make_heatmap_with_behavior_video", str(input.cfg))
        except FileNotFoundError:
            logging.warning("Could not find the behavior data, so an empty file will be created")
            open(output.figure, 'w').close()
