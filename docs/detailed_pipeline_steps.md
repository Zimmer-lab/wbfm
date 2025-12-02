# More details on each step

## Creating a single project

Working examples to create a project are available for 
[linux](wbfm/scripts/examples/0-create_new_project-linux-EXAMPLE.sh)
and [windows](wbfm/scripts/examples/0-create_new_project-windows-EXAMPLE.sh).
Recommended: run these commands on the command line.

If your data is visible locally (mounted is okay), for the initial project creation you can use a gui:

```commandline
cd /path/to/this/code/wbfm/gui
python create_project_gui.py
```

## Running the rest of the workflow for single project

This code is designed in several different scripts, which can be running using a single command.
The organization between these steps uses the workflow manager [snakemake](https://snakemake.readthedocs.io/en/stable/).
Snakemake works by keeping track of output files with special names, and only reliably works for the first run.
If you are rerunning an old project, see the next section.

Once a project is made, the analysis can be run in the following way:
1. Activate the wbfm environment
2. cd to the /snakemake folder within the project:
```commandline
cd /path/to/your/project/snakemake
```
3. Do a dry run to catch any errors in initialization:
```bash
bash RUNME.sh -n
```
4. If there are errors, there are three easy possibilities:
   1. If this is not a new project, you might have to run steps one by one (see the next subsection).
   2. If you changed the name of the project, read the *IMPORTANT* tip above
   3. If you get a permission issue, read the *IMPORTANT* tip above
   4. If you still have a problem, then it is probably a bug and you should file a GitHub issue and possibly talk to Charlie
5. Run the relevant RUNME script, either cluster or local. Probably, you want the cluster version:
```bash
bash RUNME.sh
```
6. This will run ALL steps in sequence.
   1. Note: you can't close the terminal! You may need to use a long-term terminal program like tmux or screen.
   2. If you get errors, see step 4.
   3. It will print a lot of green-colored text, and the current analysis step will be written near the top, for example: 'rule: preprocessing'. Depending on how busy the cluster is, it could take 6-24 hours / 1000 volumes.
7. Check the log files (they will be in the /snakemake folder) to make sure there were no errors.
Almost all errors will crash the program (display a lot of red text), but if you find one that doesn't, please file an issue!
8. If the program crashes and you fix the problem, then you should be able to start again from step 3 (DRYRUN). This should rerun only the steps that failed and after, not the steps that succeeded. 


## ADVANCED: running steps within an incomplete project

Snakemake works by keeping track of output files with special names, and only reliably works for the first run.
If you have simply not run all of the steps or there was a crash, then continue with the above section.
However, if you are re-running analysis steps, then see below.

In this case, you must run each step one by one. Example:

```bash
cd /path/to/this/repo/wbfm
python scripts/4-make_final_traces.py with project_path=/path/to/your/project/project_config.yaml
```

Note that you can check the current status of the project by moving to the project and running a script. Example:
```bash
cd /path/to/your/project
python log/print_project_status.py
```

You can directly run the python scripts, or, most likely, run them using sbatch using the following syntax.

### Running single steps on the cluster (sbatch)

Once the project is created, each step can be run via sbatch using this command in the scripts/cluster folder:

```commandline
sbatch single_step_dispatcher.sbatch -s 1 -t/lisc/data/scratch/neurobiology/zimmer/Charles/dlc_stacks/worm10-gui_test/project_config.yaml
```

where '-s' is a shortcut for the step to run (0b, 1, 2a, 2b, 2c, 3a, 3b, 4) and '-t' is a path to the project config file.

Note: there may also be some alternative (not main pipeline steps), for example '4-alt' which just re-extracts the traces and makes the grid plots.
This is useful for example if different preprocessing is applied to the videos (but the segmentation and tracking are unchanged).



### Creating a project: command-line details

Command:
```bash
RED_PATH="path/to/red/data"
GREEN_PATH="path/to/green/data"
PROJECT_DIR="path/to/new/project/location"

COMMAND="scripts/0a-create_new_project.py"

python $COMMAND with project_dir=$PROJECT_DIR red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH
```

This is the most complicated step, and I recommend that you create a bash script.

Speed: Fast

Output: new project folder with project_config.yaml, and with 4 numbered subfolders

### Segmentation

Preparation:
0. Make sure the project was initialized successfully!
1. Open project_config.yaml and change the variables marked CHANGE ME
..* In particular, manually check the video quality to make sure start_volume and num_frames are reasonable
..* For now, num_slices must be set manually
4. Open 1-segmentation/preprocessing_config.yaml and change the variables marked CHANGE ME
5. For now, 1-segmentation/segmentation_config.yaml should not need to be updated

Command:
```bash
python scripts/1-segment_video.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Slowest step; 3-12 hours

Output, in 1-segmentation:
1. masks.zarr
..* .zarr is the data type, similar to btf but faster
2. metadata.pickle
..* This is the centroids and brightnesses of the identified neurons

### Training data

Preparation:
1. Open 2-training_data/preprocessing_config.yaml and change the variables marked CHANGE ME
2. For now, 2-training_data/training_data_config.yaml should not need to be updated

Command:
```bash
python scripts/2ab-build_feature_and_match.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Fast, but depends on number of frames; 2-6 hours

Output, in 2-training_data/raw:
1. clust_dat_df.pickle
..* This is the dataframe that contains the partial "tracklets", i.e. tracked neurons in time
2. match_dat.pickle and frame_dat.pickle
..* These contain the matches between frames, and the frame objects themselves
..* This is mostly for debugging and visualizing, and is not used further



### Tracking

#### Part 1/2

Preparation:
1. Open 3-tracking/tracking_config.yaml and change the variables marked CHANGE ME

Command:
```bash
python scripts/3a-track_time_independent.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: 1-3 hours

Output, in 3-tracking:
1. A dataframe with positions for all neurons

#### Part 2/2

Combine the tracks and tracklets

Command:
```bash
python scripts/3b-match_tracklets_and_tracks_using_neuron_initialization.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Long; ~6 hours

Output:
1. A dataframe with positions for each neuron, corrected by the tracklets

### Traces

Preparation: None, for now

Command:
```bash
python scripts/4-make_final_traces.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: ~30 minutes

Output, in 4-traces:
1. all_matches.pickle, the matches between the DLC tracking and the original segmentation
2. red_traces.h5 and green_traces.h5
..* This is the raw time series for all neurons in each channel


## Visualization of results

I made a gui for this purpose, with everything in the gui/ folder. Read the README in that folder for more detail

Command:
```bash
python gui/trace_explorer.py --project_path PATH-TO-YOUR-PROJECT
```
