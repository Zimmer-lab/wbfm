# Running the full pipeline

This file contains information for the recommended workflow is described (using .nwb files).
If you are in the Zimmer lab and want to use lab-internal formats, see [Running the pipeline zimmer lab](docs/running_the_pipeline_zimmer_lab.md).

The following steps use the command line, and assume you have the helper scripts located in this repository (i.e., you have cloned it).

## Data preparation

The main starting point is a file in the Neurodata Without Borders (.nwb) file format.
This should contain at least raw imaging data, but may also contain segmentation, object detection, and/or object id information.

## Creating a project

While running, the raw data file will be "unpacked" into a [folder structure](docs/data_folder_organization.md) for easier processing, which can be re-exported as .nwb at any time.
Creating a project will take about 1 minute, and the scripts are:

```commandline
cd /PATH_TO_THIS_CODE/wbfm/scripts
python 0-create_new_project_from_nwb.py with project_dir=PATH_TO_NEW_PROJECT_LOCATION nwb_file=FULL_PATH_TO_NWB_FILE copy_nwb_file=True, unpack_nwb=True
```

This will create a new project from the nwb file, copy the nwb file, and unpack any analysis steps that it can find.
If successful, you will get a message like:
```
Successfully created new project at: {project_fname}
INFO - 0-create_new_project_from_nwb - Completed after 0:00:21
```

By default, the project name is "{basename(nwb_file)}".
Note that by default this will copy the .nwb file into the new project structure, which can be set to False in the command above.
In addition, if raw data is found it will be converted to zarr format for stable parallel access (this will be changed in the future); you can set unpack_nwb=False, but I have encountered strange bugs with nwb files and parallel access so I don't recommend it.


## Main pipeline workflow via snakemake

The analysis steps are organized via snakemake, although they can be run step-by-step, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md). 
Before anything, the parameters should be double-checked.
Most do not matter for most people, so I will just highlight the main ones.

Note that these yaml files are project-specific; in order to change project-wide or default parameters, see the [FAQ](docs/faq.md#changing-defaults-for-new-projects)


### Important parameters to check

The most important parameters are related to the tracking pipeline.
In particular, if you are using the newer [BarlowTrack](https://github.com/Zimmer-lab/barlow_track/) method, you must set the path to the neural network within this file (inside the project you just made):
```
PROJECT_FOLDER/snakemake/snakemake_config.yaml
```

And set the variable (the top two variables):
```
use_barlow_tracker: true
barlow_model_path: YOUR_TRAINED_NETWORK_FOLDER/resnet50.pth
```

Note that there are many other parameters in this config file, mostly related to the behavior pipeline (not covered in this document, but rather a sibling [repo](https://github.com/Zimmer-lab/centerline_behavior_annotation)).

### Actually running the pipeline

Snakemake (which is installed in the conda environment) organizes all of the analysis steps; for more information I suggest the main [docs](https://snakemake.readthedocs.io/en/stable/).
No detailed knowledge should be necessary, and our helper script should run all steps.
First, navigate to the project and activate the conda environment:

```commandline
cd /PATH_TO_THE_PROJECT/snakemake
conda activate MY_ENV
```

It is highly recommended to do look at the help first, and then do a dryrun:

```commandline
bash RUNME.sh -h
bash RUNME.sh -n
```

This will print which steps need to be run in a table format; these are dynamically determined based on the already existing files, if any.
To run **LOCALLY**, use `-c`:

```commandline
bash RUNME.sh -c
```

For a more computationally efficient workflow, we run on a cluster via slurm.
However, the exact cluster parameters can change dramatically across clusters, particularly for using gpu partitions.
Please check with your local admin and modify the `cluster_config.yaml` file (in this snakemake folder) according to their advice.
To run on a cluster (using slurm), no flags are needed:

```commandline
bash RUNME.sh
```

Then you are done!
The pipeline will run all steps, with the duration will dramatically depend on the number of objects in the video and the volume size.
Some steps scale quite poorly with these variables, so rough bounds from real projects may be helpful:
1. 3 minute video with ~5 neurons and TXYZ size ~(800x200x300x7) - 10 minutes
2. 8 minute video with ~150 neurons and TXYZ size ~(1600x600x900x22) - 8 hours


# Checking progress

## Ongoing progress

There are three ways to check progress:
1. Check the currently running jobs on your cluster or local workstation
2. Check the log files in the snakemake/ subfolder
3. Check the produced analysis files using a gui

Method 1:
Run this command on the cluster:
```commandline
squeue -u <your_username>
```

Method 2:
Use the tail command to check the log files:
```commandline
tail -f /path/to/your/project/snakemake/log/[MOST_RECENT_LOG].log
```

Note that the -f flag will keep the terminal open and update the log file as it changes.

Method 3:
Use the `progress_gui.py` gui on a local machine with mounted data to check the actual images, segmentation, and tracking produced.
```commandline
cd /PATH_TO_THIS_CODE/wbfm/gui
python progress_gui.py -p /path/to/your/project
```

Method 4:
Use `wbfm_dashboard.py` on a local machine with mounted data (i.e. access to the running projects) to keep track of multiple projects at once:
```commandline
cd /PATH_TO_THIS_CODE/wbfm/gui
python wbfm_dashboard.py -p /path/to/your/parent_folder
```

## Finished projects

When a project has finished, the best thing to check are the .png files in the 4-traces folder.
There are also several .h5 files (with different preprocessing) in this folder for further analysis.


# Manual annotation and rerunning

Tracking can be checked and corrected using the main trace_explorer [GUI](#Summary of GUIs).
NOTE: this is only compatible with tracklet-based tracking, which is not the default when using BarlowTrack networks.

However, this does not automatically regenerate the final trace dataframes.
For this, some steps must be rerun.
The steps are the same as running steps within an incomplete project, but in short:

1. Run step 3b
   - Note that setting only_use_previous_matches=True in tracking_config.yaml is suggested, and will speed the process dramatically
2. Possible: run step 1-alt (rebuild the segmentation metadata)
   - This is only necessary if segmentation was changed in the manual annotation step 
2. Run step 4

All steps can be run using multi_step_dispatcher.sh
Example usage is given within that file.

Note that it is possible to use snakemake to "know" which files need to be updated, if any files were copied to/from a local machine, snakemake will become confused and attempt to rerun the entire pipeline.

# Advanced: running steps within an incomplete project (including if it crashed)

See [detailed pipeline steps](docs/detailed_pipeline_steps.md)
