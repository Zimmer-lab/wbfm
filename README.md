# Whole Brain Freely Moving Tracking and Trace extraction

Contains code for the pipeline, GUI, and analysis code for the paper:
[An intrinsic neuronal manifold underlies brain-wide hierarchical organization of behavior in C. elegans](https://www.biorxiv.org/content/10.1101/2025.03.09.642241v1)

This repository contains python code for analyzing raw volumetric images in two channels: red (tracking) and green (activity).
Behavior analysis is also included as optional analysis, and is in a separate [repository](https://github.com/Zimmer-lab/centerline_behavior_annotation)

In order to reproduce the paper figures, please see the exported jupyter notebooks [here](wbfm/notebooks/paper).
Note that the raw data is needed, and thus any code related to loading the data would need to be updated to properly run these scripts.

# Installation

This project is designed to be installed with Anaconda, but we suggest using [mamba](https://mamba.readthedocs.io/en/latest/).
There are different use cases, some of which have easier installation steps.
Please check all sections below to determine which is best for you.


## Just running the GUI

See [GUI README](wbfm/gui/README.md)


## Running the full pipeline (non-Zimmer)

See: [Running the pipeline](docs/running_the_pipeline.md)


## Dev installation (local, for editing)

This is for developers, or if you want to run the full pipeline on your local machine.
See: [detailed installation instructions](docs/installation_instructions.md)


## Running the full pipeline (Zimmer lab)

If you just want to run the pipeline (most people), then you can use the pre-installed environments installed on the cluster, which can be activated using:
```
conda activate /lisc/data/scratch/neurobiology/zimmer/.conda/envs/wbfm/
```

# Running the pipeline

See: [Running the pipeline](docs/running_the_pipeline.md)

# Summary of GUIs

All guis are in the folder: /folder_of_this_README/wbfm/gui/example.py

1. Initial creation of project. 
```bash
python wbfm/gui/create_project_gui.py
```
See: [Running the pipeline](docs/running_the_pipeline.md) for fields to check after creating a new project

2. Visualization of most intermediate steps in the analysis is also possible, and they can be accessed via the progress gui. This also tells you which steps are completed. Note that the progress checking doesn't work with nwb files, but the gui does:
```bash
python wbfm/gui/progress_gui.py
```
Or, if you know the project already:
```bash
python wbfm/gui/progress_gui.py --project_path PATH-TO-YOUR-PROJECT
```

3. Manual annotation and more detailed visualization. 
Note, this can take minutes to load:

```bash
python wbfm/gui/trace_explorer.py --project_path PATH-TO-YOUR-PROJECT
```


# FAQ, including fixes for common problems

[FAQ](docs/faq.md)

Please also check (via search) the open and closed issues on github.

# More details

[Detailed pipeline steps](docs/detailed_pipeline_steps.md)

[Detailed installation instructions](docs/installation_instructions.md)

[Known issues](docs/known_issues.md)

[Folder organization](docs/data_folder_organization.md)

If you would like to contribute, see [how to contribute](docs/how_to_contribute.md)
