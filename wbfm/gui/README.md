# GUIs for whole body fluorescence microscopy

## Installation

Note: these instructions do not assume any knowledge of python or conda.


### Preparation: Terminal setup

If you are on linux or mac, a terminal is included. 
On Windows I suggest anaconda prompt (this comes with an anaconda installation) or git bash

Either way, you should begin by installing anaconda if you don't already have it:
https://www.anaconda.com/products/individual

However, please check the user agreements, which have recently changed for anaconda.
An alternative is mamba: https://mamba.readthedocs.io/en/latest/

### Step 1
In the folder conda-environments, there is a specific environment for this purpose: "gui_only.yaml"

If you are in a terminal, you can use this to create the proper conda environment. 
NOTE: you must clone this repository first, and cd to it (i.e. replace the path below with yours):

```commandline
cd /PATH/TO/YOUR/CODE/wbfm/conda-environments

conda env create -f gui_only.yaml --name gui_only
```

Note: this will take a while, maybe 5 minutes.

### Step 2

Activate the environment. This means that later package installations and commands you run can see all the packages you just installed!

```commandline
conda activate gui_only
```

### Step 3
After the overall packages are installed, the zimmer group private packages need to be installed:

1. git clone wbfm, centerline_behavior_annotation, and imutils (from https://github.com/Zimmer-lab)
2. cd to the main folder of each repository, and run ```pip install -e .```
   1. Note: you will run ```pip``` three times (once per folder), and it is very fast


## Summary of GUIs

All guis are in the folder: /TOP/FOLDER/wbfm/gui/example.py

Assuming you are in the top-level folder, you can access 3 guis. Please read the next sections for installation instructions!

1. Create a project. See "Summary of common problems" in main [README](../../README.md) on carefully checking paths in different operating systems:
```bash
python wbfm/gui/create_project_gui.py
```

2. Visualization of most steps in the analysis is also possible, and they can be accessed via the progress gui. This also tells you which steps are completed:
```bash
python wbfm/gui/progress_gui.py
```
Or, if you know the project already:
```bash
python wbfm/gui/progress_gui.py --project_path /path/to/your/project/project_config.yaml
```

3. Manual annotation of IDs and correction of tracking, and generally more detailed visualization. 
Note, this will take a minute or more to load:
```bash
python wbfm/gui/trace_explorer.py --project_path /path/to/your/project/project_config.yaml
```


## Example: If you have a working installation

If you do NOT have a working installation, read the next section.

All GUIs are designed to be accessed from the "progress gui" or directly from the various "explorer" guis (see list above). 
For example:

```commandline
conda activate gui_only
cd /path/to/FOLDER/of/this/README
python progress_gui.py --project_path /path/to/your/project/project_config.yaml
```

This command will bring a small GUI up that displays the status of the target project, with buttons to open more complex GUIs.
Those will be described in the next major section.

This command works with 4 assumptions:
1. Your terminal is NOT on the cluster (guis can only be run locally, but any data can be remote)
2. You are in a terminal in this folder
3. You are in the proper conda environment
4. You have initialized a project at /path/to/your/project/project_config.yaml

Instructions to satisfy these assumptions are in the next sections.


## Project initialization

See the main [README](../../README.md) file for instructions, or use a pre-generated project


## Trace_explorer GUI: two workflows

First, open the trace explorer gui:

```commandline
conda activate gui_only
cd /path/to/this/README
python trace_explorer.py --project_path /path/to/your/project/project_config.yaml
```

This opens a new Napari window with several layers, designed to be used to view and modify the tracking and segmentation.
This may take some time, ~1 minute.
Rarely, it simply won't open; if it takes longer than 5 minutes, quit it and try again.

### Detailed explanation: Identifying neurons

Neurons can be identified using the additional excel-style table that opens with the trace_explorer GUI.
The normal workflow is this:
1. Identify a neuron (how to do this is outside the scope of this repo)
2. Select the "colored_segmentation" layer on the napari GUI (shortcut: shift-s)
3. If the neuron was not tracked at the current time point, change time points to find it
4. Middle-mouse click on the neuron, which will highlight the correct cell in excel-style table
5. Type in the neuron name

At the end of a session, you can simply close the GUI. 
This will save everything by default.

#### Updating traces with IDs

None of the trace dataframes will be automatically updated with new neuron IDs.
See the [tutorial notebook](https://github.com/Zimmer-lab/wbfm/blob/dev/wbfm/notebooks/tutorials/remake_paper_traces.ipynb) for step by step instructions to regenerate these dataframes.

### Detailed explanation: tracklet/segmentation correction

We have created a [YouTube](https://youtube.com/playlist?list=PL0LLlJzm-VqQhX4Kw2KqoeccJVk4jhPS3) tutorial playlist.
The first two videos are broadly useful, and the later ones are more specific to tracklet and segmentation correction.

When you open the GUI, you will see 4 areas:
1. Left - Napari layers
2. Center - Main 3d data
3. Right - Menus and buttons
4. Bottom - Matplotlib graph

Abstractly, each area of the GUI is designed to present information at either a high level or a detailed level:
1. An overview of the entire worm body
2. Detail about a specific neuron

First I will explain the Napari layers:

#### Top level - entire body

The following layers below to this level:
1. Red data - Raw mscarlet layer
2. Green data - Raw gcamp layer
3. Neuron IDs - Numbers displayed on top of the neurons
4. Colored segmentation - Segmentation as colored by tracks. This is a subset of the next layer (Raw segmentation) 
5. Raw segmentation - Original segmentation, before tracking. Note that this layer can be interactive

#### Detailed level - Single neuron

The following layers below to this level, and are related to the currently selected neuron:
1. track_of_point - a single point on top of the currently selected neuron
2. final_track - a line showing the current and past positions of the neuron

In addition, if the user uses interactions to click on certain neurons, then more layers will be added that relate to the clicked neuron.

### Explanation of manual correction interactivity

By default, interactivity is off, and must be turned on with the checkbox in the top right.

There are two types of interactivity:
1. Tracklets 
2. Segmentation

#### Tracklet workflow

This is designed to correct the tracklets associated with neurons.
The basic steps are as follows:

1. Select a neuron under "Neuron selection"
2. Change to "tracklets" mode under "Channel and Mode selection"
   1. The graph at the bottom will display all tracklets that belong to the selected neuron
3. Find a problem (gap in tracking or error in signal)
   1. Use the plot at the bottom
4. Navigate to the time point with the problem
   1. Many shortcuts are provided
   2. Note that if the neuron is tracked, the main view will be centered on that neuron
5. Fix the problem (READ NEXT SECTION)
6. Save the current tracklet
7. Find a new problem, and repeat 4-6
8. When the neuron is fully tracked, save the manual annotations to disk
   1. NOTE: this will take some time (~10-20 seconds)
9. Choose a new neuron, and repeat 3-8

#### Categories of tracklet problems

Basically there are two types of problems, with two solutions:

1. A perfect tracklet was not assigned to the neuron
2. A tracklet had a mistake and needs to be split
3. A completely incorrect tracklet was assigned to the neuron

Use the following basic workflow to fix them:
1. Confirm the checkbox: "Turn on interactivity?"
2. With the "Raw segmentation layer" highlighted, click on a neuron
   1. This will load the tracklet associated with that segmentation
   2. In addition, a new Napari layer will appear showing the position of the tracklet across time
   3. NOTE: for the 3rd case, you are trying to click on the incorrect neuron, to select the incorrect tracklet for removal
3. Check the correctness of the tracklet
4. Case 1: the tracklet is perfect
   1. Use the button to save the current tracklet to the current neuron, and continue
   2. If there is a conflict, see below
5. Case 2: the tracklet jumps between neurons
   1. Use the "Split tracklet" buttons to remove the jumps
   2. When it is correct, save it to the current neuron
   3. If there is a conflict, see below
6. Case 3: the tracklet is added, but shouldn't be
   1. Use the "remove tracklet from all neurons" button
   2. No additional saving is needed

Note that many tracklet problems are in fact due to segmentation problems, described below.

#### Fixing tracklet conflicts

Currently, the gui will not allow you to save a tracklet if there are time points that overlap with other tracklets.
Thus, you must either a) shorten the current tracklet to fit, or b) remove conflicting tracklets.
Use the shortcut and buttons to do so, then save.

#### Segmentation workflow

This is designed to correct segmentation, and does not need to be related to the current neuron.
However, to keep track of which segmentations you have corrected, it makes sense to correct each neuron across time before moving to the next one.
In that case, the following workflow is suggested:

1. While correcting a neuron, find a segmentation problem
2. Control-click to attempt an automatic segmentation
   1. This will create a new window showing information about the splitting algorithm
3. Case 1: the automatic segmentation is good
   1. Simply save the candidate mask to RAM (button)
4. Case 2: the automatic segmentation is bad
   1. Using the pop-up or other information, choose where the neuron should actually be segmented
   2. Then alt-click to apply the manual segmentation
   3. Check that it is correct, and save to RAM (button)
5. As often as possible, Save to disk
   1. Note that this can take some time (~20 seconds)
   
## Napari tips and tricks

In the 3d view, shift-drag (mouse drag) will translate the view.
Basic click will rotate the view.

Each layer has many advanced features, like opacity and blending.
Especially when looking at multiple layers, you should experiment with these to get a good workflow.

Make sure you highlight the "Raw segmentation" layer when you want interactivity!


## Known issues

See the #gui label on the main github repository. 
Please open an issue if you find a new bug or want something to be changed!
