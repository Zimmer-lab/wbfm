#!/bin/bash

# For both gcamp and immobilized projects, first build and save the new visualizations, then copy them to the folder

# Commands
CODE_FOLDER="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"
BUILD_VIS_COMMAND=$CODE_FOLDER/"scripts/hardcoded_protocols/build_visualizations_for_paper_folders.sh"
COPY_VIS_COMMAND=$CODE_FOLDER/"scripts/hardcoded_protocols/copy_paper_summary_plots.sh"

# First step: build visualizations (sbatch jobs)
# This script waits for all the jobs to finish
echo "Building visualizations (may take a while)"
bash $BUILD_VIS_COMMAND

# Second step: copy visualizations (need to copy each file type)
bash $COPY_VIS_COMMAND
