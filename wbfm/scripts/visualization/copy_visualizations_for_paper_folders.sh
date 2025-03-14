#!/bin/bash

# Runs a script for all datasets to be used in the paper
# See copy_visualizations_for_paper_folders.sh for more details
PARENT_DIR="/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects"
DATASET_LIST=("2022-12-10_spacer_7b_2per_agar" "2022-12-05_spacer_7b_2per_agar" "2022-11-23_spacer_7b_2per_agar" "2022-11-30_spacer_7b_2per_agar" "2022-11-27_spacer_7b_2per_agar")
SCRIPT="/lisc/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/scripts/visualization/copy_visualizations_from_many_projects.sh"

# Parse arguments
while getopts s:t:d: flag
do
    case "${flag}" in
        t) target=${OPTARG};;
        d) destination_folder=${OPTARG};;
        *) echo "Invalid option";;
    esac
done

# Loop over all datasets
for DATASET in "${DATASET_LIST[@]}"; do
    # Run the command, using my folder wrapper
    bash $SCRIPT -t "$target" -d "$destination_folder" -s $PARENT_DIR/"$DATASET"
done
