#!/bin/bash

# Runs a script for all datasets to be used in the paper
PARENT_DIR="/lisc/data/scratch/neurobiology/zimmer/Charles/dlc_stacks"
DATASET_LIST=("2022-12-10_spacer_7b_2per_agar" "2022-12-05_spacer_7b_2per_agar" "2022-11-23_spacer_7b_2per_agar" "2022-11-30_spacer_7b_2per_agar" "2022-11-27_spacer_7b_2per_agar")
SCRIPT_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/scripts/cluster"

# Get the exact step to run
while getopts t:n:s: flag
do
    case "${flag}" in
        s) step_reference=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Loop over all datasets
for DATASET in "${DATASET_LIST[@]}"; do
    # Run the command, using my folder wrapper
    bash $SCRIPT_DIR/apply_step_to_all_in_folder.sh -t $PARENT_DIR/"$DATASET" -s "$step_reference"
done