#!/bin/bash

# Takes a bash command and applies it to a list of hardcoded folders
# Usage:
#   ./apply_bash_to_list_of_folders.sh command ...args...
# Note that "command" is a positional argument; the rest may be keyword
# Assumes that the folder should be passed to the command with the -s flag

# List of folders
folders=(
    "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar"
    "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-05_spacer_7b_2per_agar"
    "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-23_spacer_7b_2per_agar"
    "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-30_spacer_7b_2per_agar"
    "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar"
)

# Get command from args
command=$1

# Loop through folders and apply command
for folder in "${folders[@]}"; do
    # Call command using all args but the first (which is the command itself)
    bash "$command" "${@:2}" -s "$folder"
done