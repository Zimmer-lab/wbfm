#!/bin/bash

# Create new projects from a folder of data subfolders
# This script is meant to be run on the cluster, or on a local machine with the data mounted
# Note that if you initialize projects on a non-linux machine, you may need to change the paths to the data and
# project directories (thus cluster is suggested)
#
# Usage:
#   bash create_multiple_projects_from_data_parent_folder.sh -t <DATA_PARENT_FOLDER> -p <PROJECT_PARENT_FOLDER> -n <is_dry_run>
#
# Example:
#   bash create_multiple_projects_from_data_parent_folder.sh -t /lisc/data/scratch/neurobiology/zimmer/wbfm/data -p/lisc/data/scratch/neurobiology/zimmer/wbfm/projects

function usage {
  echo "Usage: $0 [-t DATA_PARENT_FOLDER] [-p PROJECT_PARENT_FOLDER] [-n is_dry_run] [-b run_in_background]"
  echo "  -t: parent folder of data (required)"
  echo "  -p: parent folder of projects (required)"
  echo "  -n: dry run of this script (default: false)"
  echo "  -b: run in background (default: True)"
  echo "  -h: display help (this message)"
  exit 1
}

# Get all user flags
run_in_background="True"
while getopts t:p:n:bh flag
do
    case "${flag}" in
        t) DATA_PARENT_FOLDER=${OPTARG};;
        p) PROJECT_PARENT_FOLDER=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        b) run_in_background=${OPTARG};;
        h) usage;;
        *) raise error "Unknown flag"
    esac
done

# Actually run
COMMAND="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/0a-create_new_project.py"

# Loop through the parent folder, then try to get the config file within each of these parent folders
# Counter for number of jobs actually submitted
num_jobs=0
readarray -t folders < <(find "$DATA_PARENT_FOLDER" -maxdepth 1 -type d -name "*worm*" -o -name "*animal*")
echo "Found ${#folders[@]} folders in $DATA_PARENT_FOLDER matching *worm* or *animal*"

for f in "${folders[@]}"; do
    num_jobs=$((num_jobs+1))
    EXPERIMENTER=$(cd "$f" && pwd)
    EXPERIMENTER=$(basename "$EXPERIMENTER")
    ARGS="project_dir=$PROJECT_PARENT_FOLDER experimenter=$EXPERIMENTER parent_data_folder=$f"
    if [ "$is_dry_run" ]; then
        echo "DRYRUN: Dispatching on folder: $f with EXPERIMENTER: $EXPERIMENTER"
    else
        echo "Dispatching on folder: $f with EXPERIMENTER: $EXPERIMENTER"
        if [ "$run_in_background" == "True" ]; then
            # shellcheck disable=SC2086
            python $COMMAND with $ARGS &
        else
            # shellcheck disable=SC2086
            python $COMMAND with $ARGS
        fi
    fi
done

echo "===================================================================================="
echo "Dispatched $num_jobs jobs in the background; they will finish in ~30 seconds if successful"
echo "Note that the jobs will print out their progress as they complete, and will mix messages"
echo "Expected message if successful:"
echo "INFO - 0a-create_new_project - Completed after 0:00:XX"
echo "===================================================================================="
