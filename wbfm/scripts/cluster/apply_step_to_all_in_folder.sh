#!/usr/bin/env bash

# Function to display a help message
function show_help {
    echo "Usage: $0 -t folder_of_projects -n is_dry_run -s step_reference"
    echo "  -t folder_of_projects: path to the folder containing all the projects"
    echo "  -n is_dry_run: whether to run the command or just print it"
    echo "  -s step_reference: reference to the step to run (see single_step_dispatcher.sbatch for the list of steps)"
    echo "  -h: display this help message"
}

# Get all user flags
is_dry_run=""

while getopts t:ns:h: flag
do
    case "${flag}" in
        t) folder_of_projects=${OPTARG};;
        n) is_dry_run="True";;
        s) step_reference=${OPTARG};;
        h) show_help
           exit 0;;
        *) raise error "Unknown flag"
    esac
done

# Path to the command directory
CMD_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/cluster"

# Loop through the parent folder, then try to get the config file within each of these parent folders
for f in "$folder_of_projects"/*; do
    if [ -d "$f" ] && [ ! -L "$f" ]; then
        echo "Checking folder: $f"

        for f_config in "$f"/*; do
            if [ -f "$f_config" ] && [ "${f_config##*/}" = "project_config.yaml" ]; then
                if [ "$is_dry_run" ]; then
                    echo "DRYRUN: Dispatching on config file: $f_config"
                else
                    sbatch "$CMD_DIR"/single_step_dispatcher.sbatch -s "$step_reference" -t "$f_config"
                fi
            fi
        done
    fi
done
