#!/bin/bash
# Opens tmux session and runs snakemake for all projects in a folder. Example dry run usage:
# bash run_all_projects_in_parent_folder.sh -t '/path/to/parent/folder' -n True
#
# For real usage, remove '-n True' and update the path after -t
#

# Add help function
function usage {
  echo "Usage: $0 [-t folder_of_projects] [-n] [-d] [-s rule] [-h] [-R restart_rule]"
  echo "  -t: folder of projects (required)"
  echo "  -n: dry run of this script (default: false)"
  echo "  -f: force only one rule (default: false)"
  echo "  -c: run commands locally (default: false, i.e. use sbatch)"
  echo "  -d: dry run of snakemake (default: false)"
  echo "  -s: snakemake rule to run (default: traces_and_behavior; other options: traces, behavior)"
  echo "  -R: snakemake rule to restart from (default: None)"
  echo "  -h: display help (this message)"
  exit 1
}

RULE="traces_and_behavior"
is_dry_run=""
RUNME_ARGS=""
RESTART_RULE=""
is_snakemake_dry_run=""

# Get all user flags
while getopts t:s:R:ncfdh flag
do
    case "${flag}" in
        t) folder_of_projects=${OPTARG};;
        n) is_dry_run="True";;
        d) is_snakemake_dry_run="True";;
        f) RUNME_ARGS="$RUNME_ARGS -f";;
        c) RUNME_ARGS="$RUNME_ARGS -c";;
        s) RULE=${OPTARG};;
        R) RESTART_RULE=${OPTARG};;
        h) usage;;
        *) echo "Unknown flag"; usage; exit 1;;
    esac
done

# Shared setup for each command
conda_setup_cmd="conda activate /lisc/data/scratch/neurobiology/zimmer/.conda/envs/wbfm/"

loop_through_and_analyze_folder() {
    folder_of_projects=$1
    local recursion_level
    recursion_level=$2

    if [ $recursion_level -eq 0 ]; then
        echo "Checking top level folder: $folder_of_projects"
    fi

    # Loop through the parent folder, then try to get the config file within each of these parent folders
    for f in "$folder_of_projects"/*; do
        if [ -d "$f" ] && [ ! -L "$f" ]; then
            # Echo the folder name with indentation based on recursion level
            indent=""
            for ((i=0; i<recursion_level; i++)); do
                indent="$indent  "
            done
            echo "${indent}Analyzing folder: $f (recursion level: $recursion_level)"

            # Check to make sure the project has a project_config.yaml file, i.e. is a real project
            project_found="False"
            while IFS= read -r -d '' f_config; do
                if [ -f "$f_config" ] && [ "${f_config##*/}" = "project_config.yaml" ]; then
                    project_found="True"
                    if [ "$is_dry_run" ]; then
                        # Run the snakemake dryrun
                        echo "DRYRUN: Dispatching on config file: $f_config"
                    else
                        # Get the snakemake command and run it
                        snakemake_folder="$f/snakemake"
                        snakemake_script_path="$snakemake_folder/RUNME.sh"

                        snakemake_cmd="$snakemake_script_path -s $RULE $RUNME_ARGS"
                        if [ "$is_snakemake_dry_run" ]; then
                            snakemake_cmd="$snakemake_cmd -n"
                            echo "Running snakemake dry run"
                        fi
                        if [ -n "$RESTART_RULE" ]; then
                            snakemake_cmd="$snakemake_cmd -R $RESTART_RULE"
                        fi
                        # Instead of tmux, use a controller sbatch job
                        cd "$snakemake_folder" || exit  # Move in order to create the snakemake log all together

                        # Build the job name using the folder name and the target rule
                        JOB_NAME=$(basename "$f")
                        JOB_NAME="${JOB_NAME}_${RULE}"
                        echo "Running job with name: $JOB_NAME"

                        # If the RUNME_ARGS contains -c, then run the command directly without sbatch
                        if [[ "$RUNME_ARGS" == *"-c"* ]]; then
                            # Do not run the conda setup command, which is not needed for local runs
                            echo "Running: $snakemake_cmd"
                            bash $snakemake_cmd &
                        else
                            full_cmd="$conda_setup_cmd; bash $snakemake_cmd"
                            sbatch --time 3-00:00:00 \
                                --cpus-per-task 1 \
                                --mem 24G \
                                --mail-type=FAIL,TIME_LIMIT,END \
                                --wrap="$full_cmd" \
                                --job-name="$JOB_NAME"
                            sleep .1  # Avoid overloading the scheduler
                        fi
                    fi
                fi
            done < <(find "$f" -mindepth 1 -maxdepth 1 -type f -print0)
            # If no project found, go one level deeper via recursion
            if [ "$project_found" = "False" ] && [ $recursion_level -lt 2 ]; then
                echo "No project_config.yaml found in $f, analyzing subfolders"
                loop_through_and_analyze_folder "$f" $((recursion_level + 1))
            fi
        fi
    done
}

# Actually call function
if [ -z "$folder_of_projects" ]; then
    echo "Error: folder_of_projects is required"
    usage
else
    recursion_level=0
    loop_through_and_analyze_folder "$folder_of_projects" $recursion_level
fi
