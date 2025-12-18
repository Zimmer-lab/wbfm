#!/usr/bin/env bash

#SBATCH --job-name=single_step
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=log_single_step-%j.out    # Standard output and error log
#SBATCH --time=12:00:00  # A normal video should only take <3 hours, but some could be longer
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=himem

# Example, after you are in the right directory:
# sbatch custom_step_dispatcher.sbatch -t project/path -s folder_analysis/4+make_multiproject_behavioral_encodings.py
#
# Optionally, -k can be used to add keywords (should be in quotes)
#
# Note that the subfolder is needed

pwd; hostname; date

# Get all user flags
while getopts t:n:s:k: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        s) CMD_SUFFIX=${OPTARG};;
        k) KWARGS=${OPTARG};;
        *) echo "Error: Unknown flag"; exit 1;;
    esac
done

# Unpack the flag into an actual command
CMD_DIR="/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/scripts"

CMD="${CMD_DIR}/${CMD_SUFFIX}"
FULL_CMD="python $CMD with project_path=$project_path"

echo "Running step $CMD_SUFFIX using full command:"
echo "$FULL_CMD"
echo "And kwargs: $KWARGS"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $FULL_CMD"
else
  python "$CMD" with project_path="$project_path" "$KWARGS"
fi

date
echo "Finished; check logs to determine success"
