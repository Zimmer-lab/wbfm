#!/usr/bin/env bash

#SBATCH --job-name=gpu_step
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=log_gpu_step-%j.out    # Standard output and error log
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=basic  # The basic partition has computers with single GPUs, instead of multiple

# Helper function to display a help message
function show_help {
    echo "Usage: $0 -t project_path -n is_dry_run -s step_reference"
    echo "  -t project_path: path to the project directory"
    echo "  -n is_dry_run: whether to run the command or just print it"
    echo "  -s step_reference: reference to the step to run (see single_step_dispatcher.sbatch for the list of steps)"
    echo "  -a args: additional arguments to pass to the script; only used for 3-barlow or match_ids"
    echo "  -h: display this help message"
}

pwd; hostname; date

# Get all user flags
args=""
while getopts t:n:s:a:h flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        s) step_reference=${OPTARG};;
        a) args=${OPTARG};;
        h) show_help;;
        *) raise error "Unknown flag"
    esac
done

# Unpack the flag into an actual command
CMD_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts"

declare -A CMD_DICT
CMD_DICT["0b"]="0b-preprocess_working_copy_of_data.py"
CMD_DICT["1"]="1-segment_video.py"
CMD_DICT["2a"]="2a-build_frame_objects.py"
CMD_DICT["2b"]="2b-match_adjacent_volumes.py"
CMD_DICT["2c"]="2c-postprocess_matches_to_tracklets.py"
CMD_DICT["3a"]="3a-track_time_independent.py"
CMD_DICT["3b"]="3b-match_tracklets_and_tracks_using_neuron_initialization.py"
CMD_DICT["4"]="4-make_final_traces.py"
CMD_DICT["4-alt"]="alternate/4c-extract_full_traces.py"
CMD_DICT["match_ids"]="postprocessing/4+match_two_projects.py"

# Specify additional args, if any
CMD_DICT["3-barlow"]="alternate/3-track_using_barlow.py"

CMD="${CMD_DIR}/${CMD_DICT[$step_reference]}"
FULL_CMD="$CMD with project_path=$project_path"

if [ "$step_reference" = "3-barlow" ]; then
  FULL_CMD="$FULL_CMD model_fname=/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/barlow_ZIM2165_Gcamp7b_worm1-2022_11_28_from_search/trial_0/resnet50.pth"
fi

if [ "$step_reference" = "match_ids" ]; then
  FULL_CMD="python $CMD with project_path_base=$project_path project_path_target=$args"
fi


echo "Running step $step_reference using full command:"
echo "$FULL_CMD"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $FULL_CMD"
else
  # shellcheck disable=SC2086
  python $FULL_CMD
fi

date
echo "Finished; check logs to determine success"
