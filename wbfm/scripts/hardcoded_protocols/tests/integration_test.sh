#!/bin/bash


# Function to display a help message
function show_help {
    echo "Usage: $0 [-c] [-h]"
  echo "  -c: do NOT use cluster (default: false, i.e. run on cluster)"
}

# Get all user flags
USE_CLUSTER="True"

while getopts ch flag
do
    case "${flag}" in
        c) USE_CLUSTER=${OPTARG};;
        h) show_help
           exit 0;;
        *) raise error "Unknown flag"
    esac
done

# Clean and run the integration test, i.e. shortened datasets
PARENT_DATA_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/test_data"
if [ -z "$USE_CLUSTER" ]; then
  echo "Running integration test without cluster"
  PARENT_PROJECT_DIR="/home/charles/Current_work/test_projects"
  CODE_DIR="/home/charles/Current_work/repos/dlc_for_wbfm/wbfm"
  RUNME_ARGS="-c"
else
  echo "Running integration test on the cluster"
  PARENT_PROJECT_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/test_projects"
  CODE_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"
  RUNME_ARGS=""
fi

# Define relevant subfolders
SUBFOLDERS=("freely_moving" "immobilized" "barlow" "nwb")

# For each subfolder, remove any project folders
for f in "${SUBFOLDERS[@]}"; do
    PROJECT_PATH=$PARENT_PROJECT_DIR/$f
    if [ -d "$PROJECT_PATH" ]; then
        echo "Removing existing project in subfolder $PROJECT_PATH"
        for SUBFOLDER in "$PROJECT_PATH"/*; do
            if [ -d "$SUBFOLDER" ]; then
                rm -r "$SUBFOLDER"
            fi
        done
    fi
done

# Initialize projects, again for each subfolder
COMMAND=$CODE_DIR/"scripts/cluster/create_multiple_projects_from_data_parent_folder.sh"
NWB_COMMAND=$CODE_DIR/"scripts/pipeline_alternate/0a-create_new_project_from_nwb.py"
for f in "${SUBFOLDERS[@]}"; do
    
    PROJECT_PATH=$PARENT_PROJECT_DIR/$f
    echo "Creating new project in subfolder $PROJECT_PATH"
    # The nwb file is different; it is just the file, not the same project structure
    if [ "$f" == "nwb" ]; then
        NWB_FILE=$PARENT_DATA_DIR/$f/"test_data.nwb"
        python $NWB_COMMAND with project_dir="$PROJECT_PATH" nwb_file="$NWB_FILE" copy_nwb_file=True
    else
        # If there are multiple data folders, create a project for each
        DATA_DIR=$PARENT_DATA_DIR/$f
        bash $COMMAND -t "$DATA_DIR" -p "$PROJECT_PATH" -b False
    fi
done

# Sleep, waiting for the projects to be created
echo "Projects should have been created... starting to run the integration test"

# Modify snakemake slurm options to have very short jobs, then actually run
SLURM_UPDATE_COMMAND=$CODE_DIR/"scripts/folder_analysis/copy_config_file_to_multiple_projects.sh"
NEW_CONFIG=$CODE_DIR/"alternative_project_defaults/short_video/cluster_config.yaml"

# Command to actually run
COMMAND=$CODE_DIR/"scripts/cluster/run_all_projects_in_parent_folder.sh"

# Freely moving (behavior can't be run without cluster)
PROJECT_PATH=$PARENT_PROJECT_DIR/"freely_moving"
bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_CONFIG"
if [ -z "$USE_CLUSTER" ]; then
  bash $COMMAND -t "$PROJECT_PATH" -s traces $RUNME_ARGS
else
  bash $COMMAND -t "$PROJECT_PATH" -s traces_and_behavior $RUNME_ARGS
fi

# Immobilized
PROJECT_PATH=$PARENT_PROJECT_DIR/"immobilized"
bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_CONFIG"
bash $COMMAND -t "$PROJECT_PATH" -s traces $RUNME_ARGS

# Barlow, which needs the original and an additional config change (behavior can't be run without cluster)
NEW_BARLOW_CONFIG=$CODE_DIR/"alternative_project_defaults/barlow/snakemake_config.yaml"

PROJECT_PATH=$PARENT_PROJECT_DIR/"barlow"
bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_CONFIG"
bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_BARLOW_CONFIG"
if [ -z "$USE_CLUSTER" ]; then
  bash $COMMAND -t "$PROJECT_PATH" -s traces $RUNME_ARGS
else
  bash $COMMAND -t "$PROJECT_PATH" -s traces_and_behavior $RUNME_ARGS
fi

# NWB, which is also barlow but needs a special snakemake file as well
PROJECT_PATH=$PARENT_PROJECT_DIR/"nwb"
# NEW_NWB_CONFIG=$CODE_DIR/"alternative_project_defaults/start_from_nwb_segmentation/pipeline.smk"

PROJECT_PATH=$PARENT_PROJECT_DIR/"nwb"
bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_CONFIG"
bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_BARLOW_CONFIG"
# bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_NWB_CONFIG"

bash $COMMAND -t "$PROJECT_PATH" -s traces $RUNME_ARGS

# Immobilized with barlow-generated tracklets
# PROJECT_PATH=$PARENT_PROJECT_DIR/"immobilized_barlow_tracklets"
# BARLOW_TRACKLET_CONFIG=$CODE_DIR/"alternative_project_defaults/barlow_tracklets/training_data_config.yaml"
# bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$NEW_CONFIG"
# bash $SLURM_UPDATE_COMMAND -t "$PROJECT_PATH" -c "$BARLOW_TRACKLET_CONFIG"
# bash $COMMAND -t "$PROJECT_PATH" -s traces $RUNME_ARGS
