#!/bin/bash

# Clean and run the integration test, i.e. a shortened dataset
DATA_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/test_data/immobilized/2024-04-23_19-11_worm13_7b_short_test"
PARENT_PROJECT_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/test_projects/immobilized"
CODE_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"

# Remove project if it exists
PROJECT_PATH=$PARENT_PROJECT_DIR/"pytest-2024-04-23"
if [ -d "$PROJECT_PATH" ]; then
  echo "Removing existing project at $PROJECT_PATH"
  rm -r "$PROJECT_PATH"
fi

# Initialize project
echo "Creating new project at $PROJECT_PATH"
COMMAND=$CODE_DIR/"scripts/0a-create_new_project.py"
ARGS="with parent_data_folder=$DATA_DIR project_dir=$PARENT_PROJECT_DIR experimenter=pytest"
# shellcheck disable=SC2086
python "$COMMAND" $ARGS

# Run using the snakemake pipeline from a bash (would be sbatch on the cluster) controller job
cd "$PROJECT_PATH"/snakemake || exit
echo "Running the pipeline in project $PWD"

bash RUNME.sh -c -s traces
