#!/bin/bash

# Same as build_visualizations_for_paper_folders.sh, but runs locally (without sbatch)
# Runs a script for all datasets to be used in the paper
PARENT_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects"
DATASET_PARENT_DIRS=("2022-12-10_spacer_7b_2per_agar" "2022-12-05_spacer_7b_2per_agar" "2022-11-23_spacer_7b_2per_agar" "2022-11-30_spacer_7b_2per_agar" "2022-11-27_spacer_7b_2per_agar")

CODE_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"
COMMAND=$CODE_DIR/"scripts/visualization/4+make_summary_interactive_plot.py"

# Loop over project folders and subfolders, and submit sbatch job
for DATASET_PARENT in "${DATASET_PARENT_DIRS[@]}"; do
  # Find all subfolders and loop over them
  for DATASET in "$PARENT_DIR"/"$DATASET_PARENT"/*; do
    python $COMMAND with project_path="$DATASET"/project_config.yaml &
  done
done

echo "Finished dispatching all jobs; use 'ps' to check on them."
