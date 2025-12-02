#!/bin/bash

# Runs a script for all datasets to be used in the paper
PARENT_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects"
DATASET_PARENT_DIRS=("2022-12-10_spacer_7b_2per_agar" "2022-12-05_spacer_7b_2per_agar" "2022-11-23_spacer_7b_2per_agar" "2022-11-30_spacer_7b_2per_agar" "2022-11-27_spacer_7b_2per_agar")

CODE_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"
COMMAND=$CODE_DIR/"scripts/visualization/make_summary_interactive_plot.py"

# Loop over project folders and subfolders, and submit sbatch job
for DATASET_PARENT in "${DATASET_PARENT_DIRS[@]}"; do
  # Find all subfolders and loop over them
  for DATASET in "$PARENT_DIR"/"$DATASET_PARENT"/*; do
    # Run the command, using my folder wrapper
    sbatch --job-name=my_job_"$DATASET" <<EOF
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --output=my_job_$DATASET.out
#SBATCH --error=my_job_$DATASET.err

# Your actual job commands go here
echo "Running job on folder: $DATASET"

python $COMMAND with project_path=$PARENT_DIR/"$DATASET"

echo "Job $DATASET completed"
EOF
  sleep 1  # pause to be kind to the scheduler
  done
done

# Wait for sbatch jobs to finish
while [[ $(squeue -h -u "$USER" --format="%j" | grep -c 'my_job') -gt 0 ]]; do
  sleep 60  # sleep for 60 seconds (adjust as needed)
done

echo "All your specific sbatch jobs are finished. Continue with the rest of the script."
