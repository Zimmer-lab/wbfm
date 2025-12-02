#!/bin/bash

# This script runs the Bayesian model for all neurons in the dataset

# Get an argument for whether to run gfp or not
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <gfp>"
  exit 1
fi

# Set the gfp flag
do_gfp=$1
if [ "$do_gfp" != "True" ] && [ "$do_gfp" != "False" ]; then
  echo "Invalid gfp flag: $do_gfp"
  exit 1
else
  echo "Running model with gfp: $do_gfp"
fi

# First define the list of neurons
# These are neurons that are numbered (not ided), e.g. neuron_001, neuron_100, etc.
# So we define them using a loop
neuron_list=()
for i in {1..200}
do
  neuron_list+=("neuron_$(printf "%03d" "$i")")
done

# Now loop through the list of neurons and run the model (one sbatch job per neuron)
# Note that many combinations of dataset and neuron will be empty, and will be skipped

CMD="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/utils/external/utils_pymc.py"
# Changes if running on gfp
if [ "$do_gfp" == "True" ]; then
  LOG_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/logs"
else
  LOG_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/logs"
fi

# Create an sbatch job per neuron name, which will loop over datasets
# Create a temporary file to actually dispatch
SLURM_SCRIPT=$(mktemp /tmp/slurm_script.XXXXXX)
NUM_TASKS=${#neuron_list[@]}

cat << EOF > $SLURM_SCRIPT
#!/bin/bash
#SBATCH --array=0-$(($NUM_TASKS-1))
#SBATCH --time=0-06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --partition=short,basic

# Reproduce the list for the subfile
my_list=(${neuron_list[@]})
neuron=\${my_list[\$SLURM_ARRAY_TASK_ID]}
echo "Running model for neuron: \$neuron"

log_fname="log_single_dataset_$neuron.txt"
python $CMD --neuron_name "$neuron" --dataset_name "loop" --do_gfp "$do_gfp" > "$LOG_DIR/\$log_fname"
EOF

# Submit the SLURM script
sbatch $SLURM_SCRIPT

# Clean up the temporary SLURM script
rm $SLURM_SCRIPT

