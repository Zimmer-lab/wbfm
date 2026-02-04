#!/bin/bash

# This script runs the Bayesian model for all neurons in the dataset

# Function to display a help message
function show_help {
  echo "Usage: $0 [-g] [-a] [-r] [-c] [-h] <gfp>"
  echo "  -g: Use GFP data"
  echo "  -a: Use AVB hiscl data"
  echo "  -s: Use simple eigenworms (1 and 2 only)"
  echo "  -r: Trace mode; should be one of 'None', 'pca_global', 'pca_global_1', 'cca_continuous', 'discrete'; default is pca_global"
  echo "  -c: Run temporal-split CV comparison instead of full model fitting"
  echo "  -k: Keep large deterministic variables (curvature_term, mu, etc.) in saved traces"
  echo "  -d: debug mode, which only runs a single neuron (few iterations) for testing"
  echo "  -h: Show this help message"
}

# Get all user flags
use_gfp="false"
use_avb_hiscl="false"
use_raw_trace="false"
debug="false"
simple_eigenworms="false"
cv_comparison="false"
keep_large_vars="false"
while getopts gasr:dckh flag
do
    case "${flag}" in
        g) use_gfp="true";;
        a) use_avb_hiscl="true";;
        s) simple_eigenworms="true";;
        r) residual_mode=${OPTARG};;
        d) debug="true";;
        c) cv_comparison="true";;
        k) keep_large_vars="true";;
        h) show_help
           exit 0;;
        *) echo "Error: Unknown flag"; exit 1;;
    esac
done

# First define the list of neurons
neuron_list=(
'AVEL'
'RID'
'AVBL'
'RMDVL'
'URYVL'
'BAGL'
'AUAR'
'RMED'
'RMEL'
'ALA'
'RMEV'
'RMDVR'
'URYDL'
'RMER'
'URADL'
'SMDVR'
'RIML'
'AVER'
'SMDDR'
'AVAL'
'RIVR'
'BAGR'
'RIS'
'RIBL'
'OLQVL'
'URYVR'
'SMDVL'
'URADR'
'SIADL'
'RIVL'
'URXL'
'SMDDL'
'AVAR'
'URYDR'
'SIAVL'
'AVBR'
'SIAVR'
'SIADR'
'OLQVR'
'RIMR'
'IL2'
'URXR'
'AUAL'
'OLQDL'
'AQR'
'RIBR'
'IL2VL'
'URAVL'
'URAVR'
'IL2DR'
'OLQDR'
'IL2V'
'IL2DL'
'IL1DL'
'DD01'
'IL1VL'
'IL1VR'
'IL1DR'
'VA01'
'VA02'
'VB01'
'VB02'
'VB03'
'DA01'
'DA02'
'DB01'
'DB02'
'DD01'
'SIAVL'
'SIAVR'
'SAAVL'
'SAAVR'
'SIADL'
'SIADR'
'RIAL'
'RIAR'
'RMDDL'
'RMDDR'
'RMDVL'
'RMDVR'
'AWBL'
'AWBR'
'AWAL'
'AWAR'
'IL1L'
'IL1R'
'IL2L'
'IL2R'
)

# Now loop through the list of neurons and run the model
# But parallelize so that 12 are running at a time

CMD="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/utils/external/utils_pymc.py"
# Changes if running on gfp
if [ "$use_gfp" == "true" ]; then
  LOG_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/logs"
elif [ "$use_avb_hiscl" == "true" ]; then
  LOG_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_avb_hiscl/logs"
else
  LOG_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/logs"
fi

if [ "$residual_mode" ]; then
  LOG_DIR="${LOG_DIR}_${residual_mode}"
fi

mkdir -p "$LOG_DIR"

# I don't have access to the SLURM_ARRAY_TASK_ID variable, so I'm going to use the following workaround
# Create a temporary file to actually dispatch
SLURM_SCRIPT=$(mktemp /tmp/slurm_script.XXXXXX)
NUM_TASKS=${#neuron_list[@]}

# Set of option-specific variables
# gfp datasets are much faster to run
NUM_HOURS=18
MEM_PER_TASK=128G
if [ "$use_gfp" == "true" ]; then
  CMD="$CMD --gfp"
  NUM_HOURS=6
  MEM_PER_TASK=32G
elif [ "$use_avb_hiscl" == "true" ]; then
  CMD="$CMD --avb_hiscl"
fi

if [ "$simple_eigenworms" == "true" ]; then
  CMD="$CMD --simple_eigenworms"
fi

if [ "$residual_mode" ]; then
  CMD="$CMD --residual_mode $residual_mode"
fi

if [ "$cv_comparison" == "true" ]; then
  CMD="$CMD --cv_comparison"
fi

EXTRA_SBATCH_ARGS=""
if [ "$keep_large_vars" == "true" ]; then
  CMD="$CMD --keep_large_vars"
  EXTRA_SBATCH_ARGS="#SBATCH --license=scratch-highio"
fi

if [ "$debug" == "true" ]; then
  CMD="$CMD --debug"
  NUM_TASKS=1
  NUM_HOURS=1
fi

# Actually run
cat << EOF > $SLURM_SCRIPT
#!/bin/bash
#SBATCH --array=0-$(($NUM_TASKS-1))
#SBATCH --time=0-0$NUM_HOURS:00:00
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --cpus-per-task=12
$EXTRA_SBATCH_ARGS

# Reproduce the list for the subfile
my_list=(${neuron_list[@]})
task_string=\${my_list[\$SLURM_ARRAY_TASK_ID]}
echo "Running model for neuron: \$task_string with command: $CMD"

# Fix issues with multiple pymc instances, see:
# https://github.com/pymc-devs/pymc/issues/1463
export PYTENSOR_FLAGS="base_compiledir=\$TMPDIR/.pytensor"

LOG_FILE="$LOG_DIR/log_\$task_string.txt"
if [ DEBUG == "true" ]; then
  LOG_FILE="$LOG_DIR/log_debug.txt"
fi
python $CMD --neuron_name \$task_string > \$LOG_FILE 2>&1

echo "Finished running model for neuron: \$task_string"
EOF

# Submit the SLURM script
sbatch $SLURM_SCRIPT

# Clean up the temporary SLURM script
rm $SLURM_SCRIPT
