#!/bin/bash
#SBATCH --job-name=neuropal_pipeline
#SBATCH --output=neuropal_%j.log
#SBATCH --error=neuropal_%j.err
#SBATCH --time=04:00:00              # total max runtime for the job
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard

# call this script: sbatch get_np_stack_in_project.sh /lisc/path/to/project /lisc/path/to/np/stack
# ================================
# Input arguments
# ================================
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <PROJECT_PATH> <RAW_STACK_PATH>"
    exit 1
fi

PROJECT_PATH="$1"
RAW_STACK_PATH="$2"

SCRIPT_ADD="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/postprocessing/0+add_neuropal_to_project.py"
SCRIPT_SEGMENT="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/postprocessing/0+segment_neuropal_in_project.py"

# ================================
# Run add_neuropal_to_project
# ================================
echo "=== START add_neuropal_to_project: $(date) ==="
srun python3 "$SCRIPT_ADD" with \
project_path="$PROJECT_PATH" \
raw_neuropal_path="$RAW_STACK_PATH" \
copy_data=True \
DEBUG=False
echo "=== END add_neuropal_to_project: $(date) ==="

# ================================
# Run segment_neuropal_in_project
# ================================
echo "=== START segment_neuropal_in_project: $(date) ==="
srun python3 "$SCRIPT_SEGMENT" with \
project_path="$PROJECT_PATH" \
subsample_in_z=True \
DEBUG=False
echo "=== END segment_neuropal_in_project: $(date) ==="

echo "=== ALL STEPS FINISHED: $(date) ==="