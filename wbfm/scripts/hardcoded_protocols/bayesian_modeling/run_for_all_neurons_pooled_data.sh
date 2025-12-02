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
'ANTIcorR'
'URYDR'
'SIAVL'
'AVBR'
'ANTIcorL'
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
'AIBR'
'AIBL'
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
'VG_anter_FWD_no_curve_L'
'VG_anter_FWD_no_curve_R'
'VG_middle_FWD_ramp_L'
'VG_middle_FWD_ramp_R'
'VG_middle_ramping_L'
'VG_middle_ramping_R'
'VG_post_FWD_L'
'VG_post_FWD_R'
'VG_post_turning_L'
'VG_post_turning_R'
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
'AVFL'
'AVFR'
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
if [ "$do_gfp" == "True" ]; then
  LOG_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/logs"
else
  LOG_DIR="/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/logs"
fi

for neuron in "${neuron_list[@]}"
do
  echo "Running model for neuron $neuron"
  log_fname="log_$neuron.txt"
  python $CMD --neuron_name "$neuron" --do_gfp "$do_gfp" > "$LOG_DIR/$log_fname" &
  # DEBUG: just break after one run
#  break
  sleep 1
  while [ "$(jobs | wc -l)" -ge 12 ]; do
    sleep 10
  done
done
