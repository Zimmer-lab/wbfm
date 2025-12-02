#!/bin/bash

# This script converts the Pedro's NWB files to the new format
# Pedro moved his folders to a parent folder, so we can just convert all the files in the parent folder

input_folder="/lisc/data/scratch/neurobiology/zimmer/Augusto/Charlie"

# I don't have permissions there, so I need to output the files to my folder
output_folder="/lisc/data/scratch/neurobiology/zimmer/fieseler/group_service/neurodata_without_borders"

# Loop over the subfolders in the input folder, and apply the python conversion script
for subfolder in "$input_folder"/*; do
    echo "$subfolder"
    # Apply the python script
    python -c "from wbfm.utils.general.utils_nwb import nwb_from_pedro_format; nwb_from_pedro_format('$subfolder', '$output_folder')"
done
