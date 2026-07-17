#!/bin/bash

# Get the target folder
while getopts s: flag
do
    case "${flag}" in
        s) parent_dir=${OPTARG};;
        *) echo "Error: Unknown flag"; exit 1;;
    esac
done

# Define the file path relative to each subfolder
folder_path="behavior"

# Get all subfolders as an array
subfolders=("$parent_dir"/*/)
echo "Found ${#subfolders[@]} subfolders"

# Iterate through top-level subfolders and apply permissions
for subfolder in "${subfolders[@]}"; do
    target_dir="$subfolder$folder_path"
    if [ -d "$target_dir" ]; then
        chmod -R g+w,o+rw "$target_dir"
        echo "Changed permissions for $target_dir"
    else
        echo "No such directory: $target_dir"
    fi
done
