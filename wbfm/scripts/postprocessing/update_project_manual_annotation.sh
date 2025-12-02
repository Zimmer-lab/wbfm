#!/usr/bin/env bash

# This script moves an excel file with manual annotations to the project folder
# Start point:
# - Folder of excel files with manual annotations
# - Folder of folder of projects
# - The excel file has the same name as the project folder
#
# End point:
# - The excel file is moved to the project folder, to the 3-tracking/manual_annotation folder
#

# Set the path to the folder with the excel files
excel_file_prefix="wbfm_neuron_ids"
excel_parent_folder="/home/charles/Downloads"
excel_target_folder="$excel_parent_folder/$excel_file_prefix"
log_fname="$excel_parent_folder/log.txt"

# Check if the unzipped excel folder exists, and crash if so
if [ -d "$excel_target_folder" ]; then
    echo "Excel folder already exists: $excel_target_folder"
    echo "You must manually delete it!"
    # Exit with error
    exit 1
fi

# The actual folder will be zipped, with the above path as the base name but with an additional number suffix
# First, find the exact folder and unzip it. Then continue with moving individual files.
zipped_excel_folder=$(find "$excel_parent_folder" -type f -name "$excel_file_prefix*.zip")
if [ -f "$zipped_excel_folder" ]; then
    echo "Found zipped excel folder: $zipped_excel_folder"

    # Unzip the folder
    unzip "$zipped_excel_folder" -d "$excel_target_folder" > $log_fname
else
    echo "Could not find zipped excel folder: $zipped_excel_folder"
    echo "You must manually download from google drive!"
    # Exit with error
    exit 1
fi

# The unzipped folder may have a subfolder called "wbfm_neuron_ids"; if so we need to update excel_folder
nested_excel_folder="$excel_target_folder/$excel_file_prefix"
if [ -d $nested_excel_folder ]; then
    echo "Found nested unzipped excel folder: $nested_excel_folder, updating target folder to that path"
    # Update the excel folder
    excel_target_folder="$nested_excel_folder"
else
    echo "Nested folder $nested_excel_folder was not found; target remains: $excel_target_folder"
fi

# Set the path to the folder with the projects
project_parent_folder="/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects"

# Loop over all project parent folders (actually projects are subfolders within the parent folder)
for project_parent in "$project_parent_folder"/*; do
    if [ -d "$project_parent" ] && [ ! -L "$project_parent" ]; then
        echo "Checking folder: $project_parent"

        # Loop over all projects within the parent folder
        for project in "$project_parent"/*; do
            if [ -d "$project" ] && [ ! -L "$project" ]; then
                echo "Checking folder: $project"

                # Get the name of the project
                project_name=$(basename "$project")

                # Get the path to the excel file
                excel_file="$excel_target_folder/$project_name.xlsx"

                # Check if the excel file exists
                if [ -f "$excel_file" ]; then
                    echo "Found excel file: $excel_file"

                    # Get the path to the manual annotation folder
                    manual_annotation_folder="$project/3-tracking/manual_annotation"

                    # Check if the manual annotation folder exists
                    if [ -d "$manual_annotation_folder" ]; then
                        # Print with green text
                        echo -e "\e[32mFound manual annotation folder: $manual_annotation_folder\e[0m"

                        # Move the excel file to the manual annotation folder
                        mv "$excel_file" "$manual_annotation_folder"

                        # Rename to be called "manual_annotation.xlsx" instead of "$project_name.xlsx"
                        mv "$manual_annotation_folder/$project_name.xlsx" "$manual_annotation_folder/manual_annotation.xlsx"
                    else
                        # Print with red text
                        echo -e "\e[31mCould not find manual annotation folder: $manual_annotation_folder\e[0m"
                    fi
                else
                    echo "Could not find excel file: $excel_file"
                fi
            fi
        done
    fi
done
