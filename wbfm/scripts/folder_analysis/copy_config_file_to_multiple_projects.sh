#!/bin/bash

# Add help function
function usage {
    echo "Usage: $0 "
    echo "  -t: target parent folder to search for projects"
    echo "  -c: new config file to copy to multiple projects; files will be updated based on EXACT name matches"
    echo "  -n: dry run (just print files that will be modified; default: false)"
    echo "  -h: help (print this message and quit)"
    echo "Example: $0 -t /path/to/projects -c /path/to/new_config.yaml"
    exit 1
}

# Get command line args

DRYRUN="False"
while getopts :t:c:nh flag
do
    case "${flag}" in
        t) TARGET_PARENT_FOLDER=${OPTARG};;
        c) NEW_CONFIG=${OPTARG};;
        n) DRYRUN="True";;
        h) usage;;
        *) echo "Unknown flag"; usage;;
    esac
done

# Check that the new config file exists and is not empty
if [ ! -f "$NEW_CONFIG" ]; then
    echo "Error: $NEW_CONFIG does not exist"
    exit 1
fi

if [ ! -s "$NEW_CONFIG" ]; then
    echo "Error: $NEW_CONFIG is empty"
    exit 1
fi

# Recursively within the target parent folder, find all exactly named config files and copy the new config file to them
# Regardless, print the files that will be modified
echo "Searching for files to modify..."
find "$TARGET_PARENT_FOLDER" -type f -name $(basename "$NEW_CONFIG")
if [ "$DRYRUN" == "True" ]; then
    echo "DRY RUN: the following files will be modified:"
else
    find "$TARGET_PARENT_FOLDER" -type f -name $(basename "$NEW_CONFIG") -exec cp "$NEW_CONFIG" {} \;
    echo "Successfully copied $NEW_CONFIG to all matching files"
fi
