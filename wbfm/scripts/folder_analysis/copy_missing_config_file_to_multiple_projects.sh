#!/bin/bash

# Add help function
function usage {
    echo "Usage: $0 "
    echo "  -t: target parent folder to search for projects"
    echo "  -c: full path of new config file to copy to multiple projects; files will be updated based on EXACT name matches"
    echo "  -s: sibling config file (not path) to use as a base, assuming the target config file is missing"
    echo "  -n: dry run (just print files that will be modified; default: false)"
    echo "  -o: do not overwrite existing target config files (default: overwrite existing files)"
    echo "  -h: help (print this message and quit)"
    echo "Example: $0 -t /path/to/projects -c /path/to/new_config.yaml"
    exit 1
}

DRYRUN="False"
OVERWRITE="True"
while getopts :t:c:s:nh flag
do
    case "${flag}" in
        t) TARGET_PARENT_FOLDER=${OPTARG};;
        c) NEW_CONFIG=${OPTARG};;
        s) SIBLING_CONFIG=${OPTARG};;
        n) DRYRUN="True";;
        o) OVERWRITE="False";;
        h) usage;;
        *) echo "Unknown flag"; usage;;
    esac
done

NAME_OF_CONFIG_FILE=$(basename "$NEW_CONFIG")

find "$TARGET_PARENT_FOLDER" -type d -name "snakemake" | while read -r dir; do
    sibling="$dir/$SIBLING_CONFIG"
    target="$dir/$NAME_OF_CONFIG_FILE"

    # If this sibling is found, then copy the new config file to the same directory
    if [ -f "$sibling" ]; then
        if [ -f "$target" ] && [ "$OVERWRITE" == "False" ]; then
            echo "[*] Skipping $target (already exists)"
            continue
        fi
        if [ "$DRYRUN" == "True" ]; then
            echo "[DRY RUN] Would copy $NEW_CONFIG to $target"
        else
            cp "$NEW_CONFIG" "$target"
            echo "[+] Copied missing $NEW_CONFIG to $target"
        fi
    else
        echo "[!] Sibling config $sibling not found; skipping."
        continue
    fi
done
