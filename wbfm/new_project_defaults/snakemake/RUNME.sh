#!/bin/bash

# Add help function
function usage {
    echo "Usage: $0 [-s rule] [-n] [-c] [-h] [-R restart_rule]"
    echo "  -s: snakemake rule to run until, i.e. target rule (default: traces; other options: traces_and_behavior, behavior)"
    echo "  -R: snakemake rule to restart from (default: None)"
    echo "  -n: dry run (default: false)"
    echo "  -f: force only one rule (default: false)"
    echo "  -c: do NOT use cluster (default: false, i.e. run on cluster)"
    echo "  -h: display help (this message)"
    exit 1
}

# Example from: https://hackmd.io/@bluegenes/BJPrrj7WB

# Get the snakemake rule to run as a command line arg

# Get other command line args
# Set defaults: not a dry run and on the cluster
DRYRUN=""
USE_CLUSTER="True"
RULE="traces"
RESTART_RULE=""
FORCE_ONLY_ONE_RULE="False"

while getopts :s:R:nfch flag
do
    case "${flag}" in
        s) RULE=${OPTARG};;
        R) RESTART_RULE=${OPTARG};;
        n) DRYRUN="True";;
        f) FORCE_ONLY_ONE_RULE="True";;
        c) USE_CLUSTER="";;
        h) usage;;
        *) echo "Unknown flag"; usage; exit 1;;
    esac
done

# Package options
SBATCH_OPT="sbatch -t {cluster.time} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output} --gres {cluster.gres} --job-name {rule} --constraint '{cluster.constraint}'"
SNAKEMAKE_OPT="-s pipeline.smk --latency-wait 60 --cores 56 --retries 2"
if [ -n "$RESTART_RULE" ]; then
    SNAKEMAKE_OPT="$SNAKEMAKE_OPT -R $RESTART_RULE"
fi
if [ "$FORCE_ONLY_ONE_RULE" = "True" ]; then
    SNAKEMAKE_OPT="$SNAKEMAKE_OPT --allowed-rules $RULE --force"
fi
NUM_JOBS_TO_SUBMIT=8

# Slurm doesn't properly deal with TIMEOUT errors in subjobs, so we need to create a script to deal with them
# In principle we will use a python script, but because this needs to be valid for all users, we will create a temporary
# script that will be deleted after the job is done
# Details: https://snakemake.readthedocs.io/en/v7.7.0/tutorial/additional_features.html#using-cluster-status
# More complex example: https://github.com/Snakemake-Profiles/slurm/blob/master/%7B%7Bcookiecutter.profile_name%7D%7D/slurm-status.py
CLUSTER_STATUS_SCRIPT=$(mktemp /tmp/slurm_script.XXXXXX)

cat << EOF > "$CLUSTER_STATUS_SCRIPT"
#!/usr/bin/env python
import subprocess
import sys
import time

jobid = sys.argv[1]

try:
  output = str(subprocess.check_output("sacct -j %s --format State --noheader | head -1 | awk '{print \$1}'" % jobid, shell=True).strip())
except subprocess.TimeoutExpired:
  output = "UNKNOWN"  # If the job is not found, we will consider it as unknown and check again later
except subprocess.CalledProcessError as e:
  output = "UNKNOWN"  # If the job is not found, we will consider it as unknown and check again later

# Define the possible running statuses, and check if the job is running
# Note: the print statements must be exactly as shown here for snakemake to interpret them correctly
running_status=["PENDING", "CONFIGURING", "COMPLETING", "RUNNING", "SUSPENDED", "UNKNOWN"]
if "COMPLETED" in output:
  print("success")
elif any(r in output for r in running_status):
  print("running")
else:
  print("failed")
EOF

# Make the script executable
chmod +x "$CLUSTER_STATUS_SCRIPT"

# Actual command
if [ "$DRYRUN" ]; then
    echo "DRYRUN of snakemake $RULE rule with options $SNAKEMAKE_OPT"
    snakemake "$RULE" --debug-dag -n $SNAKEMAKE_OPT
elif [ -z "$USE_CLUSTER" ]; then
    echo "Running snakemake rule $RULE locally with options $SNAKEMAKE_OPT"
    snakemake -s pipeline.smk --unlock  # Unlock the folder, just in case
    snakemake "$RULE" $SNAKEMAKE_OPT
else
    echo "Running snakemake rule $RULE on the cluster with options $SNAKEMAKE_OPT"
    snakemake -s pipeline.smk --unlock  # Unlock the folder, just in case
    snakemake "$RULE" $SNAKEMAKE_OPT --cluster "$SBATCH_OPT --parsable" --cluster-config cluster_config.yaml --jobs $NUM_JOBS_TO_SUBMIT --cluster-status "$CLUSTER_STATUS_SCRIPT"
fi
