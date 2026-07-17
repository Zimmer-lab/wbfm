import argparse
import logging
import os
import sys
import time
import traceback

from wbfm.utils.general.utils_hardcoded import load_all_paper_datasets
from wbfm.utils.projects.finished_project_data import ProjectData
from submitit import AutoExecutor, LocalJob, DebugJob


def load_project_and_create_traces(project_path, keep_old_traces=True):
    # Try to properly log; see https://github.com/facebookresearch/hydra/issues/2664
    try:
        p = ProjectData.load_final_project_data(project_path)
        if not keep_old_traces:
            p.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)
        output = p.calc_all_paper_traces()
    except BaseException as e:
        traceback.print_exc(file=sys.stderr)
        raise e
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    return {'result': output}


def main(run_locally, keep_old_traces, DEBUG=False):
    """
    Create traces for all projects in the paper, caching them in the project folder.

    Parameters
    ----------
    run_locally
    DEBUG

    Returns
    -------

    """
    print("Starting with options: run_locally=", run_locally, " keep_old_traces=", keep_old_traces, 
          "DEBUG=", DEBUG)
    # Load all paths to datasets used in the paper
    all_project_paths = load_all_paper_datasets(only_load_paths=True)

    if DEBUG:
        # Just one project
        k = list(all_project_paths.keys())[0]
        all_project_paths = {k: all_project_paths[k]}
    logging.info(f"Found {len(all_project_paths)} projects to process")

    # Set up the executor
    if run_locally:
        executor = AutoExecutor(folder="/tmp/submitit_runs", cluster='debug')
    else:
        # Can't use /tmp/submitit_runs because the cluster can't access it
        # https://github.com/facebookincubator/submitit/blob/main/docs/tips.md
        log_dir = os.path.join(os.getcwd(), 'log')
        os.makedirs(log_dir, exist_ok=True)
        executor = AutoExecutor(folder=log_dir, cluster='slurm')
        executor.update_parameters(slurm_time=f"0-04:00:00")
        executor.update_parameters(cpus_per_task=16)
        executor.update_parameters(slurm_mem="128G")
        executor.update_parameters(slurm_partition="basic,gpu")
        executor.update_parameters(slurm_job_name="create_paper_traces")
    executor.update_parameters(timeout_min=60)

    # Schedule jobs
    jobs = []
    for name, project_path in all_project_paths.items():
        # Make a new folder in the parent folder
        # Add the baseline parameters, and save in this folder
        print(f"Submitting job to build paper traces for {name}")
        job = executor.submit(load_project_and_create_traces, project_path, keep_old_traces)
        jobs.append(job)
        if DEBUG:
            break
        time.sleep(1)
    num_total_jobs = len(all_project_paths)
    # Run until all the jobs have finished and our budget is used up.
    while jobs:
        for job in jobs[:]:
            # Poll if any jobs completed
            # Local and debug jobs don't run until .result() is called.
            if job.done() or type(job) in [LocalJob, DebugJob]:
                # The log file isn't being produced, so print the stdout instead
                result = job.result()
                print(f"Job {job} finished with result shape: {result['result']['paper_traces'].shape}")
                jobs.remove(job)

        # Sleep for a bit before checking the jobs again to avoid overloading the cluster.
        # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
        print(f"Remaining jobs: {len(jobs)}/{num_total_jobs}")
        if len(jobs) > 0:
            time.sleep(5*60)
    print("All jobs finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_locally', action='store_true', help='Run the jobs locally')
    parser.add_argument('--keep_old_traces', action='store_true', help='Keep previously run traces')
    parser.add_argument('--DEBUG', action='store_true', help='Run the jobs locally')
    args = parser.parse_args()

    main(run_locally=args.run_locally, keep_old_traces=args.keep_old_traces, DEBUG=args.DEBUG)
