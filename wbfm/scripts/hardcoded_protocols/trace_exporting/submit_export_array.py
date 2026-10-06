"""
Submit NWB paper-trace exports as a SLURM job array, one dataset per task.

Unlike a hand-written sbatch file with a hard-coded --array range, this
script first enumerates the actual task list (fast: project *paths* only),
drops datasets whose export already exists, and submits an array sized to
exactly the remaining work. Run it from a login node with sbatch available:

    python submit_export_array.py --include_image_data --max_concurrent 4
    python submit_export_array.py --dry_run   # print the sbatch file without submitting
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_paper_data_as_nwb import build_tasks, filter_existing_tasks, get_parent_dir, write_taskfile

# Full cluster paths, so the job does not depend on the submitting shell's
# environment (PATH, conda activation, working directory).
CLUSTER_PYTHON = '/lisc/data/scratch/neurobiology/zimmer/.conda/envs/wbfm/bin/python'
CLUSTER_SCRIPT_DIR = ('/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm'
                      '/wbfm/scripts/hardcoded_protocols/trace_exporting')
CLUSTER_EXPORT_SCRIPT = os.path.join(CLUSTER_SCRIPT_DIR, 'export_paper_data_as_nwb.py')


def build_sbatch(n_tasks, job_name, max_concurrent, mem, time, cpus_per_task, export_flags, taskfile):
    array_spec = f'0-{n_tasks - 1}%{max_concurrent}' if n_tasks > 0 else '0-0%1'
    all_flags = export_flags + ['--taskfile', taskfile, '--only_index', '"$SLURM_ARRAY_TASK_ID"']
    export_cmd = f'{CLUSTER_PYTHON} -u {CLUSTER_EXPORT_SCRIPT} {" ".join(all_flags)}'
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array={array_spec}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --chdir={CLUSTER_SCRIPT_DIR}
#SBATCH --output={CLUSTER_SCRIPT_DIR}/slurm-%A_%a.out

{export_cmd}
"""


def main():
    parser = argparse.ArgumentParser(
        description='Enumerate NWB export tasks, then submit an exactly-sized SLURM job array')
    parser.add_argument('--include_image_data', action='store_true', help='Whether to include image data in the export')
    parser.add_argument('--delete_existing', action='store_true',
                        help='Whether to delete existing export files before exporting (default skips them)')
    parser.add_argument('--suffixes', nargs='+', default=['gfp', '', 'mutant', 'immob'],
                        help='Dataset suffixes to export')
    parser.add_argument('--max_concurrent', type=int, default=16,
                        help='Max array tasks running at once')
    parser.add_argument('--job_name', default='nwb-export')
    parser.add_argument('--mem', default='64G')
    parser.add_argument('--time', default='8:00:00')
    parser.add_argument('--cpus_per_task', default=8)
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the generated sbatch file and exit without submitting')
    args = parser.parse_args()

    tasks = build_tasks(args.suffixes, get_parent_dir(args.include_image_data))
    n_total = len(tasks)

    # Filter BEFORE submitting: drop datasets whose export already exists, so
    # the array contains exactly the remaining work and no task is wasted on
    # a start-load-and-exit no-op. (Each task still re-checks at runtime as a
    # backstop.) Uses the same shared filter as the --only_index worker, so
    # array indexes point at the same datasets in both.
    skipped = []
    if not args.delete_existing:
        tasks, skipped = filter_existing_tasks(tasks, args.include_image_data)
    n_tasks = len(tasks)
    print(f'{n_total} project(s) total, {len(skipped)} already exported, '
          f'submitting {n_tasks}; array range will be 0-{n_tasks - 1}', flush=True)
    if n_tasks == 0:
        print('Nothing remaining; nothing to submit.', flush=True)
        raise SystemExit(0)

    export_flags = []
    if args.include_image_data:
        export_flags.append('--include_image_data')
    if args.delete_existing:
        export_flags.append('--delete_existing')
    if args.suffixes != ['gfp', '', 'mutant', 'immob']:
        export_flags += ['--suffixes'] + list(args.suffixes)

    import uuid
    taskfile_name = f'export_tasks_{uuid.uuid4().hex[:8]}.txt'
    taskfile = os.path.join(CLUSTER_SCRIPT_DIR, taskfile_name)

    sbatch_text = build_sbatch(n_tasks, args.job_name, args.max_concurrent,
                               args.mem, args.time, args.cpus_per_task, export_flags, taskfile)
    if args.dry_run:
        print(sbatch_text, flush=True)
        print(f'--- taskfile WOULD be written to {taskfile} ---', flush=True)
        print('--- tasks that WOULD be dispatched ---', flush=True)
        for _, name, _, _ in tasks:
            print(f'  dispatch {name}', flush=True)
        if skipped:
            print('--- already exported (would be skipped) ---', flush=True)
            for name in skipped:
                print(f'  skip {name}', flush=True)
        return

    # Freeze the index -> dataset mapping BEFORE submitting, so array indexes
    # cannot drift when files appear mid-run. Unique name per submission so
    # concurrent submissions cannot clobber each other.
    write_taskfile(taskfile, tasks)
    print(f'Wrote taskfile {taskfile} with {n_tasks} entries', flush=True)

    proc = subprocess.run(['sbatch'], input=sbatch_text, capture_output=True, text=True)
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        print(proc.stderr, flush=True)
        raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
