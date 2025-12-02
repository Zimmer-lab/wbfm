import argparse
import subprocess
from pathlib import Path
import yaml
import itertools

SBATCH_TEMPLATES = {
    "copy_project": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

{command}
""",
    "modify_config": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

{command}
""",
    "track": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

{command}
""",
    "extract_traces": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

{command}
"""
}

def submit_job(script_path, dependency=None, debug=False):
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(script_path))

    if debug:
        print(f"[DEBUG] Submitting job: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    stdout = result.stdout.decode().strip()
    if debug:
        print(f"[DEBUG] Job submission output: {stdout}")
    job_id = stdout.strip().split()[-1]
    return job_id

def write_and_submit_job(trial_name, step_name, command, dependency=None, debug=False):
    job_name = f"{step_name}_{trial_name}"
    script_path = Path(f"sbatch_scripts/{job_name}.sh")

    script_template = SBATCH_TEMPLATES[step_name]
    script_content = script_template.format(job_name=job_name, command=command)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)

    if debug:
        print(f"[DEBUG] Wrote SBATCH script: {script_path}")
        print(f"[DEBUG] Script content:\n{script_content}")

    return submit_job(script_path, dependency=dependency, debug=debug)

def submit_copy_job(trial_name, finished_path, new_location, make_project_script, debug=False):
    cmd = (
        f"python {make_project_script} "
        f"with project_path={finished_path} "
        f"target_directory={new_location} "
        f"target_suffix={trial_name} "
        f'steps_to_keep="[\'preprocessing\', \'segmentation\']"'
    )
    return write_and_submit_job(trial_name, "copy_project", cmd, debug=debug)

def submit_modify_config_job(trial_name, new_project_path, opt_db_params, opt_umap_params, debug=False, dependency=None):
    cmd = (
        f"python - <<'EOF'\n"
        f"import yaml, pathlib\n"
        f"config_path = pathlib.Path('{new_project_path}') / '3-tracking' / 'tracking_config.yaml'\n"
        f"with open(config_path, 'r') as f:\n"
        f"    config = yaml.safe_load(f)\n"
        f"if 'opt_db' not in config or config['opt_db'] is None:\n"
        f"    config['opt_db'] = {{}}\n"
        f"if 'opt_umap' not in config or config['opt_umap'] is None:\n"
        f"    config['opt_umap'] = {{}}\n"
        f"config['opt_db'].update({opt_db_params})\n"
        f"config['opt_umap'].update({opt_umap_params})\n"
        f"with open(config_path, 'w') as f:\n"
        f"    yaml.safe_dump(config, f)\n"
        f"print('Updated', config_path)\n"
        f"EOF"
    )
    return write_and_submit_job(trial_name, "modify_config", cmd, dependency=dependency, debug=debug)


def submit_tracking_job(trial_name, new_location, barlow_model, track_script, dependency_jobid, debug=False, use_projection_space=True):
    project_path = f"{new_location}{trial_name}"
    cmd = (
        f"python {track_script} "
        f"with project_path={project_path} "
        f"model_fname={barlow_model} "
        f"use_projection_space={use_projection_space}"
    )
    return write_and_submit_job(trial_name, "track", cmd, dependency=dependency_jobid, debug=debug)

def submit_trace_job(trial_name, new_location, dispatcher_script, dependency_jobid, debug=False):
    project_path = f"{new_location}{trial_name}"
    cmd = (
        f"sbatch {dispatcher_script} "
        f"-t {project_path} -s 4"
    )
    return write_and_submit_job(trial_name, "extract_traces", cmd, dependency=dependency_jobid, debug=debug)

def parse_args():
    parser = argparse.ArgumentParser(description="Run parameter sweep for tracking configs with copy→yaml-edit→track→extract")
    parser.add_argument("--wbfm-home", required=True, help="Path to the wbfm codebase root directory")
    parser.add_argument("--finished-path", required=True, help="Path to finished project")
    parser.add_argument("--new-location", required=True, help="Base path for new projects")
    parser.add_argument("--source-trial", required=True, help="Path to a single trial directory containing the model")
    parser.add_argument("--model-fname", default="resnet50.pth", help="Model filename inside the trial folder")
    parser.add_argument("--use_projection_space", required=True, help="Using projection space or final embedding space")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def main():
    args = parse_args()
    make_project_script = Path(args.wbfm_home) / "scripts/postprocessing/make_project_like.py"
    track_script = Path(args.wbfm_home) / "scripts/pipeline_alternate/3-track_using_barlow.py"
    dispatcher_script = Path(args.wbfm_home) / "scripts/cluster/single_step_dispatcher.sbatch"
    use_projection_space = args.use_projection_space

    source_trial_dir = Path(args.source_trial)
    barlow_model_path = source_trial_dir / args.model_fname
    if not barlow_model_path.is_file():
        print(f"Error: Model file not found at {barlow_model_path}")
        return

    finished_path_parent = Path(args.finished_path).parent.name
    project_base_path = str(Path(args.new_location) / finished_path_parent)

    # === Define sweep parameters here ===
    opt_db_grid = {
        "min_cluster_size": [0.5, 0.7],
        "min_samples": [0.01, 0.05],
    }
    opt_umap_grid = {
        "n_components": [10, 20],
        "n_neighbors": [5, 10],
        "min_dist": [0.0, 0.1],
    }

    for db_vals in itertools.product(*opt_db_grid.values()):
        opt_db_params = dict(zip(opt_db_grid.keys(), db_vals))
        for umap_vals in itertools.product(*opt_umap_grid.values()):
            opt_umap_params = dict(zip(opt_umap_grid.keys(), umap_vals))

            trial_name = (
                f"{source_trial_dir.name}_db-"
                + "_".join(f"{k}{v}" for k, v in opt_db_params.items())
                + "_umap-"
                + "_".join(f"{k}{v}" for k, v in opt_umap_params.items())
            )

            if args.debug:
                print(f"[DEBUG] Starting pipeline for {trial_name}")

            # 1. Copy project
            copy_jobid = submit_copy_job(
                trial_name, args.finished_path, args.new_location, make_project_script, debug=args.debug
            )

            # 2. Modify YAML (after copy)
            new_project_path = project_base_path + trial_name
            modify_jobid = submit_modify_config_job(
                trial_name, new_project_path, opt_db_params, opt_umap_params,
                dependency=copy_jobid, debug=args.debug
            )

            # 3. Track (after yaml modified)
            track_jobid = submit_tracking_job(
                trial_name, project_base_path, str(barlow_model_path), track_script,
                dependency_jobid=modify_jobid, debug=args.debug, use_projection_space=use_projection_space
            )

            # 4. Extract traces (after tracking)
            submit_trace_job(
                trial_name, project_base_path, dispatcher_script,
                dependency_jobid=track_jobid, debug=args.debug
            )

if __name__ == "__main__":
    main()
