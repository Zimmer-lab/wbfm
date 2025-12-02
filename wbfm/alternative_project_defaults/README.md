# Alternative config file defaults for specific project types

Each folder here will contain specific config.yaml files, which can be copied to created projects using the copy_config_file_to_multiple_projects.sh script.
See that script for help and more details.

Custom config files can be created for specific project types, and then copied in the same way.

## Increase the runtime of snakemake rules for longer recordings
The runtime for each snakemake rule is determined by the cluster_config.yaml file which is located in the snakemake folder of your project. You could change the runtimes manually but if you have to increase or decrease it for multiple projects it is easier to just copy a pre-exisiting template, provided in the .. folder.

### How to do it: 
the command to copy a config file into multiple projects looks the following way. 
bash "path to copy_config_file_to_multiple_projects.sh" -t "target parent folder path that contains your projects" -c "path to the file you want to copy" -n "dryrun"

### or more concrete
increase runtime: 
bash /lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/postprocessing/copy_config_file_to_multiple_projects.sh -t "target path" -c /lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/alternative_project_defaults/long_video/cluster_config.yaml (-n)

decrease runtime:
bash /lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/postprocessing/copy_config_file_to_multiple_projects.sh -t "target path" -c /lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/alternative_project_defaults/short_video/cluster_config.yaml (-n)

You can also use this script to copy other (config) files. 

## Keeping intermediate files 
By default intermediate files are deleted after each snakemake rule is finished. This is done by a wrapper called _cleanup_helper(). If you would like to keep all intermediate files (behavior pipeline only) for your project you can simply go into the snakemake_config.yaml file in the snakemake folder of your project and set "delete_intermediate_files" to false. 
If you want to keep an intermediate file of a specific rule only you can go into the pipeline.smk file of your project (also found in snakemake folder) and remove the _cleanup_helper() wrapper from the output. 

In case you want to do this for multiple projects you can use the copy_config_file_to_multiple_projects.sh script to copy your file of choice to other projects as described . You just have to change the target path accordingly.
