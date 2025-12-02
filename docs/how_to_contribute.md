# How to contribute

Fundamentally this is a small project with few automatic tests... for now!
However, there is a full-pipeline test that can be run, which is a fast version of all of the steps.
If any code is being contributed to main, or if dev is being merged into main, this project should be run.

1. Confirm or create a project, and set the number of frames very low. Currently I use:/lisc/data/scratch/neurobiology/zimmer/Charles/dlc_stacks/project_pytest/
2. If needed, delete analysis files from a previous run using scripts/postprocessing/delete_analysis_files.py
3. Run the full snakemake pipeline as described in the main README.md. This should take about 5 minutes.
4. If no errors occur, merging should be safe!