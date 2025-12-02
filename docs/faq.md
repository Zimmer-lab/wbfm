# FAQ

## Changing defaults for new projects

There are two places with default parameters: wbfm_config.yaml and the project structure within new_project_defaults.


### wbfm_config.yaml

This is a configuration file that contains hardcoded paths to neural networks, required for segmentation, tracking, and behavior.
If you want to change these defaults, then you should copy this file with desired modifications to: 
```~/.wbfm/config.yaml```

Note that you can get the raw file via a local installation, or directly from github [here](https://github.com/Zimmer-lab/wbfm/blob/main/wbfm/utils/projects/wbfm_config.yaml)

### new_project_defaults

This is a project structure which initializes any newly created project.
If you want to change these defaults, there is not a simple beginner-friendly way to do so, but it is not so complicated.
Fundamentally you must modify the project structure wherever it is installed:
1. If you installed via the conda environment file, then the repo will be (on linux) here: ```ENVIRONMENT_NAME/lib/python3.8/site-packages/wbfm```
2. If you installed the repo locally with an editable install, then you should know where it is



# Known issues, including warnings

## Known issues in Napari

Sometimes when "show selected" is highlighted and you change the labels layer, then when you go off of "show selected" it has a too-short colormap, and gives an error.

Thus far, clicking "show selected" a couple times fixes this
