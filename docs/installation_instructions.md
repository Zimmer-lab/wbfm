
# Installation for developing

### Install Anaconda/Mamba

We suggest installing Mamba:
https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

The official page is here, but please be wary of the terms of service:
https://www.anaconda.com/products/individual

### Get the code

Clone [this repository](https://github.com/Zimmer-lab/wbfm). 
This is not strictly necessary if you know what you're doing, but we provide a conda environment yaml file for easy installation (next section).

### Install the environment

#### Pre-installed environments (Zimmer lab)

Note: there are pre-installed environments living on the cluster, at:
/lisc/data/scratch/neurobiology/zimmer/.conda/envs/wbfm

They can be activated using:
```commandline
conda activate /lisc/data/scratch/neurobiology/zimmer/.conda/envs/wbfm
```

#### Creating a new conda environment

Note: if you are just using the GUI, then you can use a simplified environment.
Detailed instructions can be found in the [README](wbfm/gui/README.md) file under the gui section
For running the full pipeline you need the environment found here:

conda-environments/wbfm.yaml

This installs the public packages and the Zimmer lab libraries directly from GitHub.

#### Installing repositories from github (not needed if using our wbfm.yaml file)

Note that you will have to install additional repositories from github:

1. centerline_behavior_annotation: https://github.com/Zimmer-lab/centerline_behavior_annotation 
2. imutils: https://github.com/Zimmer-lab/imutils

You can either clone and install locally (instructions below) or install via pip using for example:

```commandline
pip install git+[url here]
```

#### Installing local (cloned) repositories 

If you want to install local repositories

Do `conda activate YOUR_ENVIRONMENT` and install the local code in the following way:

1. cd to the repository
2. run: `pip install -e .`
3. Repeat steps 1-2 for the other repositories


#### Summary of installations

You will install 4 "things": 1 conda environment (from the yaml file) and 3 custom packages
