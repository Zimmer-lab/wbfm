##
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from wbfm.utils.nn_utils.superglue import SuperGlueFullVolumeNeuronImageFeaturesDatasetFromProject, \
    SuperGlueModel, NeuronImageFeaturesDataModuleFromMultipleProjects
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_redo_steps import correct_tracks_dataframe_using_project
from wbfm.utils.general.utils_hardcoded import load_hardcoded_neural_network_paths


def main():
    # Load data
    all_project_data = load_data_with_ground_truth()

    # Set up the network
    base_dataset_class = SuperGlueFullVolumeNeuronImageFeaturesDatasetFromProject
    batch_size = 1
    max_epochs = 20
    # Take the last gpu; for now, multiple gpus are not supported
    gpus = [torch.cuda.device_count() - 1]
    train_loader = NeuronImageFeaturesDataModuleFromMultipleProjects(batch_size=batch_size,
                                                                     all_project_data=all_project_data,
                                                                     base_dataset_class=base_dataset_class,
                                                                     dataset_kwargs=dict(
                                                                         num_to_calculate=1000))  # Calculates from each project
    # Explicitly setup to see if there are problems
    train_loader.setup()
    # Start from pretrained
    path_dict = load_hardcoded_neural_network_paths()
    superglue_parent_folder = path_dict['tracking_paths']['model_parent_folder']
    superglue_model_name = path_dict['tracking_paths']['global_tracking_model_name']
    superglue_path = os.path.join(superglue_parent_folder, superglue_model_name)

    model = SuperGlueModel.load_from_checkpoint(superglue_path)
    # model = SuperGlueModel(feature_dim=840, lr=1e-5)
    model.lr = 1e-5
    with wandb.init(project="superglue_training_multiple_projects_fixed_r2w4", entity="charlesfieseler") as run:
        wandb_logger = WandbLogger()

        trainer = Trainer(gpus=gpus, max_epochs=max_epochs, terminate_on_nan=True,
                          stochastic_weight_avg=True,
                          logger=wandb_logger)
        wandb_logger.watch(model, log='all', log_freq=1)

        trainer.fit(model, train_loader)
    out_folder = '/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/nn_checkpoints'
    model_fname = os.path.join(out_folder, 'superglue_neurons_4_datasets_06_22.ckpt')
    trainer.save_checkpoint(model_fname)


def load_data_with_ground_truth():
    ## Load the 4 datasets that have manual annotations
    folder_name = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/"
    fname = os.path.join(folder_name, "round1_worm1/project_config.yaml")
    project_data1 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round1_worm4/project_config.yaml")
    project_data2 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round2_worm6/project_config.yaml")
    project_data3 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round2_worm3/project_config.yaml")
    project_data4 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    ## Confirm that the tracks are correct
    df1 = correct_tracks_dataframe_using_project(project_data1, overwrite=False, actually_save=False)
    project_data1.final_tracks = df1
    df2 = correct_tracks_dataframe_using_project(project_data2, overwrite=False, actually_save=False)
    project_data2.final_tracks = df2
    df3 = correct_tracks_dataframe_using_project(project_data3, overwrite=False, actually_save=False)
    project_data3.final_tracks = df3
    df4 = correct_tracks_dataframe_using_project(project_data4, overwrite=False, actually_save=False)
    project_data4.final_tracks = df4
    ## Align with the manual annotation .csv file
    project_data1.finished_neurons_column_name = 'Finished?'  # round1 worm 1
    project_data2.finished_neurons_column_name = 'Finished?'  # round1 worm 4
    project_data3.finished_neurons_column_name = 'first 100 frames'  # round2 worm 6
    project_data4.finished_neurons_column_name = 'Finished?'  # round2 worm 3
    project_data1._custom_frame_indices = list(
        range(1000, 3000))  # round1 worm 1; do not include the non-moving portion
    project_data3.num_frames = 100  # round2 worm 6

    all_project_data = [project_data1, project_data2, project_data3, project_data4]

    return all_project_data



if __name__ == "__main__":
    main()

