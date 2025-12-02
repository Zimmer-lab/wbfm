# Display more informative error messages
# https://www.tutorialexample.com/fix-pyqt-gui-application-crashed-while-no-error-message-displayed-a-beginner-guide-pyqt-tutorial/
import cgitb
import os
import signal
import warnings
import logging
import sys
from functools import partial
from typing import List, Tuple, Union

import napari
import numpy as np
import pandas as pd
import tifffile
import zarr
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from backports.cached_property import cached_property
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from tqdm.auto import tqdm
from PyQt5.QtWidgets import QApplication, QProgressDialog
from wbfm.gui.utils.utils_gui_matplot import PlotQWidget
from wbfm.utils.external.custom_errors import NoBehaviorAnnotationsError, IncompleteConfigFileError
from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_neuron_names import int2name_neuron
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.gui.utils.utils_gui import zoom_using_layer_in_viewer, change_viewer_time_point, \
    zoom_using_viewer, add_fps_printer, on_close, NeuronNameEditor
from wbfm.utils.external.utils_pandas import build_tracks_from_dataframe
from wbfm.utils.projects.finished_project_data import ProjectData
import time

# cgitb.enable(format='text')


class NapariTraceExplorer(QtWidgets.QWidget):

    subplot_is_initialized = False
    tracklet_lines = None
    trace_line = None
    reference_line = None
    zoom_opt = None
    main_subplot_xlim = None

    last_time_point = 0
    current_time_point_before_callback = 0
    dict_of_saved_times = None

    # If False, will load faster but can't use tracklet correcting features
    load_tracklets = True

    # For optional napari layers
    use_track_of_point: bool = False
    manualNeuronNameEditor: NeuronNameEditor = None

    logger: logging.Logger = None

    _disable_callbacks = False

    def __init__(self, project_data: ProjectData, app: QApplication, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.load_tracklets:
            project_data.check_data_desyncing(raise_error=True)

        super(QtWidgets.QWidget, self).__init__()
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dat = project_data
        self.main_window = app

        # https://stackoverflow.com/questions/12781407/how-do-i-resize-the-contents-of-a-qscrollarea-as-more-widgets-are-placed-inside
        # scroll = QtWidgets.QScrollArea()
        # scroll.setWidgetResizable(True)  # CRITICAL
        #
        # inner = QtWidgets.QFrame()
        # inner_layout = QtWidgets.QVBoxLayout()
        # inner.setLayout(inner_layout)
        # scroll.setWidget(inner)  # CRITICAL
        #
        # inner_layout.addWidget(scroll)
        #
        # self.verticalLayoutWidget = inner
        # self.verticalLayout = inner_layout
        # self.scroll = scroll

        # Helper fields for subplots
        self.tracklet_lines = {}
        self.time_line = None
        self.outlier_line = None
        self.main_subplot_xlim = []
        self.current_subplot_xlim = None
        self.zoom_opt = {'zoom': None, 'ind_within_layer': 0, 'layer_is_full_size_and_single_neuron': False,
                         'layer_name': 'final_track'}
        try:
            self.logger = project_data.project_config.setup_logger('trace_explorer.log')
        except AttributeError:
            self.logger = project_data.logger
        if self.load_tracklets:
            project_data.tracklet_annotator.logger = self.logger
        self.logger.debug("Finished initializing Trace Explorer object")

        self.traces_mode_calculation_options = ['integration', 'z', 'volume']
        self.tracklet_mode_calculation_options = ['z', 'volume', 'likelihood', 'brightness_red']

        self.dict_of_saved_times = dict()
        self.list_of_gt_correction_widgets = []

    def setup_ui(self, viewer: napari.Viewer):

        self.logger.debug("Starting main UI setup")
        # Load dataframe and path to outputs
        self.viewer = viewer
        neuron_names = self.dat.neuron_names
        self.current_neuron_name = neuron_names[0]

        # BOX 1: Change neurons (dropdown)
        self.groupBox1NeuronSelection = QtWidgets.QGroupBox("Selection", self.verticalLayoutWidget)
        self.vbox1 = QtWidgets.QVBoxLayout(self.groupBox1NeuronSelection)
        self.changeNeuronDropdown = QtWidgets.QComboBox()
        self.changeNeuronDropdown.addItems(neuron_names)
        self.changeNeuronDropdown.setItemText(0, self.current_neuron_name)
        self.changeNeuronDropdown.currentIndexChanged.connect(self._select_neuron_using_dropdown)
        self.vbox1.addWidget(self.changeNeuronDropdown)

        # Change traces vs tracklet mode (we need the class even if we don't have the button)
        self.changeTraceTrackletDropdown = QtWidgets.QComboBox()
        self.changeTraceTrackletDropdown.addItems(['traces', 'tracklets'])
        self.changeTraceTrackletDropdown.currentIndexChanged.connect(self.change_trace_tracklet_mode)
        if self.load_tracklets:
            # Do not even add the dropdown if disallowed
            self.vbox1.addWidget(self.changeTraceTrackletDropdown)

        if self.load_tracklets:
            self.changeInteractivityCheckbox = QtWidgets.QCheckBox("Turn on interactivity? "
                                                                   "NOTE: only Raw_segmentation layer is interactive")
            self.changeInteractivityCheckbox.stateChanged.connect(self.update_interactivity)
            self.vbox1.addWidget(self.changeInteractivityCheckbox)
        else:
            self.changeInteractivityCheckbox = QtWidgets.QCheckBox("Tracklet interactivity is NOT enabled")
            self.changeInteractivityCheckbox.setEnabled(False)

        # More complex groupBoxes:
        self._setup_trace_filtering_buttons()
        self._setup_layer_creation_buttons()
        if self.load_tracklets:
            self._setup_tracklet_correction_buttons()
            # self._setup_gt_correction_shortcut_buttons()
            self._setup_segmentation_correction_buttons()

        # Move full saving button out into it's own section
        self.groupBox7SaveData = QtWidgets.QGroupBox("Saving Data", self.verticalLayoutWidget)
        self.formlayout7 = QtWidgets.QFormLayout(self.groupBox7SaveData)
        self.mainSaveButton = QtWidgets.QPushButton("Save all to disk")
        self.mainSaveButton.pressed.connect(self.save_everything_to_disk)
        msg = "IDs"
        if self.load_tracklets:
            msg = f"Masks, Tracklets, and {msg}"
        self.formlayout7.addRow(msg, self.mainSaveButton)

        self.verticalLayout.addWidget(self.groupBox1NeuronSelection)
        self.verticalLayout.addWidget(self.groupBox2TraceCalculation)
        if self.load_tracklets:
            self.verticalLayout.addWidget(self.groupBox3TrackletCorrection)
            # self.verticalLayout.addWidget(self.groupBox5)
            self.verticalLayout.addWidget(self.groupBox6SegmentationCorrection)
        self.verticalLayout.addWidget(self.groupBox5LayerCreation)
        self.verticalLayout.addWidget(self.groupBox7SaveData)

        try:
            self.initialize_track_layers()
        except KeyError:
            self.logger.warning("Failed to initialize track layers, segmentation and tracklet callbacks will be "
                                "unavailable")
        self.initialize_shortcuts()
        self.connect_napari_callbacks()
        self.initialize_trace_or_tracklet_subplot()
        if self.load_tracklets:
            self.update_interactivity()

        self.viewer.window._qt_window.closeEvent = partial(
            on_close,
            self.viewer.window._qt_window,
            widget=self,
            callbacks=[self.save_everything_to_disk]
        )

        # Set detailed titles
        self.viewer.title = f"Trace Explorer for project: {self.dat.project_dir}"

        # Open a new window with manual neuron name editing, if you have permissions in the project
        self.manualNeuronNameEditor = self.dat.build_neuron_editor_gui()
        if self.manualNeuronNameEditor is not None:
            update_func = lambda *args: self.update_neuron_id_strings_in_layer(*args, neuropal=False)
            self.manualNeuronNameEditor.annotation_updated.connect(update_func)
            self.manualNeuronNameEditor.multiple_annotations_updated.connect(update_func)
            self.manualNeuronNameEditor.setWindowTitle(f"Fluorescence Neuron Name Editor for project: {self.dat.project_dir}")
            self.manualNeuronNameEditor.show()

        # Optional: add neuropal layer interactivity
        if self.dat.neuropal_manager.has_complete_neuropal:
            self.manualNeuropalNeuronNameEditor = self.dat.build_neuron_editor_gui(neuropal_subproject=True)
            if self.manualNeuropalNeuronNameEditor is not None:
                update_func = lambda *args: self.update_neuron_id_strings_in_layer(*args, neuropal=True)
                self.manualNeuropalNeuronNameEditor.annotation_updated.connect(update_func)
                self.manualNeuropalNeuronNameEditor.multiple_annotations_updated.connect(update_func)
                self.manualNeuropalNeuronNameEditor.setWindowTitle(f"Neuropal Neuron Name Editor for project: {self.dat.project_dir}")
                # Change the background to light blue to differentiate from the other window
                self.manualNeuropalNeuronNameEditor.setStyleSheet("background-color: lightblue;")
                self.manualNeuropalNeuronNameEditor.show()

                # Also add interactivity to the segmentation layer
                self.add_neuropal_neuron_selection_callback()
        else:
            self.manualNeuropalNeuronNameEditor = None

        self.logger.debug("Finished main UI setup")

    def _setup_trace_filtering_buttons(self):
        # Change traces (dropdown)
        self.groupBox2TraceCalculation = QtWidgets.QGroupBox("Trace calculation options", self.verticalLayoutWidget)
        self.formlayout3 = QtWidgets.QFormLayout(self.groupBox2TraceCalculation)

        self.changeChannelDropdown = QtWidgets.QComboBox()
        self.changeChannelDropdown.addItems(['green', 'red', 'ratio', 'linear_model', 'df_over_f_20', 'dr_over_r_20'])
        self.changeChannelDropdown.setCurrentText('ratio')
        self.changeChannelDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Trace calculation mode:", self.changeChannelDropdown)

        if self.load_tracklets:
            self.changeSubplotMarkerDropdown = QtWidgets.QComboBox()
            self.changeSubplotMarkerDropdown.addItems(['line', 'dots'])
            self.changeSubplotMarkerDropdown.currentIndexChanged.connect(self.update_trace_or_tracklet_subplot)
            self.formlayout3.addRow("Tracklet subplot marker:", self.changeSubplotMarkerDropdown)

        self.changeTraceCalculationDropdown = QtWidgets.QComboBox()
        self.changeTraceCalculationDropdown.addItems(self.traces_mode_calculation_options)
        # , 'likelihood' ... Too short in time, so crashes
        self.changeTraceCalculationDropdown.currentIndexChanged.connect(self.update_trace_or_tracklet_subplot)
        self.formlayout3.addRow("Trace calculation (y axis):", self.changeTraceCalculationDropdown)
        # Change trace filtering (dropdown)
        self.changeTraceFilteringDropdown = QtWidgets.QComboBox()
        self.changeTraceFilteringDropdown.addItems(['no_filtering', 'rolling_mean', 'linear_interpolation',
                                                    'strong_rolling_mean', 'gaussian_moving_average'])
        self.changeTraceFilteringDropdown.setCurrentText('rolling_mean')
        self.changeTraceFilteringDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Trace filtering:", self.changeTraceFilteringDropdown)
        # Change trace filtering (dropdown)
        self.changeResidualModeDropdown = QtWidgets.QComboBox()
        self.changeResidualModeDropdown.addItems(['none', 'pca', 'nmf'])
        self.changeResidualModeDropdown.setCurrentText('none')
        self.changeResidualModeDropdown.currentIndexChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Residual Mode:", self.changeResidualModeDropdown)
        # Change trace outlier removal (checkbox)
        self.changeTraceOutlierCheckBox = QtWidgets.QCheckBox()
        self.changeTraceOutlierCheckBox.setChecked(True)
        self.changeTraceOutlierCheckBox.stateChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Remove outliers?", self.changeTraceOutlierCheckBox)
        # Do bleach correction
        self.changeBleachCorrectionCheckBox = QtWidgets.QCheckBox()
        self.changeBleachCorrectionCheckBox.setChecked(True)
        self.changeBleachCorrectionCheckBox.stateChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Do bleach correction?", self.changeBleachCorrectionCheckBox)
        # Interpolate nan or not
        self.changeInterpolationModeDropdown = QtWidgets.QCheckBox()
        self.changeInterpolationModeDropdown.setChecked(False)
        self.changeInterpolationModeDropdown.stateChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Interpolate nan?", self.changeInterpolationModeDropdown)
        # Changing display in Neuron ID layer
        # self.changeNeuronIdLayer = QtWidgets.QCheckBox()
        # self.changeNeuronIdLayer.setChecked(False)
        # self.changeNeuronIdLayer.stateChanged.connect(self.switch_neuron_id_strings)
        # self.formlayout3.addRow("Display manual IDs?", self.changeNeuronIdLayer)
        # Display ppca outlier candidates (checkbox)
        self.ppcaOutlierOverlayCheckbox = QtWidgets.QCheckBox()
        self.ppcaOutlierOverlayCheckbox.setChecked(False)
        self.ppcaOutlierOverlayCheckbox.stateChanged.connect(self.add_or_remove_tracking_outliers)
        self.formlayout3.addRow("Show PPCA outliers?", self.ppcaOutlierOverlayCheckbox)
        # REMOVE ppca outlier candidates (checkbox)
        self.ppcaOutlierRemovalCheckbox = QtWidgets.QCheckBox()
        self.ppcaOutlierRemovalCheckbox.setChecked(False)
        self.ppcaOutlierRemovalCheckbox.stateChanged.connect(self.update_trace_subplot)
        self.formlayout3.addRow("Remove PPCA outliers?", self.ppcaOutlierRemovalCheckbox)
        # Change behavior shading (dropdown)
        # Note: QListWidget allows multiple selection, but the display is very large... so use QComboBox instead
        # Note also that most updates don't change the shading, so this has to fully rebuild the plot
        self.changeSubplotShading = QtWidgets.QComboBox()
        # self.changeSubplotShading.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        try:
            unique_behaviors = self.dat.worm_posture_class.all_found_behaviors(convert_to_strings=True,
                                                                               fluorescence_fps=True)
        except (NoBehaviorAnnotationsError, AttributeError):
            unique_behaviors = []
        unique_behaviors.insert(0, 'none')
        self.changeSubplotShading.addItems(unique_behaviors)
        self.changeSubplotShading.currentIndexChanged.connect(self.update_behavior_shading)
        self.formlayout3.addRow("Shaded behaviors:", self.changeSubplotShading)
        # self.changeTrackingOutlierCheckBox = QtWidgets.QCheckBox()
        # self.changeTrackingOutlierCheckBox.stateChanged.connect(self.update_trace_subplot)
        # self.formlayout3.addRow("Remove outliers (tracking confidence)?", self.changeTrackingOutlierCheckBox)

        # Add reference neuron trace (also allows behaviors) (dropdown)
        self.changeReferenceTrace = QtWidgets.QComboBox()
        neuron_names_and_none = self.dat.neuron_names
        neuron_names_and_none.insert(0, "None")
        neuron_names_and_none.extend(WormFullVideoPosture.beh_aliases_stable())
        self.changeReferenceTrace.addItems(neuron_names_and_none)
        self.changeReferenceTrace.currentIndexChanged.connect(self.update_reference_trace)
        self.formlayout3.addRow("Reference trace:", self.changeReferenceTrace)

    def _setup_layer_creation_buttons(self):
        self.groupBox5LayerCreation = QtWidgets.QGroupBox("New layer creation", self.verticalLayoutWidget)
        self.formlayout8 = QtWidgets.QFormLayout(self.groupBox5LayerCreation)

        self.addReferenceHeatmap = QtWidgets.QPushButton("Add Layer")
        self.addReferenceHeatmap.pressed.connect(self.add_layer_colored_by_correlation_to_current_neuron)
        self.formlayout8.addRow("Correlation to current trace:", self.addReferenceHeatmap)


    def _setup_general_shortcut_buttons(self):
        self.groupBox3b = QtWidgets.QGroupBox("General shortcuts", self.verticalLayoutWidget)
        self.vbox3b = QtWidgets.QVBoxLayout(self.groupBox3b)

        # self.refreshButton = QtWidgets.QPushButton("Refresh Subplot (r)")
        # self.refreshButton.pressed.connect(self.update_trace_or_tracklet_subplot)
        # self.vbox3b.addWidget(self.refreshButton)

        self.refreshDefaultLayersButton = QtWidgets.QPushButton("Refresh Default Napari Layers")
        self.refreshDefaultLayersButton.pressed.connect(self.refresh_default_napari_layers)
        self.vbox3b.addWidget(self.refreshDefaultLayersButton)

        self.printTrackletsButton = QtWidgets.QPushButton("Print tracklets attached to current neuron (v)")
        self.printTrackletsButton.pressed.connect(self.print_tracklets)
        self.vbox3b.addWidget(self.printTrackletsButton)

        # self.zoom1Button = QtWidgets.QPushButton("Next time point (d)")
        # self.zoom1Button.pressed.connect(self.zoom_next)
        # self.vbox3b.addWidget(self.zoom1Button)
        # self.zoom2Button = QtWidgets.QPushButton("Previous time point (a)")
        # self.zoom2Button.pressed.connect(self.zoom_previous)
        # self.vbox3b.addWidget(self.zoom2Button)
        self.zoom3Button = QtWidgets.QPushButton("Zoom to next time with nan (f)")
        self.zoom3Button.pressed.connect(self.zoom_to_next_nan)
        self.vbox3b.addWidget(self.zoom3Button)

    def _setup_tracklet_correction_buttons(self):
        # BOX 4: tracklet shortcuts
        self.groupBox3TrackletCorrection = QtWidgets.QGroupBox("Tracklet Correction", self.verticalLayoutWidget)
        self.vbox4 = QtWidgets.QVBoxLayout(self.groupBox3TrackletCorrection)

        # self.trackletHint1 = QtWidgets.QLabel("Normal Click: Select tracklet attached to neuron")
        # self.vbox4.addWidget(self.trackletHint1)

        self.recentTrackletSelector = QtWidgets.QComboBox()
        self.vbox4.addWidget(self.recentTrackletSelector)
        self.recentTrackletSelector.currentIndexChanged.connect(self.change_tracklets_using_dropdown)
        self.recentTrackletSelector.setToolTip("Select from history of recent tracklets")

        # self.zoom4Button = QtWidgets.QPushButton("Zoom to next time with tracklet conflict (g)")
        # self.zoom4Button.pressed.connect(self.zoom_to_next_conflict)
        # self.zoom4Button.setToolTip("Note: does nothing if there is no tracklet selected")
        # self.vbox4.addWidget(self.zoom4Button)
        self.zoom5Button = QtWidgets.QPushButton("Jump to end of current tracklet (j)")
        self.zoom5Button.pressed.connect(self.zoom_to_end_of_current_tracklet)
        self.vbox4.addWidget(self.zoom5Button)
        self.zoom6Button = QtWidgets.QPushButton("Jump to beginning of current tracklet (h)")
        self.zoom6Button.pressed.connect(self.zoom_to_start_of_current_tracklet)
        self.vbox4.addWidget(self.zoom6Button)

        # Splitting and removing shortcuts
        self.toggleSegButton = QtWidgets.QPushButton("Toggle Raw segmentation layer (s)")
        self.toggleSegButton.pressed.connect(self.toggle_raw_segmentation_layer)
        self.splitTrackletButton1 = QtWidgets.QPushButton("Split current tracklet (keep past) (q)")
        self.splitTrackletButton1.pressed.connect(self.split_current_tracklet_keep_left)
        self.vbox4.addWidget(self.splitTrackletButton1)
        self.splitTrackletButton2 = QtWidgets.QPushButton("Split current tracklet (keep future) (e)")
        self.splitTrackletButton2.pressed.connect(self.split_current_tracklet_keep_right)
        self.vbox4.addWidget(self.splitTrackletButton2)
        self.clearTrackletButton = QtWidgets.QPushButton("Clear current tracklet (w)")
        self.clearTrackletButton.pressed.connect(self.clear_current_tracklet)
        self.vbox4.addWidget(self.clearTrackletButton)
        self.removeTrackletButton1 = QtWidgets.QPushButton("Remove OTHER tracklets with time conflicts")
        self.removeTrackletButton1.pressed.connect(self.remove_time_conflicts)
        self.vbox4.addWidget(self.removeTrackletButton1)
        self.removeTrackletButton2 = QtWidgets.QPushButton("Remove current tracklet from all neurons")
        self.removeTrackletButton2.pressed.connect(self.remove_tracklet_from_all_matches)
        self.vbox4.addWidget(self.removeTrackletButton2)
        self.appendTrackletButton = QtWidgets.QPushButton("Save current tracklet to neuron (IF conflict-free) (c)")
        self.appendTrackletButton.pressed.connect(self.save_current_tracklet_to_neuron)
        self.appendTrackletButton.setToolTip("Note: check console for more details of the conflict")
        self.vbox4.addWidget(self.appendTrackletButton)

        self.saveSegmentationToTrackletButton = QtWidgets.QPushButton("Save current segmentation "
                                                                      "to current tracklet (x)")
        self.saveSegmentationToTrackletButton.pressed.connect(self.save_segmentation_to_tracklet)
        self.vbox4.addWidget(self.saveSegmentationToTrackletButton)

        self.deleteSegmentationFromTrackletButton = QtWidgets.QPushButton("Delete current segmentation "
                                                                          "from current tracklet")
        self.deleteSegmentationFromTrackletButton.pressed.connect(self.delete_segmentation_from_tracklet)
        self.vbox4.addWidget(self.deleteSegmentationFromTrackletButton)

        self.saveTrackletsStatusLabel = QtWidgets.QLabel("STATUS: No tracklet loaded")
        self.vbox4.addWidget(self.saveTrackletsStatusLabel)

        self.list_of_tracklet_correction_widgets = [
            self.recentTrackletSelector,
            self.zoom5Button,
            self.zoom6Button,
            self.toggleSegButton,
            self.splitTrackletButton1,
            self.splitTrackletButton2,
            self.clearTrackletButton,
            self.removeTrackletButton1,
            self.removeTrackletButton2,
            self.appendTrackletButton,
            self.saveSegmentationToTrackletButton,
            self.deleteSegmentationFromTrackletButton
        ]

    def _setup_gt_correction_shortcut_buttons(self):
        self.groupBox5 = QtWidgets.QGroupBox("Ground Truth Correction (requires annotated ground truth)", self.verticalLayoutWidget)
        self.vbox5 = QtWidgets.QVBoxLayout(self.groupBox5)

        # Ground truth conflict buttons
        # self.gtDropdown = QtWidgets.QComboBox()
        # neurons_with_conflict = list(self.dat.tracklet_annotator.gt_mismatches.keys())
        # neurons_with_conflict.sort()
        # self.gtDropdown.addItems(neurons_with_conflict)
        # self.gtDropdown.setToolTip("Note: does not work if there are no ground truth neurons")
        # self.vbox5.addWidget(self.gtDropdown)

        self.zoom6Button = QtWidgets.QPushButton("Jump to next model vs. ground truth conflict")
        self.zoom6Button.setToolTip("shift-f")
        self.zoom6Button.pressed.connect(self.zoom_to_next_ground_truth_conflict)
        self.vbox5.addWidget(self.zoom6Button)
        self.resolveConflictButton = QtWidgets.QPushButton("Resolve current model vs. ground truth conflict")
        self.resolveConflictButton.setToolTip("shift-w")
        self.resolveConflictButton.pressed.connect(self.resolve_current_ground_truth_conflict)
        self.vbox5.addWidget(self.resolveConflictButton)

        self.list_of_gt_correction_widgets = [
            self.zoom6Button,
            self.resolveConflictButton
        ]

        self.update_gt_correction_interactivity()

    def _setup_segmentation_correction_buttons(self):
        self.groupBox6SegmentationCorrection = QtWidgets.QGroupBox("Segmentation Correction", self.verticalLayoutWidget)
        self.formlayout6 = QtWidgets.QFormLayout(self.groupBox6SegmentationCorrection)

        self.candidateMaskButton = QtWidgets.QPushButton("Make copy of segmentation")
        self.candidateMaskButton.pressed.connect(self.add_candidate_mask_layer)
        self.formlayout6.addRow("Produce candidate mask: ", self.candidateMaskButton)

        self.mergeSegmentationButton = QtWidgets.QPushButton("Try to merge selected")
        self.mergeSegmentationButton.pressed.connect(self.merge_segmentation)
        self.formlayout6.addRow("Produce candidate mask: ", self.mergeSegmentationButton)

        self.clearSelectedSegmentationsButton = QtWidgets.QPushButton("Clear (r)")
        self.clearSelectedSegmentationsButton.pressed.connect(self.clear_current_segmentations)
        self.formlayout6.addRow("Remove selected segmentations: ", self.clearSelectedSegmentationsButton)

        self.splitSegmentationSaveButton1 = QtWidgets.QPushButton("Save to RAM")
        self.splitSegmentationSaveButton1.pressed.connect(self.modify_segmentation_using_manual_correction)
        self.formlayout6.addRow("Save candidate mask: ", self.splitSegmentationSaveButton1)

        self.saveSegmentationStatusLabel = QtWidgets.QLabel("No segmentation loaded")
        self.formlayout6.addRow("STATUS: ", self.saveSegmentationStatusLabel)

        # self.update_segmentation_options()

        self.list_of_segmentation_correction_widgets = [
            # self.splitSegmentationManualSliceButton,
            # self.splitSegmentationKeepOriginalIndexButton,
            self.clearSelectedSegmentationsButton,
            # self.splitSegmentationManualButton,
            # self.splitSegmentationAutomaticButton,
            self.candidateMaskButton,
            self.mergeSegmentationButton,
            self.splitSegmentationSaveButton1,
            # self.mainSaveButton,
        ]

    @property
    def raw_seg_layer(self):
        try:
            return self.viewer.layers['Raw segmentation']
        except KeyError:
            return None

    @property
    def colored_seg_layer(self):
        return self.viewer.layers['Colored segmentation']

    @property
    def neuropal_seg_layer(self):
        try:
            return self.viewer.layers['Neuropal segmentation']
        except KeyError:
            return None

    @property
    def neuron_id_layer(self):
        return self.viewer.layers['Neuron IDs']

    def get_manual_id_layer(self, neuropal=False):
        if not neuropal:
            return self.viewer.layers['Manual IDs']
        else:
            return self.viewer.layers['Neuropal IDs']

    @property
    def red_data_layer(self):
        return self.viewer.layers['Red data']

    @property
    def final_track_layer(self):
        if 'final_track' in self.viewer.layers:
            return self.viewer.layers['final_track']
        else:
            return None

    @property
    def track_of_point_layer(self):
        return self.viewer.layers['track_of_point']

    def refresh_default_napari_layers(self):
        self.logger.warning("Undocumented shortcut!")
        self.dat.add_layers_to_viewer(self.viewer, which_layers='all', check_if_layers_exist=True,
                                      dask_for_segmentation=False)

    def refresh_segmentation_metadata(self):
        self.logger.warning("Undocumented shortcut!")
        t = self.t
        self.logger.info(f"Updating segmentation metadata at t={t}")
        red_volume = self.red_data_layer.data[t, ...]
        new_mask = self.raw_seg_layer.data[t, ...]
        self.dat.segmentation_metadata.modify_segmentation_metadata(t, new_mask, red_volume)
        self.logger.debug(f"Finished updating metadata")

    def _select_neuron_using_dropdown(self):
        self.logger.debug("USER: change neuron")
        self.update_gt_correction_interactivity()
        if not self._disable_callbacks:
            # self.update_dataframe_using_final_tracks_layer()
            self.update_neuron_in_tracklet_annotator()
            self.update_track_layers()
            self.update_trace_or_tracklet_subplot(preserve_xlims=False)

    def change_tracklet_to_currently_attached_tracklet(self):
        # Get tracklet that is currently attached to the selected neuron
        self.logger.debug(f"USER: change tracklet to currently attached tracklet at t={self.t}")
        if not self._disable_callbacks:
            if self.dat.tracklet_annotator is None:
                self.logger.warning("Tracklet annotator is not initialized")
                return
            target_tracklet = self.dat.tracklet_annotator.get_tracklet_attached_at_time(self.t)
            self.change_tracklets_from_gui(next_tracklet=target_tracklet)

    def change_tracklets_using_dropdown(self):
        self.change_tracklets_from_gui(self.recentTrackletSelector.currentText())

    def change_tracklets_from_gui(self, next_tracklet=None):
        self.logger.debug("USER: change tracklets from gui")
        if next_tracklet is None:
            self.logger.info("Attempted to change tracklet, but no tracklet was passed")
            return
        if not self._disable_callbacks:
            # which_tracklets_to_update = self.subplot_update_dict_for_tracklet_change(next_tracklet=next_tracklet)
            self.dat.tracklet_annotator.set_current_tracklet(next_tracklet)
            self.dat.tracklet_annotator.add_current_tracklet_to_viewer(self.viewer)
            # self.tracklet_updated_psuedo_event(which_tracklets_to_update=which_tracklets_to_update)
            self.tracklet_updated_psuedo_event()

    def change_tracklets_from_click(self):
        self.logger.debug("USER: click on segmentation")
        previous_tracklet = self.dat.tracklet_annotator.previous_tracklet_name
        next_tracklet = self.dat.tracklet_annotator.current_tracklet_name
        # which_tracklets_to_update = self.subplot_update_dict_for_tracklet_change(current_tracklet=previous_tracklet,
        #                                                                          next_tracklet=next_tracklet)
        self.tracklet_updated_psuedo_event()

    def modify_current_tracklet(self):
        self.logger.debug(f"USER: modify current tracklet: {self.current_tracklet_name}")
        # which_tracklets_to_update = self.subplot_update_dict_for_tracklet_modification()
        self.tracklet_updated_psuedo_event()

    def subplot_update_dict_for_tracklet_change(self, current_tracklet=None, next_tracklet=None):
        if current_tracklet is None:
            current_tracklet = self.current_tracklet_name
        if current_tracklet == next_tracklet:
            which_tracklets_to_update = {f"{current_tracklet}_current": 'replot'}
        else:
            which_tracklets_to_update = {f"{current_tracklet}_current": 'remove',
                                         f"{next_tracklet}_current": 'plot'}
        return which_tracklets_to_update

    def get_dict_for_tracklet_save(self, tracklet_name):
        # Remove tracklet as black, but then replot as colored
        which_tracklets_to_update = {f"{tracklet_name}_current": 'remove',
                                     tracklet_name: 'plot'}
        return which_tracklets_to_update

    def add_to_recent_tracklet_dropdown(self):
        last_tracklet = self.dat.tracklet_annotator.current_tracklet_name
        if last_tracklet is None:
            return
        current_items = [self.recentTrackletSelector.itemText(i) for i in range(self.recentTrackletSelector.count())]

        if last_tracklet in current_items:
            return

        self._disable_callbacks = True
        self.recentTrackletSelector.insertItem(0, last_tracklet)

        num_to_remember = 8
        if self.recentTrackletSelector.count() > num_to_remember:
            self.recentTrackletSelector.removeItem(num_to_remember)
        self._disable_callbacks = False

    def update_track_layers(self):
        if self.final_track_layer is None:
            return
        point_layer_data, track_layer_data = self.get_track_data()
        self.final_track_layer.data = point_layer_data
        if self.use_track_of_point:
            self.track_of_point_layer.data = track_layer_data

        self.zoom_using_current_neuron_or_tracklet()

    def zoom_using_current_neuron_or_tracklet(self):
        zoom_using_layer_in_viewer(self.viewer, **self.zoom_opt)

    def update_neuron_in_tracklet_annotator(self):
        if self.dat.tracklet_annotator is None:
            return
        self.dat.tracklet_annotator.current_neuron = self.changeNeuronDropdown.currentText()

    def update_interactivity(self):
        to_be_interactive = self.changeInteractivityCheckbox.isChecked()
        if self.dat.tracklet_annotator is not None:
            self.dat.tracklet_annotator.is_currently_interactive = to_be_interactive

        if to_be_interactive:
            self.groupBox3TrackletCorrection.setTitle("Tracklet Correction (currently enabled)")
            self.groupBox6SegmentationCorrection.setTitle("Segmentation Correction (currently enabled)")
        else:
            self.groupBox3TrackletCorrection.setTitle("Tracklet Correction (currently disabled)")
            self.groupBox6SegmentationCorrection.setTitle("Segmentation Correction (currently disabled)")

        for widget in self.list_of_segmentation_correction_widgets:
            widget.setEnabled(to_be_interactive)

        for widget in self.list_of_tracklet_correction_widgets:
            widget.setEnabled(to_be_interactive)

    def update_gt_correction_interactivity(self):
        # Initialize them as interactive or not
        if self.dat.tracklet_annotator is None:
            return
        to_be_interactive = self.dat.tracklet_annotator.gt_mismatches is not None
        if to_be_interactive:
            to_be_interactive = len(self.dat.tracklet_annotator.gt_mismatches[self.current_neuron_name]) > 0
        for widget in self.list_of_gt_correction_widgets:
            widget.setEnabled(to_be_interactive)

    def modify_segmentation_using_manual_correction(self):
        """Uses candidate mask layer to modify the segmentation in the GUI, but not on disk"""
        self.dat.modify_segmentation_using_manual_correction()
        self.dat.tracklet_annotator.update_segmentation_layer_using_buffer(self.raw_seg_layer)
        self.dat.tracklet_annotator.clear_currently_selected_segmentations()
        self.remove_layer_of_candidate_segmentation()
        self.set_segmentation_layer_visible()

    def save_everything_to_disk(self):
        """
        Uses segmentation as modified previously by candidate mask layer, tracklet dataframe, and manual IDs
        """
        # For some reason the progress bar doesn't show up until after the first segmentation_metadata call,
        # ... even if I add sleep and etc.
        if self.dat.segmentation_metadata is not None:
            dict_of_saving_callbacks = {
                'segmentation_metadata': self.dat.segmentation_metadata.overwrite_original_detection_file,
                'tracklets': self.dat.tracklet_annotator.save_manual_matches_to_disk,
                'segmentation': self.dat.modify_segmentation_on_disk_using_buffer
            }
        else:
            dict_of_saving_callbacks = {}

        if self.manualNeuronNameEditor is not None:
            dict_of_saving_callbacks['manual_ids'] = self.manualNeuronNameEditor.save_df_to_disk
        if self.manualNeuropalNeuronNameEditor:
            dict_of_saving_callbacks['neuropal_ids'] = self.manualNeuropalNeuronNameEditor.save_df_to_disk
        progress = QProgressDialog("Saving to disk, you may quit when finished", None,
                                   0, len(dict_of_saving_callbacks), self)
        progress.setWindowModality(Qt.WindowModal)

        import time
        all_flags = {}
        for i, (name, callback) in enumerate(dict_of_saving_callbacks.items()):
            progress.setValue(i)
            try:
                flag = callback()
            except PermissionError as e:
                self.logger.warning(f"Failed to save {name} to disk (other steps should succeed): {e}")
                flag = False
            all_flags[name] = flag
            # Sleep to make sure that the progress bar is updated
            time.sleep(0.1)
        # Some of the saving steps use threads, so wait for them to finish
        if len(all_flags) == 0:
            self.logger.error("No saving steps were attempted!")
            self.logger.error("If this is a NWB project, then modifications cannot be saved yet!")
            return

        with self.dat.tracklet_annotator.saving_lock:
            # Set any None values to True; this means nothing was done
            all_flags = {k: v if v is not None else True for k, v in all_flags.items()}
            if not all(all_flags.values()):
                self.logger.error("Failed to save at least one step to disk!")
                self.logger.error(f"'False' means failed step: {all_flags}")
                self.logger.error("Please try again, or ctrl-c to fully quit (if this was expected)")
            else:
                self.logger.info("================================================================")
                self.logger.info("Saving successful!")
                self.logger.info("================================================================")
            progress.setValue(len(all_flags))

    def split_segmentation_manual(self):
        # Produces candidate mask layer
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.split_current_neuron_and_add_napari_layer(self.viewer, split_method="Manual")
        self.set_segmentation_layer_invisible()

    def split_segmentation_automatic(self):
        # Produces candidate mask layer
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.split_current_neuron_and_add_napari_layer(self.viewer, split_method="Gaussian")
        self.set_segmentation_layer_invisible()

    def merge_segmentation(self):
        # Produces candidate mask layer
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.merge_current_neurons(self.viewer)
        self.set_segmentation_layer_invisible()

    def add_candidate_mask_layer(self):
        # Produces simple copy of segmentation as candidate mask layer
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.update_time_of_candidate_mask(self.t)
        self.dat.tracklet_annotator.add_candidate_mask_layer(self.viewer, new_full_mask=None)
        self.set_segmentation_layer_invisible()

    def clear_current_segmentations(self):
        self.remove_layer_of_candidate_segmentation()
        self.dat.tracklet_annotator.clear_currently_selected_segmentations()

    def initialize_track_layers(self):
        point_layer_data, track_layer_data = self.get_track_data()

        points_opt = dict(face_color='blue', size=4)
        # Note: the 'final_track' layer is used for visualization as well, e.g. zooming to a neuron
        self.viewer.add_points(point_layer_data, name="final_track", n_dimensional=True, symbol='cross', **points_opt,
                               visible=False)
        if self.use_track_of_point:
            self.viewer.add_tracks(track_layer_data, name="track_of_point", visible=False)
        self.zoom_using_current_neuron_or_tracklet()

        layer_to_add_callback = self.raw_seg_layer
        if layer_to_add_callback is not None and self.load_tracklets:
            added_segmentation_callbacks = [
                self.update_segmentation_status_label,
                self.toggle_highlight_selected_neuron
            ]
            added_tracklet_callbacks = [
                self.change_tracklets_from_click,
                self.set_segmentation_layer_invisible
            ]
            select_neuron_callback = self.select_neuron
            self.dat.tracklet_annotator.connect_tracklet_clicking_callback(
                layer_to_add_callback,
                self.viewer,
                added_segmentation_callbacks=added_segmentation_callbacks,
                added_tracklet_callbacks=added_tracklet_callbacks,
                select_neuron_callback=select_neuron_callback
            )
            self.update_neuron_in_tracklet_annotator()

        # Also add interactivity to the colored segmentation layer, for selecting neurons
        self.add_neuron_selection_callback()

    def add_neuron_selection_callback(self):
        layer_to_add_callback = self.colored_seg_layer

        @layer_to_add_callback.mouse_drag_callbacks.append
        def on_click(layer, event):
            if event.button in (1, 2, 3):
                click_modifiers = [m.name.lower() for m in event.modifiers]
                if 'shift' not in click_modifiers:
                    neuron_name = self._get_info_of_clicked_on_neuron(layer, event)
                    if neuron_name is None:
                        return

                    if event.button == 1:
                        self.select_neuron(neuron_name)
                    elif event.button == 2:
                        self.changeReferenceTrace.setCurrentText(neuron_name)
                    elif event.button == 3:
                        # Jump to the clicked row in the external name editor gui
                        if self.manualNeuronNameEditor is not None:
                            self.manualNeuronNameEditor.jump_focus_to_neuron(neuron_name)

    def add_neuropal_neuron_selection_callback(self):
        layer_to_add_callback = self.neuropal_seg_layer

        @layer_to_add_callback.mouse_drag_callbacks.append
        def on_click(layer, event):
            # Only interactivity for middle click
            if event.button in (3, ):
                click_modifiers = [m.name.lower() for m in event.modifiers]
                if 'shift' not in click_modifiers:
                    neuron_name = self._get_info_of_clicked_on_neuron(layer, event)
                    if neuron_name is None:
                        return

                    # Jump to the clicked row in the external name editor gui
                    if self.manualNeuropalNeuronNameEditor is not None:
                        self.manualNeuropalNeuronNameEditor.jump_focus_to_neuron(neuron_name)

    def _get_info_of_clicked_on_neuron(self, layer, event):
        # Get the index of the clicked segmentation

        # Get information about clicked-on neuron
        seg_index = layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True
        )
        if seg_index is None or seg_index == 0:
            self.logger.debug("Clicked on background, not a neuron")
            return None

        # The segmentation index should be the same as the name
        neuron_name = int2name_neuron(seg_index)
        self.logger.debug(f"Clicked on segmentation {seg_index}, corresponding to {neuron_name},"
                          f" with button {event.button}")
        return neuron_name

    def connect_napari_callbacks(self):
        viewer = self.viewer

        # I think this callback is actually triggered twice
        @viewer.dims.events.current_step.connect
        def update_slider(event):
            self.time_changed_callbacks()

    def initialize_shortcuts(self):
        viewer = self.viewer

        @viewer.bind_key('r', overwrite=True)
        def clear_segmentation(viewer):
            self.clear_current_segmentations()

        @viewer.bind_key('Shift-r', overwrite=True)
        def refresh_subplot(viewer):
            self.update_trace_or_tracklet_subplot()

        @viewer.bind_key('Shift-t', overwrite=True)
        def remove_tracklets(viewer):
            self.remove_all_tracklet_napari_layers()

        @viewer.bind_key('d', overwrite=True)
        def zoom_next(viewer):
            self.zoom_next()

        @viewer.bind_key('Shift-d', overwrite=True)
        def zoom_previous(viewer):
            self.zoom_next(dt=5)

        @viewer.bind_key('a', overwrite=True)
        def zoom_previous(viewer):
            self.zoom_previous()

        @viewer.bind_key('Shift-a', overwrite=True)
        def zoom_previous(viewer):
            self.zoom_previous(dt=5)

        @viewer.bind_key('Shift-z', overwrite=True)
        def zoom_last_time_point(viewer):
            self.zoom_last_time_point()

        @viewer.bind_key('Ctrl-1', overwrite=True)
        def save_time1(viewer):
            self.save_time_as_shortcut(1)

        @viewer.bind_key('1', overwrite=True)
        def jump_time1(viewer):
            self.jump_time_using_shortcut(1)

        @viewer.bind_key('Ctrl-2', overwrite=True)
        def save_time2(viewer):
            self.save_time_as_shortcut(2)

        @viewer.bind_key('2', overwrite=True)
        def jump_time2(viewer):
            self.jump_time_using_shortcut(2)

        @viewer.bind_key('Ctrl-3', overwrite=True)
        def save_time3(viewer):
            self.save_time_as_shortcut(3)

        @viewer.bind_key('3', overwrite=True)
        def jump_time3(viewer):
            self.jump_time_using_shortcut(3)

        @viewer.bind_key('Ctrl-4', overwrite=True)
        def save_time4(viewer):
            self.save_time_as_shortcut(4)

        @viewer.bind_key('4', overwrite=True)
        def jump_time4(viewer):
            self.jump_time_using_shortcut(4)

        @viewer.bind_key('f', overwrite=True)
        def zoom_to_next_nan(viewer):
            self.zoom_to_next_nan()

        @viewer.bind_key('shift-f', overwrite=True)
        def zoom_to_next_ground_truth_conflict(viewer):
            self.zoom_to_next_ground_truth_conflict()

        # @viewer.bind_key('shift-w', overwrite=True)
        # def resolve_current_ground_truth_conflict(viewer):
        #     self.resolve_current_ground_truth_conflict()

        @viewer.bind_key('Shift-e', overwrite=True)
        def toggle_manual_ids(viewer):
            self.toggle_manual_ids()

        @viewer.bind_key('g', overwrite=True)
        def zoom_to_next_nan(viewer):
            self.zoom_to_next_conflict()

        @viewer.bind_key('j', overwrite=True)
        def zoom_to_tracklet_end(viewer):
            self.zoom_to_end_of_current_tracklet()

        @viewer.bind_key('h', overwrite=True)
        def zoom_to_tracklet_end(viewer):
            self.zoom_to_start_of_current_tracklet()

        @viewer.bind_key('e', overwrite=True)
        def split_current_tracklet(viewer):
            self.split_current_tracklet_keep_right()

        @viewer.bind_key('q', overwrite=True)
        def split_current_tracklet(viewer):
            self.split_current_tracklet_keep_left()

        @viewer.bind_key('w', overwrite=True)
        def clear_current_tracklet(viewer):
            self.clear_current_tracklet()

        @viewer.bind_key('Shift-w', overwrite=True)
        def toggle_neuron_ids(viewer):
            self.toggle_neuron_ids()

        @viewer.bind_key('s', overwrite=True)
        def toggle_seg(viewer):
            self.toggle_raw_segmentation_layer()

        @viewer.bind_key('Shift-s', overwrite=True)
        def toggle_neuron_ids(viewer):
            self.toggle_colored_segmentation_layer()

        @viewer.bind_key('Shift-q', overwrite=True)
        def toggle_neuron_ids(viewer):
            self.toggle_neuropal_segmentation_layer()

        @viewer.bind_key('c', overwrite=True)
        def save_tracklet(viewer):
            self.save_current_tracklet_to_neuron()

        @viewer.bind_key('Shift-c', overwrite=True)
        def change_tracklet_to_currently_attached_tracklet(viewer):
            self.change_tracklet_to_currently_attached_tracklet()

        @viewer.bind_key('v', overwrite=True)
        def print_tracklet_status(viewer):
            self.print_tracklets()

        # @viewer.bind_key('z', overwrite=True)
        # def remove_conflict(viewer):
        #     pass
            # self.remove_time_conflicts()

        @viewer.bind_key('x', overwrite=True)
        def remove_tracklet(viewer):
            self.save_segmentation_to_tracklet()
            
        # Undocumented shortcuts just for my use
        @viewer.bind_key('Shift-p', overwrite=True)
        def refresh_napari(viewer):
            self.refresh_default_napari_layers()

        @viewer.bind_key('Shift-u', overwrite=True)
        def refresh_napari(viewer):
            self.refresh_segmentation_metadata()

        # DANGER
        @viewer.bind_key('Shift-Alt-Ctrl-t', overwrite=True)
        def refresh_napari(viewer):
            self.remove_all_tracklets_after_current_time()

    @property
    def max_time(self):
        return len(self.dat.final_tracks) - 1

    def zoom_next(self, viewer=None, dt=1):
        change_viewer_time_point(self.viewer, dt=dt, a_max=self.max_time)
        # self.time_changed_callbacks()

    def zoom_previous(self, viewer=None, dt=1):
        change_viewer_time_point(self.viewer, dt=-dt, a_max=self.max_time)
        # self.time_changed_callbacks()

    def zoom_last_time_point(self, viewer=None):
        t_target = self.last_time_point
        print(f"Trying to jump to {t_target}")
        print(self.last_time_point, self.current_time_point_before_callback, self.t)
        change_viewer_time_point(self.viewer, t_target=t_target, a_max=self.max_time)
        # self.time_changed_callbacks()

    def save_time_as_shortcut(self, i, viewer=None):
        self.dict_of_saved_times[i] = self.t

    def jump_time_using_shortcut(self, i, viewer=None):
        t_target = self.dict_of_saved_times.get(i, None)
        if t_target is not None:
            change_viewer_time_point(self.viewer, t_target=t_target, a_max=self.max_time)

    def zoom_to_next_nan(self, viewer=None):
        y_on_plot = self.y_min_max_on_plot[0]  # Don't need both min and max
        if len(y_on_plot) == 0:
            return
        t_old = self.t
        if np.isnan(y_on_plot[t_old]):
            print("Already on nan point; not moving")
            return
        for i in range(t_old+1, len(y_on_plot)):
            if np.isnan(y_on_plot[i]):
                t_target = i
                change_viewer_time_point(self.viewer, t_target=t_target - 1)
                break
        else:
            print("No nan point found; not moving")
        # self.time_changed_callbacks()

    def zoom_to_next_conflict(self, viewer=None):
        t, conflict_neuron = self.dat.tracklet_annotator.time_of_next_conflict(i_start=self.t)

        if conflict_neuron is not None:
            change_viewer_time_point(self.viewer, t_target=t)
        else:
            print("No conflict point found; not moving")
        # self.time_changed_callbacks()

    def zoom_to_end_of_current_tracklet(self, viewer=None):
        t = self.dat.tracklet_annotator.end_time_of_current_tracklet()
        if t is not None:
            change_viewer_time_point(self.viewer, t_target=t)
        else:
            print("No tracklet selected; not zooming")
        # self.time_changed_callbacks()

    def zoom_to_next_ground_truth_conflict(self):
        if self.dat.tracklet_annotator.gt_mismatches is None:
            self.logger.warning("No ground truth found; button not functional")
            return
        neuron_name = self.changeNeuronDropdown.currentText()
        remaining_mismatches = self.dat.tracklet_annotator.gt_mismatches[neuron_name]
        if len(remaining_mismatches) == 0:
            print("This neuron has no remaining conflicts")
            return
        else:
            t, tracklet_name, model_mismatch, gt_mismatch = remaining_mismatches[0]

        # Zoom to the conflict
        self.logger.info(f"Jumped to conflict at t={t} on {neuron_name} and {tracklet_name} "
                         f"with incorrect match: {model_mismatch} and correct match: {gt_mismatch}")
        change_viewer_time_point(self.viewer, t_target=t)
        self.select_neuron(neuron_name)
        self.zoom_using_current_neuron_or_tracklet()
        # Also display the incorrect match
        # incorrect_match = self.dat.napari_tracks_layer_of_single_neuron_match(neuron_name, t)
        this_pair = self.dat.raw_matches[(t, t + 1)]
        incorrect_match_tracks = this_pair.napari_tracks_of_matches([model_mismatch])
        correct_match_tracks = this_pair.napari_tracks_of_matches([gt_mismatch])
        if incorrect_match_tracks:
            # Manually rescale z
            z_scale = self.dat.physical_unit_conversion.zimmer2physical_fluorescence_single_column

            incorrect_match_tracks = np.array(incorrect_match_tracks)
            incorrect_match_tracks = z_scale(incorrect_match_tracks, which_col=2)
            correct_match_tracks = np.array(correct_match_tracks)
            correct_match_tracks = z_scale(correct_match_tracks, which_col=2)

            opt = dict(tail_width=10, head_length=0)
            self.viewer.add_tracks(incorrect_match_tracks, name="Incorrect model match", colormap='hsv', **opt)
            self.viewer.add_tracks(correct_match_tracks, name="GT match", colormap='twilight', **opt)
        else:
            self.logger.warning("Did not find match")

    def select_neuron(self, neuron_name):
        # This runs all the callbacks that happen when a neuron is selected
        self.changeNeuronDropdown.setCurrentText(neuron_name)

    def resolve_current_ground_truth_conflict(self):
        # if self.dat.tracklet_annotator.gt_mismatches is None:
        #     self.logger.warning("No ground truth found; button not functional")
        #     return
        neuron_name = self.changeNeuronDropdown.currentText()
        mismatches = self.dat.tracklet_annotator.gt_mismatches
        if len(mismatches[neuron_name]) == 0:
            self.logger.info(f"No more conflicts on neuron {neuron_name}")
        else:
            t, tracklet_name, _, _ = self.dat.tracklet_annotator.gt_mismatches[neuron_name].pop(0)
            self.logger.debug(f"Resolved conflict at t={t} on {neuron_name} and {tracklet_name}")
            self.logger.info(f"Resolved conflict; {sum(map(len, mismatches.values()))} mismatches remaining")

        # Unclear why, but multiple calls are necessary
        self.remove_match_layers()
        self.remove_match_layers()

    def remove_match_layers(self):
        for layer in self.viewer.layers:
            if 'match' in layer.name:
                self.viewer.layers.remove(layer)

    def zoom_to_start_of_current_tracklet(self, viewer=None):
        t = self.dat.tracklet_annotator.start_time_of_current_tracklet()
        if t is not None:
            change_viewer_time_point(self.viewer, t_target=t)
        else:
            print("No tracklet selected; not zooming")
        # self.time_changed_callbacks()

    def split_current_tracklet_keep_right(self):
        """
        Splits the current tracklet, updating the tracklet database.
        Keeps the right (future) half as selected in the gui, and removes old tracklet napari layer
        Also updates the subplot that displays all the tracklets

        Only difference between this and split_current_tracklet_keep_left:
            Splits at self.t vs. self.t+1
        """
        self.logger.debug("USER: split tracklet keep right")
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.remove_napari_layer_of_current_tracklet()
            # which_tracklets_to_update = self.subplot_update_dict_for_tracklet_modification()
            # Note that this function gives a log message if it fails
            successfully_split = self.dat.tracklet_annotator.split_current_tracklet(self.t,
                                                                                    set_right_half_to_current=True)
            if successfully_split:
                self.post_split_save_to_neuron_and_reselect()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def post_split_save_to_neuron_and_reselect(self):
        # Automatically save the current half to the neuron, if possible
        current_tracklet_name = self.dat.tracklet_annotator.current_tracklet_name
        self.save_current_tracklet_to_neuron(do_callback=False)
        # Reselect the tracklet, because it is removed when saved by default
        self.dat.tracklet_annotator.set_current_tracklet(current_tracklet_name)
        self.add_napari_layer_of_current_tracklet()
        # post_split_dict = self.subplot_update_dict_for_tracklet_modification()
        # which_tracklets_to_update.update(post_split_dict)
        self.tracklet_updated_psuedo_event()

    def split_current_tracklet_keep_left(self):
        """
        Splits the current tracklet, updating the tracklet database.
        Keeps the right (future) half as selected in the gui, and removes old tracklet napari layer
        Also updates the subplot that displays all the tracklets

        Only difference between this and split_current_tracklet_keep_right:
            Splits at self.t vs. self.t+1
        Returns
        -------

        """
        self.logger.debug("USER: split tracklet keep left")
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.remove_napari_layer_of_current_tracklet()
            # which_tracklets_to_update = self.subplot_update_dict_for_tracklet_modification()
            # Note that this function gives a log message if it fails
            successfully_split = self.dat.tracklet_annotator.split_current_tracklet(self.t + 1,
                                                                                    set_right_half_to_current=False)
            if successfully_split:
                self.post_split_save_to_neuron_and_reselect()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def clear_current_tracklet(self):
        self.logger.debug("USER: clear current tracklet")
        self.remove_napari_layer_of_current_tracklet()
        # current_tracklet_name = f"{self.dat.tracklet_annotator.current_tracklet_name}_current"
        self.dat.tracklet_annotator.clear_current_tracklet()
        self.tracklet_updated_psuedo_event()
        # self.tracklet_updated_psuedo_event(which_tracklets_to_update={current_tracklet_name: 'remove'})

    def toggle_raw_segmentation_layer(self):
        self._toggle_layer(self.raw_seg_layer)

    def toggle_colored_segmentation_layer(self):
        self._toggle_layer(self.colored_seg_layer)

    def toggle_neuropal_segmentation_layer(self):
        layer = self.neuropal_seg_layer
        if layer is not None:
            self._toggle_layer(layer)

    def toggle_neuron_ids(self):
        self.neuron_id_layer.visible = not self.neuron_id_layer.visible
        # self._toggle_layer(self.neuron_id_layer)

    def toggle_manual_ids(self):
        self.get_manual_id_layer().visible = not self.get_manual_id_layer().visible

    def _toggle_layer(self, layer):
        if self.viewer.layers.selection.active == layer:
            self.viewer.layers.selection.clear()
            layer.visible = False
        else:
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(layer)
            layer.visible = True

    def save_current_tracklet_to_neuron(self, do_callback=True):
        self.logger.debug("USER: save current tracklet to neuron")
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            tracklet_name = self.dat.tracklet_annotator.save_current_tracklet_to_current_neuron()
            if tracklet_name:
                self.remove_napari_layer_of_current_tracklet(tracklet_name)
                # which_tracklets_to_update = self.get_dict_for_tracklet_save(tracklet_name)
                if do_callback:
                    self.tracklet_updated_psuedo_event()
        else:
            print(f"{self.changeTraceTrackletDropdown.currentText()} mode, so this option didn't do anything")

    def save_segmentation_to_tracklet(self):
        flag = self.dat.tracklet_annotator.attach_current_segmentation_to_current_tracklet()
        self.logger.debug(f"USER: save single segmentation with outcome: {flag}")
        if flag:
            # Do callbacks because the tracklet annotator assumes I'm changing tracklets
            self.modify_current_tracklet()
            self.set_segmentation_layer_invisible()

    def delete_segmentation_from_tracklet(self):
        flag = self.dat.tracklet_annotator.delete_current_segmentation_from_tracklet()
        self.logger.debug(f"USER: delete single segmentation with outcome: {flag}")
        if flag:
            # Do callbacks because the tracklet annotator assumes I'm changing tracklets
            self.modify_current_tracklet()
            self.set_segmentation_layer_invisible()

    def remove_napari_layer_of_current_tracklet(self, layer_name=None):
        if layer_name is None:
            layer_name = self.dat.tracklet_annotator.current_tracklet_name
        if layer_name is not None and layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

    def remove_all_tracklet_napari_layers(self):
        to_remove = [layer for layer in self.viewer.layers if 'tracklet_' in layer.name]
        for layer in to_remove:
            self.viewer.layers.remove(layer)

    def add_napari_layer_of_current_tracklet(self, layer_name=None):
        if layer_name is None:
            layer_name = self.dat.tracklet_annotator.current_tracklet_name
        if layer_name is not None and layer_name not in self.viewer.layers:
            self.dat.tracklet_annotator.add_current_tracklet_to_viewer(self.viewer)

    def remove_layer_of_candidate_segmentation(self, layer_name='Candidate_mask'):
        if layer_name is not None and layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

    def print_tracklets(self):
        self.dat.tracklet_annotator.print_current_status()

    def remove_time_conflicts(self):
        conflicting_names = self.dat.tracklet_annotator.remove_tracklets_with_time_conflicts()
        # which_tracklets_to_update = {name: 'remove' for name in conflicting_names}
        # self.tracklet_updated_psuedo_event(which_tracklets_to_update=which_tracklets_to_update)
        self.tracklet_updated_psuedo_event()

    def remove_tracklet_from_all_matches(self):
        tracklet_name = self.dat.tracklet_annotator.remove_tracklet_from_all_matches()
        self.update_tracklet_status_label()
        # which_tracklets_to_update = {tracklet_name: 'remove'}
        self.tracklet_updated_psuedo_event()

    def remove_all_tracklets_after_current_time(self):
        # Just clear and update the entire plot because this should be rare and a huge change
        t = self.t
        self.dat.tracklet_annotator.remove_all_tracklets_after_time(t)
        self.tracklet_updated_psuedo_event()

    @cached_property
    def y_min_max_on_plot(self):
        """
        Return the min and max, because there could be overlapping lines in tracklet mode

        If there are no lines, return empty list

        In trace mode, returns the single line self.y_trace_mode twice

        Returns
        -------

        """
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            y_on_plot = [line.get_ydata() for line in self.static_ax.lines]
            if len(y_on_plot) == 0 or (len(y_on_plot) == 1 and len(y_on_plot[0]) == 2):
                # Empty neuron!
                print("No data found")
                return []
            proper_len = len(y_on_plot[0])  # Have to remove the time line!
            y_on_plot = [y for y in y_on_plot if len(y) == proper_len]
            # I expect to see RuntimeWarnings in this block
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y_on_plot_min = np.nanmin(np.vstack(y_on_plot), axis=0)
                y_on_plot_max = np.nanmax(np.vstack(y_on_plot), axis=0)
        else:
            y_on_plot_min = self.y_trace_mode
            y_on_plot_max = self.y_trace_mode
        return y_on_plot_min, y_on_plot_max

    def init_universal_subplot(self):
        # Note: this causes a hang when the main window is closed, even though I'm trying to set the parent
        self.mpl_widget = PlotQWidget()
        self.static_ax = self.mpl_widget.canvas.fig.subplots()
        self.reference_ax = self.static_ax.twinx()
        self.main_subplot_xlim = [0, self.dat.num_frames]
        
        # Connect clicking to a time change
        # https://matplotlib.org/stable/users/event_handling.html
        on_click = lambda event: self.on_subplot_click(event)
        cid = self.mpl_widget.canvas.mpl_connect('button_press_event', on_click)
        self.connect_time_line_callback()

    def init_subplot_post_clear(self):
        # Recreate the time line, but make sure the references are removed
        if self.time_line is not None:
            self.time_line.remove()
            del self.time_line
        time_line_options = self.calculate_time_line_options()
        self.logger.debug(f"Initializing subplot post clear with time line: {time_line_options}")
        try:
            self.time_line = self.static_ax.axvline(**time_line_options)
        except (ValueError, LinAlgError) as e:
            self.logger.warning(f"Error creating time line: {e}; creating it anyway")
            # Use dummy y values; somehow this crash is related to being unable to set the y values
            # Something about an un-invertable matrix
            time_line_options['ymin'] = 0
            time_line_options['ymin'] = 1
            try:
                self.time_line = self.static_ax.axvline(**time_line_options)
            except (ValueError, LinAlgError) as e:
                # If it doesn't work the second time, just give up
                self.logger.warning(f"Error creating time line: {e}; giving up")
                self.time_line = None

        # Try to preserve the xlimits
        self.static_ax.set_ylabel(self.changeTraceCalculationDropdown.currentText())
        self.color_using_behavior()
        if self.current_subplot_xlim is not None:
            self.static_ax.set_xlim(self.current_subplot_xlim)
        else:
            self.static_ax.set_xlim(self.main_subplot_xlim)
        self.subplot_is_initialized = True
        # Add additional annotations
        self.color_using_behavior()

    def initialize_trace_subplot(self):
        self.update_stored_trace_time_series()
        # If there is already a trace line, just empty it
        self.trace_line = self.static_ax.plot(self.tspan, self.y_trace_mode)[0]
        self.add_or_remove_tracking_outliers()
        # If there is already a reference line, just empty it
        if self.reference_line is not None:
            self.remove_reference_line()
        else:
            self.reference_line = self.reference_ax.plot([], color='tab:orange')[0]  # Initialize an empty line
        self.invalidate_y_min_max_on_plot()

    def initialize_tracklet_subplot(self):
        # Designed for traces, but reuse
        field_to_plot = self.changeTraceCalculationDropdown.currentText()
        # self.update_stored_time_series(field_to_plot)
        self.tracklet_lines = {}
        self.update_stored_tracklets_for_plotting()
        marker_opt = self.get_marker_opt()
        for name, y in self.y_tracklets_dict.items():
            new_line = y[field_to_plot].plot(ax=self.static_ax, **marker_opt).lines[0]
            self.add_tracklet_to_cache(new_line, name)
        self.update_neuron_in_tracklet_annotator()
        self.invalidate_y_min_max_on_plot()

    def add_or_remove_tracking_outliers(self):
        """
        Wrapper around add_tracking_outliers_to_plot and remove_tracking_outliers_on_plot

        Switches based on the checkbox

        Returns
        -------

        """
        if self.ppcaOutlierOverlayCheckbox.isChecked():
            self.logger.debug("Adding tracking outliers")
            self.add_tracking_outliers_to_plot()
        else:
            self.logger.debug("Removing tracking outliers")
            self.remove_tracking_outliers_from_plot()
        self.draw_subplot()

    def update_neuron_id_strings_in_layer(self, original_name: Union[list, str], old_name, new_name: Union[list, str],
                                          actually_update_gui=True, neuropal=False):
        """
        Modify the layer properties to change the displayed name of a neuron (across all time)

        Parameters
        ----------
        original_name
        old_name
        new_name
        actually_update_gui

        Returns
        -------

        """

        if isinstance(original_name, list):
            # Assume this is a multi-neuron change, and update the dataframe first before the gui
            assert len(original_name) == len(new_name), "Must have the same number of old and new names"
            for o, n in zip(original_name, new_name):
                self.update_neuron_id_strings_in_layer(o, old_name, n, actually_update_gui=False, neuropal=neuropal)
            if actually_update_gui:
                worker = self.refresh_manual_id_layer(neuropal)
                # worker.start()
            return

        msg = f"Changing neuron name {old_name} to {new_name} (original name: {original_name})"
        if neuropal:
            msg += " in neuropal layer"
        self.logger.info(msg)
        # Because the old name may have been blank, we need to use the automatic labels for indexing
        id_layer = self.get_manual_id_layer(neuropal=neuropal)
        original_name_series = id_layer.properties['automatic_label']
        original_name_series = pd.Series([int2name_neuron(n) for n in original_name_series])

        # Only modify the rows that are being changed
        rows_to_change = original_name_series == original_name

        # We need to change the features dataframe, not the properties dict
        id_layer.features.loc[rows_to_change, 'custom_label'] = new_name

        if actually_update_gui:
            # This updates the entire text layer including all strings, so it takes a while
            # Therefore do a new thread
            worker = self.refresh_manual_id_layer(neuropal)
            # worker.start()

    def refresh_manual_id_layer(self, neuropal=False):
        # This decorator makes the function return a worker, even though pycharm doesn't know it
        self.get_manual_id_layer(neuropal).refresh_text()

    def add_tracking_outliers_to_plot(self):
        # TODO: will improperly jump to selected tracklets when added; should be able to loop over self.tracklet_lines

        self.remove_tracking_outliers_from_plot()
        # Note that this function should be cached
        outlier_matrix = self.dat.calc_indices_to_remove_using_ppca()
        if len(outlier_matrix) == 0:
            self.logger.debug("No tracking outliers found; ppca algorithm probably failed")
            return

        # This is a matrix, so I need the index of this neuron name
        neuron_name = self.changeNeuronDropdown.currentText()
        neuron_index = self.dat.neuron_names.index(neuron_name)
        outlier_ind = outlier_matrix[:, neuron_index]
        x = np.array(self.tspan)[outlier_ind]
        # Take the upper line, if there are multiple overlapping lines... but there shouldn't be
        y = self.y_min_max_on_plot[1]
        y = y[outlier_ind[:len(y)]]

        self.outlier_line = self.static_ax.plot(x, y, 'o', color='tab:red')[0]
        self.logger.debug(f"Successfully added {len(x)} tracking outliers to plot")
        self.logger.debug(f"Max: {np.nanmax(y)}, min: {np.nanmin(y)}")

    def remove_tracking_outliers_from_plot(self):
        if self.outlier_line is not None:
            self.outlier_line.remove()
            del self.outlier_line
            self.outlier_line = None

    def on_subplot_click(self, event):
        t = event.xdata
        change_viewer_time_point(self.viewer, t_target=t)
        # self.time_changed_callbacks()

    def change_trace_tracklet_mode(self):
        current_mode = self.changeTraceTrackletDropdown.currentText()

        print(f"Changed mode to: {current_mode}")
        self.static_ax.clear()
        # Only show z coordinate for now

        self.changeTraceCalculationDropdown.clear()
        # Note: Setting the value of changeTraceCalculationDropdown updates the subplot
        self._disable_callbacks = True
        if current_mode == 'tracklets':
            self.changeTraceCalculationDropdown.addItems(self.tracklet_mode_calculation_options)
            self.changeTraceCalculationDropdown.setCurrentText('z')
            self.changeReferenceTrace.setCurrentText('None')
        elif current_mode == 'traces':
            self.changeTraceCalculationDropdown.addItems(self.traces_mode_calculation_options)
            self.changeTraceCalculationDropdown.setCurrentText('integration')
        self._disable_callbacks = False

        self.initialize_trace_or_tracklet_subplot()
        # Not just updating the data because we fully cleared the axes
        self.invalidate_y_min_max_on_plot()  # Force invalidation, so it is recalculated
        self.init_subplot_post_clear()
        self.finish_subplot_update_and_draw()

    def initialize_trace_or_tracklet_subplot(self):
        if not self.subplot_is_initialized:
            self.init_universal_subplot()

        # This middle block will be called when the mode is switched
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.logger.debug("Initializing tracklet mode")
            self.initialize_tracklet_subplot()
        elif self.changeTraceTrackletDropdown.currentText() == 'traces':
            self.logger.debug("Initializing trace mode")
            self.initialize_trace_subplot()

        if not self.subplot_is_initialized:
            self.init_subplot_post_clear()
            # Finally, add the traces to napari
            self.viewer.window.add_dock_widget(self.mpl_widget, area='bottom')
        # Additional annotations
        self.add_or_remove_tracking_outliers()
        self.finish_subplot_update_and_draw()

    def update_trace_or_tracklet_subplot(self, dropdown_ind=None,
                                         preserve_xlims=True, which_tracklets_to_update=None):
        """
        Update the trace or tracklet subplot, depending on the current mode

        Parameters
        ----------
        dropdown_ind
        preserve_xlims
        which_tracklets_to_update

        Returns
        -------

        """
        if self._disable_callbacks:
            return
        if self.changeTraceCalculationDropdown.currentText() == "":
            # Assume it has just been cleared, and wait
            return

        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            self.update_tracklet_subplot(preserve_xlims, which_tracklets_to_update=which_tracklets_to_update)
        elif self.changeTraceTrackletDropdown.currentText() == 'traces':
            self.update_trace_subplot()
        else:
            raise ValueError

    def update_trace_subplot(self):
        """
        Update the trace subplot, assuming that the mode is currently set to traces

        Returns
        -------

        """
        if not self.changeTraceTrackletDropdown.currentText() == 'traces':
            print("Currently on tracklet setting, so this option didn't do anything")
            return
        self.update_stored_trace_time_series()
        self.trace_line.set_ydata(self.y_trace_mode)
        self.update_reference_trace(force_draw=False)

        self.invalidate_y_min_max_on_plot()
        # Add additional annotations that change based on the y values
        self.add_or_remove_tracking_outliers()
        self.init_subplot_post_clear()
        self.finish_subplot_update_and_draw(preserve_xlims=True)

    def update_reference_trace(self, current_index=0, force_draw=True):
        """
        Extra arg to accept the unused arg passed to callbacks of comboboxes

        Parameters
        ----------
        current_index
        force_draw

        Returns
        -------

        """
        if self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            ref_name = "None"
            print("Currently on tracklet setting, so not updating reference trace")
        else:
            ref_name = self.changeReferenceTrace.currentText()

        if ref_name == "None":
            # Reset line
            # self.static_ax.lines.remove(self.reference_line)
            self.remove_reference_line()

            if force_draw:
                self.logger.debug("USER: force draw of reference line")
                # For some reason, just the second call doesn't properly delete the line
                self.update_trace_subplot()
                self.finish_subplot_update_and_draw(preserve_xlims=True)
        else:
            # Plot other trace
            t, y = self.calculate_trace(trace_name=ref_name)
            self.reference_line.set_data(t, y)
            self.reference_line.set_visible(True)
            # self.reference_line.set_ydata(y)
            if force_draw:
                self.finish_subplot_update_and_draw(preserve_xlims=True)

    def remove_reference_line(self):
        # self.reference_line.set_data([], [])
        if self.reference_line is not None:
            self.reference_line.set_visible(False)

    def get_subplot_title(self):
        ref_name = self.changeReferenceTrace.currentText()
        red_green_or_ratio = self.changeChannelDropdown.currentText()
        integration_or_not = self.changeTraceCalculationDropdown.currentText()
        traces_or_tracklets = self.changeTraceTrackletDropdown.currentText()
        neuron_name = self.changeNeuronDropdown.currentText()
        # Convert to custom IDs, if they exist
        if self.manualNeuronNameEditor is not None:
            mapping = self.manualNeuronNameEditor.original2custom(remove_empty=True)
            ref_name = mapping.get(ref_name, ref_name)
            neuron_name = mapping.get(neuron_name, neuron_name)

        if traces_or_tracklets == 'tracklets':
            title = f"Tracklets for {neuron_name}"
        else:
            title = f"{red_green_or_ratio} trace for {integration_or_not} mode for {neuron_name}"
            if ref_name != "None":
                title = f"{title} with reference {ref_name}"

        return title

    def update_tracklet_subplot(self, preserve_xlims=True, which_tracklets_to_update=None):
        """
        Update the tracklet subplot, depending on the current mode

        Currently, all calls are made with which_tracklets_to_update=None, forcing a full update of the subplot

        Parameters
        ----------
        preserve_xlims
        which_tracklets_to_update

        Returns
        -------

        """
        if not self.changeTraceTrackletDropdown.currentText() == 'tracklets':
            print("Currently on traces setting, so this option didn't do anything")
            return

        # Enhancement: Only refresh this if it needs to be updated
        self.update_stored_tracklets_for_plotting()
        if preserve_xlims:
            self.current_subplot_xlim = self.static_ax.get_xlim()
        else:
            self.current_subplot_xlim = None
        marker_opt = self.get_marker_opt()
        current_tracklet_opt = dict(color='k', lw=3)

        field_to_plot = self.changeTraceCalculationDropdown.currentText()
        if which_tracklets_to_update is None:
            # Replot ALL tracklets; takes time
            self.static_ax.clear()
            self.tracklet_lines = {}  # Remove references to old lines
            for name, y in tqdm(self.y_tracklets_dict.items(), leave=False):
                new_line = y[field_to_plot].plot(ax=self.static_ax, **marker_opt).lines[-1]
                self.add_tracklet_to_cache(new_line, name)
            if self.current_tracklet is not None:
                y = self.current_tracklet[field_to_plot]
                new_line = y.plot(ax=self.static_ax, **current_tracklet_opt, **marker_opt).lines[-1]
                line_name = f"{self.current_tracklet_name}_current"
                self.add_tracklet_to_cache(new_line, line_name)
            # Not a clear in the other branch
            self.init_subplot_post_clear()
        else:
            print(f"Updates: {which_tracklets_to_update}")
            for tracklet_name, type_of_update in which_tracklets_to_update.items():
                # If it is an already plotted (colored) tracklet
                if tracklet_name in self.tracklet_lines:
                    if type_of_update == 'remove' or type_of_update == 'replot':
                        self.remove_tracklet_from_plot_and_cache(tracklet_name)
                else:
                    if 'None' not in tracklet_name and '_current' not in tracklet_name:
                        logging.warning(f"Tried to modify {tracklet_name} on the subplot, but it wasn't found")
                        if type_of_update == 'replot':
                            # Do not allow replotting, because it would actually add a spurious line
                            type_of_update = ""
                # Should NOT be elif
                if type_of_update == 'plot' or type_of_update == 'replot':
                    # If it is an already plotted (colored) tracklet
                    y = self.y_tracklets_dict.get(tracklet_name, None)
                    extra_opt = dict()
                    if y is None:
                        if tracklet_name.endswith('_current'):
                            tracklet_name_exact = tracklet_name.replace("_current", "")
                            if tracklet_name_exact == self.current_tracklet_name:
                                y = self.current_tracklet
                                extra_opt = current_tracklet_opt

                    if y is None:
                        self.logger.debug(f"Tried to plot {tracklet_name}, but it wasn't found")
                        continue

                    new_line = y[field_to_plot].plot(ax=self.static_ax, **extra_opt, **marker_opt).lines[-1]
                    self.add_tracklet_to_cache(new_line, tracklet_name)
        self.invalidate_y_min_max_on_plot()
        self.add_or_remove_tracking_outliers()

        # self.update_stored_time_series(field_to_plot)
        # print(f"Final tracklets on plot: {self.tracklet_lines.keys()}")
        # print(self.static_ax.lines)

        self.finish_subplot_update_and_draw(preserve_xlims=preserve_xlims)

    def add_tracklet_to_cache(self, new_line, tracklet_name):
        self.tracklet_lines[tracklet_name] = new_line

    def remove_tracklet_from_plot_and_cache(self, tracklet_name):
        self.tracklet_lines[tracklet_name].remove()
        del self.tracklet_lines[tracklet_name]

    def invalidate_y_min_max_on_plot(self):
        if 'y_min_max_on_plot' in self.__dict__:
            del self.__dict__['y_min_max_on_plot']  # Force invalidation, so it is recalculated

    def get_marker_opt(self):
        if self.changeSubplotMarkerDropdown.currentText() == 'line':
            opt = dict(marker='')
        elif self.changeSubplotMarkerDropdown.currentText() == 'dots':
            opt = dict(marker='o')
        else:
            opt = {}
        return opt

    # def tracklet_updated_psuedo_event(self, which_tracklets_to_update=None):
    def tracklet_updated_psuedo_event(self):
        self.update_tracklet_status_label()
        self.update_zoom_options_for_current_tracklet()
        self.add_to_recent_tracklet_dropdown()
        self.update_trace_or_tracklet_subplot()
        # self.update_trace_or_tracklet_subplot(which_tracklets_to_update=which_tracklets_to_update)

    def update_tracklet_status_label(self):
        if self.dat.tracklet_annotator.current_neuron is None:
            update_string = "STATUS: No tracklet selected"
        else:
            if self.dat.tracklet_annotator.is_current_tracklet_confict_free:
                update_string = f"Selected: {self.dat.tracklet_annotator.current_tracklet_name}"
            else:
                types_of_conflicts = self.dat.tracklet_annotator.get_types_of_conflicts()
                update_string = f"Selected: {self.dat.tracklet_annotator.current_tracklet_name} " \
                                f"(Conflicts: {types_of_conflicts})"
        self.saveTrackletsStatusLabel.setText(update_string)

    def update_zoom_options_for_current_tracklet(self):
        tracklet_name = self.dat.tracklet_annotator.current_tracklet_name
        if tracklet_name and tracklet_name in self.viewer.layers:
            # Note that this should be called again if the layer is deleted
            self.zoom_opt['layer_name'] = self.dat.tracklet_annotator.current_tracklet_name
        else:
            # Set back to default
            self.zoom_opt['layer_name'] = 'final_track'

    def update_segmentation_status_label(self):
        if self.dat.tracklet_annotator.indices_of_original_neurons is None:
            update_string = "No segmentations selected"
        else:
            update_string = f"Selected segmentation(s): " \
                            f"{self.dat.tracklet_annotator.indices_of_original_neurons}"
        self.saveSegmentationStatusLabel.setText(update_string)

    def toggle_highlight_selected_neuron(self):
        self.dat.tracklet_annotator.toggle_highlight_selected_neuron(self.viewer)

    def center_on_selected_neuron(self):
        position = self.dat.tracklet_annotator.last_clicked_position
        # Only center if the last click was at the same time as the viewer
        if position is not None and position[0] == self.viewer.dims.current_step:
            zoom_using_viewer(position, self.viewer, zoom=None)

    def set_segmentation_layer_invisible(self):
        self.raw_seg_layer.visible = False

    def set_segmentation_layer_visible(self):
        self.raw_seg_layer.visible = True

    def set_segmentation_layer_do_not_show_selected_label(self):
        if self.raw_seg_layer is not None:
            self.raw_seg_layer.show_selected_label = False

    def time_changed_callbacks(self):
        # Check to make sure there was actually a change
        if self.current_time_point_before_callback != self.t:
            self.last_time_point = self.current_time_point_before_callback
            self.current_time_point_before_callback = self.t
            self.zoom_using_current_neuron_or_tracklet()
        self.set_segmentation_layer_do_not_show_selected_label()

    def finish_subplot_update_and_draw(self, title=None, preserve_xlims=True):
        if title is None:
            title = self.get_subplot_title()

        self.update_time_line()
        self.static_ax.set_title(title)
        print(f"Updating subplot with {len(self.static_ax.lines)} lines and title {title}")
        if preserve_xlims:
            self.static_ax.autoscale(axis='y')
            self.reference_ax.autoscale(axis='y')
        else:
            self.static_ax.autoscale(axis='both')
            self.reference_ax.autoscale(axis='both')
        self.static_ax.relim()
        self.reference_ax.relim()
        self.draw_subplot()
        y_min, y_max = self.y_min_max_on_plot
        self.logger.debug(f"Autoscaled axis: {np.nanmin(y_min)} and {np.nanmax(y_max)}")

    def draw_subplot(self):
        # self.static_ax.update_params()
        self.mpl_widget.draw()
        # self.mpl_widget.canvas.draw()

    def connect_time_line_callback(self):
        viewer = self.viewer

        @viewer.dims.events.current_step.connect
        def update_time_line(event):
            self.update_time_line()

    def update_time_line(self):
        # Doesn't work if the time line needs to be initialized
        time_options = self.calculate_time_line_options()
        try:
            self.time_line.set_xdata(time_options['x'])
            # self.time_line.set_data(time_options[:2])
            self.time_line.set_color(time_options['color'])
        except AttributeError as e:
            # Sometimes a graphics error causes the time line to fail to be initialized; if so just ignore it
            self.logger.warning("Failed to update time line; probably a graphics error; ignoring it")
            self.logger.debug(f"Error when updating time line: {e}")
        self.mpl_widget.draw()

    @property
    def t(self):
        return self.viewer.dims.current_step[0]

    def calculate_time_line_options(self):
        t = self.t
        y_min, y_max = self.y_min_max_on_plot
        if len(y_min) == 0:
            return [t, t], [0, 30], 'r'
        # ymin, ymax = np.nanmin(y_min), np.nanmax(y_max)
        if t < len(y_min):
            self.tracking_is_nan = np.isnan(y_min[t])
        else:
            self.tracking_is_nan = True
        if self.tracking_is_nan:
            line_color = 'r'
        else:
            line_color = 'k'
        # print(f"Updating time line for t={t}, y[t] = {y[t]}, color={line_color}")
        # return [t, t], [ymin, ymax], line_color
        return dict(x=t, color=line_color)

    def update_stored_trace_time_series(self):
        t, y = self.calculate_trace()
        self.y_trace_mode: pd.Series = y
        self.tspan = t

    def calculate_trace(self, trace_name=None) -> Tuple[List, pd.Series]:
        """
        Uses project_data class to calculate the trace for the current neuron, or the neuron specified by trace_name

        Can also calculate behaviors

        Parameters
        ----------
        trace_name

        Returns
        -------

        """
        if trace_name is None:
            trace_name = self.current_neuron_name

        if trace_name in self.dat.neuron_names:
            # Calculate the entire dataframe, because some options require it
            y = self.df_of_current_traces[trace_name]
            t = self.dat.x_for_plots
            # t, y = self.dat.calculate_traces(neuron_name=trace_name, **trace_opt)
        else:
            # Assume it is a behavior name
            trace_opt = self.get_trace_opt()
            t, y = self.dat.calculate_behavior_trace(trace_name, **trace_opt)
        return t, y

    def get_trace_opt(self):
        channel = self.changeChannelDropdown.currentText()
        calc_mode = self.changeTraceCalculationDropdown.currentText()
        min_confidence = None
        remove_outliers_activity = self.changeTraceOutlierCheckBox.isChecked()
        bleach_correct = self.changeBleachCorrectionCheckBox.isChecked()
        filter_mode = self.changeTraceFilteringDropdown.currentText()
        residual_mode = self.changeResidualModeDropdown.currentText()
        interpolate_nan = self.changeInterpolationModeDropdown.isChecked()
        nan_using_ppca_manifold = self.ppcaOutlierRemovalCheckbox.isChecked()
        if residual_mode != 'none':
            interpolate_nan = True
        trace_opt = dict(channel_mode=channel, calculation_mode=calc_mode,
                         remove_outliers=remove_outliers_activity,
                         filter_mode=filter_mode,
                         min_confidence=min_confidence,
                         bleach_correct=bleach_correct,
                         residual_mode=residual_mode,
                         interpolate_nan=interpolate_nan,
                         nan_using_ppca_manifold=nan_using_ppca_manifold,
                         remove_tail_neurons=False)
        return trace_opt

    @property
    def df_of_current_traces(self):
        """
        Dataframe of traces with current options

        Note that the project data class caches these dataframes, so they should only be slow once
        """
        trace_opt = self.get_trace_opt()
        return self.dat.calc_default_traces(**trace_opt, min_nonnan=None)

    def update_stored_tracklets_for_plotting(self):
        name = self.current_neuron_name
        tracklets_dict, tracklet_current, current_name = self.dat.calculate_tracklets(name)
        self.logger.debug(f"Found {len(tracklets_dict)} tracklets for {name}")
        if tracklet_current is not None:
            self.logger.debug("Additionally found 1 currently selected tracklet")
        self.y_tracklets_dict = tracklets_dict
        self.current_tracklet = tracklet_current
        self.current_tracklet_name = current_name

    def get_track_data(self):
        self.current_neuron_name = self.changeNeuronDropdown.currentText()
        return self.build_tracks_from_name()

    def color_using_behavior(self):
        if self.dat.worm_posture_class.has_beh_annotation:
            additional_shaded_state = self.changeSubplotShading.currentText()
            # Convert to BehaviorCodes, if valid
            if additional_shaded_state == 'none':
                additional_shaded_states = []
            else:
                additional_shaded_states = [BehaviorCodes[additional_shaded_state.upper()]]
            self.dat.shade_axis_using_behavior(self.static_ax, additional_shaded_states=additional_shaded_states)

    def remove_behavior_shading(self):
        ax = self.static_ax
        # Remove axvspan elements while retaining lines
        for artist in ax.get_children():
            if isinstance(artist, plt.matplotlib.patches.Polygon):  # Check if it's an axvspan
                artist.remove()

    def update_behavior_shading(self):
        self.remove_behavior_shading()
        self.color_using_behavior()
        self.draw_subplot()

    # def save_annotations(self):
    #     self.update_dataframe_using_points()
    #     # self.df[self.current_name] = new_df[self.current_name]
    #
    #     out_fname = self.annotation_output_name
    #     self.df.to_hdf(out_fname, 'df_with_missing')
    #
    #     out_fname = str(Path(out_fname).with_suffix('.csv'))
    #     #     df_old = pd.read_csv(out_fname)
    #     #     df_old[name] = df_new[name]
    #     #     df_old.to_csv(out_fname, mode='a')
    #     self.df.to_csv(out_fname)  # Just overwrite
    #
    #     print(f"Saved manual annotations for neuron {self.current_name} at {out_fname}")

    def update_dataframe_using_final_tracks_layer(self):
        logging.warning("DEPRECATION WARNING: Overwriting tracks using manual points from final_tracks layer")
        # Note: this allows for manual changing of the points
        new_df = self.build_df_of_current_points()

        self.dat.final_tracks = self.dat.final_tracks.drop(columns=self.current_neuron_name, level=0)
        self.dat.final_tracks = pd.concat([self.dat.final_tracks, new_df], axis=1)

    def build_df_of_current_points(self) -> pd.DataFrame:
        name = self.current_neuron_name
        new_points = self.final_track_layer.data

        # Initialize as dict and immediately create dataframe
        coords = ['z', 'x', 'y', 'likelihood']
        coords2ind = {'z': 1, 'x': 2, 'y': 3, 'likelihood': None}
        tmp_dict = {}
        for c in coords:
            key = (name, c)
            pts_ind = coords2ind[c]
            if pts_ind is not None:
                tmp_dict[key] = new_points[:, pts_ind]
            else:
                tmp_dict[key] = np.ones(new_points.shape[0])

        df_new = pd.DataFrame(tmp_dict)

        # col = pd.MultiIndex.from_product([[self.current_name], ['z', 'x', 'y', 'likelihood']])
        # df_new = pd.DataFrame(columns=col, index=self.dat.final_tracks.index)
        #
        # df_new[(name, 'z')] = new_points[:, 1]
        # df_new[(name, 'x')] = new_points[:, 2]
        # df_new[(name, 'y')] = new_points[:, 3]
        # df_new[(name, 'likelihood')] = np.ones(new_points.shape[0])

        return df_new

    def build_tracks_from_name(self):
        neuron_name = self.current_neuron_name
        df_single_track = self.dat.final_tracks[neuron_name]
        likelihood_threshold = self.dat.likelihood_thresh
        z_to_xy_ratio = self.dat.physical_unit_conversion.z_to_xy_ratio
        all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track,
                                                                                  likelihood_threshold,
                                                                                  z_to_xy_ratio)

        self.bad_points = to_remove
        return all_tracks_array, track_of_point

    def add_layer_colored_by_correlation_to_current_neuron(self):
        """
        Get the correlation between the current neuron and the rest...
        for now the dataframe needs to be recalculated
        """
        which_layers = [('heatmap', 'custom_val_to_plot', f'correlation_to_{self.current_neuron_name}_at_t_{self.t}')]
        y = self.y_trace_mode
        df = self.df_of_current_traces
        val_to_plot = df.corrwith(y)
        # Square but keep the sign; de-emphasizes very small correlations
        val_to_plot = val_to_plot * np.abs(val_to_plot)
        heatmap_kwargs = dict(val_to_plot=val_to_plot, t=self.t, scale_to_minus_1_and_1=True)
        self.logger.debug(f'Calculated correlation values: {val_to_plot}')
        self.dat.add_layers_to_viewer(self.viewer, which_layers=which_layers, heatmap_kwargs=heatmap_kwargs,
                                      layer_opt=dict(opacity=1.0))
        # Move manual_ids to top, so they are not obscured
        i_manual_id_layer = self.viewer.layers.index(self.get_manual_id_layer())
        # Reorder function needs the layer index, not the name
        self.viewer.layers.move(i_manual_id_layer, -1)


def napari_trace_explorer_from_config(project_path: str, app=None,
                                      load_tracklets=True, force_tracklets_to_be_sparse=True,
                                      DEBUG=False, **kwargs):
    # A parent QT application must be initialized first
    os.environ["NAPARI_ASYNC"] = "1"
    # os.environ["NAPARI_PERFMON"] = "1"
    # os.environ["NAPARI_OCTREE"] = "1" # No effect in tests; seems to only matter in 2d
    start_time = time.time()
    if app is None:
        started_new_app = True
        app = QApplication([])
    else:
        started_new_app = False

    # Build project class that has all the data
    initialization_kwargs = dict(use_custom_padded_dataframe=False,
                                 force_tracklets_to_be_sparse=force_tracklets_to_be_sparse,
                                 set_up_tracklet_interactivity=load_tracklets)
    project_data = ProjectData.load_final_project_data(project_path,
                                                       to_load_tracklets=load_tracklets,
                                                       to_load_interactivity=load_tracklets,
                                                       to_load_segmentation_metadata=True,
                                                       to_load_frames=load_tracklets,  # This is used for ground truth comparison, which requires tracklets
                                                       initialization_kwargs=initialization_kwargs, allow_hybrid_loading=True)
    if DEBUG:
        logging.debug(project_data)
    # If I don't set this to false, need to debug custom dataframe here
    project_data.use_custom_padded_dataframe = False
    # project_data.load_interactive_properties()
    ui, viewer = napari_trace_explorer(project_data, app=app, load_tracklets=load_tracklets, start_time=start_time,
                                       **kwargs)

    # Note: don't use this in jupyter
    napari.run()
    logging.info("Successfully quit napari application")
    sys.exit()


def napari_trace_explorer(project_data: ProjectData,
                          app: QApplication = None,
                          viewer: napari.Viewer = None,
                          to_print_fps: bool = False, start_time=None, **kwargs):
    """Current function for building the explorer (1/11/2022)"""
    logging.debug("Starting GUI setup")
    # Make sure ctrl-c works
    # https://python.tutorialink.com/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-console-ctrl-c/
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Build Napari and add data layers
    ui = NapariTraceExplorer(project_data, app, **kwargs)
    if viewer is None:
        ui.logger.info("Creating a new Napari window")
        viewer = napari.Viewer(ndisplay=3)
    ui.dat.add_layers_to_viewer(viewer, dask_for_segmentation=False)

    if project_data.neuropal_manager.has_complete_neuropal:
        ui.dat.add_layers_to_viewer(viewer, which_layers=['Neuropal', 'Neuropal segmentation', 'Neuropal Ids'])

    # Actually dock my additional gui elements
    ui.setup_ui(viewer)
    viewer.window.add_dock_widget(ui)
    ui.show()
    change_viewer_time_point(viewer, t_target=10)

    if to_print_fps:
        add_fps_printer(viewer)

    ui.logger.info("Finished GUI setup. If nothing is showing, trying quitting and running again")
    if start_time is not None:
        ui.logger.info(f"Time to initialize: {time.time() - start_time:.2f} s")
    return ui, viewer


def napari_behavior_explorer_from_config(project_path, fluorescence_fps=True, DEBUG=False):
    # Load project
    project_data = ProjectData.load_final_project_data_from_config(project_path)

    # Load specific data
    try:
        df_kymo = project_data.worm_posture_class.curvature(fluorescence_fps=fluorescence_fps)
    except NoBehaviorAnnotationsError as e:
        print("No behavior annotations were found; showing only the video")
        df_kymo = None
        
    video_fname = project_data.worm_posture_class.behavior_video_btf_fname(raw=True)
    try:
        # store = tifffile.imread(video_fname, aszarr=True)
        # video_zarr = zarr.open(store, mode='r')
        video_zarr = project_data.worm_posture_class.raw_behavior_video
    except ValueError:
        print("Failed to open video as tiff; trying zarr format")
        print(video_fname)
        return
    # Subset video to be the same fps as the fluorescence
    if fluorescence_fps:
        video_zarr = video_zarr[::project_data.physical_unit_conversion.frames_per_volume, :, :]
    video = video_zarr

    # dask_chunk = list(video_zarr.chunks).copy()
    # dask_chunk[0] = 1000
    # video = dask.array.from_zarr(video_zarr, chunks=dask_chunk)

    # Main viewer
    viewer = napari.view_image(video)

    # Kymograph subplot
    if fluorescence_fps:
        aspect = 1
    else:
        aspect = 24
    mpl_widget = PlotQWidget()
    static_ax = mpl_widget.canvas.fig.subplots()
    # Get vmin and vmax dynamically
    if df_kymo is not None:
        vmin = np.nanquantile(df_kymo.values, 0.01)
        vmax = np.nanquantile(df_kymo.values, 0.99)
        static_ax.imshow(df_kymo.T, aspect=aspect, vmin=vmin, vmax=vmax, cmap='RdBu')
    viewer.window.add_dock_widget(mpl_widget, area='bottom')

    # Callback: click on the kymograph to change the viewer time
    def on_subplot_click(event):
        t = event.xdata
        change_viewer_time_point(viewer, t_target=t)
    mpl_widget.canvas.mpl_connect('button_press_event', on_subplot_click)

    # Callback: when the time is changed, update a vertical line on the kymograph
    def get_time_line_options():
        return dict(x=viewer.dims.current_step[0], color='black')

    time_line = static_ax.axvline(**get_time_line_options())

    @viewer.dims.events.current_step.connect
    def update_time_line(event):
        time_options = get_time_line_options()
        time_line.set_xdata(time_options['x'])
        time_line.set_color(time_options['color'])
        mpl_widget.draw()

    # Start gui loop
    napari.run()
