import logging
import os
import subprocess
import sys
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, Qt, QItemSelectionModel
from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QMessageBox, QStyledItemDelegate
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QTableView, QAbstractItemView, QListWidget, QLabel, QPushButton
from PyQt5.QtGui import QStandardItemModel, QStandardItem


def _fix_dimension_for_plt(crop_sz, dat_crop):
    # Final output should be XYC
    if len(dat_crop.shape) == 3:
        if crop_sz is None:
            # Just visualize center of worm
            dat_crop = dat_crop[15]  # Remove z
        else:
            dat_crop = dat_crop[0]
    return np.array(dat_crop)


def array2qt(img):
    # From: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
    h, w, channel = img.shape
    # bytesPerLine = 3 * w
    # return QtGui.QPixmap(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    new_img = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(new_img)


def zoom_using_layer_in_viewer(viewer: napari.Viewer, layer_name='pts_with_future_and_past', zoom=None,
                               layer_is_full_size_and_single_neuron=True, ind_within_layer=None) -> None:
    # Get current point
    t = viewer.dims.current_step[0]
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]

        if layer_is_full_size_and_single_neuron:
            tzxy = get_zxy_from_single_neuron_layer(layer, t)
        else:
            tzxy = get_zxy_from_multi_neuron_layer(layer, t, ind_within_layer)
        print(f"Centering screen using: tzxy={tzxy} from layer {layer}")
    else:
        print(f"Layer {layer_name} not found; no zooming")
        return

    # Enhancement: better way to check for nesting
    if len(tzxy) == 1:
        tzxy = tzxy[0]
    if len(tzxy) == 1:
        tzxy = tzxy[0]
    # Data may be actually a null value (but t should be good)
    tzxy[0] = t
    zoom_using_viewer(tzxy, viewer, zoom)


def zoom_using_viewer(tzxy, viewer, zoom):
    try:
        is_positive = tzxy[2] > 0 and tzxy[3] > 0
        is_finite = not all(np.isnan(tzxy))
        # Center to the neuron in xy
        if zoom is not None:
            viewer.camera.zoom = zoom
        if is_positive and is_finite:
            viewer.camera.center = tzxy[1:]
        # Center around the neuron in z
        if is_positive and is_finite:
            viewer.dims.current_step = (tzxy[0], tzxy[1], 0, 0)
    except IndexError:
        logging.warning("Index error in zooming; skipping")
        return


def get_zxy_from_single_neuron_layer(layer, t, ind_within_layer=None):
    return layer.data[t]


def get_zxy_from_multi_neuron_layer(layer, t, ind_within_layer=None):
    # e.g. text labels, with all neurons in a time point in a row (thus t is no longer a direct index)
    # Or, if nans have been dropped from an otherwise full-size layer
    # Note: if ind_within_layer is None, it has no effect
    dat = layer.data
    if dat.shape[1] == 5:
        # Tracks layer; neuron index is now first column
        dat = dat[:, 1:]
    elif dat.shape[1] == 4:
        # Points layer
        pass
    else:
        raise ValueError(f"Unrecognized layer shape {dat.shape}")
    ind = dat[:, 0] == t

    if len(np.where(ind)[0]) == 0:
        fake_dat = np.zeros_like(layer.data[0, :])
        fake_dat[0] = t
        # logging.warning(f"Time {t} not found in layer: {layer.data[:, 0]}")
        return fake_dat
    else:
        if ind_within_layer is not None:
            return layer.data[ind, :][ind_within_layer, :]
        else:
            return layer.data[ind, :]


def change_viewer_time_point(viewer: napari.Viewer,
                             dt: int = None, t_target: int = None, a_max: int = None) -> tuple:
    # Increment time
    if dt is not None:
        t = np.clip(viewer.dims.current_step[0] + dt, a_min=0, a_max=a_max)
    elif t_target is not None:
        t = np.clip(t_target, a_min=0, a_max=a_max)
    else:
        raise ValueError("Must pass either target time or dt")
    tzxy = (t,) + viewer.dims.current_step[1:]
    viewer.dims.current_step = tzxy

    return tzxy


def add_fps_printer(viewer):
    # From: https://github.com/napari/napari/issues/836
    def fps_status(viewer, x):
        # viewer.help = f'{x:.1f} frames per second'
        print(f'{x:.1f} frames per second')

    viewer.window.qt_viewer.canvas.measure_fps(callback=lambda x: fps_status(viewer, x))


def build_gui_for_grid_plots(parent_folder, DEBUG=False):
    # Build a GUI for selecting which grid plots to view
    # Each grid plot is a png file
    # Subfolder structure is:
    # parent_folder
    #  - folder of projects
    #    - folder of single project
    #      - folder called "4-traces"
    #        - individual png files
    # There can be many png files, and we want a dropdown to select which one to view

    # Get all the project parent folders
    project_parent_folders = [os.path.join(parent_folder, x) for x in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, x))]
    # Get individual project_folders as a nested dictionary of {project_parent_folder: {project_name: project_folder}}
    project_folders = {}
    for project_parent_folder in project_parent_folders:
        # Set the outer key as just the folder name
        key = os.path.basename(project_parent_folder)
        project_folders[key] = {os.path.basename(x): os.path.join(project_parent_folder, x)
                                    for x in os.listdir(project_parent_folder)
                                    if os.path.isdir(os.path.join(project_parent_folder, x))}

    # Remove any folders that have not subfolders
    project_folders = {k: v for k, v in project_folders.items() if len(v) > 0}

    # Get all the png files as a nested dictionary of {project_parent_folder: {project_name: {png_basename: png_full_path}}}
    png_files = {}
    for project_parent_folder, project_name_dict in project_folders.items():
        png_files[project_parent_folder] = {}
        for project_name, project_full_path in project_name_dict.items():
            this_subfolder = os.path.join(project_full_path, "4-traces")
            # There could be other folders with different subfolders, so skip if this isn't a folder
            if not os.path.isdir(this_subfolder):
                continue
            png_basename = [x for x in os.listdir(this_subfolder) if x.endswith(".png")]
            png_full_path = [os.path.join(this_subfolder, x) for x in png_basename]
            # Do not save if there are no png files
            if len(png_basename) > 0:
                png_files[project_parent_folder][project_name] = dict(zip(png_basename, png_full_path))

    # Remove any folders that have no subfolders
    png_files = {k: v for k, v in png_files.items() if len(v) > 0}
    # Remove subfolders that have no png files
    for k, v in png_files.items():
        png_files[k] = {k2: v2 for k2, v2 in v.items() if len(v2) > 0}

    # Build the GUI using qtwidgets
    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout()
    window.setLayout(layout)
    # Add a dropdown to select the folder of projects
    project_parent_dropdown = QComboBox()
    project_parent_dropdown.addItems(png_files.keys())
    layout.addWidget(project_parent_dropdown)

    # Add a dropdown to select the project within the folder
    project_dropdown = QComboBox()
    layout.addWidget(project_dropdown)

    # Add a dropdown to select the png file
    png_dropdown = QComboBox()
    layout.addWidget(png_dropdown)
    # Add a button to view the selected png file
    view_button = QPushButton("View")
    layout.addWidget(view_button)
    # Add a popup if the image isn't found
    popup = QMessageBox()
    popup.setWindowTitle("Error")
    popup.setText("Image not found")
    popup.setIcon(QMessageBox.Critical)
    popup.setStandardButtons(QMessageBox.Ok)

    # Callback to update the png dropdown when the project dropdown is changed
    def update_png_dropdown():
        project_name = project_dropdown.currentText()
        project_parent_name = project_parent_dropdown.currentText()
        # Do not try if no project is selected
        if project_name == "" or project_parent_name == "":
            return
        # print("Updating png dropdown for: ", project_parent_name, project_name)
        keys = list(png_files[project_parent_name][project_name].keys())
        png_dropdown.clear()
        png_dropdown.addItems(keys)
        # Set the default value as an item that contains the word 'beh'
        for i, key in enumerate(keys):
            if "beh" in key.lower():
                png_dropdown.setCurrentIndex(i)
                break

    # Callback to view the selected png file
    def view_png():
        project_parent_name = project_parent_dropdown.currentText()
        project_name = project_dropdown.currentText()
        png_name = png_dropdown.currentText()
        # Do not try if no file is selected
        if png_name == "" or project_name == "" or project_parent_name == "":
            return
        png_path = png_files[project_parent_name][project_name][png_name]
        # Display the image if it exists using the system default image viewer
        if os.path.exists(png_path):
            print("Opening: ", png_path)
            path = os.path.realpath(png_path)
            if sys.platform == "win32":
                os.startfile(path)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, path])
        else:
            popup.exec()

    # Callback to update the project dropdown when the project parent dropdown is changed
    def update_project_dropdown():
        project_parent_folder = project_parent_dropdown.currentText()
        keys = list(png_files[project_parent_folder].keys())
        project_dropdown.clear()  # Note that this triggers the update_png_dropdown callback
        project_dropdown.addItems(keys)
        project_dropdown.setCurrentIndex(0)
        # print("Updating project dropdown for: ", project_parent_folder)
        # print(png_files[project_parent_folder])

    # Connect the callbacks, ensuring that the parent folder is updated first
    project_parent_dropdown.currentTextChanged.connect(update_project_dropdown)
    project_dropdown.currentTextChanged.connect(update_png_dropdown)
    view_button.clicked.connect(view_png)

    # Set the parent dropdown folder, in order to trigger the callbacks
    project_parent_dropdown.setCurrentIndex(1)
    project_parent_dropdown.setCurrentIndex(0)

    window.show()
    app.exec()


def on_close(self, event, widget, callbacks):
    # Copied from deeplabcut-napari
    # https://github.com/DeepLabCut/napari-deeplabcut/blob/c05d4a8eb58716da96b97d362519d4ee14e21cac/src/napari_deeplabcut/_widgets.py#L121
    choice = QMessageBox.warning(
        widget,
        "Warning",
        "Automatically saving data... Are you certain you want to quit? \n"
        "Regardless, please do NOT kill the terminal until you see the message: 'Saving successful!'",
        QMessageBox.Yes | QMessageBox.No,
    )
    # To be used for auto-saving
    for callback in callbacks:
        callback()

    if choice == QMessageBox.Yes:
        event.accept()
        # Call the main quit function
        widget.main_window.quit()
    else:
        event.ignore()


class NonEditableDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return None  # Disable editing


class NeuronNameEditor(QWidget):
    """
    Opens a table GUI for editing neuron names

    Has additional information on the right side, including:
    - A list of all neurons with the same Manual Annotation (duplicates)
    - A list of which neurons are "targeted" for manual annotation, which updates as neurons are named
    - A list of neurons that have been manually annotated, but are not on the "targeted" list

    """

    annotation_updated = pyqtSignal(str, str, str)
    multiple_annotations_updated = pyqtSignal(list, list, list)

    def __init__(self, neurons_to_id=None, DEBUG=False):
        super().__init__()

        # Save options
        self.neurons_to_id = neurons_to_id

        # Set up the GUI
        layout = QGridLayout()

        self.tableView = QTableView()
        self.model = QStandardItemModel(self)
        self.tableView.setModel(self.model)

        # Make sure that selecting programmatically overwrites the previous selection
        self.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectItems)

        # Set the initial size of the widget window, rows, and columns
        screen = QApplication.desktop().screenGeometry()
        fraction_of_screen_width = 0.4
        fraction_of_screen_height = 1.0
        initial_width = int(screen.width() * fraction_of_screen_width)
        initial_height = int(screen.height() * fraction_of_screen_height)
        self.resize(initial_width, initial_height)

        # The titles (top) and buttons (bottom) should be small
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 8)
        layout.setRowStretch(2, 1)

        # The first column should be the largest
        layout.setColumnStretch(0, 7)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

        # Set up widgets
        self.duplicatesList = QListWidget()
        self.notIdedList = QListWidget()
        self.customIdedList = QListWidget()

        # Set up buttons
        self.swapLRButton = QPushButton("Swap L/R")
        self.swapLRButton.clicked.connect(self.swap_left_right_annotations)

        # Set up each column in proper location
        layout.addWidget(QLabel("Editable table of neuron names"), 0, 0)
        layout.addWidget(self.tableView, 1, 0, 2, 1)
        layout.addWidget(QLabel("Duplicated IDs (please fix!)"), 0, 1)
        layout.addWidget(self.duplicatesList, 1, 1, 2, 1)
        layout.addWidget(QLabel("Neurons to ID"), 0, 2)
        layout.addWidget(self.notIdedList, 1, 2, 2, 1)
        layout.addWidget(QLabel("Custom (non-list) IDs"), 0, 3)
        layout.addWidget(self.customIdedList, 1, 3, 2, 1)

        layout.addWidget(self.swapLRButton, 3, 0)

        self.setLayout(layout)

        # Set up dummy data, if using
        if DEBUG:
            self.set_up_dummy_data()
        else:
            self.df = None
            self.df_datatypes = None
            self.filename = None

        # Set up signal

        # When data is changed, update the duplicates list and the stored dataframe
        self.model.dataChanged.connect(self.update_dataframe_range_from_table)
        self.model.dataChanged.connect(self.update_all_widgets)

        # Set custom delegate to prevent editing of first column
        non_editable_delegate = NonEditableDelegate()
        self.tableView.setItemDelegateForColumn(0, non_editable_delegate)

        # Remove the close button, if part of a larger gui (default)
        if not DEBUG:
            self.setWindowFlag(Qt.WindowCloseButtonHint, False)

    def set_up_dummy_data(self):
        # Dummy data
        column_names = [self.original_id_column_name, self.manual_id_column_name, "Metadata"]
        data = [
            ["neuron_001", "VB02", "Metadata 1"],
            ["neuron_002", "VB02", "Metadata 2"],
            ["neuron_003", "AVAL", "Metadata 3"],
            ["neuron_004", "AVAL", "Metadata 4"],
            ["neuron_005", "AVAR", "Metadata 5"]
        ]
        self.df = pd.DataFrame(data, columns=column_names)
        self.df_datatypes = None
        # Actually set up
        self.import_dataframe(self.df, None)

    def get_set_of_neurons_to_id(self):
        """Initial creation should load from a yaml file"""
        neurons_to_id = self.neurons_to_id
        if neurons_to_id is None:
            return None
        else:
            return set(neurons_to_id)

    def _set_column_edit_flags(self):
        # Set all columns to be editable, except the first one (the original names)
        for row in range(self.model.rowCount()):
            for col in range(self.model.columnCount()):
                if col > 0:
                    self.model.item(row, col).setFlags(
                        self.model.item(row, col).flags() | Qt.ItemIsEditable
                    )
                else:
                    self.model.item(row, col).setFlags(
                        self.model.item(row, col).flags() | ~Qt.ItemIsEditable
                    )

    def jump_focus_to_neuron(self, neuron_name, column_name=None, column_offset=0):
        """
        Select based on the 'Neuron ID' column in the table

        Parameters
        ----------
        neuron_name
        column_name: str or None (default: None) - if None, use the manual_id_column_idx column
        column_offset: int (default: 0) - offset from the column index

        Returns
        -------

        """
        if column_name is None:
            column_idx = self.manual_id_column_idx
        else:
            column_idx = list(self.df.columns).index(column_name)
        if column_offset != 0:
            column_idx += column_offset
        row_index = self.df.index[self.df[self.original_id_column_name] == neuron_name].tolist()[0]
        # print(f"Selecting row: {row_index} with original name {neuron_name}")
        selection_model = self.tableView.selectionModel()
        cell_index = self.model.index(row_index, column_idx)
        # Use setCurrentIndex instead of select, to overwrite the previous selection
        selection_model.setCurrentIndex(cell_index, QItemSelectionModel.Select)
        # selection_model.select(cell_index, QItemSelectionModel.Select)

        # Jump to that cell in this window (not focus of main gui)
        self.setFocus()
        self.activateWindow()
        self.raise_()

        # Center the scroll bar on the first column of the selected cell
        first_col_cell_index = self.model.index(row_index, 0)
        self.tableView.scrollTo(first_col_cell_index, QAbstractItemView.PositionAtCenter)

        # Start editing the selected cell
        self.tableView.edit(cell_index)

    def update_table_from_dataframe(self):
        """
        Note that this doesn't emit the signal for updating manual annotations

        Returns
        -------

        """
        self.model.clear()
        self.model.setHorizontalHeaderLabels(list(self.df.columns))
        for row in self.df.values:
            row_items = [QStandardItem(str(item)) for item in row]
            self.model.appendRow(row_items)

    def update_dataframe_from_table(self):
        """
        Update the dataframe from the table

        Should preserve the original datatypes of the dataframe

        Returns
        -------

        """
        for row in range(self.model.rowCount()):
            for col in range(self.model.columnCount()):
                self.update_dataframe_cell_from_table(col, row)

    def update_dataframe_cell_from_table(self, col, row, emit_signal=True):
        dtype = self.df_datatypes[col]
        string_data = self.model.item(row, col).text()
        # Also emit a signal if this is the "ID1" column
        # print(f"Updating dataframe cell: {row}, {col} to {string_data}")
        if emit_signal and col == self.manual_id_column_idx:
            original_name = str(self.df.at[row, self.original_id_column_name])
            old_name = str(self.df.at[row, self.manual_id_column_name])
            new_name = string_data
            if old_name != new_name:
                logging.info(f"Changing neuron name: {old_name} -> {new_name}")
                self.annotation_updated.emit(original_name, old_name, new_name)

        if dtype == np.int64:
            self.df.iat[row, col] = int(string_data)
        elif dtype == np.float64:
            self.df.iat[row, col] = float(string_data)
        else:
            # Probably object
            self.df.iat[row, col] = string_data

    @property
    def manual_id_column_idx(self):
        return list(self.df.columns).index(self.manual_id_column_name)

    @property
    def manual_id_column_name(self):
        return "ID1"

    @property
    def original_id_column_name(self):
        return "Neuron ID"

    def update_dataframe_range_from_table(self, top_left, bottom_right):
        for row in range(top_left.row(), bottom_right.row() + 1):
            for column in range(top_left.column(), bottom_right.column() + 1):
                item = self.model.item(row, column)
                if item:
                    self.update_dataframe_cell_from_table(column, row)

    def import_dataframe(self, df, filename):
        # Set up data table
        self.df = df
        self.update_table_from_dataframe()
        self.update_all_widgets()
        self.df_datatypes = df.dtypes
        self._set_column_edit_flags()

        # Set up filename, which will be used to save the dataframe
        self.filename = filename
        self.save_df_to_disk(also_save_h5=True)  # Save initial version as .h5

    def update_all_widgets(self):
        self.update_duplicates_list()
        self.update_not_ided_list()
        self.update_custom_ided_list()

    def save_df_to_disk(self, also_save_h5=True):
        """
        Saves the dataframe as a .h5 or .xlsx file, overwriting any existing file

        In principle the .h5 version would be used as a backup, but the .xlsx version is actually read

        """
        if self.filename is None:
            logging.warning("No filename set; not saving")
            return False
        df = self.df.copy()
        # Do not save 'ID1' column values that are the same as the original name
        id_col = self.manual_id_column_name
        df[id_col] = df[id_col].where(df[id_col] != df[self.original_id_column_name], '')

        # Save
        if also_save_h5:
            fname = str(Path(self.filename).with_suffix('.h5'))
            df.to_hdf(fname, key='df_with_missing', mode='w')

        fname = self.filename
        try:
            if not fname.endswith(".xlsx"):
                raise FileNotFoundError(f"Filename must end with .xlsx; found {fname}")
            df.to_excel(fname, index=False)
        except (PermissionError, FileNotFoundError):
            logging.warning(f"Error when saving {fname}; "
                            f"Do you have the file open in another program?")
            return False

        return True

    def update_duplicates_list(self):
        # Clear the duplicate list
        self.duplicatesList.clear()

        # Get all duplicates of any Manual Annotation
        df = self.df

        custom_name_counts = df[self.manual_id_column_name].value_counts()
        duplicate_series = custom_name_counts[custom_name_counts > 1]
        duplicate_names = list(duplicate_series.index)
        # Remove the name 'nan'
        if 'nan' in duplicate_names:
            duplicate_names.remove('nan')
        elif np.nan in duplicate_names:
            duplicate_names.remove(np.nan)
        elif '' in duplicate_names:
            duplicate_names.remove('')

        for name in duplicate_names:
            # Get the original name as well
            original_name_list = df[df[self.manual_id_column_name] == name][self.original_id_column_name]
            str_to_add = f"{name} ({', '.join(original_name_list)})"
            self.duplicatesList.addItem(str_to_add)

    def update_not_ided_list(self):
        self.notIdedList.clear()

        # Remove any id's that have been finished
        df = self.df
        unique_ids = set(df[self.manual_id_column_name].unique())
        ids_to_do = self.get_set_of_neurons_to_id()
        if ids_to_do is None:
            return

        not_yet_done_ids = list(ids_to_do.difference(unique_ids))

        # Make sure all entries are strings
        not_yet_done_ids = [str(x) for x in not_yet_done_ids]

        # Update widget
        self.notIdedList.addItems(not_yet_done_ids)

    def update_custom_ided_list(self):
        # Clear the duplicate list
        self.customIdedList.clear()

        # Remove any id's that aren't on the hardcoded list
        df = self.df
        unique_ids = set(df[self.manual_id_column_name].unique())
        ids_to_do = self.get_set_of_neurons_to_id()
        if ids_to_do is None:
            return

        custom_ids = list(unique_ids.difference(ids_to_do))
        # Make sure all entries are strings
        custom_ids = [str(x) for x in custom_ids]

        # Update widget
        self.customIdedList.addItems(custom_ids)

    def original2custom(self, remove_empty=False):
        """
        Mapping between original names and custom ids, as currently annotated
        """
        df = self.df
        id_col = self.manual_id_column_name
        orig_col = self.original_id_column_name

        # Create dictionary from these two columns
        mapping = dict(zip(df[orig_col], df[id_col]))
        mapping = {str(k): str(v) for k, v in mapping.items() if not (remove_empty and v == '')}
        return mapping

    def swap_left_right_annotations(self):
        """
        Swap the ID'ed neurons that are left/right

        Specifically, swap all suffixes that are 'L' and 'R' AND are in a neuron that has more than 3 characters
        """
        df = self.df
        id_col = self.manual_id_column_name
        # For updating the GUI
        all_original_names = []
        all_old_names = []
        all_new_names = []

        for i, row in df.iterrows():
            starting_id = row[id_col]
            did_swap = False
            if isinstance(starting_id, str) and len(starting_id) > 3:
                if starting_id.endswith('L'):
                    df.at[i, id_col] = starting_id[:-1] + 'R'
                    did_swap = True
                elif starting_id.endswith('R'):
                    df.at[i, id_col] = starting_id[:-1] + 'L'
                    did_swap = True
                if did_swap:
                    logging.info(f"Swapping {starting_id} to {df.at[i, id_col]}")
                    all_original_names.append(str(row[self.original_id_column_name]))
                    all_old_names.append(str(starting_id))
                    all_new_names.append(str(df.at[i, id_col]))

        # Update these GUI elements
        self.update_table_from_dataframe()
        self.update_all_widgets()

        # Update parent GUI elements, if any
        self.multiple_annotations_updated.emit(all_original_names, all_old_names, all_new_names)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = NeuronNameEditor(DEBUG=True)
    example.show()
    sys.exit(app.exec_())
