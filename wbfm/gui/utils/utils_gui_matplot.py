import os
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_project
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd


class PlotQWidget(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = PlotCanvas(self, width=10, height=3)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

    def draw(self):
        self.canvas.draw()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class ClickableGridPlot:
    """
    Interactive matplotlib GUI for selecting and categorizing neurons from grid plots.

    This class creates a clickable grid visualization where each subplot represents a neuron's
    activity trace. Users can interactively select neurons and assign them to different 
    categorical lists (1-3) using mouse clicks and keyboard inputs. Selections are color-coded
    and persist across sessions.

    Features:
    ---------
    - **Left-click**: Select a neuron and assign it to the current list (1, 2, or 3)
    - **Right-click**: Deselect a neuron (reset to list 0)
    - **Keyboard (1, 2, 3)**: Switch between list categories before clicking
      - List 1: Green shading
      - List 2: Blue shading  
      - List 3: Red shading
    - **Auto-save**: Selections are saved to CSV/XLSX when the plot window is closed
    - **Auto-load**: Previous selections are restored when reopening the same project

    Parameters:
    -----------
    project_data : object
        Project data object containing neuron information and project directory.
        Must have attributes: project_dir, neuron_names, project_config
    verbose : int, optional (default=3)
        Verbosity level for debug output (higher = more detailed)
    editor : NeuronNameEditor, optional (default=None)
        Optional NeuronNameEditor instance for syncing neuron selections to Notes column.
        When None, grid plot operates independently without editor integration.

    Attributes:
    -----------
    fig : matplotlib.figure.Figure
        The grid plot figure containing all neuron subplots
    selected_neurons : dict
        Dictionary mapping neuron names to their selection metadata:
        {"neuron_name": {"List ID": int, "Proposed Name": str}}
    current_list_index : int
        Currently active list (1=green, 2=blue, 3=red, 0=none)
    current_selected_label : str
        Name of the most recently clicked neuron

    Workflow:
    ---------
    1. Press '1', '2', or '3' to select which list to assign neurons to
    2. Left-click on a neuron subplot to assign it to the current list
    3. Right-click to deselect/remove a neuron from any list
    4. Close the window to save selections to 'selected_neurons.csv' and '.xlsx'
    5. Reopen to automatically restore previous selections

    Output Files:
    -------------
    Saves to: {project_dir}/visualization/selected_neurons.csv (and .xlsx)
    Format: Index=neuron_name, Columns=['List ID', 'Proposed Name']

    Example:
    --------
    >>> plot = ClickableGridPlot(project_data)
    >>> # Interactive selection happens in GUI
    >>> # Files are auto-saved on window close
    """
    def __init__(self, project_data, verbose=3, editor=None):

        # Set up grid plot
        opt = dict(channel_mode='ratio',
                   calculation_mode='integration',
                   filter_mode='rolling_mean',
                   to_save=False)

        mplstyle.use('fast')
        with safe_cd(project_data.project_dir):
            fig = make_grid_plot_from_project(project_data, **opt)

        self.fig = fig
        self.project_data = project_data
        if editor is None:
            editor = self.project_data.build_neuron_editor_gui()
            if editor is not None:
                editor.setWindowTitle(f"Neuron Name Editor for project: {self.project_data.project_dir}")
                editor.show()
        
        if editor is None:
            project_data.logger.warning("No editor available; grid plot will operate without editor integration (metadata syncing to Notes column will be disabled)")
        else:
            project_data.logger.info("Editor detected; grid plot will sync neuron selections to editor Notes column with smart class replacement")
        self.editor = editor

        # Set up metadata objects
        names = project_data.neuron_names
        self.selected_neurons = {n: {"List ID": 0, "Proposed Name": n} for n in names}
        self.current_list_index = 1
        self.current_selected_label = None
        self.verbose = verbose

        # Set up text box for modifying names
        # plt.subplots_adjust(bottom=0.2)
        # axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        # self.text_box = TextBox(axbox, 'Modify neuron name', initial="initial_text")
        # self.text_box.on_submit(self.modify_neuron_name)

        # Finish
        self.connect()
        # Load file and add initial colors, if any
        self.load_previous_file()
        plt.show()

    def connect(self):
        cid = self.fig.canvas.mpl_connect('button_press_event', self.shade_selected_subplot_callback)
        cid = self.fig.canvas.mpl_connect('key_press_event', self.update_current_list_index)
        cid = self.fig.canvas.mpl_connect('close_event', self.write_file)

    def update_current_list_index(self, event):
        if event.key in ['1', '2', '3']:
            self.current_list_index = int(event.key)
        else:
            pass

        print(f"Current list index: {self.current_list_index}")

    def modify_neuron_name(self, text):
        self.selected_neurons[self.current_selected_label]["Proposed Name"] = text

    def update_selected_label(self, new_label):
        self.current_selected_label = new_label
        # self.text_box.set_val(new_label)

    def get_color_from_list_index(self):
        print(f"Getting color: {self.current_list_index}")
        if self.current_list_index == 1:
            return 'green'
        elif self.current_list_index == 2:
            return 'blue'
        else:
            return 'red'

    def shade_selected_subplot_callback(self, event):
        ax = event.inaxes
        if self.verbose >= 3:
            print(event)
            print(ax)
        if ax is None or len(ax.lines) == 0:
            return
        button_pressed = event.button

        self.shade_selected_subplot(ax, button_pressed)

    def shade_selected_subplot(self, ax, button_pressed):

        line = ax.lines[0]
        label = line.get_label()
        self.update_selected_label(label)

        # Button codes: https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.MouseButton
        if button_pressed == 1:
            # Left click = select neuron
            if self.selected_neurons[label]["List ID"] == self.current_list_index:
                print(f"{label} already selected")
            else:
                print(f"Selecting {label}")
                self._reset_shading(ax)

                y = line.get_ydata()
                color = self.get_color_from_list_index()

                shading = ax.axhspan(np.nanmin(y), np.nanmax(y), xmax=len(y), facecolor=color, alpha=0.25, zorder=-100)
                ax.draw_artist(shading)

                self.selected_neurons[label]["List ID"] = self.current_list_index
                
                # Update editor notes if editor is available
                code_string = self._get_code_string_from_list_index()
                self._update_editor_notes(label, code_string)

        elif button_pressed == 3:
            # Right click = deselect
            if self.selected_neurons[label]["List ID"] == 0:
                print(f"{label} not selected")
            else:
                print(f"Deselecting {label}")
                self._reset_shading(ax)
                plt.draw()
                self.selected_neurons[label]["List ID"] = 0
        else:
            print("Button press detected, but did nothing")
        # From: https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
        ax.figure.canvas.blit(ax.bbox)
        # if verbose >= 2:
        #     print("Currently selected neuron:")
        #     print(self.selected_neurons)

    def _reset_shading(self, ax):
        if len(ax.patches) > 0:
            [p.remove() for p in ax.patches]
            # ax.patches = []

    def _get_code_string_from_list_index(self):
        """
        Map the current list index to a code string for the Notes column.
        
        Returns:
            str: "Good" (List 1), "Custom" (List 2), or "Invalid" (List 3)
        """
        if self.current_list_index == 1:
            return "Good"
        elif self.current_list_index == 2:
            return "Custom"
        else:  # current_list_index == 3
            return "Invalid"

    def _find_neuron_row_in_editor(self, neuron_name):
        """
        Find the row index of a neuron in the editor's dataframe.
        
        Parameters:
            neuron_name (str): The name of the neuron to find
            
        Returns:
            int or None: Row index if found, None otherwise
        """
        if self.editor is None or self.editor.df is None:
            return None
            
        try:
            matching_rows = self.editor.df.index[
                self.editor.df[self.editor.original_id_column_name] == neuron_name
            ].tolist()
            if matching_rows:
                return matching_rows[0]
            else:
                if self.verbose >= 2:
                    print(f"Warning: Neuron '{neuron_name}' not found in editor dataframe")
                return None
        except Exception as e:
            if self.verbose >= 2:
                print(f"Error finding neuron '{neuron_name}' in editor: {e}")
            return None

    def _update_editor_notes(self, neuron_name, code_string):
        """
        Update the Notes column in the editor for a given neuron with smart class replacement.
        
        If the Notes field already contains a class string ("Good", "Custom", or "Invalid"),
        it will be replaced with the new code_string. Otherwise, the code_string is appended
        with a semicolon separator.
        
        Parameters:
            neuron_name (str): The name of the neuron
            code_string (str): The code string to add ("Good", "Custom", or "Invalid")
        """
        if self.editor is None or self.editor.df is None:
            if self.verbose >= 2:
                print(f"Editor not available; skipping notes update for {neuron_name}")
            return
            
        row_idx = self._find_neuron_row_in_editor(neuron_name)
        if row_idx is None:
            return
            
        try:
            # Get the current notes content
            current_notes = str(self.editor.df.at[row_idx, self.editor.notes_column_name])
            if current_notes == 'nan' or current_notes == '':
                current_notes = ''
            
            # Define the class strings to search for
            class_strings = ["Good", "Custom", "Invalid"]
            
            # Check if any class string exists in current_notes
            found_class = None
            for cs in class_strings:
                if cs in current_notes:
                    found_class = cs
                    break
            
            # Smart replacement: replace existing class or append if none exists
            if found_class is not None:
                # Replace the old class string with the new one
                new_notes = current_notes.replace(found_class, code_string)
            else:
                # Append the new code string
                if current_notes.strip():
                    new_notes = current_notes + ";" + code_string
                else:
                    new_notes = code_string
            
            # Update the dataframe
            self.editor.df.at[row_idx, self.editor.notes_column_name] = new_notes
            
            # Update the table view to reflect the change
            notes_col_idx = self.editor.notes_column_idx
            model_item = self.editor.model.item(row_idx, notes_col_idx)
            if model_item is not None:
                model_item.setText(new_notes)
            else:
                # If item doesn't exist in model, rebuild the table
                self.editor.update_table_from_dataframe()
            
            if self.verbose >= 2:
                print(f"Updated notes for {neuron_name}: '{current_notes}' -> '{new_notes}'")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"Error updating notes for neuron '{neuron_name}': {e}")

    def write_file(self, event):
        log_dir = self.project_data.project_config.get_visualization_config(make_subfolder=True).absolute_subfolder
        fname = os.path.join(log_dir, 'selected_neurons.csv')
        print(f"Saving: {fname}")

        df = pd.DataFrame(self.selected_neurons)
        df.T.to_csv(path_or_buf=fname, index=True)
        fname = Path(fname).with_suffix('.xlsx')
        df.T.to_excel(fname, index=True)

        if self.editor is not None:
            try:
                print("Saving editor annotations...")
                self.editor.save_df_to_disk(also_save_h5=True)
                print("Editor annotations saved successfully")
            except PermissionError as e:
                print(f"Warning: Failed to save editor annotations (file may be open in another program): {e}")
            except Exception as e:
                print(f"Error saving editor annotations: {e}")
            finally:
                # Close the editor window after saving
                self.editor.close()

    def load_previous_file(self):
        visualization_directory = self.project_data.project_config.get_visualization_config().absolute_subfolder
        fname = os.path.join(visualization_directory, 'selected_neurons.csv')
        if not os.path.exists(fname):
            print(f"Did not find previous state at: {fname}")
            return
        else:
            # plt.show(block=False)
            self.fig.canvas.draw()
            print(f"Reading previous state from: {fname}")
            df = pd.read_csv(fname, index_col=0)

            axes = self.fig.axes
            button_pressed = 1

            for ax, (name, list_index) in zip(axes, df.iterrows()):
                if list_index[0] == 0:
                    continue

                # Add the shading to this axis
                print(f"Shading {name} with index {list_index[0]} and name {list_index[1]}")
                self.current_list_index = list_index[0]
                self.shade_selected_subplot(ax, button_pressed)

                # Also add the info to the dict
                self.selected_neurons[name]["List ID"] = list_index[0]
                self.selected_neurons[name]["Proposed Name"] = list_index[1]

        plt.draw()
        self.current_list_index = 1

    # def plot(self):
    #     data = [random.random() for i in range(250)]
    #     ax = self.figure.add_subplot(111)
    #     ax.plot(data, 'r-', linewidth = 0.5)
    #     ax.set_title('PyQt Matplotlib Example')
    #     self.draw()
