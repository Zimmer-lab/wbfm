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
    def __init__(self, project_data, verbose=3):

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

    def write_file(self, event):
        log_dir = self.project_data.project_config.get_visualization_config(make_subfolder=True).absolute_subfolder
        fname = os.path.join(log_dir, 'selected_neurons.csv')
        # fname = get_sequential_filename(fname)
        print(f"Saving: {fname}")

        df = pd.DataFrame(self.selected_neurons)
        df.T.to_csv(path_or_buf=fname, index=True)
        fname = Path(fname).with_suffix('.xlsx')
        df.T.to_excel(fname, index=True)
        # df = pd.DataFrame(self.selected_neurons, index=[0])
        # df.to_csv(path_or_buf=fname, header=True, index=False)

        print(df.T)

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
