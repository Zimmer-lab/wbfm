import numpy as np
import pandas as pd
import plotly.express as px

from scipy import stats
from statsmodels.stats.multitest import multipletests

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.general.utils_hardcoded import neurons_with_less_confident_ids
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation

# NOTE: calc_statistics_for_pc1_comparison_plots and plot_pc1_comparison
# were moved to the 505_laser_wavelength repo
# (paper_figure_notebooks/src/utils_paper_revisions.py); they are only
# used there.
