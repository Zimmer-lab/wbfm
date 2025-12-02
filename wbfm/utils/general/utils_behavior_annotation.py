import functools
import logging
import os
import re
from enum import Flag, auto
from pathlib import Path
from typing import List, Union, Optional, Dict

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences, peak_widths
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, make_binary_vector_from_starts_and_ends, \
    remove_short_state_changes, get_contiguous_blocks_from_two_columns, resample_categorical, \
    combine_columns_with_suffix, extend_short_states
from wbfm.utils.external.custom_errors import InvalidBehaviorAnnotationsError, NeedsAnnotatedNeuronError
import plotly.graph_objects as go

from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe, filter_gaussian_moving_average
from wbfm.utils.general.hardcoded_paths import get_summary_visualization_dir
from wbfm.utils.external.utils_pandas import pad_events_in_binary_vector


@functools.total_ordering
class BehaviorCodes(Flag):
    """
    Top-level behaviors that are discretely annotated.
    Designed to work with Ulises' automatic annotations via a hardcoded mapping. See also: from_ulises_int

    NOTE: behaviors should always be loaded using this mapping, not directly from integers!

    Example (See also FullDatasetTriggeredAverages):
        trigger_opt = dict(state=BehaviorCodes.REV)
        triggered_class = FullDatasetTriggeredAverages.load_from_project(project_data_gcamp, trigger_opt=trigger_opt)

    The following is the auto-generated description of all behaviors
    ----------------------------------------------------------------------------------------------

    """
    # Basic automatically annotated behaviors
    FWD = auto()
    REV = auto()

    VENTRAL_TURN = auto()
    DORSAL_TURN = auto()
    SUPERCOIL = auto()  # Manually annotated
    QUIESCENCE = auto()  # Manually annotated
    SELF_COLLISION = auto()  # Annotated using Charlie's pipeline
    HEAD_CAST = auto()  # Manually annotated
    SLOWING = auto()  # Annotated using Charlie's pipeline
    PAUSE = auto()  # Annotated using Charlie's pipeline
    STIMULUS = auto()  # Annotated using a manual excel file

    NOT_ANNOTATED = auto()
    UNKNOWN = auto()
    TRACKING_FAILURE = auto()

    CUSTOM = auto()  # Used when annotations are manually passed as a numpy array; can only deal with one custom state

    @classmethod
    def _ulises_int_2_flag(cls, flip: bool = False):
        original_mapping = {
            -1: cls.FWD,
            1: cls.REV,
            2: cls.FWD | cls.VENTRAL_TURN,
            3: cls.FWD | cls.DORSAL_TURN,
            4: cls.REV | cls.VENTRAL_TURN,
            5: cls.REV | cls.DORSAL_TURN,
            6: cls.SUPERCOIL,
            7: cls.QUIESCENCE,
            0: cls.NOT_ANNOTATED,
            -99: cls.UNKNOWN,  # Should not be in any files that Ulises produces
        }
        if flip:
            original_mapping = {v: k for k, v in original_mapping.items()}
        return original_mapping

    @classmethod
    def ulises_int_to_enum(cls, value: int) -> 'BehaviorCodes':
        """
        Convert from Ulises' integer value to the corresponding BehaviorCodes value

        HARDCODED!

        Parameters
        ----------
        value

        Returns
        -------

        """
        original_mapping = cls._ulises_int_2_flag()
        return original_mapping.get(value, cls.UNKNOWN)

    @classmethod
    def enum_to_ulises_int(cls, value: 'BehaviorCodes') -> int:
        """
        Convert from BehaviorCodes to the corresponding Ulises' integer value

        HARDCODED!

        Parameters
        ----------
        value

        Returns
        -------

        """
        original_mapping = cls._ulises_int_2_flag(flip=True)
        return original_mapping[value]

    def __add__(self, other):
        # Allows adding vectors as well
        # Note that pandas will add np.nan values if the vectors are different lengths
        if other in (BehaviorCodes.NOT_ANNOTATED, BehaviorCodes.UNKNOWN, np.nan):
            return self
        elif self in (BehaviorCodes.NOT_ANNOTATED, BehaviorCodes.UNKNOWN, np.nan):
            return other
        else:
            return self | other

    def __radd__(self, other):
        # Required for sum to work
        # https://stackoverflow.com/questions/5082190/typeerror-after-overriding-the-add-method
        return self.__add__(other)

    def __eq__(self, other):
        # Allows equality comparisons, but only between this enum
        if isinstance(other, BehaviorCodes):
            return self.value == other.value
        else:
            return False

    def __lt__(self, other):
        # Allows sorting (rest are generated via functools.total_ordering)
        return self.value < other.value

    def __hash__(self):
        # Allows this enum to be used as a key in a dictionary
        return hash(self.value)

    @classmethod
    def _load_from_list(cls, vec: List[int]) -> pd.Series:
        """
        Load from a list of int; DO NOT USE DIRECTLY!

        Returns
        -------

        """
        return pd.Series([cls(i) for i in vec])

    @classmethod
    def load_using_dict_mapping(cls, vec: Union[pd.Series, pd.DataFrame, List[int]],
                                mapping: dict=None) -> pd.Series:
        """
        Create a pd.Series from a list of integers, using a hardcoded mapping between Ulises' integers and BehaviorCodes

        Load using the hardcoded mapping between Ulises' integers and BehaviorCodes

        See also: from_ulises_int

        Returns
        -------

        """
        if isinstance(vec, pd.Series):
            vec = vec.values
        elif isinstance(vec, pd.DataFrame):
            # Check that there is only one column, then convert to series
            assert len(vec.columns) == 1, "Can only convert one column at a time"
            vec = vec.iloc[:, 0].values
        if mapping is None:
            mapping = cls._ulises_int_2_flag()
        # Map all values using the dict, unless they are already BehaviorCodes
        beh_vec = pd.Series([mapping[i] if not isinstance(i, BehaviorCodes) else i for i in vec])
        cls.assert_all_are_valid(beh_vec)
        return beh_vec

    @classmethod
    def vector_equality(cls, enum_list: Union[list, pd.Series], query_enum, exact=False) -> pd.Series:
        """
        Compares a query enum to a list of enums, returning a binary vector

        By default allows complex comparisons, e.g. FWD_VENTRAL_TURN == FWD because FWD_VENTRAL_TURN is a subset of FWD

        Parameters
        ----------
        enum_list
        query_enum
        exact

        Returns
        -------

        """
        if isinstance(enum_list, pd.DataFrame):
            # Check that there is only one column, then convert to series
            assert len(enum_list.columns) == 1, "Can only compare to one column at a time"
            enum_list = enum_list.iloc[:, 0]

        if exact:
            binary_vector = [query_enum == e for e in enum_list]
        else:
            binary_vector = [query_enum in e for e in enum_list]
        if isinstance(enum_list, pd.Series):
            return pd.Series(binary_vector, index=enum_list.index)
        else:
            return pd.Series(binary_vector)

    @classmethod
    def vector_diff(cls, enum_list: Union[list, pd.Series], exact=False) -> pd.Series:
        """
        Calculates the vector np.diff, which can't be done directly because these aren't integers

        Returns
        -------

        """
        if exact:
            return pd.Series([e2 != e1 for e1, e2 in zip(enum_list.iloc[:-1], enum_list.iloc[1:])])
        else:
            return pd.Series([e2 not in e1 for e1, e2 in zip(enum_list.iloc[:-1], enum_list.iloc[1:])])

    @classmethod
    def possible_colors(cls, include_complex_states=True):
        # Because I'm refactoring the colormaps to functions not dictionaries, I want a list to loop over
        states = [cls.FWD, cls.REV, cls.SELF_COLLISION]
        if include_complex_states:
            states.extend([cls.FWD | cls.VENTRAL_TURN, cls.FWD | cls.DORSAL_TURN,
                           cls.REV | cls.VENTRAL_TURN, cls.REV | cls.DORSAL_TURN,
                           cls.QUIESCENCE, cls.PAUSE])
        return states

    @classmethod
    def possible_behavior_aliases(cls) -> List[str]:
        """A list of aliases for all the states. Each alias is just the lowercase version of the state name"""
        return [state.name.lower() for state in cls]

    @classmethod
    def shading_cmap_func(cls, query_state: 'BehaviorCodes',
                          additional_shaded_states: Optional[List['BehaviorCodes']] = None,
                          default_reversal_shading: bool = True,
                          force_string_output=False) -> Optional[str]:
        """
        Colormap for shading on top of traces, but using 'in' logic instead of '==' logic

        There are two common use cases: shading behind a trace, and shading a standalone ethogram.
            By default, shading behind a trace will use a gray color for reversals, and no shading for anything else
        See ethogram_cmap for shading a standalone ethogram

        The first color uses a gray shading
        Additionally passed states will use a matplotlib colormap

        force_string_output is required if this is being used as a plotly colormap
        """
        base_cmap = matplotlib.cm.get_cmap('Pastel1')
        cmap_dict = {}
        if default_reversal_shading:
            cmap_dict[cls.REV] = 'lightgray'

        if additional_shaded_states is not None:
            # Add states and colors using the matplotlib colormap
            num_added = 0
            for state in additional_shaded_states:
                if state not in cmap_dict:
                    cmap_dict[state] = base_cmap(num_added)
                    num_added += 1

        for state, color in cmap_dict.items():
            if state in query_state:
                output_color = color
                break
        else:
            output_color = None

        if force_string_output:
            if isinstance(output_color, tuple):
                output_color = f"rgba({output_color[0]}, {output_color[1]}, {output_color[2]}, {output_color[3]})"

        return output_color

        # Otherwise use a hardcoded colormap
        # if cls.FWD in query_state:
        #     return None
        # elif cls.REV in query_state:
        #     return 'lightgray'
        # elif cls.SELF_COLLISION in query_state and include_collision:
        #     return 'red'
        # else:
        #     return None

    @classmethod
    def base_colormap(cls) -> List[str]:
        # See: https://plotly.com/python/discrete-color/
        # cmap = px.colors.qualitative.Set1.copy()
        # # Manually reorder some things to match better with prior work
        # cmap[0], cmap[1] = cmap[1], cmap[0]  # Blue REV and red FWD
        # cmap[3], cmap[6] = cmap[6], cmap[3]  # Switch brown and purple
        # cmap.pop(3)  # Remove purple because it's hard to distinguish from blue

        # return px.colors.qualitative.Set1_r.copy()
        # cmap = px.colors.qualitative.Set2.copy()
        # cmap = px.colors.qualitative.Vivid_r.copy()
        # cmap = px.colors.qualitative.Dark2.copy()
        cmap = px.colors.qualitative.Dark2.copy()
        cmap[3], cmap[5] = cmap[5], cmap[3]  # Switch pink and gold
        # Move gray to the front
        # cmap.insert(0, cmap.pop(7))
        return cmap

    @classmethod
    def ethogram_cmap(cls, include_turns=True, include_reversal_turns=False, include_quiescence=False,
                      include_collision=False, additional_shaded_states=None, include_pause=True,
                      use_plotly_style_strings=True, include_custom=False, include_stimulus=False) -> Dict['BehaviorCodes', str]:
        """
        Colormap for shading as a stand-alone ethogram

        Returns a dictionary mapping each state to a color, using a plotly colormap
        Example:
            {BehaviorCodes.FWD: 'rgb(228,26,28)'}

        Alternative output is hex format (use_plotly_style_strings=False):
            {BehaviorCodes.FWD: '#E41A1C'}

        """
        base_cmap = cls.base_colormap()
        cmap = {cls.UNKNOWN: None, cls.TRACKING_FAILURE: None,
                cls.FWD: base_cmap[0],
                cls.REV: base_cmap[1],
                # Same as FWD by default
                cls.FWD | cls.VENTRAL_TURN: base_cmap[0],
                cls.FWD | cls.DORSAL_TURN: base_cmap[0],
                cls.FWD | cls.SELF_COLLISION: base_cmap[0],
                cls.FWD | cls.SELF_COLLISION | cls.DORSAL_TURN: base_cmap[0],
                cls.FWD | cls.SELF_COLLISION | cls.VENTRAL_TURN: base_cmap[0],
                # Same as REV by default
                cls.REV | cls.VENTRAL_TURN: base_cmap[1],
                cls.REV | cls.DORSAL_TURN: base_cmap[1],
                cls.REV | cls.PAUSE: base_cmap[1],
                cls.REV | cls.SELF_COLLISION: base_cmap[1],
                cls.REV | cls.SELF_COLLISION | cls.DORSAL_TURN: base_cmap[1],
                cls.REV | cls.SELF_COLLISION | cls.VENTRAL_TURN: base_cmap[1],
                # Unclear, but same as FWD by default
                cls.QUIESCENCE: base_cmap[0],
                cls.QUIESCENCE | cls.VENTRAL_TURN: base_cmap[0],
                cls.QUIESCENCE | cls.DORSAL_TURN: base_cmap[0],
                cls.PAUSE | cls.QUIESCENCE: base_cmap[0],
                cls.SLOWING: base_cmap[4],
                cls.SLOWING | cls.FWD: base_cmap[4],
                }
        if include_turns:
            # Turns during FWD are differentiated, but not during REV
            # Ventral is purple, dorsal is gold
            cmap[cls.VENTRAL_TURN] = base_cmap[2]
            cmap[cls.DORSAL_TURN] = base_cmap[3]
            cmap[cls.FWD | cls.VENTRAL_TURN] = base_cmap[2]
            cmap[cls.FWD | cls.DORSAL_TURN] = base_cmap[3]
            cmap[cls.REV | cls.VENTRAL_TURN] = base_cmap[1]
            cmap[cls.REV | cls.DORSAL_TURN] = base_cmap[1]
        if include_reversal_turns:
            # Turns during REV are differentiated, but not during FWD
            cmap[cls.REV | cls.VENTRAL_TURN] = base_cmap[4]
            cmap[cls.REV | cls.DORSAL_TURN] = base_cmap[5]
        if include_quiescence:
            # Brown
            cmap[cls.QUIESCENCE] = base_cmap[6]
            cmap[cls.QUIESCENCE | cls.VENTRAL_TURN] = base_cmap[6]
            cmap[cls.QUIESCENCE | cls.DORSAL_TURN] = base_cmap[6]
            cmap[cls.PAUSE | cls.QUIESCENCE] = base_cmap[6]
        if include_pause:
            # Brown (same as quiescence)
            cmap[cls.PAUSE] = base_cmap[6]
        if include_collision:
            # Gray
            cmap[cls.SELF_COLLISION] = base_cmap[7]
            cmap[cls.FWD | cls.SELF_COLLISION] = base_cmap[7]
            cmap[cls.FWD | cls.SELF_COLLISION | cls.DORSAL_TURN] = base_cmap[7]
            cmap[cls.FWD | cls.SELF_COLLISION | cls.VENTRAL_TURN] = base_cmap[7]
            cmap[cls.REV | cls.SELF_COLLISION] = base_cmap[7]
            cmap[cls.REV | cls.SELF_COLLISION | cls.DORSAL_TURN] = base_cmap[7]
            cmap[cls.REV | cls.SELF_COLLISION | cls.VENTRAL_TURN] = base_cmap[7]
        if include_custom:
            # Green
            cmap[cls.CUSTOM] = base_cmap[4]
        if include_stimulus:
            # Pink (switched with gold)
            cmap[cls.STIMULUS] = base_cmap[5]
        if additional_shaded_states is not None:
            # Add states and colors using the matplotlib colormap
            # Start at the first color that isn't in the cmap
            for i in range(10):
                if base_cmap[i] not in cmap.values():
                    i_color_offset = i
                    break
            else:
                raise ValueError(f"Could not find a color in the base colormap that is not already in the ethogram "
                                 f"colormap. Base colormap: {base_cmap}, ethogram colormap: {cmap}")
            for i, state in enumerate(additional_shaded_states):
                cmap[state] = base_cmap[i_color_offset + i]

        if not use_plotly_style_strings:
            # Convert to rgb strings
            # The strings are by default a string like 'rgb(228,26,28)'
            # First we need to convert the strings to a tuple of integers
            def str_2_tuple(s):
                integers = re.findall(r'\d+', s)
                # Convert the found integers to integers (they are initially strings)
                return [int(num)/255 for num in integers]
            cmap = {k: matplotlib.colors.to_hex(str_2_tuple(v)) if v else v for k, v in cmap.items()}

        return cmap

    # @classmethod
    # def __contains__(cls, value):
    #     # NOTE: I would have to do a metaclass instead of a normal override to make this work
    #     # Backport the python 3.12 feature of allowing "in" to work with non-integers
    #     try:
    #         super().__contains__(value)
    #         return True
    #     except TypeError:
    #         return False

    @classmethod
    def assert_is_valid(cls, value):
        if not isinstance(value, BehaviorCodes):
            raise InvalidBehaviorAnnotationsError(f"Value {value} is not a valid behavioral code "
                                                  f"({cls._value2member_map_})")

    @classmethod
    def assert_all_are_valid(cls, vec):
        for v in vec:
            cls.assert_is_valid(v)

    @classmethod
    def is_successful_behavior(cls, value):
        """Returns True if the behavior is a successful behavior, i.e. not a tracking or other pipeline failure"""
        return value not in (cls.NOT_ANNOTATED, cls.UNKNOWN, cls.TRACKING_FAILURE)

    @classmethod
    def must_be_manually_annotated(cls, value):
        """Returns True if the behavior must be manually annotated"""
        if value is None:
            return False
        return value in (cls.SUPERCOIL, cls.QUIESCENCE)

    @property
    def full_name(self):
        """
        Simple states will properly return a name, but if it is a compound state it will be None by default
        ... unfortunately the enum class relies on certain names being None, so I have to have a separate
        property for this

        See convert_to_simple_states

        Returns
        -------

        """
        if self._name_ is not None:
            return self._name_
        else:
            # Convert a string like 'BehaviorCodes.DORSAL_TURN|REV' to 'DORSAL_TURN and REV'
            split_name = self.__str__().split('.')[-1]
            full_name = split_name.replace('|', ' and ')
            return full_name

    @property
    def individual_names(self):
        """
        Returns a list of the individual names in the compound state, or the simple name if it is a simple state

        Returns
        -------

        """
        if self._name_ is not None:
            return [self._name_]
        else:
            # Convert a string like 'BehaviorCodes.DORSAL_TURN|REV' to ['DORSAL_TURN', 'REV']
            split_name = self.__str__().split('.')[-1]
            individual_names = split_name.split('|')
            return individual_names

    @classmethod
    def default_state_hierarchy(cls, use_strings=False,
                                include_slowing=True, include_self_collision=False):
        """
        Returns the default state hierarchy for this behavior

        Returns
        -------

        """
        vec = [cls.REV, cls.VENTRAL_TURN, cls.DORSAL_TURN, cls.PAUSE, cls.FWD,
               cls.TRACKING_FAILURE, cls.UNKNOWN]
        if include_slowing:
            vec.insert(3, cls.SLOWING)
        if include_self_collision:
            vec.insert(1, cls.SELF_COLLISION)
        if use_strings:
            return [v.name for v in vec]
        else:
            return vec

    @classmethod
    def convert_to_simple_states(cls, query_state: 'BehaviorCodes'):
        """
        Collapses simultaneous states into one-state-at-a-time, using a hardcoded hierarchy

        Returns
        -------

        """

        for state in cls.default_state_hierarchy():
            if state in query_state:
                return state
        return cls.UNKNOWN

    @classmethod
    def use_pause_to_filter_vector(cls, query_vec: pd.Series):
        """
        Collapses simultaneous PAUSE + other states into just PAUSE

        Returns
        -------

        """
        return query_vec.apply(lambda x: cls.PAUSE if cls.PAUSE in x else x)

    @classmethod
    def convert_to_simple_states_vector(cls, query_vec: pd.Series):
        """
        Uses convert_to_simple_states on a vector

        Parameters
        ----------
        query_vec

        Returns
        -------

        """
        return query_vec.apply(cls.convert_to_simple_states)

    @classmethod
    def plot_behaviors(cls, vec: pd.Series):
        """
        Plots a vector of behaviors as a series of colored dots

        Returns
        -------

        """
        vec_strings = vec.apply(lambda x: x.full_name)
        # Create dataframe that plotly can use
        df = pd.DataFrame({'time': vec.index, 'behavior': vec_strings}).assign(y=0)
        fig = px.scatter(df, x='time', y='y', color='behavior')
        return fig


def options_for_ethogram(beh_vec, shading=False, include_reversal_turns=False, include_collision=False,
                         additional_shaded_states: Optional[List['BehaviorCodes']] = None,
                         to_extend_short_states=False,
                         yref='paper', DEBUG=False, **kwargs):
    """
    Returns a list of dictionaries that can be passed to plotly to draw an ethogram

    if shading is True, then the ethogram will be partially transparent, to be drawn on top of a trace

    Parameters
    ----------
    beh_vec
    shading
    include_reversal_turns
    include_collision
    additional_shaded_states
    yref: str - either 'paper' or a specific axis label (default 'paper'). If 'paper', then shades all subplots
        See fig.add_shape on a plotly figure for more details
    DEBUG

    Returns
    -------

    """
    all_shape_opt = []
    if shading:
        cmap_func = lambda state: BehaviorCodes.shading_cmap_func(state,
                                                                  additional_shaded_states=additional_shaded_states,
                                                                  force_string_output=True,
                                                                  **kwargs)
    else:
        cmap_func = lambda state: \
            BehaviorCodes.ethogram_cmap(include_reversal_turns=include_reversal_turns,
                                        additional_shaded_states=additional_shaded_states,
                                        **kwargs).get(state, None)

    # Loop over all behaviors in the colormap (some may not be present in the vector)
    possible_behaviors = BehaviorCodes.possible_colors()
    if additional_shaded_states is not None:
        possible_behaviors.extend(additional_shaded_states)
    for behavior_code in possible_behaviors:
        binary_behavior = BehaviorCodes.vector_equality(beh_vec, behavior_code)
        if cmap_func(behavior_code) is None:
            # Do not draw anything for this behavior
            if DEBUG:
                print(f'No color for behavior {behavior_code}')
            continue
        starts, ends = get_contiguous_blocks_from_column(binary_behavior, already_boolean=True)
        if to_extend_short_states and behavior_code not in (BehaviorCodes.NOT_ANNOTATED, BehaviorCodes.UNKNOWN):
            starts, ends = extend_short_states(starts, ends, len(binary_behavior), state_length_minimum=10)
        color = cmap_func(behavior_code)
        if DEBUG:
            print(f'Behavior {behavior_code} is {color}')
            print(f"starts: {starts}, ends: {ends}")
        for s, e in zip(starts, ends):
            # If there is an index in the behavior vector, convert the starts and ends
            # to the corresponding time
            s = beh_vec.index[s]
            if e < len(beh_vec):
                e = beh_vec.index[e]
            else:
                # If the last behavior is the same as the one we are plotting, then we need to
                # extend the end of the last block to the end of the vector
                e = beh_vec.index[-1]
            # Note that yref is ignored if this is a subplot. If yref is manually set, then it refers to the entire plot
            shape_opt = dict(type="rect", x0=s, x1=e, yref=yref, y0=0, y1=1,
                             fillcolor=color, line_width=0, layer="below")
            all_shape_opt.append(shape_opt)

    return all_shape_opt


def shade_using_behavior_plotly(beh_vector, fig, shape_opt=None, index_conversion=None, **kwargs):
    """
    Plotly version of shade_using_behavior

    See options_for_ethogram for more details on the kwargs


    Parameters
    ----------
    beh_vector
    fig
    index_conversion

    Returns
    -------

    """
    if shape_opt is None:
        shape_opt = {}
    if index_conversion is not None:
        beh_vector.index = index_conversion
    ethogram_opt = options_for_ethogram(beh_vector, shading=True, **kwargs)
    for opt in ethogram_opt:
        fig.add_shape(**opt, **shape_opt)


def shade_stacked_figure_using_behavior_plotly(beh_df, fig, **kwargs):
    """
    NOTE: DOESN'T WORK

    Expects a dataframe with a column 'dataset_name' that will be used to annotate a complex figure with multiple
    subplots

    Parameters
    ----------
    beh_df
    fig
    kwargs

    Returns
    -------

    """

    # Assume each y axis is named like 'y', y2', 'y3', etc. (skips 'y1')
    # Gather the behavior dataframe by 'dataset_name', and loop over each dataset (one vector)
    for i, (dataset_name, df) in tqdm(enumerate(beh_df.groupby('dataset_name'))):
        yref = f'y{i+1} domain' if i > 0 else 'y domain'
        print(dataset_name)
        shade_using_behavior_plotly(df['raw_annotations'].reset_index(drop=True), fig, yref=yref, **kwargs)


def plot_stacked_figure_with_behavior_shading_using_plotly(all_projects: dict,
                                                           names_to_plot: Union[str, List[str]],
                                                           to_shade=True, to_save=False, fname_suffix='',
                                                           trace_kwargs=None, combine_neuron_pairs=True,
                                                           DEBUG=False, full_path_title=False, **kwargs):
    """
    Loads the traces and behaviors from each project, producing a stack of plotly figures that

    Example:
        from wbfm.utils.visualization.hardcoded_paths import load_paper_datasets
        from wbfm.utils.general.utils_behavior_annotation import plot_stacked_figure_with_behavior_shading_using_plotly

        all_projects = load_paper_datasets('gcamp_good')
        neurons_to_plot = ['AVBL', 'AVBR', 'RIS']
        trace_kwargs=dict(use_paper_options=True)

        fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_good, neurons_to_plot,
            trace_kwargs=trace_kwargs)
        fig.show()


    Parameters
    ----------
    all_projects - Dict[str, ProjectData]; dictionary of project names and ProjectData objects
    names_to_plot - str or List[str]; names of the traces (neurons or behaviors) to plot
        See calc_behavior_from_alias for valid behaviors
    to_shade - bool; if True, then shades the behaviors on top of the traces
    to_save - bool; if True, then saves the figure to a hardcoded directory (you must have permissions)
        See get_summary_visualization_dir
    fname_suffix - str; suffix to add to the filename
    trace_kwargs - dict; kwargs to pass to build_trace_time_series_from_multiple_projects
        See ProjectData.calc_default_traces for more details
    kwargs - dict; kwargs to pass to shade_using_behavior_plotly

    Returns
    -------

    """
    # First build the overall dataframe with traces and behavior
    if trace_kwargs is None:
        trace_kwargs = {}
    trace_kwargs['min_nonnan'] = trace_kwargs.get('min_nonnan', None)
    trace_kwargs['rename_neurons_using_manual_ids'] = trace_kwargs.get('rename_neurons_using_manual_ids', True)
    trace_kwargs['manual_id_confidence_threshold'] = trace_kwargs.get('manual_id_confidence_threshold', 0)
    trace_kwargs['nan_tracking_failure_points'] = trace_kwargs.get('nan_tracking_failure_points', True)
    trace_kwargs['nan_using_ppca_manifold'] = trace_kwargs.get('nan_using_ppca_manifold', 0)

    from wbfm.utils.visualization.multiproject_wrappers import build_behavior_time_series_from_multiple_projects, \
        build_trace_time_series_from_multiple_projects
    from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture
    beh_columns = ['raw_annotations']
    # Get any behaviors required in column names
    for name in names_to_plot:
        if name not in beh_columns and name in WormFullVideoPosture.beh_aliases_stable():
            beh_columns.append(name)
    df_all_beh = build_behavior_time_series_from_multiple_projects(all_projects, beh_columns)
    df_all_traces = build_trace_time_series_from_multiple_projects(all_projects, **trace_kwargs)
    if combine_neuron_pairs:
        df_all_traces = combine_columns_with_suffix(df_all_traces)
    # Note: if there is one extra frame in the traces, it will be dropped here
    df_traces_and_behavior = pd.merge(df_all_traces, df_all_beh, how='inner', on=['dataset_name', 'local_time'])
    # Check if the physical time is set in the projects
    if not all([project.use_physical_time for project in all_projects.values()]):
        logging.warning(f"Physical time is not set in all projects, so the x axis will be in frames instead of time. "
                        f"This will cause problems if the traces are indexed using physical time")
    if DEBUG:
        print(f"df_all_traces: {df_all_traces}")
        print(f"df_all_beh: {df_all_beh}")
        print(f"df_traces_and_behavior: {df_traces_and_behavior}")

    # Prepare for plotting
    all_dataset_names = df_traces_and_behavior['dataset_name'].unique()
    n_datasets = len(all_dataset_names)

    if isinstance(names_to_plot, str):
        names_to_plot = [names_to_plot]
    cmap = px.colors.qualitative.D3

    # Initialize the plotly figure with subplots
    if full_path_title:
        # Do not use the full path, because it is too long; take the last 3 folders
        f = lambda path: '/'.join(Path(path).parts[-4:])
        subplot_titles = [f(all_projects[n].project_dir) for n in all_dataset_names]
    else:
        subplot_titles = [f"placeholder" for dataset_name in all_dataset_names]
    fig = make_subplots(rows=n_datasets, cols=1, #row_heights=[500]*len(all_dataset_names),
                        vertical_spacing=0.01, subplot_titles=subplot_titles)

    for i_dataset, (dataset_name, df) in tqdm(enumerate(df_traces_and_behavior.groupby('dataset_name')), total=n_datasets):

        opt = dict(row=i_dataset + 1, col=1)
        # Add traces
        for i_trace, name in enumerate(names_to_plot):
            if name not in df.columns:
                continue
            if DEBUG:
                print(f"Plotting {name}, x={df['local_time'].values}, y={df[name].values}")
            line_dict = go.Scatter(x=df['local_time'], y=df[name], name=name,
                                   legendgroup=name, showlegend=(i_dataset == 0),
                                   line_color=cmap[i_trace % len(cmap)])
            fig.add_trace(line_dict, **opt)
        # Update the axes
        fig.update_yaxes(title_text="dR/R", **opt)
        # Remove x ticks
        fig.update_xaxes(showticklabels=False, **opt)
        if not full_path_title:
            # Goofy way to update the subplot titles: https://stackoverflow.com/questions/65563922/how-to-change-subplot-title-after-creation-in-plotly
            fig.layout.annotations[i_dataset].update(text=f"{dataset_name}")
        # Add shapes
        if to_shade:
            beh_vector = df['raw_annotations'].reset_index(drop=True)
            beh_vector.index = df['local_time']
            shade_using_behavior_plotly(beh_vector, fig, shape_opt=opt, yref='y domain', **kwargs)
            if DEBUG:
                binary_beh_vector = BehaviorCodes.vector_equality(beh_vector, BehaviorCodes.REV)
                starts, ends = get_contiguous_blocks_from_column(binary_beh_vector, already_boolean=True)
                print(f"dataset {dataset_name}: starts: {starts}, ends: {ends}")
                break

    # Show x ticks on the last subplot
    fig.update_xaxes(title_text="Time", showticklabels=True, row=n_datasets, col=1)
    # Update the fig to be taller
    fig.update_layout(height=200*n_datasets)

    if to_save:
        folder = get_summary_visualization_dir()
        fname = os.path.join(folder, "multi_dataset_IDed_neurons_and_behavior", f"{names_to_plot}-{fname_suffix}.html")
        print(f"Saving to {fname}")
        fig.write_html(str(fname))
        fname = Path(fname).with_suffix('.png')
        fig.write_image(str(fname))

    return fig


def detect_peaks_and_interpolate(dat, to_plot=False, fig=None, height="mean", height_factor=1.0, width=5):
    """
    Builds a time series approximating the highest peaks of an oscillating signal

    Returns the interpolation class, which has the location and value of the peaks themselves

    Parameters
    ----------
    dat
    to_plot

    Returns
    -------

    """

    # Get peaks
    if height == "mean":
        height = height_factor * np.mean(dat)
    elif height == "std":
        height = height_factor * np.std(dat)
    peaks, properties = find_peaks(dat, height=height, width=width)
    y_peaks = dat[peaks]

    # Add a dummy peak at the beginning and end of the vector to help edge artifacts
    peaks = np.concatenate([[0], peaks, [len(dat)-1]])
    y_peaks = np.concatenate([[y_peaks.iat[0]], y_peaks, [y_peaks.iat[-1]]])

    # Interpolate
    interp_obj = interp1d(peaks, y_peaks, kind='cubic', bounds_error=False, fill_value="extrapolate")
    x = np.arange(len(dat))
    y_interp = interp_obj(x)

    if to_plot:
        if fig is None:
            plt.figure(dpi=200, figsize=(10, 5))
        plt.plot(dat, label="Raw data")
        plt.scatter(peaks, y_peaks, c='r', label="Detected peaks")
        plt.plot(x, y_interp, label="Interpolated envelope")#, c='tab:purple')
        plt.title("Envelope signal interpolated between peaks")
        plt.legend()

    return x, y_interp, interp_obj


def detect_peaks_and_interpolate_using_inter_event_intervals(dat, to_plot=False, fig=None,
                                                             beh_vector=None, include_zero_crossings=False,
                                                             min_time_between_peaks=2, prominence_factor=0.25,
                                                             height=None, width=5, verbose=1, DEBUG=False):
    """
    Somewhat similar to detect_peaks_and_interpolate, but instead of using the peaks themselves, uses the
    inter-event intervals to interpolate between the peaks, troughs, and zero crossings

    Parameters
    ----------
    dat
    to_plot
    fig
    height
    width

    Returns
    -------

    """
    # Make sure there are no nans
    dat = fill_nan_in_dataframe(dat)

    # Get peaks
    if height == "mean":
        height = np.mean(dat)
    # As of 2023, should be around 0.002 to detect nose wiggles for a 100-segment kymograph
    # And 0.02 for normal oscillations
    prominence = prominence_factor*np.std(dat)
    if DEBUG:
        print(prominence)
    ind_peaks, peak_properties = find_peaks(dat, height=height, width=width, prominence=prominence)

    # Get troughs
    ind_troughs, trough_properties = find_peaks(-dat, height=height, width=width, prominence=prominence)

    # Get zero crossings and combine
    if include_zero_crossings:
        ind_zero_crossings = np.where(np.diff(np.sign(dat)))[0]
        all_ind = np.concatenate([ind_peaks, ind_troughs, ind_zero_crossings])
    else:
        all_ind = np.concatenate([ind_peaks, ind_troughs])

    # Sort and get the inter-event intervals
    all_ind = np.sort(all_ind)
    all_ind_with_removals = all_ind.copy()
    raw_inter_event_intervals = np.diff(all_ind, append=len(dat)-1)

    # Remove any events that cross a behavior boundary
    if beh_vector is not None:
        starts, ends = get_contiguous_blocks_from_column(beh_vector, already_boolean=True)
        ind_to_remove = []
        for s, e in zip(starts, ends):
            # Remove any events that cross a behavior START boundary
            for i in range(len(all_ind)-1):
                this_ind, next_ind = all_ind[i], all_ind[i+1]
                if this_ind > s:
                    break
                if next_ind != -1 and this_ind != -1:
                    crosses_behavior_start = this_ind < s <= next_ind
                    if crosses_behavior_start:
                        if DEBUG:
                            print(f"Removing {all_ind[i+1]} (previous: {all_ind[i]}) at index {i+1} "
                                  f"because it crosses a behavior start ({s} {e})")
                        # all_ind_with_removals[i] = -1  # Mark for removal
                        # all_ind_with_removals[i+1] = -1
                        # We will take a diff, so we need to remove the previous index as well
                        ind_to_remove.extend([i-1, i])
                        break
        for s, e in zip(starts, ends):
            # Remove any events that cross a behavior END boundary
            for i in range(len(all_ind)-1):
                this_ind, next_ind = all_ind[i], all_ind[i+1]
                if this_ind > e:
                    break
                if next_ind != -1 and this_ind != -1:
                    crosses_behavior_end = this_ind < e <= next_ind
                    if crosses_behavior_end:
                        if DEBUG:
                            print(f"Removing {all_ind[i+1]} (previous: {all_ind[i]}) at index {i+1} "
                                  f"because it crosses a behavior end ({s} {e})")
                        # all_ind_with_removals[i] = -1  # Mark for removal
                        # all_ind_with_removals[i+1] = -1
                        # We will take a diff, so we need to remove the previous index as well
                        ind_to_remove.extend([i-1, i])
                        break
        # If there are any intervals marked for removal, remove them from the diff vector directly
        if len(ind_to_remove) > 0:
            ind_to_remove = np.unique(ind_to_remove)
            inter_event_intervals_with_removals = np.delete(raw_inter_event_intervals, ind_to_remove)
            all_ind_with_removals = np.delete(all_ind_with_removals, ind_to_remove)
        else:
            inter_event_intervals_with_removals = raw_inter_event_intervals.copy()

    else:
        if verbose >= 1:
            logging.warning("No behavior vector provided, so not removing any events that cross behavior boundaries... "
                            "This will likely lead to edge artifacts")
        inter_event_intervals_with_removals = raw_inter_event_intervals.copy()

    # Remove too-short intervals (regardless of above behavior crossing issues)
    valid_ind = np.where(inter_event_intervals_with_removals > min_time_between_peaks)[0]
    all_ind_with_removals = all_ind_with_removals[valid_ind]
    inter_event_intervals_with_removals = inter_event_intervals_with_removals[valid_ind]
    # Also process raw intervals for possible debugging
    valid_ind = np.where(raw_inter_event_intervals > min_time_between_peaks)[0]
    raw_inter_event_intervals = raw_inter_event_intervals[valid_ind]
    all_ind = all_ind[valid_ind]

    # Convert to a frequency
    inter_event_frequency = 1 / inter_event_intervals_with_removals
    # Repeat an event at the beginning and end of the vector to help edge artifacts
    all_ind_with_removals = np.concatenate([[0], all_ind_with_removals, [len(dat)-1]])
    inter_event_frequency = np.concatenate([[inter_event_frequency[0]], inter_event_frequency,
                                            [inter_event_frequency[-1]]])

    # Interpolate
    interp_obj = interp1d(all_ind_with_removals, inter_event_frequency, kind='linear', bounds_error=False,
                          fill_value="extrapolate")
    x = np.arange(len(dat))
    y_interp = interp_obj(x)
    # Clip the final output to be positive
    y_interp = np.clip(y_interp, 0, None)

    if to_plot:
        # Plot using plotly, with the interpolated series on a different plot
        # Make two subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        # First plot the raw data
        fig.add_trace(go.Scatter(x=x, y=dat, mode='lines'), row=1, col=1)
        # Also add the peaks, troughs, and zero crossings
        fig.add_scatter(x=ind_peaks, y=dat[ind_peaks], mode='markers', name='Peaks', row=1, col=1)
        fig.add_scatter(x=ind_troughs, y=dat[ind_troughs], mode='markers', name='Troughs', row=1, col=1)
        if include_zero_crossings:
            fig.add_scatter(x=ind_zero_crossings, y=dat[ind_zero_crossings], mode='markers', name='Zero crossings',
                            row=1, col=1)
        # Another scatter for the events that weren't removed
        fig.add_scatter(x=all_ind_with_removals, y=dat[all_ind_with_removals], mode='markers', name='Valid events', row=1, col=1)

        # Second subplot: interpolated time series
        fig.add_trace(go.Scatter(x=x, y=y_interp, mode='lines'), row=2, col=1)
        # Also add the raw inter-event intervals
        raw_inter_event_frequency = 1 / raw_inter_event_intervals
        fig.add_scatter(x=all_ind, y=raw_inter_event_frequency, mode='markers', name='Raw Inter-event frequency',
                        row=2, col=1)
        # Add the actually used events
        fig.add_scatter(x=all_ind_with_removals, y=inter_event_frequency, mode='markers',
                        name='Behavior-edge artifact filtered',
                        row=2, col=1)
        fig.show()

        # Also plot the peak and trough prominences as a histogram (plotly)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=peak_properties['prominences'], nbinsx=200, name="Peak prominences"))
        fig.add_trace(go.Histogram(x=trough_properties['prominences'], nbinsx=200, name="Trough prominences"))
        # Add vertical line where the minimum prominence is
        fig.add_vline(x=prominence, line_width=3, line_dash="dash", line_color="green", name="Minimum prominence")
        fig.show()

    return x, y_interp, interp_obj


def approximate_behavioral_annotation_using_pc1(project_cfg, trace_kwargs=None, to_save=True):
    """
    Uses the first principal component of the traces to approximate annotations for forward and reversal
    IMPORTANT: Although pc0 should correspond to rev/fwd, the sign of the PC is arbitrary, so we need to check
    that the sign is correct. Currently there's no way to do that without ID'ing a neuron that should correlate to fwd
    or rev, and checking that the sign is correct

    Saves an excel file within the project's behavior folder, and updates the behavioral config

    This file should be found by get_manual_behavior_annotation_fname

    Parameters
    ----------
    project_cfg

    Returns
    -------

    """
    # Load project
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # Calculate pca_modes of the project
    opt = dict(use_paper_options=True, interpolate_nan=True)
    if trace_kwargs is not None:
        opt.update(trace_kwargs)
    pca_modes, _ = project_data.calc_pca_modes(n_components=2, flip_pc1_to_have_reversals_high=True,
                                               **opt)
    pc0 = pca_modes.loc[:, 0]

    # df_traces = project_data.calc_default_traces(**opt)
    # from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
    # df_traces_no_nan = fill_nan_in_dataframe(df_traces, do_filtering=True)
    # # Then PCA
    # pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
    # pipe.fit(df_traces_no_nan.T)
    # pc0 = pipe.steps[1][1].components_[0, :]

    # Using a threshold of 0, assign forward and reversal
    starts, ends = get_contiguous_blocks_from_column(pd.Series(pc0) > 0, already_boolean=True)
    beh_vec = pd.DataFrame(make_binary_vector_from_starts_and_ends(starts, ends, pc0, pad_nan_points=(5, 0)),
                           columns=['Annotation'])
    # Should save using Ulises' convention, because that's what all other files are using
    beh_vec[beh_vec == 1] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.REV)
    beh_vec[beh_vec == 0] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.FWD)

    # Save within the behavior folder
    if to_save:
        beh_cfg = project_data.project_config.get_behavior_config()
        fname = 'pc1_generated_reversal_annotation'
        beh_cfg.save_data_in_local_project(beh_vec, fname,
                                           prepend_subfolder=True, suffix='.xlsx', sheet_name='behavior')
        beh_cfg.config['manual_behavior_annotation'] = str(Path(fname).with_suffix('.xlsx'))
        beh_cfg.update_self_on_disk()

    return beh_vec


def approximate_behavioral_annotation_using_ava(project_cfg, return_raw_rise_high_fall=False,
                                                trace_kwargs=None, min_length=8, to_save=True, DEBUG=False):
    """
    Uses AVAL/R to approximate annotations for forward and reversal
    Specifically, detects a "rise" "plateau" and "fall" state in the trace, and defines:
    - Start of rise is start of reversal
    - Start of fall is end of reversal

    Saves an excel file within the project's behavior folder, and updates the behavioral config

    This file should be found by get_manual_behavior_annotation_fname

    Should be more consistent with prior work than approximate_behavioral_annotation_using_pc1

    Parameters
    ----------
    project_cfg - str or ProjectData; path to the project config file (or loaded project)
    return_raw_rise_high_fall - bool; if True, then returns the raw rise/high/fall/low vector, not the converted
        fwd/rev vector
    min_length - int; minimum length of a state (in volumes) to be considered a state
    to_save - bool; if True, then saves the behavior vector to disk inside the project
    DEBUG - bool; if True, then plots debugging information

    Returns
    -------

    """
    # Load project
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # Calculate pca_modes of the project
    opt = dict(use_paper_options=True)
    if trace_kwargs is not None:
        opt.update(trace_kwargs)

    df_traces = project_data.calc_default_traces(**opt)
    y = combine_pair_of_ided_neurons(df_traces, 'AVA')
    beh_vec = calculate_rise_high_fall_low(y, DEBUG=DEBUG)

    if return_raw_rise_high_fall:
        return beh_vec

    # Convert this to ulises REV/FWD (because it will be saved to disk)
    beh_vec[beh_vec == 'rise'] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.REV)
    beh_vec[beh_vec == 'high'] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.REV)
    beh_vec[beh_vec == 'fall'] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.FWD)
    beh_vec[beh_vec == 'low'] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.FWD)

    # Remove very short states
    beh_vec = remove_short_state_changes(beh_vec, min_length=min_length)
    beh_vec = pd.DataFrame(beh_vec, columns=['Annotation'])

    # Save within the behavior folder
    if to_save:
        beh_cfg = project_data.project_config.get_behavior_config()
        fname = 'behavior/ava_generated_reversal_annotation'
        beh_cfg.save_data_in_local_project(beh_vec, fname,
                                           prepend_subfolder=False, suffix='.xlsx', sheet_name='behavior')
        beh_cfg.config['manual_behavior_annotation'] = str(Path(fname).with_suffix('.xlsx'))
        beh_cfg.update_self_on_disk()

    return beh_vec


def approximate_slowing_using_speed_from_config(project_cfg, min_length=3, return_raw_rise_high_fall=False, DEBUG=False):
    """
    Uses worm speed to define slowing periods
    Specifically, detects a "rise" "plateau" and "fall" state in the trace, and defines:
    - Rise if the speed is negative or crosses 0
    - Fall if the speed is positive or crosses 0

    Parameters
    ----------
    project_cfg

    Returns
    -------

    """
    # Load project
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    y = project_data.worm_posture_class.worm_angular_velocity(fluorescence_fps=False)
    beh_vec, beh_vec_raw = calc_slowing_using_peak_detection(y, min_length)

    if return_raw_rise_high_fall:
        return beh_vec_raw
    else:
        return beh_vec


def calc_slowing_using_peak_detection(y, min_length, DEBUG=False, **kwargs):
    kwargs['width'] = kwargs.get('width', 5)
    kwargs['height'] = kwargs.get('height', 0.1)
    kwargs['deriv_epsilon'] = kwargs.get('deriv_epsilon', 0.4)
    beh_vec_raw = calculate_rise_high_fall_low(y - y.mean(), min_length=min_length, verbose=0, DEBUG=DEBUG, **kwargs)
    # Convert this to slowing periods
    # For each period check what the mean speed is
    beh_vec = pd.Series(np.zeros_like(beh_vec_raw), index=beh_vec_raw.index, dtype=bool)
    rise_starts, rise_ends = get_contiguous_blocks_from_column(beh_vec_raw == 'rise', already_boolean=True)
    fall_starts, fall_ends = get_contiguous_blocks_from_column(beh_vec_raw == 'fall', already_boolean=True)
    for s, e in zip(rise_starts, rise_ends):
        # If the mean speed is negative, then it is a slowing period
        # Get a zero crossing, if any
        zero_crossings = np.where(np.diff(np.sign(y[s:e])))[0]
        if len(zero_crossings) > 0:
            # Only annotate the state up until the zero crossing
            # The zero crossing is the end of the rise in both rises and falls
            i_zero = s + zero_crossings[0]
            beh_vec[s:i_zero] = True
        elif np.mean(y[s:e]) < 0:
            beh_vec[s:e] = True

    for s, e in zip(fall_starts, fall_ends):
        zero_crossings = np.where(np.diff(np.sign(y[s:e])))[0]
        if len(zero_crossings) > 0:
            i_zero = s + zero_crossings[0]
            beh_vec[s:i_zero] = True
        elif np.mean(y[s:e]) > 0:
            beh_vec[s:e] = True
    # Remove very short states (requires the vector to be integers)
    beh_vec = remove_short_state_changes(beh_vec, min_length=min_length)
    return beh_vec, beh_vec_raw


def calc_slowing_using_threshold(y, min_length, threshold, only_negative_deriv=True, DEBUG=False):
    """
    Uses a threshold to define slowing periods, and then (optionally) only keep periods with a negative derivative

    Note that this expects an all-positive speed, i.e. not signed by reversals

    Parameters
    ----------
    y
    min_length
    DEBUG

    Returns
    -------

    """

    # Get the periods both below the threshold and with negative derivative
    beh_vec = pd.Series(np.zeros_like(y), index=y.index, dtype=bool)
    beh_vec[y < threshold] = True

    if only_negative_deriv:
        dy = np.gradient(y)
        beh_vec[dy > 0] = False

    # Remove very short states (requires the vector to be integers)
    if min_length > 0:
        beh_vec = remove_short_state_changes(beh_vec, min_length=min_length)

    return beh_vec


def calculate_rise_high_fall_low(y, min_length=5, height=0.5, width=5, prominence=0.0,
                                 signal_delta_threshold=0.15, high_assignment_threshold=0.4,
                                 deriv_epsilon=0.4, smoothing_std=2, verbose=1, DEBUG=False) -> pd.Series:
    """
    From a time series, calculates the "rise", "high", "fall", and "low" states

    Algorithm:
    - Find the peaks in the derivative in two steps:
        - Find peaks in a strongly smoothed signal
        - Find peaks in the original signal
        - Keep peaks that are in both (or close)
        - Remove peaks that do not have a large enough delta in the original signal
    - Same for negative derivative
    - Assign the positive peak regions as "rise" and the negative peak regions as "fall"
    - Assign intermediate regions based on two passes:
        - If it is after a rise and before a fall and the amplitude is > high_assignment_threshold, it is "high"
        - If it is after a fall and before a rise and the amplitude is < high_assignment_threshold, it is "low"
        - Otherwise it is "ambiguous" and assigned based on the mean amplitude (closer to previously assigned high or
        low)

    Parameters
    ----------
    y - pandas series (float)
    min_length - int (default 5). Minimum length of a state to be considered valid
    verbose
    height - See scipy.signal.find_peaks
    width - See scipy.signal.find_peaks
    prominence - See scipy.signal.find_peaks
    DEBUG - bool. If True, plots the derivative and the peaks

    Returns
    -------
    A pandas series with the same index as y, with the states as strings:

    """
    # Check if it was mean subtracted
    eps = 1e-3
    if np.abs(np.nanmean(y)) > eps:
        logging.warning("The input vector was not mean subtracted; this may lead to incorrect results")
    # Reset the index, because everything in this function uses raw indices
    y = y.copy().reset_index(drop=True)
    # Take derivative and standardize
    # Don't z score, because the baseline should be 0, not the mean
    dy = np.gradient(y)
    dy = dy / np.nanstd(dy)
    derivative_state_shift = 1
    # Define the start of the rise and fall as the width of the peak at the absolute height of 0
    # Unfortunately scipy only allows relative height calculations, so we have to hack an absolute height
    # https://stackoverflow.com/questions/53778703/python-scipy-signal-peak-widths-absolute-heigth-fft-3db-damping
    beh_vec = pd.Series(np.zeros_like(y))
    peak_eps = width
    opt_find_peaks = dict(height=height, width=width, prominence=prominence)
    for i, this_dy in enumerate([dy, -dy]):
        # First find peaks in the smoothed signal
        df_smooth = filter_gaussian_moving_average(pd.Series(this_dy), smoothing_std)
        peaks_smooth, properties_smooth = find_peaks(df_smooth, **opt_find_peaks)
        # Second find the peaks in the original signal
        peaks_raw, properties_raw = find_peaks(this_dy, **opt_find_peaks)
        # Build a consensus list of peaks found in both signals
        peaks, heights = [], []
        for peak, prop in zip(peaks_raw, properties_raw['peak_heights']):
            if np.any(np.abs(peaks_smooth - peak) < peak_eps):
                peaks.append(peak)
                heights.append(prop)
        heights = np.array(heights)

        prominences, left_bases, right_bases = peak_prominences(this_dy, peaks)
        # Instead of prominences, pass the peaks heights to get the intersection at 0
        # But, because the derivative might not exactly be 0, pass an epslion value
        # Note that this epsilon is quite high; some "high" periods can have a negative slope almost as high as a "fall"

        widths, h_eval, left_ips, right_ips = peak_widths(
            this_dy, peaks,
            rel_height=1,
            prominence_data=(heights - deriv_epsilon, left_bases, right_bases)
        )

        # Filter: remove peaks that do not have a large enough delta in the original signal
        peaks_filtered, heights_filtered = [], []
        for i_left, i_right, peak, height in zip(left_ips, right_ips, peaks, heights):
            delta = np.abs(y[int(i_left)] - y[int(i_right)])
            if delta > signal_delta_threshold:
                peaks_filtered.append(peak)
                heights_filtered.append(height)
                if DEBUG and verbose >= 1:
                    print(f"Keeping peak at {int(i_left)} because delta ({delta}) is large enough")
            else:
                if DEBUG and verbose >= 1:
                    print(f"Removing peak at {int(i_left)} because delta ({delta}) is too small "
                          f"({signal_delta_threshold})")
        heights = np.array(heights_filtered)
        peaks = np.array(peaks_filtered)

        # Recalculate metadata using the final filtered peaks
        prominences, left_bases, right_bases = peak_prominences(this_dy, peaks)
        widths, h_eval, left_ips, right_ips = peak_widths(
            this_dy, peaks,
            rel_height=1,
            prominence_data=(heights - deriv_epsilon, left_bases, right_bases)
        )

        if DEBUG:
            # Plot the derivative with the peaks and widths
            print(h_eval)
            fig = px.line({'dy': dy, 'dy_smoothed': df_smooth},
                          title="Positive peaks" if i == 0 else "Negative peaks")
            fig.show()
            plt.figure(dpi=200)
            fig = plt.plot(dy)
            for i_left, i_right, i_prom in zip(left_ips, right_ips, prominences):
                plt.plot(np.arange(int(i_left), int(i_right)), dy[int(i_left): int(i_right)], "x")
                print(f"left: {int(i_left)}, right: {int(i_right)}, "
                      f"height_delta: {this_dy[int(i_left)] - this_dy[int(i_right)]}, prominence: {i_prom}")

        # Actually assign the state
        if i == 0:
            state = 'rise'
        else:
            state = 'fall'
        for i_left, i_right in zip(left_ips, right_ips):
            if i_right - i_left < min_length:
                continue
            beh_vec[int(i_left) + derivative_state_shift: int(i_right) + derivative_state_shift] = state
    # Then define the intermediate regions
    # If it is after a rise and before a fall it is "high", otherwise "low"
    # There should be no regions that are surrounded by "rise" or "fall"
    starts, ends = get_contiguous_blocks_from_column(beh_vec == 0, already_boolean=True)
    if len(starts) <= 1:
        # If there is only one region, then it is either all high or all low... but probably something is wrong
        logging.warning(f"Only one region detected in the derivative; probably something is wrong")
        if np.mean(y) > 0:
            beh_vec[:] = 'high'
        else:
            beh_vec[:] = 'low'
        return beh_vec

    ambiguous_periods = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        # Special cases for the first and last regions, based on the next start or previous end
        is_above_high_threshold = np.mean(y[s:e]) > high_assignment_threshold
        if s == 0:
            # First
            if beh_vec[e + 1] == 'fall' and is_above_high_threshold:
                beh_vec[s:e] = 'high'
            else:
                beh_vec[s:e] = 'low'
        elif e == len(beh_vec):
            # Last
            if beh_vec[s - 1] == 'rise' and is_above_high_threshold:
                beh_vec[s:e] = 'high'
            else:
                beh_vec[s:e] = 'low'
        elif (beh_vec[s - 1] == 'rise' and beh_vec[e + 1] == 'fall') and is_above_high_threshold:
            # In between
            beh_vec[s:e] = 'high'
        elif (beh_vec[s - 1] == 'fall' and beh_vec[e + 1] == 'rise') and not is_above_high_threshold:
            beh_vec[s:e] = 'low'
        else:
            if beh_vec[s - 1] == beh_vec[e + 1] and verbose > 0:
                logging.warning(f"Region from {s} to {e} is surrounded by rise or fall; "
                                f"probably a rise or fall was missed")
            # In this case, save to be split later, based on similarity to other high or fall periods
            ambiguous_periods.append((s, e))

    # First get the means of all the high and low periods
    high_mean = np.mean(y[beh_vec == 'high'])
    low_mean = np.mean(y[beh_vec == 'low'])
    if DEBUG:
        print(f"High mean: {high_mean}, low mean: {low_mean}")
    for s, e in ambiguous_periods:
        # Assign this period to the high or low state based on which mean it is closer to
        this_mean = np.mean(y[s:e])
        if np.abs(this_mean - high_mean) < np.abs(this_mean - low_mean):
            beh_vec[s:e] = 'high'
            if DEBUG:
                print(f"Region from {s} to {e} with mean {this_mean} assigned to high")
        else:
            beh_vec[s:e] = 'low'
            if DEBUG:
                print(f"Region from {s} to {e} with mean {this_mean} assigned to low")

    if DEBUG:
        df = pd.DataFrame({'y': y, 'dy': dy, 'state': beh_vec})
        px.scatter(df, y='y', color='state').show()
    return beh_vec


def shade_using_behavior(beh_vector, ax=None, behaviors_to_ignore=(BehaviorCodes.SELF_COLLISION, ),
                         cmap=None, index_conversion=None,
                         additional_shaded_states: Optional[List['BehaviorCodes']]=None, alpha=1.0,
                         DEBUG=False, **kwargs):
    """
    Shades current plot using a 3-code behavioral annotation:
        Invalid data (no shade)
        FWD (no shade)
        REV (gray)

    See BehaviorCodes for valid codes

    See options_for_ethogram for a plotly-compatible version

    Parameters
    ----------
    beh_vector - vector of behavioral codes
    ax - axis to plot on
    behaviors_to_ignore - list of behaviors to ignore. See BehaviorCodes for valid codes
    cmap - colormap to use. See BehaviorCodes for default
    additional_shaded_states - list of additional states to shade
    alpha - transparency of the shading
    index_conversion - function to convert indices from the beh_vector to the plot indices
    DEBUG
    kwargs - Ignored

    Returns
    -------

    """
    if cmap is None:
        cmap = lambda state: BehaviorCodes.shading_cmap_func(state,
                                                             additional_shaded_states=additional_shaded_states)
    if ax is None:
        ax = plt.gca()

    # Get all behaviors that exist in the data and the cmap
    beh_vector = pd.Series(beh_vector)
    # data_behaviors = beh_vector.unique()
    # cmap_behaviors = pd.Series(BehaviorCodes.possible_colors(include_complex_states=include_complex_states))
    # Note that this returns a numpy array in the end
    # all_behaviors = pd.concat([pd.Series(data_behaviors), pd.Series(cmap_behaviors)]).unique()

    # Define the list of simple behaviors we will shade
    all_behaviors = [BehaviorCodes.REV]  # Default
    # Insert new states at the front, so they are shaded first
    if additional_shaded_states is not None:
        if BehaviorCodes.REV in additional_shaded_states:
            # This means there is a user-defined order for REV, e.g. plotted behind everything else
            all_behaviors = additional_shaded_states
        else:
            all_behaviors = additional_shaded_states + all_behaviors
        # Make sure these are not removed in the next step
        behaviors_to_ignore = tuple(set(behaviors_to_ignore) - set(additional_shaded_states))
    all_behaviors = pd.Series(all_behaviors)
    if DEBUG:
        print("behavior list before removals: ", all_behaviors, ' behaviors to ignore: ', behaviors_to_ignore)

    # Remove behaviors to ignore
    if behaviors_to_ignore is not None:
        for b in behaviors_to_ignore:
            all_behaviors = all_behaviors[all_behaviors != b]
    for b in [BehaviorCodes.UNKNOWN, BehaviorCodes.NOT_ANNOTATED, BehaviorCodes.TRACKING_FAILURE]:
        all_behaviors = all_behaviors[all_behaviors != b]
    if DEBUG:
        print("all_behaviors: ", all_behaviors)
        print(f"Cmap: {[cmap(b) for b in all_behaviors]}")

    # Loop through the remaining behaviors, and use the binary vector to shade per behavior
    beh_vector = pd.Series(beh_vector)
    for b in all_behaviors:
        binary_vec = BehaviorCodes.vector_equality(beh_vector, b)
        color = cmap(b)
        if color is None:
            if DEBUG:
                print(f'No color for behavior {b}')
            continue
        # Get the start and end indices of the binary vector
        starts, ends = get_contiguous_blocks_from_column(binary_vec, already_boolean=True)
        if DEBUG and len(starts) == 0:
            print(f'No behavior {b} found')
        for start, end in zip(starts, ends):
            if index_conversion is not None:
                ax_start = index_conversion[start]
                if end >= len(index_conversion):
                    # Often have an off by one error
                    ax_end = index_conversion[-1]
                else:
                    ax_end = index_conversion[end]
            else:
                ax_start = start
                ax_end = end
            if DEBUG:
                print(f'Behavior {b} from {ax_start} to {ax_end}')
            ax.axvspan(ax_start, ax_end, alpha=alpha, color=color, zorder=-10)


def add_behavior_shading_to_plot(ind_preceding, index_conversion=None,
                                 behavior_shading_type='fwd', ax=None, use_plotly=False, DEBUG=False):
    if not use_plotly:
        if True: #xlim is None:
            # Instead of the xlim, we want the length of the vector
            if ax is None:
                # Get data from current figure
                lines = plt.gca().get_lines()
            else:
                lines = ax.get_lines()
            if len(lines) == 0:
                # If there is more than one line, it should be fine
                raise ValueError("No lines found in the axis, cannot shade")
            x = lines[0].get_xdata()
            xlim = (0, len(x))
    else:
        xlim = (0, len(index_conversion))
    # Shade using behavior either before or after the ind_preceding line
    if behavior_shading_type is not None:
        # Initialize empty (FWD = no annotation)
        beh_vec = np.array([BehaviorCodes.FWD for _ in range(xlim[1] - xlim[0])])
        # beh_vec = np.array([BehaviorCodes.FWD for _ in range(int(np.ceil(xlim[1])))])
        if behavior_shading_type == 'fwd':
            # If 'fwd' triggered, the shading should go BEFORE the line
            beh_vec[:ind_preceding] = BehaviorCodes.REV
        elif behavior_shading_type == 'rev':
            # If 'rev' triggered, the shading should go AFTER the line
            beh_vec[ind_preceding:] = BehaviorCodes.REV
        elif behavior_shading_type == 'both':
            # If 'both' triggered, the shading should go BEFORE and AFTER the line
            beh_vec[:] = BehaviorCodes.REV
        else:
            raise ValueError(f"behavior_shading must be 'rev' or 'fwd', not {behavior_shading_type}")

        if DEBUG:
            print(behavior_shading_type)
            print(ind_preceding)
            print(index_conversion)
            # print(beh_vec)
        # else:
        #     # NOTE: ind_preceding is not used
        #     assert ax is not None, "For plotly shading, ax (fig) must be provided"
        #     beh_vec = pd.Series(index=index_conversion, data=False)
        #     if behavior_shading_type == 'rev':
        #         beh_vec.loc[0:] = BehaviorCodes.REV
        #         beh_vec.loc[:0] = BehaviorCodes.FWD
        #     elif behavior_shading_type == 'fwd':
        #         beh_vec.loc[0:] = BehaviorCodes.FWD
        #         beh_vec.loc[:0] = BehaviorCodes.REV
        #     else:
        #         raise ValueError(f"behavior_shading must be 'rev' or 'fwd', not {behavior_shading_type}")

        # Actual shading
        if use_plotly:
            beh_vec = pd.Series(beh_vec, index=index_conversion)
            shade_using_behavior_plotly(beh_vec, fig=ax)
        else:
            shade_using_behavior(beh_vec, ax=ax, index_conversion=index_conversion)


def get_same_phase_segment_pairs(t, df_phase, min_distance=10, similarity_threshold=0.1, DEBUG=False):
    """
    Finds pairs of body segments that have the same hilbert phase

    Parameters
    ----------
    t
    df_phase
    min_distance
    similarity_threshold
    DEBUG

    Returns
    -------

    """
    phase_slice = df_phase.loc[t, :]

    start_segment = 10
    end_segment = 90

    seg_pairs = []
    for i_seg in range(start_segment, end_segment):
        phase_subtracted = np.abs(phase_slice - phase_slice[i_seg])

        i_seg_pair = np.argmax(phase_subtracted[i_seg + min_distance:] < similarity_threshold) + i_seg + min_distance
        if i_seg_pair > end_segment:
            break
        seg_pairs.append([i_seg, i_seg_pair])

        if DEBUG:
            print(i_seg_pair, phase_subtracted[i_seg_pair - 1:i_seg_pair + 2])

    return seg_pairs


def get_heading_vector_from_phase_pair_segments(t, seg_pairs, df_pos):
    """
    Uses pairs of segments and gets the average vector (heading) of them

    Parameters
    ----------
    t
    seg_pairs
    df_pos

    Returns
    -------

    """
    pos_slice = df_pos.loc[t, :]
    all_vectors = []
    for pair in seg_pairs:
        vec_seg0 = [pos_slice[pair[0]]['X'], pos_slice[pair[0]]['Y']]
        vec_seg1 = [pos_slice[pair[1]]['X'], pos_slice[pair[1]]['Y']]

        vec_delta = np.array(vec_seg0) - np.array(vec_seg1)
        all_vectors.append(vec_delta)
    if len(all_vectors) == 0:
        return np.array([np.nan, np.nan])
    else:
        return np.mean(all_vectors, axis=0)


def rgb_to_hex(rgb: List[int]):
    return '#%02x%02x%02x' % rgb


def plot_dataframe_of_transitions(df_probabilities, df_raw_number=None, output_folder=None, to_view=True, engine=None,
                                  use_behavior_codes_colors=True, verbose=1, DEBUG=False):
    """

    Parameters
    ----------
    df_probabilities
    output_folder
    to_view
    engine - See https://graphviz.org/docs/layouts/ for options

    Returns
    -------

    """
    # Create a Digraph object
    from graphviz import Digraph
    dot = Digraph(comment='State Transition Diagram')

    # Add nodes to the graph
    num_nodes = 0
    for state in df_probabilities.index:
        # Set the size parameter based on df_raw_number, if present
        # See https://www.graphviz.org/pdf/dotguide.pdf for parameters
        opt = dict()
        if df_raw_number is not None:
            max_sz = df_raw_number.max().max()
            size = 10*np.log((df_raw_number.loc[state, state] / max_sz + 1))
            print(state, size)
            _opt = dict(width=str(size), height=str(size), shape='circle', fixedsize='true')
            opt.update(_opt)
        if use_behavior_codes_colors:
            # Also set the color based on the state
            state_enum = BehaviorCodes[state]
            color = state_enum.ethogram_cmap(include_turns=True, use_plotly_style_strings=False)[state_enum]
        else:
            color = 'white'
        opt['fillcolor'] = color
        opt['style'] = 'filled'
        # opt['fontcolor'] = color

        dot.node(state, **opt)
        num_nodes += 1
        if DEBUG:
            print(state, opt)
    if verbose >= 1:
        print(f"Added {num_nodes} nodes")

    # Add edges to the graph with labels and widths based on transition probabilities
    eps = 0.01
    num_edges = 0
    for from_state in df_probabilities.index:
        for to_state in df_probabilities.columns:
            probability = df_probabilities.loc[from_state, to_state]
            if probability > eps:
                edge_opt = dict(tail_name=from_state, head_name=to_state,
                                label=f'{probability:.2f}', width=str(probability * 5))
                dot.edge(**edge_opt)
                num_edges += 1
                if DEBUG:
                    print(edge_opt)
    if verbose >= 1:
        print(f"Added {num_edges} edges")

    # Render the graph to a file or display it
    if output_folder is not None:
        fname = os.path.join(output_folder, 'state_transition_diagram')
        dot.render(fname, view=False, format='png', engine=engine)
        dot.render(fname, view=to_view, format='pdf', engine=engine)
    # else:
    #     dot.render(view=to_view, format='pdf', engine=engine)

    return dot


def approximate_turn_annotations_using_ids(project_cfg, min_length=4, post_reversal_padding=10,
                                           to_save=True, DEBUG=False):
    """
    Use case is for immobilized recordings where there is no real behavior, but there are ID's

    Defines a turn in this way:
    - AVA fall, annotated using calculate_rise_high_fall_low
    - Ventral if SMDVL/R are in a rise state
    - Dorsal if SMDDL/R are in a rise state
    - If both, take the one with higher amplitude

    Parameters
    ----------
    project_data

    Returns
    -------

    """
    # Load project
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # First get the AVA rise/fall
    ava_vec = approximate_behavioral_annotation_using_ava(project_cfg, return_raw_rise_high_fall=True,
                                                          min_length=2*min_length, to_save=False, DEBUG=DEBUG)

    # Also need the traces, which should be the same as the traces used for the AVA rise/fall
    opt = dict(use_paper_options=True)
    df_traces = project_data.calc_default_traces(**opt)

    y_turns = {}
    for base_name in ['SMDD', 'SMDV', 'RIV']:
        try:
            y_turns[base_name] = combine_pair_of_ided_neurons(df_traces, base_name=base_name)
        except NeedsAnnotatedNeuronError:
            logging.warning(f"Could not find {base_name}; no turns can be annotated")
    y_ventral = y_turns.get('SMDV', None)
    if y_ventral is None:
        y_ventral = y_turns.get('RIV', None)
    y_dorsal = y_turns.get('SMDD', None)
    if y_ventral is None and y_dorsal is None:
        raise NeedsAnnotatedNeuronError("Could not find either SMDV, SMDD, or RIV; no turns can be annotated")

    if y_dorsal is not None:
        dorsal_vec = calculate_rise_high_fall_low(y_dorsal - y_dorsal.mean())
        # dorsal_rise_starts, dorsal_rise_ends = get_contiguous_blocks_from_column(dorsal_vec == 'rise', already_boolean=True)
    else:
        dorsal_vec = y_dorsal.copy()
        dorsal_vec[:] = 'low'

    if y_ventral is not None:
        ventral_vec = calculate_rise_high_fall_low(y_ventral - y_ventral.mean())
        # ventral_rise_starts, ventral_rise_ends = get_contiguous_blocks_from_column(ventral_vec == 'rise',
        #                                                                            already_boolean=True)
    else:
        ventral_vec = y_ventral.copy()
        ventral_vec[:] = 'low'
    # if DEBUG:
    #     print(f"Found {len(dorsal_rise_starts)} dorsal rise periods: {dorsal_rise_starts} to {dorsal_rise_ends}")
    #     print(f"Found {len(ventral_rise_starts)} ventral rise periods: {ventral_rise_starts} to {ventral_rise_ends}")

    # Loop over ava falls, and try to assign them to dorsal or ventral
    # The one in a greater (longer) rise state is the one that wins
    # If both dorsal and ventral are equally in a rise state, take the one with higher amplitude
    # If neither dorsal nor ventral are in a rise state, then skip it
    ava_fall_starts, ava_fall_ends = get_contiguous_blocks_from_column(ava_vec == 'fall', already_boolean=True)
    turn_vec = pd.Series(np.zeros_like(ava_vec), index=ava_vec.index, dtype=object)
    for s, e in zip(ava_fall_starts, ava_fall_ends):
        if s <= 1:
            continue
        # Check if dorsal or ventral are in a rise state, including some time after
        e_padding = e + post_reversal_padding
        if DEBUG:
            print(f"Checking for dorsal/ventral rise from {s} to {e_padding}")
        len_dorsal_rise = len(np.where(dorsal_vec[s:e_padding] == 'rise')[0])
        len_ventral_rise = len(np.where(ventral_vec[s:e_padding] == 'rise')[0])
        # if len_ventral_rise > 0:
        #     end_of_ventral_turn = ventral_rise_ends[np.where(ventral_rise_ends > e)[0][0]]
        # if len_dorsal_rise > 0:
        #     end_of_dorsal_turn = dorsal_rise_ends[np.where(dorsal_rise_ends > e)[0][0]]
        if len_ventral_rise > len_dorsal_rise:
            # Define the extent of the behavior as starting from the end of the AVA rise until the end of the ventral
            # (or dorsal) rise
            turn_vec[s:e] = 'ventral'
            if DEBUG:
                print(f"ventral annotation from {s} to {e}")
        elif len_ventral_rise < len_dorsal_rise:
            turn_vec[s:e] = 'dorsal'
            if DEBUG:
                print(f"dorsal annotation from {s} to {e}")
        elif len_ventral_rise == 0 and len_dorsal_rise == 0:
            continue
        else:
            # This means they were both rising the same non-zero amount
            if np.mean(y_ventral[s:e_padding]) > np.mean(y_dorsal[s:e_padding]):
                turn_vec[s:e] = 'ventral'
                if DEBUG:
                    print(f"tie-breaker ventral annotation from {s} to {e}")
            else:
                turn_vec[s:e] = 'dorsal'
                if DEBUG:
                    print(f"tie-breaker dorsal annotation from {s} to {e}")

    # Convert this to Turn annotations, which will be saved to disk after combination with the reversal annotations
    turn_vec[turn_vec == 'ventral'] = BehaviorCodes.VENTRAL_TURN
    turn_vec[turn_vec == 'dorsal'] = BehaviorCodes.DORSAL_TURN
    turn_vec[turn_vec == 0] = BehaviorCodes.NOT_ANNOTATED

    # Remove very short states
    # turn_vec = remove_short_state_changes(turn_vec, min_length=min_length)
    # turn_vec = pd.DataFrame(turn_vec, columns=['Annotation'])

    # Save within the behavior folder
    if to_save and not DEBUG:
        # Add to the reversal annotation
        beh_cfg = project_data.project_config.get_behavior_config()
        fname = beh_cfg.config['manual_behavior_annotation']
        if fname is None or not Path(fname).exists():
            # must be produced if it doesn't exist already
            approximate_behavioral_annotation_using_ava(project_cfg, return_raw_rise_high_fall=False,
                                                        min_length=2*min_length, to_save=True, DEBUG=DEBUG)
            beh_cfg = project_data.project_config.get_behavior_config()
            fname = beh_cfg.config['manual_behavior_annotation']
            abs_fname = beh_cfg.resolve_relative_path_from_config('manual_behavior_annotation')
            if fname is None or not Path(abs_fname).exists():
                raise FileNotFoundError(f"Could not find {fname} even after generating it")

        # Load the existing annotation
        from wbfm.utils.general.postures.centerline_classes import parse_behavior_annotation_file
        beh_vec_existing, _ = parse_behavior_annotation_file(project_data.project_config,
                                                             convert_to_behavior_codes=True)

        # Add this new annotation to the existing one, and convert to ulises integers for disk saving
        beh_vec = beh_vec_existing + turn_vec
        beh_vec_disk = beh_vec.apply(BehaviorCodes.enum_to_ulises_int)
        beh_vec_disk = pd.DataFrame(beh_vec_disk, columns=['Annotation'])

        # Save, overwriting the previous one
        beh_cfg.save_data_in_local_project(beh_vec_disk, fname, allow_overwrite=True, make_sequential_filename=False,
                                           prepend_subfolder=False, suffix='.xlsx', sheet_name='behavior')

    return turn_vec


def combine_pair_of_ided_neurons(df_traces, base_name='AVA'):
    y = 0
    num_y = 0
    suffixes = ['L', 'R']
    col_names = set(get_names_from_df(df_traces))
    for suffix in suffixes:
        this_name = f"{base_name}{suffix}"
        if this_name in col_names:
            num_y += 1
            # If both L/R are present, average them
            y = (y + df_traces[this_name]) / num_y
    if num_y == 0 and base_name in col_names:
        # Check for just that name, without the suffix
        num_y = 1
        y = df_traces[base_name]
    if num_y == 0:
        raise NeedsAnnotatedNeuronError(base_name)
    return y


def annotate_turns_from_reversal_ends(rev_ends, y_curvature: pd.Series, pad_up_to=None):
    """
    Uses the reversal ends and curvature to annotate turns in the following way:
    1. A turn starts at the end of the reversal
    2. A turn ends at the next zero crossing of the curvature
    3. If the curvature is positive at the end of the reversal, it is a ventral turn, otherwise dorsal

    Parameters
    ----------
    rev_ends
    y_curvature

    Returns
    -------

    """
    ventral_starts = []
    ventral_ends = []
    dorsal_starts = []
    dorsal_ends = []
    sign_flips = np.where(np.diff(np.sign(y_curvature)))[0]
    for e in rev_ends:
        if e == len(y_curvature):
            break
        # Determines ventral or dorsal turn
        y_initial = y_curvature.iat[e]  # Should I change this if there is a collision?

        # Get the next approximate zero crossing
        _next_flip_array = sign_flips[sign_flips > e]
        if len(_next_flip_array) == 0:
            break
        i_next_flip = _next_flip_array[0] + 1
        if i_next_flip > 1:
            if np.sign(y_initial) > 0:
                ventral_starts.append(e)
                ventral_ends.append(i_next_flip)
            else:
                dorsal_starts.append(e)
                dorsal_ends.append(i_next_flip)
    # See alias: ventral_only_head_curvature
    # opt = dict(fluorescence_fps=False, start_segment=2, end_segment=10, do_abs=False)
    # thresh = 0.035  # Threshold from looking at histograms of peaks
    # _raw_ventral = (self.summed_curvature_from_kymograph(only_positive=True, **opt) > thresh)
    # _raw_dorsal = (self.summed_curvature_from_kymograph(only_negative=True, **opt) > thresh)
    #
    # # Remove any turns that are too short (less than about 0.5 seconds)
    # _raw_ventral = remove_short_state_changes(_raw_ventral, min_length=30)
    # _raw_dorsal = remove_short_state_changes(_raw_dorsal, min_length=30)
    #
    # Combine
    _raw_ventral = make_binary_vector_from_starts_and_ends(ventral_starts, ventral_ends, y_curvature)
    _raw_dorsal = make_binary_vector_from_starts_and_ends(dorsal_starts, dorsal_ends, y_curvature)
    # Pad the edges of the surviving states so that they aren't completely removed after downsampling
    _raw_ventral = pad_events_in_binary_vector(pd.Series(_raw_ventral), pad_up_to=pad_up_to)
    _raw_dorsal = pad_events_in_binary_vector(pd.Series(_raw_dorsal), pad_up_to=pad_up_to)

    _raw_vector = pd.Series(_raw_ventral.astype(int) - _raw_dorsal.astype(int))
    _raw_vector = _raw_vector.replace(1, BehaviorCodes.VENTRAL_TURN)
    _raw_vector = _raw_vector.replace(0, BehaviorCodes.NOT_ANNOTATED)
    _raw_vector = _raw_vector.replace(-1, BehaviorCodes.DORSAL_TURN)
    _raw_vector = _raw_vector.replace(np.nan, BehaviorCodes.NOT_ANNOTATED)
    BehaviorCodes.assert_all_are_valid(_raw_vector)
    return _raw_vector


def plot_behavior_syncronized_discrete_states_from_traces(df_traces, neuron_group, neuron_plot, plot_style='bar',
                                                          normalize=True, target_len=100, DEBUG=False):
    """
    Calculates discrete states using neuron_group, then plots the discretized states of neuron_plot

    Parameters
    ----------
    df_traces
    neuron_group
    neuron_plot

    Returns
    -------

    """
    idx_list = ['low', 'rise', 'high', 'fall']

    df = calculate_behavior_syncronized_discrete_states(df_traces, neuron_group, neuron_plot, idx_list, target_len,
                                                        DEBUG)
    df_counts = convert_discrete_state_df_to_counts(df, idx_list, normalize, target_len)

    if plot_style is not None:
        if plot_style == 'imshow':
            # Plot each state separately
            # First, make a plotly figure with subplots
            fig = make_subplots(cols=len(idx_list), rows=1, shared_yaxes=True, vertical_spacing=0.02,
                                subplot_titles=idx_list)
            grouped = df_counts.T.groupby('state')

            for i, key in enumerate(grouped.groups.keys()):
                g = grouped.get_group(key)
                # Add this dataframe as a heatmap
                fig.add_trace(go.Heatmap(z=g.drop(columns='state').T, showscale=False,),
                              row=1, col=i + 1)
            fig.update_yaxes(tickvals=np.arange(len(df_counts.index)), ticktext=df_counts.index,
                             overwrite=True, row=1, col=1, title=neuron_plot)
            fig.update_layout(title=f"Activity of {neuron_plot} seperated by {neuron_group}")
        elif plot_style == 'bar':
            fig = plot_fractional_state_annotations(df_counts, neuron_group, neuron_plot)
        else:
            raise NotImplementedError(f"plot_style {plot_style} not implemented")
        fig.show()
    else:
        fig = None

    return fig, (df, df_counts)


def convert_discrete_state_df_to_counts(df, idx_list=None, normalize=True, target_len=100):
    """Designed to be used with output from calculate_behavior_syncronized_discrete_states"""
    if idx_list is None:
        idx_list = ['low', 'rise', 'high', 'fall']
    df_counts = df.apply(lambda x: x.value_counts()).fillna(0)
    if normalize:
        df_counts = df_counts / df_counts.sum()
    # Add a row for the state of the grouping variable, which is a constant for target_len frames at a time
    idx_row = []
    for idx in idx_list:
        idx_row.extend([idx for _ in range(target_len)])
    df_counts.loc['state', :] = idx_row
    return df_counts


def calculate_behavior_syncronized_discrete_states(df_traces, neuron_group, neuron_plot, idx_list=None, target_len=100,
                                                   DEBUG=False):
    if idx_list is None:
        idx_list = ['low', 'rise', 'high', 'fall']
    # if neuron_group not in df_traces or neuron_plot not in df_traces:
    #     raise NeedsAnnotatedNeuronError(f"neuron_group ({neuron_group}) or neuron_plot ({neuron_plot}) not found")
    # Calculate discrete states ('low', 'rise', 'high', 'fall')
    y_ava = combine_pair_of_ided_neurons(df_traces, neuron_group)
    y_ava = filter_gaussian_moving_average(y_ava, 1)
    beh_ava = calculate_rise_high_fall_low(y_ava, verbose=0, DEBUG=False)
    y_riv = combine_pair_of_ided_neurons(df_traces, neuron_plot)
    y_riv = filter_gaussian_moving_average(y_riv, 1)
    beh_riv = calculate_rise_high_fall_low(y_riv, verbose=0, DEBUG=False)
    if DEBUG:
        # Plot both traces with their discrete states as colors

        df_ava = pd.DataFrame({'y': y_ava.values, 'state': beh_ava.values}).reset_index()
        df_ava['id'] = neuron_group
        df_riv = pd.DataFrame({'y': y_riv.values, 'state': beh_riv.values}).reset_index()
        df_riv['id'] = neuron_plot
        # New column for both states simultaneously
        df_ava['state_combined'] = neuron_group + df_ava['state'] + '_' + neuron_plot + df_riv['state']
        df_riv['state_combined'] = neuron_group + df_ava['state'] + '_' + neuron_plot + df_riv['state']
        # Stack the two dataframes, keeping the local indices
        df = pd.concat([df_ava, df_riv], axis=0, ignore_index=True)
        fig = px.scatter(df, x='index', y='y', color='state', facet_row='id')
        fig.show()
        fig = px.scatter(df, x='index', y='y', color='state_combined', facet_row='id')
        fig.show()
    df_combined = pd.DataFrame({f'beh_{neuron_group}': beh_ava, f'beh_{neuron_plot}': beh_riv})
    # Get the variable length series from each bout
    df_each_bout = get_contiguous_blocks_from_two_columns(df_combined, f'beh_{neuron_group}', f'beh_{neuron_plot}')
    # Resample each bout to be the same length
    func = lambda x: resample_categorical(x, target_len=target_len)
    result_synced = df_each_bout.map(func)
    # Combine into single dataframe
    all_dfs = []
    for idx in idx_list:
        if idx not in result_synced:
            continue
        s = result_synced[idx]
        df = pd.DataFrame.from_dict(dict(zip(s.index, s.values))).T
        all_dfs.append(df.reset_index(drop=True))
    df = pd.concat(all_dfs, axis=1)
    df.columns = np.arange(len(df.columns))
    return df


def plot_fractional_state_annotations(df_counts, neuron_group, neuron_plot):
    """Designed to be used with output from convert_discrete_state_df_to_counts"""
    df_counts_melted = df_counts.T.reset_index().drop(columns='state')
    var_name = f'{neuron_plot}_state'
    df_counts_melted = pd.melt(df_counts_melted, id_vars='index', var_name=var_name,
                               value_name='fraction')
    fig = px.bar(df_counts_melted, x='index', y='fraction', color=var_name, orientation='v')
    # Vertical black lines at the transition points
    x_list = [100, 200, 300]
    for x in x_list:
        fig.add_shape(type='line', x0=x, x1=x, y0=0, y1=1, line=dict(color='black', width=1))
    fig.update_layout(barmode='stack', bargap=0)
    # Update the xticks to use the 'state' column
    x_ticks = np.arange(50, len(df_counts.columns), step=100)
    x_tick_text = df_counts.loc['state', x_ticks]
    fig.update_xaxes(tickvals=x_ticks, ticktext=x_tick_text,
                     overwrite=True, title=f"{neuron_group} state")
    return fig


def approximate_background_using_video(behavior_video, num_frames=1000):
    """
    Approximates the background of a behavior video using the mean of the first 1000 frames

    Should only be used for old projects where a proper background was not measured

    Parameters
    ----------
    behavior_video

    Returns
    -------

    """
    import tifffile

    with tifffile.TiffFile(behavior_video, 'r') as behavior_dat:
        # Get the first 1000 frames
        frames = behavior_dat.asarray(key=np.arange(num_frames))
        # Take the mean
        background = np.mean(frames, axis=0)

    return np.array(background, dtype=frames.dtype)


def save_background_in_project(cfg, **kwargs):
    """

    Parameters
    ----------
    project_cfg

    Returns
    -------

    """

    from wbfm.utils.projects.project_config_classes import ModularProjectConfig
    cfg = ModularProjectConfig(cfg)

    # Get the .btf of the behavioral video
    behavior_video, _ = cfg.get_behavior_raw_file_from_red_fname()
    background = approximate_background_using_video(behavior_video, **kwargs)

    # Save in the raw data background folder
    background_raw_data_folder = cfg.get_folder_with_background()
    # Get subfolder for behavior
    subfolder = [f for f in os.listdir(background_raw_data_folder) if f.endswith('-BH')][0]
    fname = os.path.join(background_raw_data_folder, subfolder, 'AVG_approximate_background_Ch0-BHbigtiff.btf')

    # Save (btf)
    print(f"Saving background to {fname} with dtype {background.dtype}")
    import tifffile
    tifffile.imwrite(fname, background)

    return background


def convert_starts_and_ends_to_behavior_vector(csv_fname, num_frames, min_duration=0, DEBUG=False):
    """
    Converts starts and ends (read from a csv) to a binary vector, which is usually forward or reverse

    Parameters
    ----------
    csv_fname
    num_frames
    DEBUG

    Returns
    -------

    """
    df = pd.read_csv(csv_fname)
    print(df)
    all_starts = df['start'].values
    all_ends = df['end'].values
    
    state_trace = np.zeros(num_frames)
    for start, end in zip(all_starts, all_ends):
        if end - start < min_duration:
            continue

        state_trace[start:end] = 1
    
    # Write it back as a new csv
    new_fname = csv_fname.replace('.csv', f'_binary_vector.csv')
    pd.DataFrame(state_trace).to_csv(new_fname, index=True)
