from dataclasses import dataclass
from functools import cached_property
from typing import Optional, List

import pandas as pd
import zarr
from dask import array as da
from imutils import MicroscopeDataReader

from wbfm.utils.external.custom_errors import IncompleteConfigFileError
from wbfm.utils.general.utils_filenames import load_file_according_to_precedence, read_if_exists
from wbfm.utils.projects.project_config_classes import SubfolderConfigFile
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons


@dataclass
class NeuropalManager:

    config: Optional[SubfolderConfigFile] = None

    df_id_fname: str = None

    @cached_property
    def data(self) -> Optional[da.Array]:
        """
        Data from the neuropal dataset, if present
        """
        if self.config is None:
            return None
        # Should always exist if the config exists
        neuropal_path = self.config.resolve_relative_path_from_config('neuropal_data_path')
        neuropal_data = MicroscopeDataReader(neuropal_path)
        return da.squeeze(neuropal_data.dask_array)

    @cached_property
    def segmentation(self) -> Optional[da.Array]:
        """
        Segmentation from the neuropal dataset, if present
        """
        if self.config is None:
            return None
        # May not exist if the segmentation step has not been run
        neuropal_path = self.config.resolve_relative_path_from_config('neuropal_segmentation_path')
        if neuropal_path is None:
            return None
        neuropal_segmentation = zarr.open(neuropal_path)
        return neuropal_segmentation

    @cached_property
    def segmentation_metadata(self) -> Optional[DetectedNeurons]:
        """
        Metadata from the neuropal segmentation

        Returns
        -------

        """
        if self.config is None:
            return None
        # May not exist if the segmentation step has not been run
        neuropal_path = self.config.resolve_relative_path_from_config('segmentation_metadata_path')
        if neuropal_path is None:
            return None
        neuropal_segmentation_metadata = DetectedNeurons(neuropal_path)
        return neuropal_segmentation_metadata

    @property
    def has_complete_neuropal(self):
        return self.data is not None and self.segmentation is not None
    
    @property
    def segmentation_succeeded(self):
        return self.has_complete_neuropal and self.segmentation_metadata.get_all_neuron_metadata_for_single_time(0, as_dataframe=True, use_mean_intensity=True) is not None
    
    @property
    def neuron_names(self) -> List[str]:
        """Get the names from the segmentation metadata"""
        df = self.segmentation_metadata.get_all_neuron_metadata_for_single_time(0, as_dataframe=True)
        # Get the top level of the multiindex
        return df.index.get_level_values(0).unique().tolist()

    @cached_property
    def df_ids(self) -> Optional[pd.DataFrame]:
        """
        Load a dataframe corresponding to manual iding based on the neuropal stacks, if they exist

        This will not exist for every project

        """
        if self.config is None:
            return None
        # Manual annotations take precedence by default
        excel_fname = self.get_default_manual_annotation_fname()
        try:
            possible_fnames = dict(excel=excel_fname)
        except ValueError:
            self.df_id_fname = ''
            return None
        possible_fnames = {k: str(v) for k, v in possible_fnames.items()}
        fname_precedence = ['newest']
        try:
            df_neuropal_id, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                      reader_func=read_if_exists,
                                                                      na_filter=False)
        except ValueError:
            # Then the file was corrupted... try to load from the h5 file (don't worry about other file types)
            self.logger.warning(f"Found corrupted neuropal annotation file ({excel_fname}), "
                                f"with no backup h5 file; returning None")
            fname = ''
            df_neuropal_id = None
        self.df_id_fname = fname
        return df_neuropal_id

    def get_default_manual_annotation_fname(self) -> Optional[str]:
        if self.config is None:
            raise IncompleteConfigFileError("No project config found; cannot load or save manual annotations")
        excel_fname = self.config.resolve_relative_path("manual_annotation.xlsx", prepend_subfolder=True)
        return excel_fname
