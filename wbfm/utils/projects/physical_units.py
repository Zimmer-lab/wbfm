import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from wbfm.utils.external.custom_errors import IncompleteConfigFileError


@dataclass
class PhysicalUnitConversion:
    """Converts from pixels to micrometers, but also to Leifer-specific scaling"""

    zimmer_fluroscence_um_per_pixel_xy: float = 0.325
    zimmer_behavior_um_per_pixel_xy: float = 2.4
    zimmer_um_per_pixel_z: float = 1.5
    zimmer_um_per_pixel_z_neuropal: float = 0.2

    leifer_um_per_unit: float = 84

    volumes_per_second: float = None
    exposure_time: int = 12  # Only used if volumes_per_second is not specified

    num_z_slices: int = None
    num_flyback_planes_discarded: int = None  # This should give an error if not properly set

    @property
    def frames_per_second(self):
        return self.volumes_per_second * self.frames_per_volume

    @property
    def frames_per_volume(self):
        if self.num_flyback_planes_discarded is None:
            raise IncompleteConfigFileError("num_flyback_planes_discarded not found; "
                                            "cannot calculate frames_per_volume")
        return self.num_z_slices + self.num_flyback_planes_discarded

    @property
    def time_delta_frame(self):
        return 1 / self.frames_per_second

    @property
    def time_delta_volume(self):
        return 1 / self.volumes_per_second

    @property
    def z_to_xy_ratio(self):
        return self.zimmer_um_per_pixel_z / self.zimmer_fluroscence_um_per_pixel_xy

    @property
    def grid_spacing(self):
        return np.array([self.zimmer_fluroscence_um_per_pixel_xy, self.zimmer_fluroscence_um_per_pixel_xy,
                         self.zimmer_um_per_pixel_z])

    def zimmer2physical_fluorescence(self, vol0_zxy: np.ndarray) -> np.ndarray:
        """
        Assumes that z is the 0th column, and x/y are 1, 2

        Parameters
        ----------
        vol0_zxy - shape is N x 3, where N=number of neurons (or objects)

        Returns
        -------
        zxy_in_phyical - shape is same as input

        """
        # xy, then z
        xy_in_um = vol0_zxy[:, [1, 2]] * self.zimmer_fluroscence_um_per_pixel_xy
        xy_in_physical = xy_in_um

        z_in_um = vol0_zxy[:, [0]] * self.zimmer_um_per_pixel_z
        z_in_physical = z_in_um

        zxy_in_phyical = np.hstack([z_in_physical, xy_in_physical])

        return zxy_in_phyical
    
    def zimmer2physical_final_tracks(self, df_tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the tracks from zimmer coordinates to physical coordinates.
        Assumes that the DataFrame has multiindex columns; top level is the neuron, and second level contains (at least) 'z', 'x', 'y' in pixel space.

        Parameters
        ----------
        df_tracks : pd.DataFrame
            DataFrame containing tracks with columns 'z', 'x', 'y'.

        Returns
        -------
        pd.DataFrame
            DataFrame with converted coordinates in physical units.
        """
        # Convert z, x, y columns to physical units
        df_tracks = df_tracks.copy()
        df_tracks.loc[:, (slice(None), 'z')] *= self.zimmer_um_per_pixel_z
        df_tracks.loc[:, (slice(None), 'x')] *= self.zimmer_fluroscence_um_per_pixel_xy
        df_tracks.loc[:, (slice(None), 'y')] *= self.zimmer_fluroscence_um_per_pixel_xy

        return df_tracks

    def zimmer2physical_fluorescence_single_column(self, dat0: np.ndarray, which_col=0) -> np.ndarray:
        """Converts just a single column, in place"""

        dat0[:, which_col] *= self.z_to_xy_ratio
        return dat0

    def zimmer2physical_behavior(self, frame0_xy: np.ndarray) -> np.ndarray:
        # Assume no z coordinate
        xy_in_um = frame0_xy * self.zimmer_behavior_um_per_pixel_xy

        return xy_in_um

    def zimmer2leifer(self, vol0_zxy: np.ndarray) -> np.ndarray:
        """ Target: 1 unit = 84 um, and xyz from zxy"""

        # xy, then z
        xy_in_um = vol0_zxy[:, [1, 2]] * self.zimmer_fluroscence_um_per_pixel_xy
        xy_in_leifer = xy_in_um / self.leifer_um_per_unit

        z_in_um = vol0_zxy[:, [0]] * self.zimmer_um_per_pixel_z
        z_in_leifer = z_in_um / self.leifer_um_per_unit

        zxy_in_leifer = np.hstack([z_in_leifer, xy_in_leifer])
        xyz_in_leifer = zxy_in_leifer[:, [2, 1, 0]]

        xyz_in_leifer -= np.mean(xyz_in_leifer, axis=0)

        return xyz_in_leifer

    def leifer2zimmer(self, vol0_xyz_leifer: np.ndarray) -> np.ndarray:
        """Tries to invert zimmer2leifer, but does not know the original mean value"""

        # xy, then z
        xy_in_um = vol0_xyz_leifer[:, [0, 1]] * self.leifer_um_per_unit
        xy_in_zimmer = xy_in_um / self.zimmer_fluroscence_um_per_pixel_xy

        z_in_um = vol0_xyz_leifer[:, [2]] * self.leifer_um_per_unit
        z_in_zimmer = z_in_um / self.zimmer_um_per_pixel_z

        zxy_in_zimmer = np.hstack([z_in_zimmer, xy_in_zimmer])
        xyz_in_zimmer = zxy_in_zimmer[:, [2, 1, 0]]

        xyz_in_zimmer -= np.min(xyz_in_zimmer, axis=0)

        return xyz_in_zimmer

    @staticmethod
    def load_from_config(project_cfg, DEBUG=False):

        from wbfm.utils.general.postures.centerline_classes import get_behavior_fluorescence_fps_conversion
        # First, load from the main project config file
        if 'physical_units' in project_cfg.config:
            if DEBUG:
                print("Using physical unit conversions from project config")
            # Main units
            opt = project_cfg.config['physical_units'].copy()
            if 'volumes_per_second' not in opt:
                project_cfg.logger.debug("Using hard coded camera fps; this depends on the exposure time")
                camera_fps = opt.get('camera_fps', 1000)
                if 'exposure_time' not in opt:
                    logging.debug("exposure_time not found in physical_units or project config; using default")
                exposure_time = opt.get('exposure_time', 12)
                frames_per_volume = get_behavior_fluorescence_fps_conversion(project_cfg)
                opt['volumes_per_second'] = camera_fps / exposure_time / frames_per_volume
                if DEBUG:
                    print(f"Calculated volumes_per_second: {opt['volumes_per_second']} "
                          f"from camera_fps: {camera_fps}, exposure_time: {exposure_time}, "
                          f"frames_per_volume: {frames_per_volume}")
            else:
                if DEBUG:
                    print(f"Using volumes_per_second: {opt['volumes_per_second']} from project config")
            # Additional dataset unit
            num_slices = project_cfg.get_num_slices_robust()
            if num_slices is not None:
                opt['num_z_slices'] = num_slices
            else:
                # This is a very old parameter, and should be in all projects
                raise ValueError("num_slices not found in dataset_params")
        else:
            project_cfg.logger.warning("Using default physical unit conversions")
            opt = dict()

        # Second, load from the raw data config file (only needed for flyback removal, i.e. data that isn't included)
        raw_data_cfg = project_cfg.get_raw_data_config()
        if not raw_data_cfg.has_valid_self_path:
            opt['num_flyback_planes_discarded'] = 0
            logging.debug("No raw data config found; assuming no flyback planes discarded")
        elif not raw_data_cfg.config.get('flyback_saved', False):
            num_flyback_planes_discarded = raw_data_cfg.config.get('num_flyback_planes_discarded', None)
            if num_flyback_planes_discarded is None:
                raise IncompleteConfigFileError("num_flyback_planes_discarded not found in raw data config")
            opt['num_flyback_planes_discarded'] = num_flyback_planes_discarded

        return PhysicalUnitConversion(**opt)
