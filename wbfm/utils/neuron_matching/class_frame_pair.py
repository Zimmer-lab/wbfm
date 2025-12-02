import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
import scipy.ndimage as ndi
import cv2
import numpy as np
import pandas as pd
from wbfm.utils.visualization.utils_napari import napari_tracks_from_match_list
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons
from wbfm.utils.external.utils_cv2 import cast_matches_as_array
from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame
from wbfm.utils.external.custom_errors import NoMatchesError, AnalysisOutOfOrderError
from wbfm.utils.neuron_matching.utils_affine import calc_matches_using_affine_propagation
from wbfm.utils.general.utils_features import match_known_features, build_features_and_match_2volumes
from wbfm.utils.neuron_matching.utils_gaussian_process import calc_matches_using_gaussian_process
from wbfm.utils.general.utils_networkx import calc_bipartite_from_candidates
from wbfm.utils.general.distance_functions import dist2conf
from wbfm.utils.nn_utils.data_formatting import flatten_nested_list
from wbfm.utils.projects.physical_units import PhysicalUnitConversion
from wbfm.utils.projects.project_config_classes import ModularProjectConfig


@dataclass
class FramePairOptions:
    """Options for matching neurons between two frames; See FramePair for the actual matching process"""
    # Flag and options for each method
    # First: default feature-embedding method
    embedding_matches_to_keep: float = 1.0
    embedding_use_GMS: bool = False
    crossCheck: bool = True

    # Second: local affine method
    add_affine_to_candidates: bool = False
    start_plane: int = 4
    num_features_per_plane: int = 10000
    affine_matches_to_keep: float = 0.8
    affine_use_GMS: bool = True
    min_matches: int = 20
    allow_z_change: bool = False
    affine_num_candidates: int = 1
    affine_generate_additional_keypoints: bool = False

    # Third: gaussian process method
    add_gp_to_candidates: bool = False
    starting_matches: str = 'affine_matches'
    gp_num_candidates: int = 1

    # Fourth: neural network (transformer)
    add_fdnc_to_candidates: bool = False
    fdnc_options: dict = None

    # For filtering / postprocessing the matches
    matching_method: str = 'bipartite'
    z_threshold: float = None
    min_confidence: float = 0.001
    z_to_xy_ratio: float = None  # Deprecated; will be removed after Ulises projects
    apply_tanh_to_confidence: bool = True

    # These are mutually exclusive
    use_superglue: bool = True

    # Physical unit conversion; required for leifer network
    physical_unit_conversion: PhysicalUnitConversion = None

    # New: rotation of the entire image as preprocessing
    preprocess_using_global_rotation: bool = False

    _already_warned = False  # To avoid spamming installation warnings

    def __post_init__(self):
        # All of this is for the deprecated fdnc method... will be removed
        try:
            from wbfm.utils.nn_utils.fdnc_predict import load_fdnc_options
            default_options = load_fdnc_options()
        except ImportError:
            if not self._already_warned:
                logging.warning("fDNC is not installed. Skipping prediction using this method")
                self._already_warned = True
            default_options = {}

        if self.fdnc_options is None:
            self.fdnc_options = {}
        else:
            default_options.update(self.fdnc_options)
        self.fdnc_options = default_options

    @staticmethod
    def load_from_config_file(cfg: ModularProjectConfig, training_config=None):
        if training_config is None:
            training_config = cfg.get_training_config()
        pairwise_matches_params = training_config.config['pairwise_matching_params'].copy()
        frame_pair_options = FramePairOptions(**pairwise_matches_params)

        # Check that the network used to create the embeddings is consistent
        if training_config.config['tracker_params'].get('use_barlow_network', False) and frame_pair_options.use_superglue:
            frame_pair_options.use_superglue = False
            logging.warning("use_superglue was set to True, but is inconsistent with the feature space of the Frames (use_barlow_network=True); setting use_superglue=False")

        physical_unit_conversion = PhysicalUnitConversion.load_from_config(cfg)
        frame_pair_options.physical_unit_conversion = physical_unit_conversion

        return frame_pair_options


@dataclass
class FramePair:
    """
    Information connecting neurons in two ReferenceFrame objects

    Also implements an ensemble of methods to match neurons between the two frames:
    - Feature matching using opencv feature embedding
    - Local affine matching using opencv (performance constrained by feature matching)
    - Gaussian process matching (performance constrained by feature matching)
    - Neural network matching (superglue and fDNC)

    """
    options: FramePairOptions = None

    # Final output, with confidences
    final_matches: list = None

    # Intermediate products, with confidences
    feature_matches: list = None
    affine_matches: list = None
    affine_pushed_locations: list = None
    gp_matches: list = None
    gp_pushed_locations: list = None

    all_gps: list = None  # The actual gaussian processes; may get warning from scipy versioning

    # New method:
    fdnc_matches: list = None

    # Original keypoints
    keypoint_matches: list = None

    # Frame classes
    frame0: ReferenceFrame = None
    frame1: ReferenceFrame = None

    # For global rigid pre-rotation
    rigid_rotation_matrix: np.ndarray = None
    _dat0_preprocessed: np.ndarray = None
    _pts0_preprocessed: np.ndarray = None
    _dat0: np.ndarray = None
    _dat1: np.ndarray = None

    @property
    def all_candidate_matches(self) -> list:
        if self.feature_matches is not None:
            all_matches = self.feature_matches.copy()
        else:
            all_matches = []
        if self.options.add_affine_to_candidates:
            if self.affine_matches is not None:
                all_matches.extend(self.affine_matches)
        if self.options.add_gp_to_candidates:
            if self.gp_matches is not None:
                all_matches.extend(self.gp_matches)
        if self.options.add_fdnc_to_candidates:
            if self.fdnc_matches is not None:
                all_matches.extend(self.fdnc_matches)
        return all_matches

    @property
    def num_possible_matches(self) -> int:
        return num_possible_matches_between_two_frames(self.frame0, self.frame1)

    @property
    def dat0(self):
        if self._dat0 is None:
            self._dat0 = self.frame0.get_raw_data()
        return self._dat0

    @property
    def dat1(self):
        if self._dat1 is None:
            self._dat1 = self.frame1.get_raw_data()
        return self._dat1

    @property
    def pts0(self):
        return self.frame0.neuron_locs

    @property
    def pts1(self):
        return self.frame1.neuron_locs

    @property
    def dat0_preprocessed(self):
        # Returns most-updated version
        if self._dat0_preprocessed is None:
            return self.dat0
        else:
            return self._dat0_preprocessed

    @property
    def pts0_preprocessed(self):
        # Returns most-updated version
        if self._pts0_preprocessed is None:
            return np.array(self.pts0)
        else:
            return self._pts0_preprocessed

    @property
    def vector_field(self):
        pts0 = self.pts0_preprocessed
        pts1 = self.pts1
        vec_field = []
        for m in self.final_matches:
            vec_field.append(pts1[m[1]] - pts0[m[0]])
        return np.array(vec_field), pts0, pts1

    def check_both_frames_valid(self):
        """
        For now, just check the number of neurons detected in the frames.
        Note that if only one neuron is detected, it will lead to size errors
        """
        is_valid = True
        if self.num_possible_matches <= 1 or np.isnan(self.num_possible_matches):
            is_valid = False
        return is_valid

    def load_raw_data(self, dat0=None, dat1=None):
        """Loads raw data into subclasses (Reference Frames)"""
        if dat0 is None:
            _ = self.dat0
        else:
            self.frame0._raw_data = dat0
        if dat1 is None:
            _ = self.dat1
        else:
            self.frame1._raw_data = dat1

    def preprocess_data(self, force_rotation=False):
        """Preprocesses the volumetric data, if applicable options are True"""
        if force_rotation or self.options.preprocess_using_global_rotation:
            self._dat0_preprocessed, _ = self.rigidly_align_volumetric_images()
            self._pts0_preprocessed, _, _ = self.rigidly_align_point_clouds()
        else:
            # No other options at this time; leave the entries as None
            pass

    def __getstate__(self):
        # Modify pickling, following:
        # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        # Also similar to this solution: https://www.ianlewis.org/en/pickling-objects-cached-properties
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if '_dat0' in state:
            del state['_dat0']
        if '_dat1' in state:
            del state['_dat1']
        if '_dat0_preprocessed' in state:
            del state['_dat0_preprocessed']
        return state

    def prep_for_pickle(self):
        self.frame0.prep_for_pickle()
        self.frame1.prep_for_pickle()

    def rebuild_keypoints(self):
        self.frame0.rebuild_keypoints()
        self.frame1.rebuild_keypoints()

    def calc_final_matches(self, method=None, **kwargs):
        if method is None:
            method = self.options.matching_method
        if method == 'bipartite':
            return self.calc_final_matches_using_bipartite_matching(**kwargs)
        elif method == 'unanimous':
            return self.calc_final_matches_using_unanimous_voting(**kwargs)
        else:
            raise NotImplementedError

    def calc_final_matches_using_bipartite_matching(self, min_confidence: float = None,
                                                    z_threshold=None) -> list:
        if len(self.all_candidate_matches) == 0:
            return []
        z_threshold, min_confidence = self.use_defaults_if_none(min_confidence, z_threshold)

        try:
            matches, conf, _ = calc_bipartite_from_candidates(self.all_candidate_matches,
                                                              min_confidence_after_sum=min_confidence,
                                                              apply_tanh_to_confidence=self.options.apply_tanh_to_confidence)
            final_matches = [(m[0], m[1], c) for m, c in zip(matches, conf)]
            final_matches = self.filter_matches_using_z_threshold(final_matches, z_threshold)
        except NoMatchesError:
            final_matches = []
        self.final_matches = final_matches
        return final_matches

    def calc_final_matches_using_unanimous_voting(self, min_confidence: float = None,
                                                  z_threshold=None) -> list:
        if len(self.all_candidate_matches) == 0:
            return []
        z_threshold, min_confidence = self.use_defaults_if_none(min_confidence, z_threshold)

        candidates = self.all_candidate_matches
        candidates = [c for c in candidates if c[2] > min_confidence]
        candidates = self.filter_matches_using_z_threshold(candidates, z_threshold)

        match_dict = defaultdict(list)
        conf_dict = defaultdict(list)
        for c in candidates:
            if match_dict[c[0]]:
                # Evaluates false if None or empty
                if match_dict[c[0]] == c[1]:
                    # Was the same match
                    conf_dict[(c[0], c[1])].append(c[2])
                else:
                    # Must remove the match from both dictionaries
                    match_dict[c[0]] = None
                    conf_dict[(c[0], c[1])] = None
            elif match_dict[c[0]] is None:
                continue
            else:
                # Is empty (not yet matched)
                match_dict[c[0]] = c[1]
                conf_dict[(c[0], c[1])].append(c[2])

        # conf_dict = {k: np.tanh(v) for k, v in conf_dict.items()}
        final_matches = [[k0, k1, np.mean(v)] for (k0, k1), v in conf_dict.items() if v is not None]
        # Use bipartite matching to remove overmatching
        matches, conf, _ = calc_bipartite_from_candidates(final_matches)
        final_matches = [(m[0], m[1], c) for m, c in zip(matches, conf)]

        self.final_matches = final_matches
        return final_matches

    def use_defaults_if_none(self, min_confidence, z_threshold):
        if min_confidence is None:
            min_confidence = self.options.min_confidence
        else:
            self.options.min_confidence = min_confidence
        if z_threshold is None:
            z_threshold = self.options.z_threshold
        else:
            self.options.z_threshold = z_threshold
        return z_threshold, min_confidence

    def get_f0_to_f1_dict(self, matches=None):
        if matches is None:
            matches = self.final_matches
        return {n0: n1 for n0, n1, _ in matches}

    def get_f1_to_f0_dict(self, matches=None):
        if matches is None:
            matches = self.final_matches
        return {n1: n0 for n0, n1, _ in matches}

    def get_pair_to_conf_dict(self, matches=None):
        if matches is None:
            matches = self.final_matches
        return {(n0, n1): c for n0, n1, c in matches}

    def get_metadata_dict(self) -> FramePairOptions:
        return self.options

    def save_matches_as_excel(self, target_dir='.'):
        f0_ind = self.frame0.frame_ind
        f1_ind = self.frame1.frame_ind
        df = pd.DataFrame(self.final_matches, columns=[f'Neuron_in_f{f0_ind}', f'Neuron_in_f{f1_ind}', 'Confidence'])
        fname = f'matches_f{f0_ind}_f{f1_ind}.xlsx'
        fname = os.path.join(target_dir, fname)
        df.to_excel(fname, index=False)

    def filter_matches_using_z_threshold(self, matches, z_threshold) -> list:
        if z_threshold is None:
            return matches
        n0 = self.pts0.copy()
        n1 = self.pts1.copy()

        def _delta_z(m):
            return np.abs(n0[m[0]][0] - n1[m[1]][0])

        return [m for m in matches if _delta_z(m) < z_threshold]

    def modify_confidences_using_image_features(self, metadata: DetectedNeurons, gamma=1.0, mode='brightness'):
        # Get brightness... this object doesn't know the object, because it is full-video information
        i0, i1 = self.frame0.frame_ind, self.frame1.frame_ind
        if mode == 'brightness':
            x0, x1 = metadata.get_normalized_intensity(i0), metadata.get_normalized_intensity(i1)
        elif mode == 'volume':
            x0, x1 = metadata.get_all_volumes(i0), metadata.get_all_volumes(i1)
        else:
            raise ValueError

        # Per match, calculate similarity score based on delta
        matches = self.final_matches
        distances = [x0[m[0]] - x1[m[1]] for m in matches]
        multipliers = dist2conf(distances, gamma)

        # Multiply the original confidence
        matches[:, 2] *= multipliers

        return matches

    def matched_neurons_as_point_clouds(self):
        """Returns 2 numpy arrays of zxy point clouds, aligned as matched by final_matches"""
        pts0, pts1 = [], []
        n0, n1 = self.pts0, self.pts1
        for m in self.final_matches:
            pts0.append(n0[m[0]])
            pts1.append(n1[m[1]])

        pts0, pts1 = np.array(pts0), np.array(pts1)
        return pts0, pts1

    def napari_tracks_of_matches(self, list_of_matches=None):
        if list_of_matches is None:
            list_of_matches = self.final_matches
        n0_zxy = self.pts0
        n1_zxy = self.pts1

        tracks = napari_tracks_from_match_list(list_of_matches, n0_zxy, n1_zxy, t0=self.frame0.frame_ind)
        return tracks

    def calc_or_get_alignment_between_matched_neurons(self, pts0=None, pts1=None, recalculate_alignment=True):

        if recalculate_alignment:
            if pts0 is None:
                pts0, pts1 = self.matched_neurons_as_point_clouds()
            val, h, inliers = cv2.estimateAffine3D(pts0, pts1, confidence=0.999)
            self.rigid_rotation_matrix = h

        else:
            h = self.rigid_rotation_matrix

        if h is None:
            raise AnalysisOutOfOrderError("match.align_point_clouds(recalculate_alignment=True)")

        return h

    def rigidly_align_point_clouds(self, index_to_align=0, recalculate_alignment=True) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if index_to_align == 0:
            raw_cloud = self.pts0
            target_cloud = self.pts1
        else:
            raise NotImplementedError
        h = self.calc_or_get_alignment_between_matched_neurons(recalculate_alignment=recalculate_alignment)

        transformed_cloud = cv2.transform(np.array([raw_cloud]), h)[0]

        return np.array(transformed_cloud), np.array(target_cloud), np.array(raw_cloud)

    def rigidly_align_volumetric_images(self, volume0=None, recalculate_alignment=True) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Aligns the entire volume using the (possibly non-final) neuron matches

        Returns the 3d Affine transformed version of the volume0, which can be passed
        Also returns the intermediate products (rotation) and the raw
        """
        from napari.utils.transforms import Affine
        h = self.calc_or_get_alignment_between_matched_neurons(recalculate_alignment=recalculate_alignment)
        napari_affine = Affine(affine_matrix=np.vstack([h, [0, 0, 0, 1]]))

        if volume0 is None:
            volume0 = self.dat0

        # For some reason this doesn't work directly
        # volume0_rotated = napari_affine.func(volume0)
        volume0_rotated = ndi.affine_transform(volume0,
                                               np.linalg.inv(napari_affine.affine_matrix),
                                               output_shape=volume0.shape, order=5)

        return volume0_rotated, volume0

    def print_candidates_by_method(self):
        """Prints the total number of candidates, but not the exact matches"""
        num_matches = len(self.feature_matches)
        print(f"Found {num_matches} candidates via feature matching")
        if self.affine_matches is not None:
            num_matches = len(self.affine_matches)
            print(f"Found {num_matches} candidates via affine matching")
        if self.gp_matches is not None:
            num_matches = len(self.gp_matches)
            print(f"Found {num_matches} candidates via gaussian process matching")

        num_matches = len(self.final_matches)
        print(f"Processed these into {num_matches} final matches candidates")

    def print_candidates_for_neuron(self, i_neuron, i_frame=0):
        for m in self.all_candidate_matches:
            if m[i_frame] == i_neuron:
                print(f"Candidate: {m}")

    def print_reason_for_match(self, test_match):
        f0_to_1 = self.get_f0_to_f1_dict()

        if np.isscalar(test_match):
            # Assume the user gave the first time point ID
            m0 = test_match
            m1 = f0_to_1.get(m0, None)
            if m1 is None:
                print(f"Found no match for {m0}")
                return
            test_match = (m0, m1)
        else:
            m0, m1 = test_match

        all_match_types = [
            (self.feature_matches, "feature"),
            (self.affine_matches, "affine"),
            (self.gp_matches, "gaussian process"),
            (self.fdnc_matches, "fdnc (neural network)"),
        ]

        if f0_to_1[m0] == m1:
            f_to_conf = self.get_pair_to_conf_dict()
            print(f"Found match {test_match} with confidence {f_to_conf[test_match]}")

            for match_type in all_match_types:
                if match_type[0] is None:
                    continue
                self.print_match_by_method(match_type[0], m0, m1, match_type[1])

    def print_match_by_method(self, this_method_matches, m0, m1, method_name):
        aff_dict = self.get_f0_to_f1_dict(this_method_matches)
        if m0 in aff_dict:
            if aff_dict[m0] == m1:
                conf = self.get_pair_to_conf_dict(this_method_matches)[(m0, m1)]
                print(f"Same match as {method_name} method with confidence: {conf}")
            else:
                conf = self.get_pair_to_conf_dict(this_method_matches)[(m0, aff_dict[m0])]
                print(f"Different match than {method_name} method: {aff_dict[m0]} with confidence: {conf}")
        else:
            print(f"Neuron {m0} not matched using {method_name} method")

    def match_using_feature_embedding(self):
        logging.warning("DEPRECATION WARNING: Using old opencv based feature matching code. Should use FeatureSpaceTemplateMatcher instead")
        # Default method; always call this
        obj = self.options
        opt = dict(matches_to_keep=obj.embedding_matches_to_keep,
                   use_GMS=obj.embedding_use_GMS,
                   crossCheck=obj.crossCheck)
        self._match_using_feature_embedding(**opt)

    def _match_using_feature_embedding(self, matches_to_keep=1.0, use_GMS=False, crossCheck=True):
        """
        Requires the frame objects to have been correctly initialized, i.e. their neurons need a feature embedding

        Uses direct brute force matching to match the neurons given these embeddings, then optionally postprocesses using GMS to
        make sure they reflect a locally consistent motion field
        """
        frame0, frame1 = self.frame0, self.frame1
        # New: Might pre-align the features using rigid rotation
        #   This should not affect the core matches, but may strongly affect the GMS postprocessing
        pts0, pts1 = self.pts0_preprocessed, self.pts1
        # First, get feature matches
        neuron_embedding_matches = match_known_features(frame0.all_features,
                                                        frame1.all_features,
                                                        pts0,
                                                        pts1,
                                                        frame0.vol_shape[1:],
                                                        frame1.vol_shape[1:],
                                                        matches_to_keep=matches_to_keep,
                                                        use_GMS=use_GMS,
                                                        crossCheck=crossCheck)
        # With neuron embeddings, the keypoints are the neurons
        neuron_embedding_matches_with_conf = cast_matches_as_array(neuron_embedding_matches, gamma=1.0)
        self.feature_matches = neuron_embedding_matches_with_conf
        self.keypoint_matches = neuron_embedding_matches_with_conf  # Overwritten by affine match, if used

    def match_using_local_affine(self):
        if self.options.add_affine_to_candidates:
            obj = self.options
            opt = dict(start_plane=obj.start_plane,
                       num_features_per_plane=obj.num_features_per_plane,
                       matches_to_keep=obj.affine_matches_to_keep,
                       use_GMS=obj.affine_matches_to_keep,
                       min_matches=obj.min_matches,
                       allow_z_change=obj.allow_z_change,
                       num_candidates=obj.affine_num_candidates,
                       generate_additional_keypoints=obj.affine_generate_additional_keypoints)
            try:
                self._match_using_local_affine(**opt)
            except NoMatchesError:
                # Probably just a low quality image, no major problem
                return

    def _match_using_local_affine(self, start_plane,
                                  num_features_per_plane,
                                  matches_to_keep,
                                  use_GMS,
                                  min_matches,
                                  allow_z_change,
                                  num_candidates,
                                  generate_additional_keypoints):
        frame0, frame1 = self.frame0, self.frame1
        if generate_additional_keypoints:
            # Generate keypoints and match per slice
            # New: dat0 may be rigidly rotated to align with dat1
            # dat0, dat1 = self.dat0_preprocessed, self.dat1
            dat0, dat1 = frame0.get_uint8_data(), frame1.get_uint8_data()
            # Transpose because opencv needs it
            dat0 = np.transpose(dat0, axes=(0, 2, 1))
            dat1 = np.transpose(dat1, axes=(0, 2, 1))
            opt = dict(start_plane=start_plane,
                       num_features_per_plane=num_features_per_plane,
                       matches_to_keep=matches_to_keep,
                       use_GMS=use_GMS,
                       detect_new_keypoints=False,
                       kp0_zxy=self.frame0.keypoint_locs,
                       kp1_zxy=self.frame1.keypoint_locs)
            kp0_locs, kp1_locs, all_kp0, all_kp1, kp_matches, all_match_offsets = \
                build_features_and_match_2volumes(dat0, dat1, **opt)
            # Save intermediate data in objects
            frame0.keypoint_locs = kp0_locs
            frame1.keypoint_locs = kp1_locs
            frame0.keypoints = all_kp0
            frame1.keypoints = all_kp1
            # kp_matches = recursive_cast_matches_as_array(kp_matches, all_match_offsets, gamma=1.0)
            kp_matches = [(i, i, 1.0) for i in range(len(frame0.keypoint_locs))]
            self.keypoint_matches = kp_matches
        else:
            self.keypoint_matches = self.feature_matches
        # Then match using distance from neuron position to keypoint cloud
        options = {'all_feature_matches': self.keypoint_matches,
                   'min_matches': min_matches,
                   'allow_z_change': allow_z_change,
                   'num_candidates': num_candidates}
        affine_matches, _, affine_pushed = calc_matches_using_affine_propagation(frame0, frame1, **options)
        # Above code requires that the keypoint_locs are actually the full keypoints
        # frame0.keypoint_locs = kp0_locs
        # frame1.keypoint_locs = kp1_locs
        self.affine_matches = affine_matches
        self.affine_pushed_locations = affine_pushed

    def match_using_gp(self):
        self._match_using_gp(self.options.gp_num_candidates, self.options.starting_matches)

    def _match_using_gp(self, n_neighbors, starting_matches_name='best'):
        # err
        if starting_matches_name == 'best':
            if self.affine_matches is None:
                starting_matches_name = 'feature_matches'
            elif self.feature_matches is None:
                starting_matches_name = 'affine_matches'
            elif len(self.affine_matches) > len(self.feature_matches):
                starting_matches_name = 'affine_matches'
            else:
                starting_matches_name = 'feature_matches'
        if starting_matches_name in ['affine_matches', 'feature_matches', 'final_matches']:
            starting_matches = getattr(self, starting_matches_name)
        else:
            raise ValueError(f"Unknown starting matches: {starting_matches_name}")

        # Can start with any matched point clouds, but not more than ~100 matches otherwise it's way too slow
        # New: n0 may be rigidly prealigned
        n0, n1 = self.pts0_preprocessed, self.pts1.copy()
        n0 = self.options.physical_unit_conversion.zimmer2physical_fluorescence(n0)
        n1 = self.options.physical_unit_conversion.zimmer2physical_fluorescence(n1)
        # Actually match
        options = {'matches_with_conf': starting_matches, 'n_neighbors': n_neighbors}
        gp_matches, all_gps, gp_pushed = calc_matches_using_gaussian_process(n0, n1, **options)
        self.gp_matches = gp_matches
        self.all_gps = all_gps
        self.gp_pushed_locations = gp_pushed

    def match_using_fdnc(self):
        self._match_using_fdnc(self.options.fdnc_options)

    def _match_using_fdnc(self, prediction_options):
        try:
            from fDNC.src.DNC_predict import predict_matches
        except ImportError:
            logging.warning("fDNC is not installed. Skipping prediction using this method")
            self.options._already_warned = True
        # New: n0 may be rigidly prealigned
        n0, n1 = self.pts0_preprocessed, self.pts1.copy()
        template_pos = self.options.physical_unit_conversion.zimmer2leifer(np.array(n0))
        test_pos = self.options.physical_unit_conversion.zimmer2leifer(np.array(n1))

        _, matches_with_conf = predict_matches(test_pos=test_pos, template_pos=template_pos, **prediction_options)
        if prediction_options['topn'] is not None:
            matches_with_conf = flatten_nested_list(matches_with_conf)
        self.fdnc_matches = matches_with_conf

    def match_using_all_methods(self):
        """Assumes that feature matches have already been calculated (superglue or basic)"""
        if self.feature_matches is None:
            raise NoMatchesError

        if self.options.add_affine_to_candidates:
            self.match_using_local_affine()
        if self.options.add_gp_to_candidates:
            self.match_using_gp()
        if self.options.add_fdnc_to_candidates:
            self.match_using_fdnc()

        self.calc_final_matches()

    def print_reason_for_all_final_matches(self, which_set_of_matches=None):
        dict_of_matches = self.get_pair_to_conf_dict(which_set_of_matches)
        for k in dict_of_matches.keys():
            print("==================================")
            self.print_reason_for_match(k)

    def __repr__(self):
        return f"FramePair with {len(self.final_matches)}/{self.num_possible_matches} matches \n"


def calc_FramePair_from_FeatureSpaceTemplates(template_base, template_target,
                                              frame_pair_options: FramePairOptions = None) -> FramePair:
    """
    Calculates a FramePair from two FeatureSpaceTemplateMatcher objects. Note that this uses the matcher from the template_base object

    See also calc_FramePair_from_Frames

    Parameters
    ----------
    template0
    template1

    Returns
    -------
    FramePair
    """
    if frame_pair_options is None:
        frame_pair_options = FramePairOptions()
    frame_pair = FramePair(frame0=template_base.template_frame, frame1=template_target.template_frame, options=frame_pair_options)
    if not frame_pair.check_both_frames_valid():
        # In particular, no neurons detected in at least one frame
        return frame_pair

    # Get the matched frames from the templates
    matches_class = template_base.match_target_frame(template_target.template_frame)

    # Create a FramePair from the matched frames
    frame_pair.feature_matches = matches_class.array_matches_with_conf.tolist()

    # Add additional candidates; the class checks if they are used
    frame_pair.match_using_all_methods()

    return frame_pair


def calc_FramePair_from_Frames(frame0: ReferenceFrame, frame1: ReferenceFrame, frame_pair_options: FramePairOptions,
                               verbose: int = 1,
                               DEBUG: bool = False) -> FramePair:
    """
    Similar to older function, but this doesn't assume the features are
    already matched

    Main constructor for the class FramePair

    See also: calc_2frame_matches
    """
    # Frames are 'desynced' because affine matching overwrites the keypoints
    # frame0.check_data_desyncing()
    # frame1.check_data_desyncing()

    # Create class, then call member functions
    frame_pair = FramePair(options=frame_pair_options, frame0=frame0, frame1=frame1)
    if not frame_pair.check_both_frames_valid():
        # In particular, no neurons detected in at least one frame
        return frame_pair
    # Core matching algorithm
    frame_pair.match_using_feature_embedding()

    # May not change anything, based on frame_pair_options
    # Should I redo the feature alignment here?
    # frame_pair.calc_final_matches()  # Temporary, using just the feature matches above
    # frame_pair.preprocess_data()

    # Add additional candidates; the class checks if they are used
    frame_pair.match_using_all_methods()

    return frame_pair


def calc_FramePair_like(pair: FramePair, frame0: ReferenceFrame = None, frame1: ReferenceFrame = None) -> FramePair:
    """
    Calculates a new frame pair using the metadata from the FramePair, and new frames (optional)

    Parameters
    ----------
    pair
    frame0
    frame1

    Returns
    -------

    """

    metadata = pair.get_metadata_dict()

    if frame0 is None:
        frame0 = pair.frame0
    if frame1 is None:
        frame1 = pair.frame1

    new_pair = calc_FramePair_from_Frames(frame0, frame1, metadata)
    new_pair.calc_final_matches()

    return new_pair


def num_possible_matches_between_two_frames(frame0: ReferenceFrame, frame1: ReferenceFrame):
    if frame0 is None or frame1 is None:
        return np.nan
    return min(frame0.num_neurons, frame1.num_neurons)
