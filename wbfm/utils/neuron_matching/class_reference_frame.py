import logging
from dataclasses import dataclass
import cv2
import numpy as np
from tqdm.auto import tqdm
from wbfm.utils.external.utils_cv2 import get_keypoints_from_3dseg
from wbfm.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from wbfm.utils.external.custom_errors import OverwritePreviousAnalysisError, DataSynchronizationError, \
    AnalysisOutOfOrderError, DeprecationError, NoNeuronsError
from wbfm.utils.general.utils_features import convert_to_grayscale, detect_keypoints_and_features, \
    build_feature_tree, build_neuron_tree, build_f2n_map, detect_only_keypoints
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons

##
## Basic class definition
##


@dataclass
class ReferenceFrame:
    """ Information for registered reference frames"""

    # Data for registration
    neuron_locs: list = None
    keypoints: list = None
    keypoint_locs: np.ndarray = None  # Includes the z coordinate
    all_features: np.ndarray = None   # Vector per neuron
    features_to_neurons: dict = None

    # Metadata
    frame_ind: int = None
    video_fname: str = None
    vol_shape: tuple = None
    z_depth: int = None

    # From the preprocessing settings
    alpha_red: float = None

    # To be finished with a set of other registered frames
    neuron_ids: list = None  # global neuron index

    # For adding new keypoints
    alternate_2d_encoder: callable = lambda x: None
    options_2d_encoder: dict = None

    _all_features_non_neurons: np.ndarray = None
    _raw_data: np.ndarray = None  # Note that this is actually preprocessed

    def get_default_base_2d_encoder(self):
        self.alternate_2d_encoder = self.get_default_base_2d_encoder
        if self.options_2d_encoder is not None:
            opt = self.options_2d_encoder
        else:
            opt = {}
        return cv2.xfeatures2d.VGG_create(**opt)

    def get_metadata(self):
        return {'frame_ind': self.frame_ind,
                'video_fname': self.video_fname,
                'vol_shape': self.vol_shape}

    def iter_neurons(self):
        # Practice with yield
        for neuron in self.neuron_locs:
            yield neuron

    def get_features_of_neuron(self, which_neuron):
        iter_tmp = self.features_to_neurons.items()
        return [key for key, val in iter_tmp if val == which_neuron]

    @property
    def num_neurons(self):
        if self.neuron_locs is not None:
            return self.neuron_locs.shape[0]
        else:
            return 0

    def get_raw_data(self) -> np.ndarray:
        if self._raw_data is None:
            logging.warning(
                "Getting raw data using this method is not recommended (possible filename desynchronization)")
            self._raw_data = zarr_reader_folder_or_zipstore(self.video_fname)[self.frame_ind, ...]
        return self._raw_data

    def get_uint8_data(self) -> np.ndarray:
        raw_dat = self.get_raw_data()
        if raw_dat.dtype == np.uint8:
            dat = raw_dat
        elif raw_dat.dtype == np.uint16:
            # Assume it needs to be scaled
            dat = (raw_dat * self.alpha_red).astype('uint8')
        else:
            raise NotImplementedError(f"Datatype should be uint8 or uint16, found {raw_dat.dtype} instead")
        return dat

    def import_neurons_from_metadata(self, detected_neurons: DetectedNeurons) -> list:
        """

        Parameters
        ----------
        detected_neurons

        Returns
        -------
        neuron_locs - also saved as self.neuron_locs and self.keypoint_locs

        """
        neuron_locs = detected_neurons.detect_neurons_from_file(self.frame_ind, numpy_not_list=False)

        if len(neuron_locs) == 0:
            raise NoNeuronsError("No neurons detected... check data settings")

        self.neuron_locs = neuron_locs
        return neuron_locs

    def copy_neurons_to_keypoints_locs(self):
        """ Explicitly a different method for backwards compatibility"""
        self.keypoint_locs = self.neuron_locs.copy()

    def detect_non_neuron_keypoints(self,
                                    dat,
                                    num_features_per_plane=1000,
                                    start_plane=0,
                                    append_to_existing_keypoints=False,
                                    verbose=0):
        """ See: detect_keypoints_and_build_features"""
        if not append_to_existing_keypoints:
            if self.keypoints is not None:
                raise OverwritePreviousAnalysisError('keypoints')

        all_locs = []
        all_kps = []
        for i in range(dat.shape[0]):
            if i < start_plane:
                continue
            im = np.squeeze(dat[i, ...])
            kp = detect_only_keypoints(im, num_features_per_plane)

            all_kps.extend(kp)
            locs_3d = np.array([np.hstack((i, row.pt)) for row in kp])
            all_locs.extend(locs_3d)

        all_locs = np.array(all_locs)

        if append_to_existing_keypoints:
            self.keypoints.append(all_kps)
            self.keypoint_locs = np.vstack([self.keypoint_locs, all_locs])
        else:
            self.keypoints = all_kps
            self.keypoint_locs = all_locs

        return all_kps, all_locs

    def detect_keypoints_and_build_features(self,
                                            dat,
                                            num_features_per_plane=1000,
                                            start_plane=0,
                                            use_saved_detector=False,
                                            verbose=0):
        """

        Uses ORB features to match neurons, based on nearest neighbor association between neurons and keypoints

        keypoint-neuron associate is performed elsewhere; see:
            build_trivial_keypoint_to_neuron_mapping
            build_nontrivial_keypoint_to_neuron_mapping

        As of 06.06.2021, not part of main traces (only postprocessing)

        Parameters
        ----------
        dat
        num_features_per_plane
        start_plane
        verbose

        Returns
        -------
        all_kps, all_locs, all_features
        (also saved)

        """

        if self.keypoints is not None:
            raise OverwritePreviousAnalysisError('keypoints')

        if use_saved_detector:
            detector = self.alternate_2d_encoder()
            if detector is None:
                raise AnalysisOutOfOrderError('set detector')
        else:
            detector = self.get_default_base_2d_encoder()

        all_features = []
        all_locs = []
        all_kps = []
        for i_z in range(dat.shape[0]):
            if i_z < start_plane:
                continue
            im = np.squeeze(dat[i_z, ...])
            kp, features = detect_keypoints_and_features(im, num_features_per_plane, detector=detector)

            if features is None:
                continue
            all_features.extend(features)
            all_kps.extend(kp)
            locs_3d = np.array([np.hstack((i_z, row.pt)) for row in kp])
            all_locs.extend(locs_3d)

        all_locs, all_features = np.array(all_locs), np.array(all_features)

        self.keypoints = all_kps
        self.keypoint_locs = all_locs
        self.all_features = all_features

        return all_kps, all_locs, all_features

    def build_nontrivial_keypoint_to_neuron_mapping(self, neuron_feature_radius):
        """
        Matches keypoints and features based purely on distance (max=neuron_feature_radius)

        Designed when the keypoints are detected separately from the neurons
        Can also be used when the keypoints are a superset of neurons (e.g. ORB keypoints + neurons)

        """

        kp_3d_locs, neuron_locs = self.keypoint_locs, self.neuron_locs

        # Requires some open3d subfunctions; will not work on a cluster
        num_f, pc_f, _ = build_feature_tree(kp_3d_locs, which_slice=None)
        _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
        kp2n_map = build_f2n_map(kp_3d_locs,
                                 num_f,
                                 pc_f,
                                 neuron_feature_radius,
                                 tree_neurons,
                                 verbose=0)

        self.features_to_neurons = kp2n_map

        return kp2n_map

    def encode_neurons_using_3d_network(self, model, gpu, dataset, use_projection_space=True) -> None:
        """
        Build a feature vector for each neuron using a 3d CNN. 
        Note that this function doesn't know the neuron locations, but rather uses pre-built crops from the dataset class

        Designed to be used with a trained BarlowTrack network; see also embed_using_barlow in the BarlowTrack repo

        Note: network_path is needed as an arg for compatibility with the yaml file
        """
        import torch
        batch, ids = dataset[self.frame_ind]
        batch = batch.to(gpu)
        all_embeddings = []

        def _embed_single_neuron(name):
            idx = ids.index(name)
            crop = torch.unsqueeze(batch[:, idx, ...], 0)
            embeddings = model.embed(crop) if use_projection_space else model.backbone(crop)
            all_embeddings.append(embeddings.cpu().detach().numpy())

        with torch.no_grad():
            for n in ids:
                _embed_single_neuron(n)
        
        # Convert to expected format (numpy array, neurons x features)
        self.all_features = np.vstack(all_embeddings).squeeze()
        self.keypoints = list()
        
    def encode_neurons_using_2d_network(self, base_2d_encoder=None,
                                        use_keypoint_locs=True,
                                        transpose_images=True, **kwargs) -> None:
        """
        Builds a feature vector for each neuron (zxy location) in a 3d volume
        Default: Uses opencv VGG as a 2d encoder for a number of slices above and below the exact z location

        Note: overwrites the keypoints using only the locations

        Performance note: because this loops over keypoints, it only works for a small number, ~100-300
            Designed to be used with detected neurons, not ORB keypoints (which could be >10000)

        Creates feature vectors of length z_depth * 128 (default for VGG)
        """

        z_depth = self.z_depth
        im_3d = self.get_uint8_data()

        if use_keypoint_locs:
            locs_zxy = self.keypoint_locs
        else:
            locs_zxy = self.neuron_locs
        num_kps = locs_zxy.shape[0]

        if transpose_images:
            im_3d_gray = [convert_to_grayscale(xy).astype('uint8').transpose() for xy in im_3d]
        else:
            im_3d_gray = [convert_to_grayscale(xy).astype('uint8') for xy in im_3d]
        all_embeddings = []
        all_keypoints = []
        if base_2d_encoder is None:
            base_2d_encoder = self.get_default_base_2d_encoder()

        for i_loc in tqdm(range(num_kps), total=num_kps, leave=False):
            z, x, y = locs_zxy[i_loc, :]
            kp_2d = cv2.KeyPoint(x, y, 31.0)

            z = int(z)
            slices_around_keypoint = np.arange(z - z_depth, z + z_depth + 1)
            slices_around_keypoint = np.clip(slices_around_keypoint, 0, len(im_3d_gray) - 1)
            # Generate features on neighboring z slices as well
            # Repeat slices if near the edge
            one_kp_embedding = []
            for i in slices_around_keypoint:
                im_2d = im_3d_gray[int(i)]
                _, this_ds = base_2d_encoder.compute(im_2d, [kp_2d])
                one_kp_embedding.append(this_ds)

            one_kp_embedding = np.hstack(one_kp_embedding)
            all_embeddings.extend(one_kp_embedding)
            all_keypoints.append(kp_2d)

        all_embeddings = np.array(all_embeddings)
        self.all_features = all_embeddings
        self.keypoints = all_keypoints
        # self.check_data_desyncing()

    def build_trivial_keypoint_to_neuron_mapping(self, ignore_keypoints=False):
        # This is now just a trivial mapping
        self.check_data_desyncing(ignore_keypoints=ignore_keypoints)
        kp2n_map = {i: i for i in range(len(self.neuron_locs))}
        self.features_to_neurons = kp2n_map

    def prep_for_pickle(self):
        """Deletes the cv2.Keypoints (the locations are stored though)"""
        if self.keypoints is not None and len(self.keypoints) > 0:
            # self.check_data_desyncing()
            self.keypoints = []
        self._raw_data = None
        self.alternate_2d_encoder = None

    def rebuild_keypoints(self):
        """
        Rebuilds keypoints from keypoint_locs
        see also self.prep_for_pickle()
        """
        if len(self.keypoints) > 0:
            logging.warning("Overwriting existing keypoints...")
        k = get_keypoints_from_3dseg(self.keypoint_locs)
        self.keypoints = k

    def check_data_desyncing(self, ignore_keypoints=False):
        if not ignore_keypoints:
            if len(self.keypoint_locs) != len(self.keypoints):
                logging.warning(f"{len(self.keypoint_locs)} != {len(self.keypoints)}")
                raise DataSynchronizationError('keypoint_locs', 'keypoints', 'rebuild_keypoints')

            if len(self.keypoints) != len(self.all_features):
                logging.warning(f"{len(self.keypoints)} != {len(self.all_features)}")
                raise DataSynchronizationError('all_features', 'keypoints')
        
        # Always check neuron_locs
        if len(self.neuron_locs) != len(self.all_features):
            logging.warning(f"{len(self.neuron_locs)} != {len(self.all_features)}")
            raise DataSynchronizationError('neuron_locs', 'all_features')

    def __str__(self):
        return f"=======================================\n\
ReferenceFrame:\n\
Frame index: {self.frame_ind} \n\
Number of neurons: {self.num_neurons} \n"

    def __repr__(self):
        return f"ReferenceFrame with {self.num_neurons} neurons \n"


##
## Class for Set of reference frames
##

@dataclass
class RegisteredReferenceFrames:
    """Data for matched reference frames"""

    # Intermediate products
    reference_frames: list = None
    pairwise_matches: dict = None
    pairwise_conf: dict = None

    # More detailed intermediates and pipeline_alternate matchings
    feature_matches: dict = None
    bipartite_matches: list = None

    # Global neuron coordinate system
    neuron_cluster_mode: str = None
    global2local: dict = None
    local2global: dict = None

    verbose: int = 0

    def __str__(self):
        return f"RegisteredReferenceFrames with {len(self.reference_frames)} Frames \n"

    def __repr__(self):
        [print(r) for r in self.reference_frames]
        return f"=======================================\n\
                RegisteredReferenceFrames:\n\
                Number of frames: {len(self.reference_frames)} \n"


def build_reference_frame_encoding(metadata=None, all_detected_neurons: DetectedNeurons = None,
                                   encoder_opt=None, use_barlow_network=False, verbose=0):
    """
    Directly builds an embedding for each neuron, instead of detecting keypoints

    See: build_reference_frame
    """
    if encoder_opt is None:
        encoder_opt = {}
    if metadata is None:
        metadata = {}

    frame = ReferenceFrame(**metadata)

    # Build keypoints (in this case, neurons directly)
    try:
        # Sets neuron_locs
        frame.import_neurons_from_metadata(all_detected_neurons)
    except NoNeuronsError:
        return frame
    # Sets keypoint_locs
    frame.copy_neurons_to_keypoints_locs()

    # Calculate encodings
    if use_barlow_network:
        frame.encode_neurons_using_3d_network(**encoder_opt)
    else:
        frame.encode_neurons_using_2d_network(**encoder_opt)

    # Set up mapping between neurons and keypoints
    frame.build_trivial_keypoint_to_neuron_mapping(ignore_keypoints=use_barlow_network)

    return frame
