import sys

from pynwb import NWBFile, NWBHDF5IO
import argparse
import os


class TestNWB:
    """
    Updates from Andrew Kirjner

    Copied (with minor changes) from Zimmer wbfm's copy from: https://github.com/focolab/NWBelegans/blob/main/check_NWB.py

    """

    def __init__(self, nwbfile):
        with NWBHDF5IO(nwbfile, mode='r', load_namespaces=True) as io:
            if isinstance(io, NWBFile):
                print('NWB file loaded successfully')
                read_nwbfile = io
            else:
                read_nwbfile = io.read()

            print("\n=== File Structure ===")
            print("\nGeneral Metadata:")
            print(f"Session ID: {read_nwbfile.session_id}")
            print(f"Session Description: {read_nwbfile.session_description}")
            print(f"Identifier: {read_nwbfile.identifier}")

            if read_nwbfile.subject is not None:
                print("\nSubject Information:")
                print(f"Subject ID: {read_nwbfile.subject.subject_id}")
                print(f"Growth Stage: {getattr(read_nwbfile.subject, 'growth_stage', 'Not specified')}")

            print("\nAcquisition Groups:")
            for name, acq in read_nwbfile.acquisition.items():
                print(f"- {name}: {type(acq).__name__}")
                if hasattr(acq, 'data'):
                    try:
                        shape = acq.data.shape
                        print(f"  Shape: {shape}")
                    except AttributeError:
                        print("  Shape: Not available")

            print("\nProcessing Modules:")
            for module_name, module in read_nwbfile.processing.items():
                print(f"\n- Module: {module_name}")
                for data_interface_name, data_interface in module.data_interfaces.items():
                    print(f"  - {data_interface_name}: {type(data_interface).__name__}")

            # Now run the original tests
            has_neuropal = False
            has_calcium_imaging = False
            has_calcium_traces = False
            has_segmentation = False
            has_segmentation_untracked = False
            has_centroids = False
            has_segmentation_ids = False
            try:
                image = read_nwbfile.acquisition['NeuroPALImageRaw'].data[:]  # get the neuroPAL image as a np array
                channels = read_nwbfile.acquisition['NeuroPALImageRaw'].RGBW_channels[
                           :]  # get which channels of the image correspond to which RGBW pseudocolors
                im_vol = read_nwbfile.acquisition[
                    'NeuroPALImageRaw'].imaging_volume  # get the metadata associated with the imaging acquisition

                # get the locations of neuron centers
                seg = read_nwbfile.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons'].voxel_mask[:]
                labels = read_nwbfile.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons']['ID_labels'][:]
                # get information about all of the optical channels used in acquisition
                optchans = im_vol.optical_channel_plus
                # get the order of the optical channels in the image... not in flavell data?
                try:
                    chan_refs = read_nwbfile.processing['NeuroPAL']['OpticalChannelRefs'].channels[:]
                except KeyError:
                    chan_refs = read_nwbfile.processing['NeuroPAL']['order_optical_channels'].channels[:]
                has_neuropal = True
            except (KeyError, TypeError) as e:
                pass
                # print(f"Error with neuropal stacks: {e}")

                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc_type, fname, exc_tb.tb_lineno)

            try:
                # load the first 15 frames of the calcium images
                calcium_frames = read_nwbfile.acquisition['CalciumImageSeries'].data[0:15, :, :, :]
                size = read_nwbfile.acquisition['CalciumImageSeries'].data.shape
                print(f"Size of calcium imaging data: {size}")
                has_calcium_imaging = True
            except KeyError as e:
                # print(f"Error with calcium imaging: {e}")
                pass

            try:
                activity = read_nwbfile.processing['CalciumActivity']
                if 'SignalDFoF' in activity.data_interfaces:
                    fluor = activity['SignalDFoF']['SignalCalciumImResponseSeries'].data
                elif 'SignalFluorescence' in activity.data_interfaces:
                    fluor = activity['SignalFluorescence']['SignalCalciumImResponseSeries'].data
                elif 'SignalRawFluor' in activity.data_interfaces:
                    fluor = activity['SignalRawFluor']['SignalCalciumImResponseSeries'].data
                else:
                    raise KeyError
                # print(f"Size of calcium imaging traces: {fluor.shape}")
                has_calcium_traces = True

            except KeyError as e:
                # print(f"Error with calcium traces: {e}")
                pass

            try:
                activity = read_nwbfile.processing['CalciumActivity']
                if 'CalciumSeriesSegmentation' in activity.data_interfaces:
                    try:
                        # Try to access the segmentation data in my format
                        calc_seg = activity['CalciumSeriesSegmentation'].data
                        has_segmentation = True
                    except (TypeError, AttributeError):
                        if 'SegmentationVol0' in activity.data_interfaces:
                            calc_seg = activity['CalciumSeriesSegmentation']['SegmentationVol0'].voxel_mask[:]
                            has_segmentation = True
                        else:
                            has_segmentation = False
                            try:
                                print(list(activity['CalciumSeriesSegmentation'].data.shape))
                            except AttributeError:
                                pass

                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

                if 'CalciumSeriesSegmentationUntracked' in activity.data_interfaces:
                    try:
                        calc_seg = activity['CalciumSeriesSegmentationUntracked'].data
                        print(f"Size of untracked segmentation: {calc_seg.shape}")
                        has_segmentation_untracked = True
                    except:
                        pass
                        
                if 'NeuronCentroids' in activity.data_interfaces:
                    try:
                        from wbfm.utils.nwb.utils_nwb_export import load_per_neuron_position
                        calc_seg = load_per_neuron_position(activity['NeuronCentroids'])
                        has_centroids = True
                    except:
                        pass
                        
                if 'NeuronSegmentationID' in activity.data_interfaces:
                    try:
                        calc_seg = activity['NeuronSegmentationID'].to_dataframe()
                        has_segmentation_ids = True
                    except:
                        pass
                        

            except (TypeError, KeyError) as e:
                # print(f"Error with segmentation: {e}")

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                # print(read_nwbfile.processing['CalciumActivity']['CalciumSeriesSegmentation'].keys())


        print(f"==========================================\n"
              f"Found the following data in the NWB file: \n"
              f"===============Fluorescence===============\n"
              f"Video calcium imaging:  {has_calcium_imaging}\n"
              f"Raw video segmentation: {has_segmentation_untracked}\n"
              f"Neuron centroids:       {has_centroids}\n"
              f"Segmentation IDs:       {has_segmentation_ids}\n"
              f"Video segmentation:     {has_segmentation}\n"
              f"Video calcium traces:   {has_calcium_traces}\n"
              f"===============  Neuropal  ===============\n"
              f"NeuroPAL image:         {has_neuropal}\n"
              f"===============  Behavior  ===============\n"
              f"Behavior video:         (Not yet implemented)\n"
              f"Behavior time series:   (Not yet implemented)\n")


def main():
    parser = argparse.ArgumentParser(description='Test an NWB file and check its contents')
    parser.add_argument('--nwb_file', '-n', help='Path to the NWB file to test')
    args = parser.parse_args()

    # Expand user path if it contains ~
    nwb_path = args.nwb_file

    if not os.path.exists(nwb_path):
        print(f"Error: File {nwb_path} does not exist")
        return

    # Create instance and test the file
    tester = TestNWB(nwb_path)


if __name__ == '__main__':
    main()
