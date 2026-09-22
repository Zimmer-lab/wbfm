# NWB file format used by this package

This package uses Neurodata Without Borders (NWB) files as the primary interchange format:
raw imaging data goes **in**, and finished analysis (segmentation, tracking, traces) can be
exported **out**. This document describes the exact layout this code expects, which channel
means what, and how to diagnose/fix common channel mix-ups.

Relevant code:

- Reading: `wbfm/utils/projects/finished_project_data.py` (`from_nwb_file`)
- Writing: `wbfm/utils/nwb/utils_nwb_export.py`
- Validation: `python wbfm/utils/nwb/test_nwb.py PATH/TO/file.nwb`
- Project creation: `wbfm/scripts/0-create_new_project_from_nwb.py`

---

## Quick reference

| Property | Convention |
|---|---|
| Imaging data location | `acquisition['CalciumImageSeries']` (type `MultiChannelVolumeSeries`) |
| Axis order **on disk (in the NWB)** | **(T, X, Y, Z, C)** — channel axis **last** |
| Axis order **inside the pipeline** (zarr / in-memory) | **(T, Z, X, Y)** — channels split into separate arrays |
| Channel 0 | **Red** reference channel (mScarlet; used for tracking by default) |
| Channel 1 | **Green** signal channel (GCaMP; calcium activity) |
| Channel identity is determined by… | **Position/index only**, *not* by wavelength or channel name |
| `order_optical_channels` metadata | Written on export for provenance; **not used when reading** calcium data |

Example: a recording with 460 timepoints, 157×184 pixels, 10 z-planes, 2 channels has
shape `(460, 157, 184, 10, 2)` = (T, X, Y, Z, C) with `data[..., 0]` = red and
`data[..., 1]` = green.

> **There is no config option to remap channel indices when reading an NWB file.**
> If your file has the channels in the opposite order, you must reorder them (see
> [Fixing swapped channels](#fixing-swapped-channels)).

---

## Minimal required contents

To create a project and run the pipeline, the NWB must contain at least:

1. **`acquisition['CalciumImageSeries']`** — the dual-channel volumetric time series
   (5-D, TXYZC; or 4-D/single-channel, which is loaded into *both* red and green).
2. **`imaging_volume.rate`** — volumes per second (used as the acquisition rate).
3. **`imaging_volume.grid_spacing`** — pixel sizes in µm; the first two entries must be
   equal (isotropic in x/y). The last entry is interpreted as z spacing.

Everything else (segmentation, tracks, traces, behavior) is optional and will be
unpacked into the project if present:

| NWB location | Interface name | Meaning |
|---|---|---|
| `acquisition` | `CalciumImageSeries` | Preprocessed dual-channel movie (required) |
| `acquisition` | `RawCalciumImageSeries` | Optional raw (unpreprocessed) movie |
| `acquisition` | `NeuroPALImageRaw` | Optional NeuroPAL raw image |
| `processing['CalciumActivity']` | `order_optical_channels` | Channel-order provenance (not read for calcium data) |
| | `CalciumSeriesSegmentation` | Tracked segmentation labels (T, X, Y, Z) |
| | `CalciumSeriesSegmentationUntracked` | Raw (untracked) segmentation; if missing, the tracked one is used for both |
| | `CalciumSeriesSegmentationCoords` | Per-timepoint segmentation coordinates |
| | `NeuronIDs` | Segmentation label IDs |
| | `SignalFluorescence` / `SignalCalciumImResponseSeries` | Green traces |
| | `ReferenceFluorescence` / `ReferenceCalciumImResponseSeries` | Red traces |
| | `SignalDFoF` / `SignalCalciumImResponseSeries` | ΔF/F (ratio) traces |
| `processing` (`Position`) | `NeuronCentroids` | Tracked neuron centroids → `final_tracks` |
| `processing` (`…`) | `NeuronSegmentationID` | Per-neuron segmentation IDs |
| `processing['BF_NIR']` | `BrightFieldNIR` | Behavior video (T, X, Y) |
| `processing['Behavior']` | various | Behavior timeseries (kymograph, eigenworms, etc.) |

You can check which of these a file has with:

```commandline
python wbfm/utils/nwb/test_nwb.py /path/to/file.nwb
```

---

## Axis order in detail

### On disk (inside the NWB)

Calcium imaging data is stored as a 5-D array with **channel last**:

```
data.shape = (T, X, Y, Z, C)
#            time, x, y, z-plans, channels (C = 2 for red+green)
```

The export code stacks the two channel volumes and transposes explicitly to this order
(`utils_nwb_export.py`: *"Reshape to be TXYZC from TZXYC"*).

Segmentation videos are stored as **(T, X, Y, Z)** (no channel axis).

### Inside the pipeline

When loading an NWB, the reader splits the channel axis and transposes to the package's
internal layout (`finished_project_data.py`: *"Transpose data from TXYZC to TZXY"*):

```
red_data   = data[..., 0].transpose((0, 3, 1, 2))   # (T, X, Y, Z) → (T, Z, X, Y)
green_data = data[..., 1].transpose((0, 3, 1, 2))   # (T, X, Y, Z) → (T, Z, X, Y)
```

All downstream steps (preprocessing zarr files, segmentation, tracking, GUI layers) use
this **(T, Z, X, Y)** layout with red and green as separate arrays. When data is unpacked
from NWB into the project folder, the zarr files therefore have shape (T, Z, X, Y) even
though the NWB itself is (T, X, Y, Z, C).

A 4-D array (or C = 1) is treated as a single channel: it is loaded into **both** red and
green (with a log warning). More than 2 channels: extras are ignored (with a warning).

---

## Channel convention (red vs green)

### The code does not read channel names or wavelengths

Channel identity is **purely positional** when reading:

| Channel index | Loaded as | Intended fluorophore | Typical lasers |
|---|---|---|---|
| `[..., 0]` | `red_data` / "Red data" | mScarlet (reference) | 561 ex / ~617 em |
| `[..., 1]` | `green_data` / "Green data" | GFP-GCaMP (signal) | 488 ex / ~525 em |

The `order_optical_channels` field (e.g. `0: 488-525-50m`, `1: 561-605-70m`) and the
`OpticalChannelPlus` objects on the imaging volume are written by the exporter so that
the file is self-describing for other tools, **but this package ignores them for calcium
imaging data** (they are only consulted for NeuroPAL data in `test_nwb.py`).

Consequences:

- If your NWB has green-first ordering, the pipeline will **silently swap** red and green.
- The napari layers **"Red data"** and **"Green data"** will then show the wrong channel
  (green looks red and vice versa).
- Tracking will run on whatever sits in channel 0, even if that is actually GCaMP.

### Which channel is used for what

| Step | Default channel | Config override |
|---|---|---|
| Segmentation (StarDist) | **Red** (ch 0) | `segment_and_track_on_green_channel: true` → green; or `sum_red_and_green_channels: true` → red+green |
| Tracking / frame-to-frame matching | **Red** (ch 0) | same flag: `segment_and_track_on_green_channel` |
| Segmentation metadata (brightness, centroids) | **Always red** (hardcoded) | none |
| Trace extraction | **Both** → `red_traces.h5` + `green_traces.h5` | per-channel file overrides in `traces_config.yaml` |
| Napari GUI layers | Layer name maps to the corresponding array | none |

The relevant keys live in the project's `project_config.yaml`:

```yaml
dataset_params:
  segment_and_track_on_green_channel: False  # Optional; may improve performance for dim red channels
```

and in `1-segmentation/segment_config.yaml`:

```yaml
segmentation_params:
  sum_red_and_green_channels: false
```

If both green-segmentation and summing are enabled, green wins and summing is ignored
(a warning is logged).

> **Note:** one internal code path (SuperGlue frame-pair loading in
> `tracklet_pipeline.py`) always reads the red channel regardless of
> `segment_and_track_on_green_channel`. Keep this in mind if you rely on green-only
> tracking with the tracklet-based tracker.

---

## Why neurons can disappear when the calcium signal drops

Segmentation and tracking are designed to run on the **red reference channel**, which is
roughly constant (mScarlet), so normal GCaMP fluctuations should not remove neurons.

Neurons disappear when the activity channel is (wrongly) used for detection/tracking:

1. **Swapped channels in the NWB** (most common for external data): channel 0 is actually
   GCaMP, so the pipeline "tracks on red" but is really tracking on the activity signal.
   Dim frames → lost objects → neurons vanish and reappear with the calcium transient.
2. **`segment_and_track_on_green_channel: true`** with a dim green channel.
3. Brightness-based filtering: segmentation metadata brightness is always measured on
   channel 0 (`red_data`), so a dim channel 0 also causes drops.

**Diagnosis:** open the project in the progress GUI and toggle the "Red data" /
"Green data" layers. If the layer labeled "Red data" shows GCaMP-like activity (or if
neurons flicker in the layer used for tracking), your channel order is wrong.

### Fixing swapped channels

Reorder the channel axis so that **index 0 = red, index 1 = green**, then recreate the
project. Sketch with h5py (adjust paths to your file):

```python
import h5py
import numpy as np

src = "/path/to/file.nwb"
tmp = "/path/to/file_fixed.nwb"

with h5py.File(src, "r") as f_in, h5py.File(tmp, "w") as f_out:
    # Copy the whole file, then fix the channel axis of the imaging data
    # (do this carefully for large files — e.g. copy dataset-by-dataset with chunks)
    ...

# Simpler alternative for small files: load with pynwb, swap data[..., ::-1] on
# acquisition['CalciumImageSeries'], and write a new NWB file.
```

After swapping, verify with `test_nwb.py` and by checking in Python that
`data[0, ..., 0]` looks like the red/reference channel (roughly constant) and
`data[0, ..., 1]` looks like GCaMP (structured, activity-dependent).

**Workaround without rewriting the file:** set
`segment_and_track_on_green_channel: true` in `project_config.yaml`. Because the reader
already mislabeled the channels, this makes tracking use the real red channel. The GUI
layer names ("Red data" / "Green data") will still be swapped, and exported NWB files
will keep the wrong order — reordering the source file is the clean fix.

---

## Creating a project from an NWB file

See [Running the pipeline](running_the_pipeline.md). In short:

```commandline
cd wbfm/scripts
python 0-create_new_project_from_nwb.py with project_dir=NEW_PROJECT nwb_file=/path/to/file.nwb copy_nwb_file=True unpack_nwb=True
```

Unpacking converts the NWB imaging data into zarr files (`preprocessed_red.zarr`,
`preprocessed_green.zarr`) with the internal (T, Z, X, Y) layout. Subsequent pipeline
steps read the zarr files, not the NWB (except for hybrid loading of analysis results).

---

## Exporting analysis back to NWB

From a finished project:

```commandline
python wbfm/scripts/postprocessing/4+export_as_nwb.py with project_path=... output_folder=... include_image_data=True
```

This writes (among other things):

- `acquisition['CalciumImageSeries']` — channels stacked as **red then green** (dict
  insertion order: `{'red': …, 'green': …}`), so exported files always follow the
  ch0=red / ch1=green convention.
- Segmentation, centroids, and red/green/ratio traces under `processing['CalciumActivity']`.
- Behavior data under `processing['Behavior']` / `processing['BF_NIR']`.

---

## Related documentation

- [Running the pipeline](running_the_pipeline.md) — end-to-end workflow
- [Data folder organization](data_folder_organization.md) — non-NWB raw file layout
- [FAQ](faq.md) — common problems
- GUI layer meanings: [GUI README](../wbfm/gui/README.md)
