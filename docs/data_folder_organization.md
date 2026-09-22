# Raw data

Fundamentally, there should be one folder per recording, and one subfolder for red, green, and behavior (optional):

- Green: `*_worm*`/`*CH0`/`*`.btf
- Red: `*_worm*`/`*CH1`/`*`.btf
- Behavior: `*_worm*`/`*CH0-BH`/`*`.btf

If there are other .btf files in these subfolders, the program might not work properly. 

# NWB files

If your raw data is (or will be) an NWB file instead of the folder layout above, see
[NWB file format](nwb_format.md) for the expected axis order, channel naming, and required fields.
