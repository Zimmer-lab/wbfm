# WormWideWeb dataset info
## Details
To get information about the recordings and the model-determined values, please check out the [paper](https://doi.org/10.1016/j.cell.2023.07.035).
## Citation
Please cite the [paper](https://doi.org/10.1016/j.cell.2023.07.035) if you use the datasets and findings from this website.
## JSON file keys:
### Dataset overview keys:
`uid`: dataset uid 
`max_t`: number of time points recorded  
`avg_timestep`: average length of a confocal volume, in minutes
`num_neurons`: number of neurons recorded  
`dataset_type`: type(s) of the dataset
`timestamp_confocal`: Time stamps of each confocal frame, in seconds.

### Behavioral keys:
`velocity`: velocity trace over time
`head_curvature`: head curvature trace over time  
`pumping`: pumping/feeding trace over time
`angular_velocity`: angular velocity trace over time
`body_curvature`: body curvature trace over time
`reversal_events`: list of reversal start/stop time points, in confocal frames

### Neural data keys:
`trace_array`: neural trace array, neurons x time, z-scored
`trace_original`: neural trace array, neurons x time, original ratiometric un-normalized values
`labeled`: labeling/identity information for NeuroPAL dataset  

### CePNEM-determined keys:
`ranges`: time ranges where CePNEM was fit for this dataset
`rel_enc_str_v`: relative encoding strength of each neuron to velocity
`rel_enc_str_Î¸h`: relative encoding strength of each neuron to head curvature 
`rel_enc_str_P`: relative encoding strength of each neuron to feeding/pumping 
`forwardness`: forward tuning slope of each neuron 
`dorsalness`: dorsal tuning slope of each neuron
`feedingness`: pumping tuning slope of each neuron 
`tau_vals`: half-decay constants for the neurons, in seconds  
`encoding_changing_neurons`: list of encoding changing neurons  
`neuron_categorization`: model-determined neuron tuning information, expressed as a list of neurons that are tuned to each behavioral category in each time range
