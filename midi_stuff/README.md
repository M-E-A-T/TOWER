## Install Requirements:
`python -m pip install -r requirements.txt`

## Usage:
`python main.py`

### Message structure:
this can be a midi on / off or a controller change, I chose controller change because I can just link it to the volume of a specific track and send a 0 or (127 * .8) 80% of the slider.

where t is the controller number and dict[t] is where we store the state
cc_msg = mido.Message('control_change', channel=0, control=t, value=dict[t])
