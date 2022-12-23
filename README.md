# SLEAPyFaces

This is a package for extracting facial expressions from SLEAP analyses.

----

[![PyPI - Version](https://img.shields.io/pypi/v/sleapyfaces.svg)](https://pypi.org/project/sleapyfaces)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleapyfaces.svg)](https://pypi.org/project/sleapyfaces)

-----

## To Do:

Things that it does:
1. iterate (repeatedly) over each mouse and each week (each mouse and each experiment)
    - [ ] get project files (experimental) structure
    - [ ] initialize an iterator over project structure
2. get daq data from CSV file
    - [x] read CSV files
    - [x] save each column from CSV file
        * Note: CSV columns are of differing lengths
3. get “beh_metadata” from json metadata
    - [x] read JSON file
    - [x] grab the values for key “beh_metadata”
        - [x] get the values of sub key “trialArray”
        - [x] get the values of subkey “ITIArray”
4. get video metadata from *.mp4 file (with ffmpeg.probe)
    - [x] read in the *.mp4 metadata
    - [x] select the correct video stream
    - [x] get the average frames per second
5. get SLEAP data from *.h5 file
    - [x] open h5 file
    - [x] get transposed values of key “tracks” (tracking_locations)
    - [x] fill missing locations (linear regress. fit)
    - [x] get transposed values of key “edge_inds”
    - [x] get values of key “edge_names”
    - [x] get transposed values of “instance_scores”
    - [x] get transposed values of “point_scores”
    - [x] get values of “track_occupancy”
    - [x] get transposed values of “tracking_scores”
    - [x] get decoded values of “node_names” (make sure theres no encoding issues)
6. deconstruct SLEAP points into x and y points (across all frames)
    - [ ] iterate over each node
    - [ ] breakup the 4D array “tracks” into 1D array for x and y values respectively
        * Note: [frame, node, x/y, color] for greyscale the color dimension is 1D (i.e. essentially the 4D array is 3D because the color dimension is constant)
    - [ ] iterate over each frame
    - [x] assign mouse, week, frame #, and timestamp (using average frames per second)
7. Split data into individual trials by trial type using the Speaker and LED data from the CSV daq data
    - [ ] initialize trial iterators for the consistently documented points from the daq CSV
    - [ ] iterate over each trial in “trialArray”
    - [ ] get the index of 10sec before and 13sec after trial start
    - [x] for each feature, grab the start and end indices
    - [x] store data from each trial in a pd.dataframe
    - [x] concatenate all pd.dataframes together for each video
    - [ ] concatenate the pd.dataframes from each video together for each mouse (base expr split)
8. Prepare the data
    - [ ] (opt.) mean center across all points for a single trial
    - [ ] mean center across all trials for a single experiment
    - [ ] mean center across all experiments for a single mouse
    - [ ] mean center across all mice
    - [ ] (opt.) z-score mean centered data
9. Analyze the data
    - [ ] Perform 2D and 3D PCAs on all data (raw, centered, by trial, by week, by mouse, etc…)
    - [ ] apply gaussian kernel to PCA outputs
10. Save the data
    - [x] write everything to HDF5 file(s)

----

**Table of Contents**

- [SLEAPyFaces](#sleapyfaces)
	- [To Do:](#to-do)
	- [Installation](#installation)
	- [License](#license)

## Installation

```console
pip install sleapyfaces
```

## License

`sleapyfaces` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
