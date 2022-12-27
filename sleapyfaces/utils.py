import h5py as h5
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import json

from typing import Dict, List, Sequence, MutableSequence


def json_loads(json_str: str) -> Dict:
    """
    Cite:
        From: https://sleap.ai/
        By: Talmo Pereira

    Summary:
        A simple wrapper around the JSON decoder we are using.

    Args:
        json_str: JSON string to decode.

    Returns:
        Result of decoding JSON string.
    """
    return json.loads(json_str)


def json_dumps(d: Dict, filename: str = None):
    """
    Cite:
        From: https://sleap.ai/
        By: Talmo Pereira

    Summary:
        A simple wrapper around the JSON encoder we are using.

    Args:
        d: The dict to write.
        filename: The filename to write to.

    Returns:
        None
    """

    encoder = json

    if filename:
        with open(filename, "w") as f:
            encoder.dump(d, f, ensure_ascii=False)
    else:
        return encoder.dumps(d)


def save_dt_to_hdf5(hdfstore: pd.HDFStore, dt: pd.DataFrame, path: str):
    """
    Summary:
        Saves a pandas DataFrame to an HDF5 file.

    Args:
        hdfstore (pd.HDFStore Buffer): The HDF5 filename object to save the data to. Assume it is open.
        dt (pd.DataFrame): The DataFrame to save.
        path (str): The path to group save the DataFrame under.

    Returns:
        None
    """
    hdfstore.put(path, dt, format="table", data_columns=True)


def save_dict_to_hdf5(h5file: h5.File, path: str, dic: dict):
    """
    Cite:
        From: https://sleap.ai/
        By: Talmo Pereira

    Summary:
        Saves dictionary to an HDF5 file.

        Calls itself recursively if items in dictionary are not
        `np.ndarray`, `np.int64`, `np.float64`, `str`, or `bytes`.
        Objects must be iterable.

    Args:
        h5file (h5.file): The HDF5 filename object to save the data to.
            Assume it is open.
        path (str): The path to group save the dict under.
        dic (dict): The dict to save.

    Raises:
        ValueError: If type for item in dict cannot be saved.

    Returns:
        None
    """
    for key, item in list(dic.items()):
        print(f"Saving {key}:")
        if item is None:
            h5file[path + key] = ""
        elif isinstance(item, bool):
            h5file[path + key] = int(item)
        elif isinstance(item, list):
            items_encoded = []
            for it in item:
                if isinstance(it, str):
                    items_encoded.append(it.encode("utf8"))
                else:
                    items_encoded.append(it)

            h5file[path + key] = np.asarray(items_encoded)
        elif isinstance(item, (str)):
            h5file[path + key] = item.encode("utf8")
        elif isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, float)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            save_dict_to_hdf5(h5file, path + key + "/", item)
        elif isinstance(item, int):
            h5file[path + key] = item
        else:
            raise ValueError("Cannot save %s type" % type(item))


def fill_missing(Y, kind="linear") -> np.ndarray:
    """
    Cite:
        From: https://sleap.ai/notebooks/Analysis_examples.html
        By: Talmo Pereira

    Summary:
        Fills missing values independently along each dimension after the first.

    Args:
        Y (np.array): any dimensional array with missing values.
        kind (str): Interpolation kind.

    Returns:
        Y (np.array): Original array with missing values filled.
    """

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def smooth_diff(node_loc: np.ndarray, win=25, poly=3) -> np.ndarray:
    """
    Cite:
        From: https://sleap.ai/notebooks/Analysis_examples.html
        By: Talmo Pereira

    Summary:
        Computes the velocity of a node by taking the
        derivative of the smoothed position of the node and then taking the norm of the velocity vector at each frame.

    Args:
        node_loc (np.array): is a [frames, 2] array

        win (int): defines the window to smooth over

        poly (int): defines the order of the polynomial
        to fit with

    Returns:
        node_vel (np.array): is a [frames, 1] array

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def corr_roll(
    datax: List | pd.Series | np.ndarray, datay: List | pd.Series | np.ndarray, win: int
) -> np.ndarray:
    """
    Cite:
            From: https://sleap.ai/notebooks/Analysis_examples.html
            By: Talmo Pereira

    Summary:
        Computes the rolling correlation between two timeseries

        Args:
        datax (List or np.array or pd.Series): the x-dimensional timeseries
        datay (List np.array or pd.Series): the y-dimeansional timeseries

        win (int): sets the number of frames over which the covariance is computed

        Returns:
                np.array: returns a numpy array of the rolling correlation between the two timeseries over time

    """

    s1 = pd.Series(datax)
    s2 = pd.Series(datay)

    return np.array(s2.rolling(win).corr(s1))


def into_trial_format(
    var: pd.DataFrame,
    trial_types: list,
    trial_start_idx: np.ndarray[np.int64],
    trial_end_idx: np.ndarray[np.int64],
) -> pd.DataFrame:
    """
    Summary:
        Splits an array or dataframe into individual trials.

        Assumes that the index of the array or dataframe is the frame number.

    Args:
        var (np.array or pd.DataFrame): array or dataframe to split into trials
        trial_start_idx (list[int]): the list of frame indecies where the trials start
        trial_end_idx (list[int]): the list of frame indecies where the trials end

    Returns:
        pd.DataFrame: returns a DataFrame with a metaindex of trial number and frame number
    """

    if len(trial_start_idx) != len(trial_end_idx):
        raise ValueError("trial_start_idx and trial_end_idx must be the same length")
    var_trials = [0] * len(trial_start_idx)
    for trial, start, end, trial_type in enumerate(
        zip(trial_start_idx, trial_end_idx, trial_types)
    ):
        var_trials[trial] = pd.DataFrame(var[start:end, :])
        trial_type = [trial_type] * len(var_trials[trial])
        var_trials[trial] = pd.concat(
            [var_trials[trial], pd.DataFrame(trial_type, columns=["trial_type"])],
            axis=1,
        )
    return pd.concat(var_trials, keys=range(len(var_trials)))


# create gaussian kernel for smoothing
def gaussian_kernel(window_size: int, sigma=1) -> np.ndarray:
    """
        Summary:
        this function creates a gaussian kernel for back smoothing

    Args:
        window_size (int): how many frames to smooth over
        sigma (int, optional): relative standard deviation. Defaults to 1.

    Returns:
        np.array: returns a kernel to smooth over with shape (window_size,)
    """
    x_vals = np.arange(window_size)
    to_ret = np.exp(-((x_vals - window_size // 2) * 2) / (2 * sigma * 2))
    to_ret[: window_size // 2] = 0
    return to_ret


def reduce_daq(iterable: list, ms=4000) -> list[float]:
    """
    Summary:

        Reduces rapid succession TTL pulses to a single pulse.

    Args:
        iterable (list): the list of TTL pulse times (preferably in ms)
        ms (int, optional): the minimum time between pulses. Defaults to 4000ms (or 5 seconds) between pulses.

    Returns:
        list[float]: a reduced list of TTL pulse times
    """
    list: list[float] = []
    j: int = 0
    list.append(iterable[j])
    for i in range(0, len(iterable)):
        if iterable[j] < (iterable[i] - ms):
            j = i
            list.append(iterable[j])
    return list


def tracks_deconstructor(
    tracks: np.ndarray | pd.DataFrame | List | Sequence | MutableSequence,
    nodes: np.ndarray | pd.DataFrame | List | Sequence | MutableSequence,
) -> pd.DataFrame:
    """takes the tracks array from a SLEAP analysis file and converts it into a pandas DataFrame

    Args:
        tracks (np.ndarray | pd.DataFrame | List | Sequence | MutableSequence): the 4D array of tracks from a SLEAP analysis file
        nodes (np.ndarray | pd.DataFrame | List | Sequence | MutableSequence): the list of nodes from a SLEAP analysis file

    Returns:
        pd.DataFrame: the tracks DataFrame
    """
    new_tracks = [pd.DataFrame()] * (len(nodes) * 2)
    for n, node in enumerate(nodes):
        new_tracks[n] = pd.concat(
            [pd.DataFrame(tracks[:, n, 0, 0]), pd.DataFrame(tracks[:, n, 1, 0])],
            columns=[f"{node}_x", f"{node}_y"],
            axis=1,
        )
    return pd.concat(new_tracks, axis=1)
