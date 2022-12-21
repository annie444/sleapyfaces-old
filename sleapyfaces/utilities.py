import h5py as h5
import numpy as np
import pandas as pd
import rapidjson
import json

from typing import Any, Dict, Hashable, Iterable, List, Optional


def json_loads(json_str: str) -> Dict:
    """
    A simple wrapper around the JSON decoder we are using.

    Args:
        json_str: JSON string to decode.

    Returns:
        Result of decoding JSON string.
    """
    try:
        return rapidjson.loads(json_str)
    except:
        return json.loads(json_str)


def json_dumps(d: Dict, filename: str = None):
    """
    A simple wrapper around the JSON encoder we are using.

    Args:
        d: The dict to write.
        filename: The filename to write to.

    Returns:
        None
    """

    encoder = rapidjson

    if filename:
        with open(filename, "w") as f:
            encoder.dump(d, f, ensure_ascii=False)
    else:
        return encoder.dumps(d)


def save_dt_to_hdf5(hdfstore: pd.HDFStore, dt: pd.DataFrame, path: str):
    """
    Saves a pandas DataFrame to an HDF5 file.

    Args:
            hdfstore: The HDF5 filename object to save the data to.
                    Assume it is open.
            dt: The DataFrame to save.
            path: The path to group save the DataFrame under.

    Returns:
            None
    """
    dt.to_hdf(hdfstore, path, format="table", mode="a")


def save_dict_to_hdf5(h5file: h5.File, path: str, dic: dict):
    """
    Saves dictionary to an HDF5 file.

    Calls itself recursively if items in dictionary are not
    `np.ndarray`, `np.int64`, `np.float64`, `str`, or `bytes`.
    Objects must be iterable.

    Args:
        h5file: The HDF5 filename object to save the data to.
            Assume it is open.
        path: The path to group save the dict under.
        dic: The dict to save.

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
