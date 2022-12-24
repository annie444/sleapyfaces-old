from dataclasses import dataclass
from collections.abc import MutableSequence
from os import PathLike
import pandas as pd
import numpy as np
import glob
from typing import List, Text, List, Iterable, Dict, Sequence, Any
from io import BufferedWriter
import h5py
from sleapyfaces.utilities import (
    fill_missing,
    json_dumps,
    save_dict_to_hdf5,
    save_dt_to_hdf5,
)
import json
import ffmpeg
import h5py as h5


@dataclass
class DAQData(MutableSequence):
    """
    Summary:
        Cache for DAQ data.

    Class Attributes:
        BasePath (Text of PathLike[Text]): Path to the directory containing the DAQ data.

    Instance Attributes:
        cache (pd.DataFrame): Pandas DataFrame containing the DAQ data.
        columns (List): List of column names in the cache.
    """

    BasePath: Text | PathLike[Text]

    def __init__(self, BasePath):
        self.BasePath = BasePath
        self.cache = pd.read_csv(f"{self.BasePath}/{glob.glob('*.csv')[0]}")
        self.columns = self.cache.columns

    def __post_init__(self):
        self.BasePath = self.BasePath
        self.cache = pd.read_csv(f"{self.BasePath}/{glob.glob('*.csv')[0]}")
        self.columns = self.cache.columns

    def __getitem__(self, col: Text) -> List:
        return self.cache[col].dropna().tolist()

    def __setitem__(self, col: Text, value: float) -> None:
        self.cache[col] = value

    def __delitem__(self, col: Text) -> None:
        del self.cache[col]

    def __len__(self) -> int:
        return len(self.cache)

    def insert(self, index: int, name: Text, value: List) -> None:
        self.cache.insert(index, name, value)

    def append(self, name: Text, value: List) -> None:
        if len(list) == len(self.cache.iloc[:, 0]):
            self.cache = pd.concat(
                [self.cache, pd.DataFrame(value, columns=[name])], axis=1
            )
        elif len(list) == len(self.cache.iloc[0, :]):
            self.cache.columns = value
        else:
            raise ValueError("Length of list does not match length of cached data.")

    def reverse(self) -> None:
        self.cache.sort_index(ascending=False, inplace=True, kind="quicksort")

    def sort(self, col=None, ascending=True) -> None:
        self.cache.sort_values(
            key=col, ascending=ascending, inplace=True, kind="quicksort"
        )

    def __contains__(self, col: Text) -> bool:
        return self.cache.__contains__(col)

    def extend(self, values: Iterable) -> None:
        return super().extend(values)

    def pop(self, col: Text) -> pd.Series:
        return self.cache.pop(col)

    def __repr__(self) -> Text:
        return self.cache.__repr__()

    def __str__(self) -> Text:
        return self.cache.__str__()

    def save_data(self, filename: Text | PathLike[Text] | BufferedWriter) -> None:
        if (
            filename.endswith(".csv")
            or filename.endswith(".CSV")
            or isinstance(filename, BufferedWriter)
        ):
            self.cache.to_csv(filename, index=True)
        else:
            self.cache.to_csv(f"{filename}.csv", index=True)


@dataclass
class SLEAPanalysis:
    """
    Summary:
        a class for reading and storing SLEAP analysis files

    Class Attributes:
        BasePath (Text | PathLike[Text]): path to the directory containing the SLEAP analysis file

    Instance Attributes:
        datasets (Dict): dictionary of datasets in the SLEAP analysis file
        tracks (pd.DataFrame): a pandas DataFrame containing the tracks from the SLEAP analysis file
                (with missing frames filled in using a linear interpolation method)
                NOTE: more information about the tracks DataFrame can be found in the tracks property docstring
    """

    BasePath: Text | PathLike[Text]

    def __init__(self, BasePath):
        self.BasePath = BasePath
        self.datasets = self.getDatasets()
        self.tracks = self.tracks_deconstructor(
            self.datasets["tracks"], self.datasets["node_names"]
        )

    @classmethod
    def getDatasets(
        cls,
    ) -> Dict[str, np.ndarray | pd.DataFrame | List | Sequence | MutableSequence]:
        with h5py.File(f"{cls.BasePath}", "r") as f:
            datasets = list(f.keys())
            data = dict()
            for dataset in datasets:
                if dataset == "tracks":
                    data[dataset] = fill_missing(f[dataset][:].T)
                elif "name" in dataset:
                    data[dataset] = [n.decode() for n in f[dataset][:].flatten()]
                else:
                    data[dataset] = f[dataset][:].T
        return data

    @staticmethod
    def tracks_deconstructor(
        tracks: np.ndarray | pd.DataFrame | List | Sequence | MutableSequence,
        nodes: np.ndarray | pd.DataFrame | List | Sequence | MutableSequence,
    ) -> pd.DataFrame:
        new_tracks = [pd.DataFrame()] * (len(nodes) * 2)
        for n, node in enumerate(nodes):
            new_tracks[n] = pd.concat(
                [pd.DataFrame(tracks[:, n, 0, 0]), pd.DataFrame(tracks[:, n, 1, 0])],
                columns=[f"{node}_x", f"{node}_y"],
                axis=1,
            )
        return pd.concat(new_tracks, axis=1)

    def save_data(self, filename: Text | PathLike[Text], path="SLEAP") -> None:
        if filename.endswith(".h5") or filename.endswith(".H5"):
            with h5py.File(filename) as f:
                save_dict_to_hdf5(f, path, self.datasets)
            with pd.HDFStore(filename, mode="a") as store:
                save_dt_to_hdf5(store, self.tracks, f"{path}/tracks")

    def __annotations__(self) -> Dict[str, type]:
        return {
            "BasePath": str | Text | PathLike[str] | PathLike[Text] | BufferedWriter,
            "datasets": Dict[
                str, np.ndarray | pd.DataFrame | List | Sequence | MutableSequence
            ],
            "tracks": pd.DataFrame,
        }

    def __post_init__(self):
        self.BasePath = self.BasePath
        self.datasets = self.getDatasets()
        self.tracks = self.tracks_deconstructor(
            self.datasets["tracks"], self.datasets["node_names"]
        )

    def __iter__(self):
        return iter(self.__dict__.values())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


@dataclass
class BehMetadata(MutableSequence):
    """
    Summary:
        Cache for JSON data.

    Class Attributes:
        BasePath (Text of PathLike[Text]): Path to the directory containing the JSON data.
        MetaDataKey (Text): Key for the metadata in the JSON data. Defaults to "beh_metadata" based on bruker_control.
        TrialArrayKey (Text): Key for the trial array in the JSON data. Defaults to "trialArray" based on bruker_control.
        ITIArrayKey (Text): Key for the ITI array in the JSON data. Defaults to "ITIArray" based on bruker_control.

        Bruker Control Repository:
            Link: https://github.com/Tyelab/bruker_control
            Author: Jeremy Delahanty

    Instance Attributes:
        cache (pd.DataFrame): Pandas DataFrame containing the JSON data.
        columns (List): List of column names in the cache."""

    BasePath: str | Text | PathLike[str] | PathLike[Text]
    MetaDataKey: Text
    TrialArrayKey: Text
    ITIArrayKey: Text

    def __init__(
        self,
        BasePath,
        MetaDataKey="beh_metadata",
        TrialArrayKey="trialArray",
        ITIArrayKey="ITIArray",
    ):
        self.BasePath = BasePath
        self.MetaDataKey = MetaDataKey
        self.TrialArrayKey = TrialArrayKey
        self.ITIArrayKey = ITIArrayKey
        self.cache = self.getCache()
        self.columns = self.cache.columns

    def __post_init__(self):
        self.BasePath = self.BasePath
        self.MetaDataKey = self.MetaDataKey
        self.TrialArrayKey = self.TrialArrayKey
        self.ITIArrayKey = self.ITIArrayKey
        self.cache = self.getCache()
        self.columns = self.cache.columns

    def __init_subclass__(
        cls, BasePath: str | Text | PathLike[str] | PathLike[Text], *args, **kwargs
    ) -> None:
        super(BehMetadata, cls).__init__(*args, **kwargs)

    @classmethod
    def getCache(cls) -> pd.DataFrame:
        with open(f"{cls.BasePath}/{glob.glob('*.json')[0]}", "r") as js:
            js = json.load(js)
            trialArray = js.get(cls.MetaDataKey)[cls.TrialArrayKey]
            ITIArray = js.get(cls.MetaDataKey)[cls.ITIArrayKey]
        return pd.DataFrame(
            [trialArray, ITIArray], columns=[cls.TrialArrayKey, cls.ITIArrayKey]
        )

    def __getitem__(self, col: Text) -> List:
        return self.cache[col].dropna().tolist()

    def __setitem__(self, col: Text, value: float) -> None:
        self.cache[col] = value

    def __delitem__(self, col: Text) -> None:
        del self.cache[col]

    def __len__(self) -> int:
        return len(self.cache)

    def insert(self, index: int, name: Text, value: List) -> None:
        self.cache.insert(index, name, value)

    def append(self, name: Text, value: List) -> None:
        if len(list) == len(self.cache.iloc[:, 0]):
            self.cache = pd.concat(
                [self.cache, pd.DataFrame(value, columns=[name])], axis=1
            )
        elif len(list) == len(self.cache.iloc[0, :]):
            self.cache.columns = value
        else:
            raise ValueError("Length of list does not match length of cached data.")

    def reverse(self) -> None:
        self.cache.sort_index(ascending=False, inplace=True, kind="quicksort")

    def sort(self, col=None, ascending=True) -> None:
        self.cache.sort_values(
            key=col, ascending=ascending, inplace=True, kind="quicksort"
        )

    def __contains__(self, col: Text) -> bool:
        return self.cache.__contains__(col)

    def extend(self, values: Iterable) -> None:
        return super().extend(values)

    def pop(self, col: Text) -> pd.Series:
        return self.cache.pop(col)

    def __repr__(self) -> Text:
        return self.cache.__repr__()

    def __str__(self) -> Text:
        return self.cache.__str__()

    def save_data(self, filename: Text | PathLike[Text] | BufferedWriter) -> None:
        if (
            filename.endswith(".csv")
            or filename.endswith(".CSV")
            or isinstance(filename, BufferedWriter)
        ):
            self.cache.to_csv(filename, index=True)
        else:
            self.cache.to_csv(f"{filename}.csv", index=True)


@dataclass
class VideoMetadata:
    """
    Summary:
        class for caching the video metadata.

    Class Attributes:
        BasePath (Text of PathLike[Text]): Path to the directory containing the video data.

    Instance Attributes:
        fps (float): Frames per second of the video data.
    """

    BasePath: Text | PathLike[Text]

    def __init__(self, BasePath):
        self.BasePath = BasePath
        self.cache = self.getCache()
        self.fps = self.cache.get("avg_frame_rate")

    def __post_init__(self):
        self.cache = self.getCache()
        self.fps = self.cache.get("avg_frame_rate")

    @classmethod
    def getCache(cls) -> Dict[Any, Any]:
        for video in glob.glob("*.mp4"):
            if len(glob.glob("*.mp4")) > 1:
                continue
            else:
                return ffmpeg.probe(f"{cls.BasePath}/{video}")["streams"][
                    (
                        int(
                            ffmpeg.probe(f"{cls.BasePath}/{video}")["format"][
                                "nb_streams"
                            ]
                        )
                        - 1
                    )
                ]

    def save_data(self, filename: Text | PathLike[Text] | BufferedWriter) -> None:
        if (
            filename.endswith(".json")
            or filename.endswith(".JSON")
            or isinstance(filename, BufferedWriter)
        ):
            json_dumps(self.cache, filename)
        else:
            json_dumps(self.cache, f"{filename}.csv")

    def __repr__(self) -> Text:
        return self.cache.__repr__()

    def __str__(self) -> Text:
        return self.cache.__str__()

    def __getitem__(self, key: Text) -> Any:
        return self.cache[key]

    def __setitem__(self, key: Text, value: Any) -> None:
        self.cache[key] = value

    def __delitem__(self, key: Text) -> None:
        del self.cache[key]

    def __len__(self) -> int:
        return len(list(self.cache.keys()))

    def __contains__(self, key: Text) -> bool:
        return self.cache.__contains__(key)

    def __iter__(self) -> Iterable:
        return self.cache.__iter__()

    def __next__(self) -> Any:
        return self.cache.__next__()

    def __reversed__(self) -> Iterable:
        return self.cache.__reversed__()

    def __hash__(self) -> int:
        return hash(self.cache)

    def __eq__(self, other: Any) -> bool:
        return self.cache.__eq__(other)

    def __ne__(self, other: Any) -> bool:
        return self.cache.__ne__(other)

    def __annotations__(self) -> Dict[Any, type]:
        return {
            "BasePath": str | PathLike[str] | Text | PathLike[Text] | BufferedWriter,
            "fps": float,
            "cache": Dict[Any, Any],
        }
