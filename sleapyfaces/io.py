from dataclasses import dataclass
from collections.abc import MutableSequence
from os import PathLike
import pandas as pd
import numpy as np
import glob
from io import BufferedWriter
from sleapyfaces.utils import (
    fill_missing,
    json_dumps,
    save_dict_to_hdf5,
    save_dt_to_hdf5,
    tracks_deconstructor,
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
        path (Text of PathLike[Text]): Path to the directory containing the DAQ data.

    Instance Attributes:
        cache (pd.DataFrame): Pandas DataFrame containing the DAQ data.
        columns (List): List of column names in the cache.

    Methods:
        append: Append a column to the cache.
        save_data: Save the cache to a csv file.
    """

    path: str | PathLike[str]

    def __init__(self, path: str | PathLike[str]):
        self.path = path

    def __post_init__(self):
        self.cache = pd.read_csv(self.path)
        self.columns = self.cache.columns

    def append(self, name: str, value: list) -> None:
        if len(list) == len(self.cache.iloc[:, 0]):
            self.cache = pd.concat(
                [self.cache, pd.DataFrame(value, columns=[name])], axis=1
            )
        elif len(list) == len(self.cache.iloc[0, :]):
            self.cache.columns = value
        else:
            raise ValueError("Length of list does not match length of cached data.")

    def save_data(self, filename: str | PathLike[str] | BufferedWriter) -> None:
        """saves the cached data to a csv file

        Args:
            filename (Text | PathLike[Text] | BufferedWriter): the name of the file to save the data to
        """
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
        path (Text | PathLike[Text]): path to the directory containing the SLEAP analysis file

    Instance Attributes:
        data (Dict): dictionary of all the data from the SLEAP analysis file
        track_names (List): list of the track names from the SLEAP analysis file
        tracks (pd.DataFrame): a pandas DataFrame containing the tracks from the SLEAP analysis file
                (with missing frames filled in using a linear interpolation method)
                NOTE: more information about the tracks DataFrame can be found in the tracks property docstring
    """

    path: str | PathLike[str]

    def __init__(self, path: str | PathLike[str]):
        self.path = path

    def __post_init__(self):
        self.data = self.getDatasets()
        self.tracks = tracks_deconstructor(self.data["tracks"], self.data["node_names"])
        self.track_names = [""] * len(self.data["node_names"]) * 2
        for name, i in zip(
            self.data["node_names"], range(0, (len(self.data["node_names"]) * 2), 2)
        ):
            self.track_names[i] = f"{name}_x"
            self.track_names[i + 1] = f"{name}_y"

    def getDatasets(
        self,
    ) -> dict[
        str,
        np.ndarray[str]
        | np.ndarray[int]
        | np.ndarray[float]
        | pd.DataFrame[str]
        | pd.DataFrame[int]
        | pd.DataFrame[float]
        | list[str]
        | list[int]
        | list[float],
    ]:
        """gets the datasets from the SLEAP analysis file

        Returns:
            data (Dict): a dictionary containing the datasets from the SLEAP analysis file
        """
        with h5.File(f"{self.BasePath}", "r") as f:
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

    def __len__(self) -> int:
        return len(self.tracks.index)

    def append(self, item: pd.Series[str] | pd.Series[int] | pd.Series[float]) -> None:
        if len(item) == len(self.tracks.index):
            self.tracks = pd.concat([self.tracks, item], axis=1)
        else:
            raise ValueError("Length of list does not match length of cached data.")

    def save_data(self, filename: str | PathLike[str], path="SLEAP") -> None:
        """saves the SLEAP analysis data to an HDF5 file

        Args:
            filename (Text | PathLike[Text]): the name of the file to save the data to
            path (str, optional): the internal HDF5 path to save the data to. Defaults to "SLEAP".
        """
        if filename.endswith(".h5") or filename.endswith(".hdf5"):
            with h5.File(filename) as f:
                save_dict_to_hdf5(f, path, self.datasets)
            with pd.HDFStore(filename, mode="a") as store:
                save_dt_to_hdf5(store, self.tracks, f"{path}/tracks")


@dataclass
class BehMetadata(MutableSequence):
    """
    Summary:
        Cache for JSON data.

    Class Attributes:
        path (str of PathLike[str]): Path to the directory containing the JSON data.
        MetaDataKey (str): Key for the metadata in the JSON data. Defaults to "beh_metadata" based on bruker_control.
        TrialArrayKey (str): Key for the trial array in the JSON data. Defaults to "trialArray" based on bruker_control.
        ITIArrayKey (str): Key for the ITI array in the JSON data. Defaults to "ITIArray" based on bruker_control.

        Bruker Control Repository:
            Link: https://github.com/Tyelab/bruker_control
            Author: Jeremy Delahanty

    Instance Attributes:
        cache (pd.DataFrame): Pandas DataFrame containing the JSON data.
        columns (List): List of column names in the cache."""

    path: str | PathLike[str]
    MetaDataKey: str
    TrialArrayKey: str
    ITIArrayKey: str

    def __init__(
        self,
        path: str | PathLike[str],
        MetaDataKey="beh_metadata",
        TrialArrayKey="trialArray",
        ITIArrayKey="ITIArray",
    ):
        self.path = path
        self.MetaDataKey = MetaDataKey
        self.TrialArrayKey = TrialArrayKey
        self.ITIArrayKey = ITIArrayKey

    def __post_init__(self):
        self.cache = self.getCache()
        self.columns = self.cache.columns

    def getCache(self) -> pd.DataFrame:
        with open(self.path, "r") as fp:
            js = json.load(fp)
            trialArray = js.get(self.MetaDataKey)[self.TrialArrayKey]
            ITIArray = js.get(self.MetaDataKey)[self.ITIArrayKey]
        return pd.DataFrame([trialArray, ITIArray], columns=["trialArray", "ITIArray"])

    def save_data(self, filename: str | PathLike[str] | BufferedWriter) -> None:
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
        path (str of PathLike[str]): Path to the directory containing the video data.

    Instance Attributes:
        fps (float): Frames per second of the video data.
    """

    path: str | PathLike[str]

    def __init__(self, path: str | PathLike[str]):
        self.path = path

    def __post_init__(self):
        self.cache = self.getCache()
        self.fps = float(eval(self.cache.get("avg_frame_rate")))

    def getCache(self) -> dict:
        for video in glob.glob("*.mp4"):
            if len(glob.glob("*.mp4")) > 1:
                continue
            else:
                return ffmpeg.probe(f"{self.path}/{video}")["streams"][
                    (
                        int(
                            ffmpeg.probe(f"{self.path}/{video}")["format"]["nb_streams"]
                        )
                        - 1
                    )
                ]

    def save_data(self, filename: str | PathLike[str] | BufferedWriter) -> None:
        if (
            filename.endswith(".json")
            or filename.endswith(".JSON")
            or isinstance(filename, BufferedWriter)
        ):
            json_dumps(self.cache, filename)
        else:
            json_dumps(self.cache, f"{filename}.csv")
