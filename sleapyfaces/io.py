from dataclasses import dataclass
from collections.abc import MutableSequence
from os import PathLike
from re import T
import pandas as pd
import glob
from typing import List, Text, List, Iterable, Dict
from io import BufferedWriter
import h5py
from sleapyfaces.utilities import fill_missing
import json
import ffmpeg


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

    def __init_subclass__(cls, BasePath, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.BasePath = BasePath
        cls.cache = pd.read_csv(f"{cls.BasePath}/{glob.glob('*.csv')[0]}")
        cls.columns = cls.cache.columns

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


class SLEAPanalysis:
    """
    Summary:
        a class for reading and storing SLEAP analysis files

    Class Attributes:
        BasePath (Text | PathLike[Text]): path to the directory containing the SLEAP analysis file
        Filename (Text): name of the SLEAP analysis file

    Instance Attributes:
        datasets (Dict): dictionary of datasets in the SLEAP analysis file
        tracks (Array): 4D array of tracks in the SLEAP analysis file (frame, node index, x=0 & y=1, color index)
            Note: the color index is the index of the color in video, for greyscale videos this is always 0, so the color index can be ignored
    """

    BasePath: Text | PathLike[Text]
    Filename: Text

    def __init__(self, BasePath, Filename=""):
        self.BasePath = BasePath
        self.Filename = Filename
        self.datasets = self.getDatasets(self.BasePath, self.Filename)
        self._tracks = self.datasets["tracks"]

    def __init_subclass__(cls, BasePath, Filename="", **kwargs):
        super().__init_subclass__(**kwargs)
        cls.BasePath = BasePath
        cls.Filename = Filename
        cls.datasets = cls.getDatasets(cls.BasePath, cls.Filename)
        cls._tracks = cls.datasets["tracks"]

    @staticmethod
    def getDatasets(BasePath: Text | PathLike[Text], Filename="") -> Dict:
        if Filename != "" and not Filename.endswith(".h5"):
            Filename = Filename + ".h5"
        with h5py.File(f"{BasePath}/{Filename}", "r") as f:
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

    BasePath: Text | PathLike[Text]
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
        self.cache = self.getCache(
            self.BasePath, self.MetaDataKey, self.TrialArrayKey, self.ITIArrayKey
        )
        self.columns = self.cache.columns

    def __init_subclass__(
        cls,
        BasePath,
        MetaDataKey="beh_metadata",
        TrialArrayKey="trialArray",
        ITIArrayKey="ITIArray",
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls.BasePath = BasePath
        cls.MetaDataKey = MetaDataKey
        cls.TrialArrayKey = TrialArrayKey
        cls.ITIArrayKey = ITIArrayKey
        cls.cache = cls.getCache(
            cls.BasePath, cls.MetaDataKey, cls.TrialArrayKey, cls.ITIArrayKey
        )
        cls.columns = cls.cache.columns

    @staticmethod
    def getCache(
        BasePath: Text | PathLike[Text],
        MetaDataKey: Text,
        TrialArrayKey: Text,
        ITIArrayKey: Text,
    ) -> pd.DataFrame:
        with open(f"{BasePath}/{glob.glob('*.json')[0]}", "r") as js:
            js = json.load(js)
            trialArray = js.get(MetaDataKey)[TrialArrayKey]
            ITIArray = js.get(MetaDataKey)[ITIArrayKey]
        return pd.DataFrame(
            [trialArray, ITIArray], columns=[TrialArrayKey, ITIArrayKey]
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
        self.fps = self.getFPS(self.BasePath)

    def __init_subclass__(cls, BasePath, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.BasePath = BasePath
        cls.fps = cls.getFPS(cls.BasePath)

    @staticmethod
    def getFPS(BasePath: Text | PathLike[Text]) -> float:
        for video in glob.glob("*.mp4"):
            if len(glob.glob("*.mp4")) > 1:
                continue
            else:
                return ffmpeg.probe(f"{BasePath}/{video}")["streams"][
                    (
                        int(ffmpeg.probe(f"{BasePath}/{video}")["format"]["nb_streams"])
                        - 1
                    )
                ].get("avg_frame_rate")
