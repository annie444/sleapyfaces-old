from sleapyfaces.io import (
    DAQData,
    BehMetadata,
    VideoMetadata,
    VideoMetadata,
    SLEAPanalysis,
)
import pandas as pd
from dataclasses import dataclass
from typing import List, Set, Text, Dict
from os import PathLike
import numpy as np


@dataclass
class Filenames:

    DAQDataFilename: str | Text
    SLEAPanalysisFilename: str | Text
    BehMetadataFilename: str | Text
    VideoMetadataFilename: str | Text

    def __init__(
        self,
        DAQDataFilename,
        SLEAPanalysisFilename,
        BehMetadataFilename,
        VideoMetadataFilename,
    ):
        self.DAQDataFilename = DAQDataFilename
        self.SLEAPanalysisFilename = SLEAPanalysisFilename
        self.BehMetadataFilename = BehMetadataFilename
        self.VideoMetadataFilename = VideoMetadataFilename
        self.__files = [
            self.DAQDataFilename,
            self.SLEAPanalysisFilename,
            self.BehMetadataFilename,
            self.VideoMetadataFilename,
        ]

    @property
    def files(self) -> List[str | Text]:
        return self.__files

    @files.setter
    def files(self, files: List[str | Text]) -> None:
        self.__files.append[files]


@dataclass
class BasePaths:

    DAQDataBasePath: str | Text | PathLike[Text] | PathLike[str] | DAQData.BasePath
    SLEAPanalysisBasePath: str | Text | PathLike[Text] | PathLike[
        str
    ] | SLEAPanalysis.BasePath
    BehMetadataBasePath: str | Text | PathLike[Text] | PathLike[
        str
    ] | BehMetadata.BasePath
    VideoMetadataBasePath: str | Text | PathLike[Text] | PathLike[
        str
    ] | VideoMetadata.BasePath

    def __init__(
        self,
        DAQDataBasePath,
        SLEAPanalysisBasePath,
        BehMetadataBasePath,
        VideoMetadataBasePath,
    ):
        self.DAQDataBasePath = DAQDataBasePath
        self.SLEAPanalysisBasePath = SLEAPanalysisBasePath
        self.BehMetadataBasePath = BehMetadataBasePath
        self.VideoMetadataBasePath = VideoMetadataBasePath
        self.__files = [
            self.DAQDataBasePath,
            self.SLEAPanalysisBasePath,
            self.BehMetadataBasePath,
            self.VideoMetadataBasePath,
        ]

    @property
    def files(self) -> List[str | Text]:
        return self.__files

    @files.setter
    def files(self, files: List[str | Text]) -> None:
        self.__files.append[files]


@dataclass
class FileConstructor(BasePaths, Filenames):
    def __init__(self):
        self.DAQFile = f"{BasePaths.DAQDataBasePath}/{Filenames.DAQDataFilename}"
        self.SLEAPFile = (
            f"{BasePaths.SLEAPanalysisBasePath}/{Filenames.SLEAPanalysisFilename}"
        )
        self.BehFile = (
            f"{BasePaths.BehMetadataBasePath}/{Filenames.BehMetadataFilename}"
        )
        self.VideoFile = (
            f"{BasePaths.VideoMetadataBasePath}/{Filenames.VideoMetadataFilename}"
        )

    def __annotations__(self):
        return {
            "DAQFile": str | Text,
            "SLEAPFile": str | Text,
            "BehFile": str | Text,
            "VideoFile": str | Text,
        }

    def __post_init__(self):
        self.DAQFile = f"{BasePaths.DAQDataBasePath}/{Filenames.DAQDataFilename}"
        self.SLEAPFile = (
            f"{BasePaths.SLEAPanalysisBasePath}/{Filenames.SLEAPanalysisFilename}"
        )
        self.BehFile = (
            f"{BasePaths.BehMetadataBasePath}/{Filenames.BehMetadataFilename}"
        )
        self.VideoFile = (
            f"{BasePaths.VideoMetadataBasePath}/{Filenames.VideoMetadataFilename}"
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


class DataContainer(FileConstructor):

    TrackedData: List[str]
    CustomColumns: List[str] | pd.Series | pd.DataFrame | np.ndarray
    TrialArrayKey: BehMetadata.TrialArrayKey
    ITIArrayKey: BehMetadata.ITIArrayKey
    MetadataKey: BehMetadata.MetaDataKey

    def __init__(
        self,
        TrackedData,
        CustomColumns,
        TrialArrayKey,
        ITIArrayKey,
        MetadataKey,
    ):
        self.DAQ = DAQData(self.DAQFile)
        self.Beh = BehMetadata(
            self.BehFile,
            **{
                "MetadataKey": MetadataKey,
                "TrialArrayKey": TrialArrayKey,
                "ITIArrayKey": ITIArrayKey,
            },
        )
        self.Video = VideoMetadata(self.VideoFile)
        self.SLEAP = SLEAPanalysis(self.SLEAPFile)
        self.TrackedData = TrackedData
        self.CustomColumns = CustomColumns

    @classmethod
    def initialize_dataframe(cls) -> pd.DataFrame:
        for i in range(len(cls.SLEAP.tracks)):
            df = pd.DataFrame()
