from sleapyfaces.io import (
    DAQData,
    BehMetadata,
    VideoMetadata,
    SLEAPanalysis,
)
from dataclasses import dataclass
from os import PathLike
import pandas as pd


class File:
    """A structured file object that contains the base path and filename of a file.

    Class Attributes:
        file_str: the location of the file.

    Instance Attributes:
        file: the location of the file.
        filename: the name of the file.
        basepath: the base path of the file.
        iPath: the ith base path of the file path. (i.e. iPath(1) returns the second to last path in the file path.)
    """

    file_str: str | PathLike[str]

    def __init__(self, file_str):
        self._file = file_str

    @property
    def file(self) -> str | PathLike[str]:
        return self._file

    @property
    def filename(self) -> str:
        return self._file.split("/")[-1]

    @property
    def basepath(self) -> str:
        return "/".join(self._file.split("/")[:-1])

    def iPath(self, i: int) -> str:
        """Returns the ith path in the file path."""
        return "/".join(self._file.split("/")[:-i])


@dataclass
class FileConstructor:

    """Takes in the base paths and filenames of the experimental data and returns them as a structured object.

    Class Attributes:
        DAQFile (str): The location of the DAQ data file.
        SLEAPFile (str): The location of the SLEAP analysis file.
        BehFile (str): The location of the behavioral metadata file.
        VideoFile (str): The location of the video file.

    Instance Attributes:
        DAQ (File): The location of the DAQ file as a structured File object.
        SLEAP (File): The location of the SLEAP analysis file as a structured File object.
        Beh (File): The location of the behavioral metadata file as a structured File object.
        Video (File): The location of the video file as a structured File object.
    """

    DAQFile: str | PathLike[str] | DAQData.path
    SLEAPFile: str | PathLike[str] | SLEAPanalysis.path
    BehFile: str | PathLike[str] | BehMetadata.path
    VideoFile: str | PathLike[str] | VideoMetadata.path

    def __init__(
        self,
        DAQFile,
        SLEAPFile,
        BehFile,
        VideoFile,
    ):
        self.DAQ = File(DAQFile)
        self.SLEAP = File(SLEAPFile)
        self.Beh = File(BehFile)
        self.Video = File(VideoFile)


class CustomColumn:
    """Builds an annotation column for the base dataframe.

    Class Attributes:
        ColumnTitle (str): The title of the column.
        ColumnData (str | int | float | bool): The data to be added to the column.

    Instance Attributes:
        column (pd.Series): The column to be added to the base dataframe.
    """

    ColumnTitle: str
    ColumnData: str | int | float | bool

    def __init__(self, ColumnTitle: str, ColumnData: str | int | float | bool):
        self.ColumnTitle = ColumnTitle
        self.ColumnData = ColumnData

    def column(self, length: int) -> pd.Series:
        col = [self.ColumnData] * length
        return pd.Series(col, name=self.ColumnTitle)
