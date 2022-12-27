from sleapyfaces.io import (
    DAQData,
    BehMetadata,
    VideoMetadata,
    SLEAPanalysis,
)
from dataclasses import dataclass
from os import PathLike
import os
import glob
import pandas as pd


@dataclass(slots=True)
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

    get_glob: bool
    file: str | PathLike[str]
    basepath: str | PathLike[str]
    filename: str
    iPath: callable

    def __init__(self, basepath: str | PathLike[str], filename: str, get_glob=False):
        self.basepath = basepath
        self.filename = filename

    def __post_init__(self):
        if self.get_glob:
            self.file = glob.glob(os.path.join(self.basepath, self.filename))[0]
        else:
            self.file = os.path.join(self.basepath, self.filename)

    def iPath(self, i: int) -> str:
        """Returns the ith path in the file path."""
        return "/".join(self.file.split("/")[:-i])


@dataclass(slots=True)
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

    daq: File
    sleap: File
    beh: File
    video: File


@dataclass(slots=True)
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
    Column: pd.Series
    buildColumn: callable

    def __init__(self, ColumnTitle: str, ColumnData: str | int | float | bool):
        self.ColumnTitle = ColumnTitle
        self.ColumnData = ColumnData

    def buildColumn(self, length: int) -> None:
        self.Column = [self.ColumnData] * length
        self.Column = pd.Series(self.Column, name=self.ColumnTitle)
