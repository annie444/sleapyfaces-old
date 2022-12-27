from typing import Sequence
import glob
import os
from sleapyfaces.structs import FileConstructor, CustomColumn
from sleapyfaces.experiment import Experiment
from sleapyfaces.io import SLEAPanalysis, DAQData, BehMetadata, VideoMetadata


class BaseProject(Sequence):
    """Base class for project

    Args:
        base (str): Base path of the project (e.g. "/specialk_cs/2p/raw/CSE009")
        iterator (dict[str, str]): Iterator for the project files, with keys as the label and values as the folder name (e.g. {"week 1": "20211105", "week 2": "20211112"})
        DAQFile (str): The naming convention for the DAQ files (e.g. "*_events.csv" or "DAQOutput.csv")
        ExprMetaFile (str): The naming convention for the experimental structure files (e.g. "*_config.json" or "BehMetadata.json")
        SLEAPFile (str): The naming convention for the SLEAP files (e.g. "*_sleap.h5" or "SLEAP.h5")
        VideoFile (str): The naming convention for the video files (e.g. "*.mp4" or "video.avi")
        glob (bool): Whether to use glob to find the files (e.g. True or False)
            NOTE: if glob is True, make sure to include the file extension in the naming convention

    """

    def __init__(
        self,
        base: str,
        iterator: dict[str, str],
        DAQFile: str,
        ExprMetaFile: str,
        SLEAPFile: str,
        VideoFile: str,
        glob: bool = False,
    ):
        self.base = base
        self.iterators = iterator
        self.DAQFile = DAQFile
        self.ExprMetaFile = ExprMetaFile
        self.SLEAPFile = SLEAPFile
        self.VideoFile = VideoFile
        self.glob = glob

    def __post_init__(self):
        self.files = ["" for _ in range(len(self.iterators.keys()))]
        self.names = self.iterators.keys()
        for i, key in enumerate(self.iterators.keys()):
            if self.glob:
                self.files[i] = FileConstructor(
                    DAQFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.DAQFile)
                    )[0],
                    SLEAPFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.SLEAPFile)
                    )[0],
                    BehFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.ExprMetaFile)
                    )[0],
                    VideoFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.VideoFile)
                    )[0],
                )
            else:
                self.files[i] = FileConstructor(
                    DAQFile=os.path.join(self.base, self.iterators[key], self.DAQFile),
                    SLEAPFile=os.path.join(
                        self.base, self.iterators[key], self.SLEAPFile
                    ),
                    BehFile=os.path.join(
                        self.base, self.iterators[key], self.ExprMetaFile
                    ),
                    VideoFile=os.path.join(
                        self.base, self.iterators[key], self.VideoFile
                    ),
                )

    def build(self, key: str):
        """Build the project"""
        for expr in self.exprs:
            expr["Experiment"] = Experiment(SLEAPanalysis(expr["Files"].SLEAP))

    @property
    def files(self, key: str):
        if key in self.iterators.keys():
            if self.glob:
                return FileConstructor(
                    DAQFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.DAQFile)
                    )[0],
                    SLEAPFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.SLEAPFile)
                    )[0],
                    BehFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.ExprMetaFile)
                    )[0],
                    VideoFile=glob.glob(
                        os.path.join(self.base, self.iterators[key], self.VideoFile)
                    )[0],
                )
            else:
                return FileConstructor(
                    DAQFile=os.path.join(self.base, self.iterators[key], self.DAQFile),
                    SLEAPFile=os.path.join(
                        self.base, self.iterators[key], self.SLEAPFile
                    ),
                    BehFile=os.path.join(
                        self.base, self.iterators[key], self.ExprMetaFile
                    ),
                    VideoFile=os.path.join(
                        self.base, self.iterators[key], self.VideoFile
                    ),
                )

    @property
    def exprs(self):
        return [
            {
                "Name": name,
                "Files": files,
                "Experiment": Experiment(SLEAPanalysis(files.SLEAP)),
            }
            for name, files in zip(self.names, self.files)
        ]
