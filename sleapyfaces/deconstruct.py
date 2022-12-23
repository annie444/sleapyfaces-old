from sleapyfaces.io import (
    DAQData,
    BehMetadata,
    VideoMetadata,
    VideoMetadata,
    SLEAPanalysis,
)
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class DataContainer:

    TrackedData: List[str]
    BasePath: str

    def __init__(self, BasePath: str):
        self.BasePath = BasePath
        self.DAQ = DAQData(BasePath)
        self.Beh = BehMetadata(BasePath)
        self.Video = VideoMetadata(BasePath)
        self.SLEAP = SLEAPanalysis(BasePath)
        self.__columns = self.DAQ.columns + self.Beh.columns + zip([f"{node}_x" for node in self.SLEAP.datasets["node_names"]], [f"{node}_y" for node in self.SLEAP.datasets["node_names"]])

    @property
    def columns(self) -> List[str]:
        return self.__columns

    @columns.setter
    def columns(self, value: str | List[str]) -> None:
        self.__columns = self.__columns.insert(0, value)

    @columns.deleter
    def columns(self) -> None:
        del self.__columns




    def initialize_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for frame in
