from dataclasses import dataclass
from sleapyfaces.io import SLEAPanalysis, BehMetadata, VideoMetadata, DAQData
from sleapyfaces.structs import FileConstructor, CustomColumn

from sleapyfaces.utils import into_trial_format, reduce_daq

import pandas as pd
import numpy as np


@dataclass(slots=True)
class Experiment:
    name: str
    files: FileConstructor
    sleap: SLEAPanalysis
    beh: BehMetadata
    video: VideoMetadata
    daq: DAQData
    data: pd.DataFrame
    trials: pd.DataFrame
    trialData: property
    buildData: callable
    buildTrials: callable

    def __init__(self, name: str, files: FileConstructor):
        self.name = name
        self.files = files
        self.sleap = SLEAPanalysis(self.files.sleap.file)
        self.beh = BehMetadata(self.files.beh.file)
        self.video = VideoMetadata(self.files.video.file)
        self.daq = DAQData(self.files.daq.file)

    def __post_init__(self):
        self.data = self.sleap.tracks
        self.numeric_columns = self.sleap.track_names

    def buildData(self, CustomColumns: list[CustomColumn]):
        for col in CustomColumns:
            col.buildColumn(len(self.data.index))
            self.append(col.Column)

        ms_per_frame = (self.video.fps**-1) * 1000
        for i in range(len(self.sleap)):
            self.append(pd.Series([i * ms_per_frame], name="Timestamps"))
            self.append(pd.Series([i], name="Frames"))

    def append(self, item: pd.Series[str] | pd.Series[int] | pd.Series[float]):
        if len(item) == len(self.data.index):
            self.data = pd.concat([self.data, item], axis=1)
        else:
            raise ValueError("Length of list does not match length of cached data.")

    def buildTrials(
        self,
        TrackedData: list[str],
        Reduced: list[bool],
        start_buffer: int = 10000,
        end_buffer: int = 13000,
    ):
        """Converts the data into trial by trial format.

        Args:
            TrackedData (list[str]): the list of columns from the DAQ data that signify the START of each trial.
            DAQ (DAQData): the DAQ data object.
            Reduced (list[bool]): a boolean list with the same length as the TrackedData list that signifies the columns from the tracked data with quick TTL pulses that occour during the trial.
                (e.g. the LED TTL pulse may signify the beginning of a trial, but during the trial the LED turns on and off, so the LED TTL column should be marked as True)
            start_buffer (int, optional): The time in miliseconds you want to capture before the trial starts. Defaults to 10000 (i.e. 10 seconds).
            end_buffer (int, optional): The time in miliseconds you want to capture after the trial starts. Defaults to 13000 (i.e. 13 seconds).

        Raises:
            ValueError: if the length of the TrackedData and Reduced lists are not equal.

        Exposes the instance attribute:
                trials (pd.DataFrame): the dataframe with the data in trial by 	trial format, with a metaindex of trial number and frame number
        """

        if len(Reduced) != len(TrackedData):
            raise ValueError(
                "The number of Reduced arguments must be equal to the number of TrackedData arguments. NOTE: If you do not want to reduce the data, pass in a list of False values."
            )

        start_indecies = [0] * len(TrackedData)
        end_indecies = [0] * len(TrackedData)
        timestamps = self.data.loc[:, "Timestamps"].to_numpy(dtype=np.float256)

        for i, data, reduce in enumerate(zip(TrackedData, Reduced)):

            if reduce:
                times = np.array(
                    reduce_daq(pd.Series(self.daq.cache.loc[:, data]).tolist()),
                    dtype=np.float256,
                )

            else:
                times = pd.Series(self.daq.cache.loc[:, data]).to_numpy(
                    dtype=np.float256
                )

            start_indecies[i] = [0] * len(times)
            end_indecies[i] = [0] * len(times)

            for j, time in enumerate(times):
                start_indecies[i][j] = np.absolute(
                    timestamps - (time - start_buffer)
                ).argmin()
                end_indecies[i][j] = (
                    np.absolute(timestamps - (time + end_buffer)).argmin() + 1
                )

            start_indecies[i] = np.array(start_indecies[i], dtype=np.int64).flatten()
            end_indecies[i] = np.array(end_indecies[i], dtype=np.int64).flatten()

        start_indecies = np.unique(
            np.array(start_indecies, dtype=np.int64).flatten()
        ).sort()
        end_indecies = np.unique(
            np.array(end_indecies, dtype=np.int64).flatten()
        ).sort()

        if len(start_indecies) != len(end_indecies):
            raise ValueError(
                "The number of start indecies does not match the number of end indecies."
            )
        if len(start_indecies) != len(self.beh.cache):
            raise ValueError(
                "The number of start indecies does not match the number of trials in the behavior data. Maybe reduce?"
            )
        self.trialData = into_trial_format(
            self.data,
            self.beh.cache.loc[:, "trialArray"],
            start_indecies,
            end_indecies,
        )

    @property
    def trialData(self) -> list[pd.DataFrame]:
        return self._trialData

    @trialData.setter
    def trialData(self, value):
        self._trialData = value
        self.trials = pd.concat(self.trialData, axis=0, keys=range(len(self.trialData)))
