from dataclasses import dataclass
from sleapyfaces.io import SLEAPanalysis, BehMetadata, VideoMetadata, DAQData
from sleapyfaces.structs import FileConstructor, CustomColumn

from sleapyfaces.utils import into_trial_format, reduce_daq

import pandas as pd
import numpy as np


@dataclass
class Experiment:
    name: str
    files: FileConstructor
    cust_cols: list[CustomColumn]
    sleap: SLEAPanalysis
    beh: BehMetadata
    video: VideoMetadata
    daq: DAQData


class DataContainer:
    def __init__(
        self,
        SLEAP_instance: SLEAPanalysis,
    ):
        self.SLEAP = SLEAP_instance

    def data(self, CustomColumns: list[CustomColumn], Video: VideoMetadata):
        for col in CustomColumns:
            self.SLEAP.append(col.column(len(self.SLEAP)))

        ms_per_frame = (Video.fps**-1) * 1000
        for i in range(len(self.SLEAP)):
            self.SLEAP.append(pd.Series([i * ms_per_frame], name="Timestamps"))
            self.SLEAP.append(pd.Series([i], name="Frames"))
        return self.SLEAP

    def to_trial(
        self,
        TrackedData: list[str],
        DAQ: DAQData,
        Beh: BehMetadata,
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
        timestamps = self.SLEAP.tracks.loc[:, "Timestamps"].to_numpy(dtype=np.float256)

        for i, data, reduce in enumerate(zip(TrackedData, Reduced)):

            if reduce:
                times = np.array(
                    reduce_daq(pd.Series(DAQ.cache.loc[:, data]).tolist()),
                    dtype=np.float256,
                )

            else:
                times = pd.Series(DAQ.cache.loc[:, data]).to_numpy(dtype=np.float256)

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
        if len(start_indecies) != len(Beh.cache):
            raise ValueError(
                "The number of start indecies does not match the number of trials in the behavior data. Maybe reduce?"
            )

        self.trials = into_trial_format(
            self.SLEAP.tracks,
            Beh.cache.loc[:, "trialArray"],
            start_indecies,
            end_indecies,
        )
