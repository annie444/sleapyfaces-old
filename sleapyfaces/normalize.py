import pandas as pd


def mean_center(data: pd.DataFrame, track_names: list[str]) -> pd.DataFrame:
    """Mean center the data."""
    num_data = data.loc[:, track_names]
    num_data = num_data - num_data.mean()
    data.loc[:, track_names] = num_data


def z_score(data: pd.DataFrame, track_names: list[str]) -> pd.DataFrame:
    """Mean center the data."""
    data = mean_center(data, track_names)
    for track in track_names:
        data.loc[:, track] = data.loc[:, track] / data.loc[:, track].std()
    return data
