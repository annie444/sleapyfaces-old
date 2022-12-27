import pandas as pd
from sklearn.decomposition import PCA


def mean_center(data: pd.DataFrame, track_names: list[str]) -> pd.DataFrame:
    """Mean center the data."""
    num_data = data.loc[:, track_names]
    num_data = num_data - num_data.mean()
    data.loc[:, track_names] = num_data
    return data


def z_score(data: pd.DataFrame, track_names: list[str]) -> pd.DataFrame:
    """Mean center the data."""
    data = mean_center(data, track_names)
    for track in track_names:
        data.loc[:, track] = data.loc[:, track] / data.loc[:, track].std()
    return data


def pca(data: pd.DataFrame, track_names: list[str]) -> dict[str, pd.DataFrame]:
    num_data = data.loc[:, track_names]
    qual_data = data.drop(columns=track_names)
    pcas = {}

    pca2d = PCA(n_components=2)
    pca3d = PCA(n_components=2)

    num_data_2d = pca2d.fit_transform(num_data)
    num_data_3d = pca3d.fit_transform(num_data)

    num_data_2d = pd.DataFrame(
        num_data_2d, columns=["principal component 1", "principal component 2"]
    )
    num_data_3d = pd.DataFrame(
        num_data_3d,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )

    pcas["pca2d"] = pd.concat([qual_data.reset_index(), num_data_2d], axis=1)
    pcas["pca3d"] = pd.concat([qual_data.reset_index(), num_data_3d], axis=1)

    return pcas
