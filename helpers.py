import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

body_movement_code = {
    0: "standing #1",
    1: "standing #2",
    2: "walking",
    3: "walking_fast",
    4: "running",
}


def load_data() -> pd.DataFrame:
    """Load and reindex the dataset to get each "gesture" in MultiIndex format

    Returns:
        Pandas dataframe indexed by the body state label and repetition
    """
    data_path = Path().absolute().joinpath('data').joinpath("pison_data_interview 2.csv")
    columns = [
        "time_ms",
        "ch0",
        "ch1",
        "ch0_hp",
        "ch1_hp",
        "qx",
        "qy",
        "qz",
        "qw",
        "gx",
        "gy",
        "gz",
        "ax",
        "ay",
        "az",
        "body_label",
        "rep",
    ]
    dataset = pd.read_csv(data_path, names=columns)
    # Reindex the dataframe by body label and repetition to get each "gesture" series set independently
    dataset.set_index(["body_label", "rep"], inplace=True)
    # Sort multi-index to avoid performance warnings
    dataset.sort_index(inplace=True)
    return dataset
