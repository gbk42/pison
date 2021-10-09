"""Module containing basic helper functions"""
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import itertools

EMG_CHANNELS = ["ch0", "ch1"]
EMG_HP_CHANNELS = ["ch0_hp", "ch1_hp"]
IMU_CHANNELS = ["qx", "qy", "qz", "qw"]
GYRO_CHANNELS = ["gx", "gy", "gz"]
ACC_CHANNELS = ["ax", "ay", "az"]

body_movement_code = {
    0: "standing #1",
    1: "standing #2",
    2: "walking",
    3: "walking_fast",
    4: "running",
}


def load_data(uniform_length: bool = True) -> pd.DataFrame:
    """Load and reindex the dataset to get each "gesture" in MultiIndex format

    Args:
        uniform_length: Find the minimum length of each observation and crop the rest to this length

    Returns:
        Pandas dataframe indexed by the body state label and repetition
    """
    data_path = (
        Path().absolute().joinpath("data").joinpath("pison_data_interview 2.csv")
    )
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
    # Add a samples column and add it to the index
    for observation in itertools.product(range(5), range(1, 4)):
        dataset.loc[observation, "sample_num"] = range(len(dataset.loc[observation]))

    min_length = min(
        [
            dataset.loc[combo].shape[0]
            for combo in itertools.product(range(5), range(1, 4))
        ]
    )
    if uniform_length:
        print(f"Trimming observations to {min_length} samples")
        dataset = dataset[dataset.sample_num < min_length]
    return dataset
