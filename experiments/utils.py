"""
utils.py
=====================
Helpful functions for the experiments.
"""
import os.path
from pathlib import Path
import pandas as pd
from typing import Tuple

PROCESSED_DIR = Path("../data/processed")


def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the stat train and test data in that order."""
    names = ["frame_stat_table", "frame_train", "frame_test"]

    frames = []
    for name in names:
        file = PROCESSED_DIR / "{}.csv".format(name)
        assert os.path.isfile(file), "Cannot find file: {}. Have you run get_data/admission_prediction.py?".format(file)
        frames.append(pd.read_csv(PROCESSED_DIR / "{}.csv".format(name), index_col=0))

    return frames
