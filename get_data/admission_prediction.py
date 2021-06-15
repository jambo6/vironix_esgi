"""
admission_prediction.py
=======================
Processing script for the "admission prediction" dataset.

Dataset can be found as an R script at https://github.com/yaleemmlc/admissionprediction. Here we assume that the data
has already been converted from .Rdata -> .csv.

Note that the raw data is very large (~1.7gb). The data is of shape 560,486 x 972. It is unlikely all 972 features will
be relevant to us here so we will generate a reduced version.
"""
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
DATA_FILE = DATA_DIR / "admissionprediction.csv"

# Check data is setup correctly
assert os.path.isdir(
    "../data"
), "You must make a /data/ directory from root and add the admissionprediction.csv file in /data/raw"
assert os.path.exists(DATA_FILE), "Cannot find file at {}.".format(DATA_FILE)
if not os.path.isdir(PROCESSED_DIR):
    os.mkdir(PROCESSED_DIR)

COLUMN_SUBSET = [
    # Features
    "cc_headache",
    "cc_shortnessofbreath",
    "cc_breathingdifficulty",
    "cc_breathingproblem",
    "cc_chestpain",
    "pulse_last",
    "resp_last",
    "spo2_last",
    "temp_last",
    "sbp_last",
    "dbp_last",
    # Labels
    "influenza",
    "cc_influenza",
]


def create_influenza_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Creates influenza labels from influenza and cc_influenza columns by marking 1 if either are true."""
    assert all([x in frame.columns for x in ("influenza", "cc_influenza")]), (
        "This function requires both influenza " "and cc_influenza labels."
    )
    frame["label"] = frame[["influenza", "cc_influenza"]].max(axis=1)
    frame.drop(["influenza", "cc_influenza"], axis=1, inplace=True)
    frame['label'] = frame['label'].astype(int)
    return frame


def create_splits(
    frame: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    shuffle: bool = True,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates three splits for the data according to the specified ratios.

    Arguments:
        frame: The dataframe to split.
        ratios: The ratios for which to make the splits, must sum to 1.
        shuffle: Whether to first shuffle the dataset.
        stratify: Whether to stratify the splits. If this is set, there must be a column called 'label' that will be
            stratified on.

    Returns:
        Three dataframes that have been split according to the corresponding sizes.
    """
    assert sum(ratios) == 1
    stratify_indexes = frame["label"] if stratify else None

    # Perform the first split to get the test data
    inner_frame, frame_3 = train_test_split(
        frame, train_size=sum(ratios[:-1]), test_size=ratios[-1], stratify=stratify_indexes, shuffle=shuffle
    )

    # Perform the statistical/train split
    if stratify:
        stratify_indexes = inner_frame["label"]
    frame_1, frame_2 = train_test_split(
        inner_frame, train_size=ratios[0], test_size=ratios[1], stratify=stratify_indexes, shuffle=shuffle
    )

    return frame_1, frame_2, frame_3


def create_statistical_table(frame: pd.DataFrame, mode='mean_std') -> pd.DataFrame:
    """Creates a table of statistics from an ML-feature style dataframe with label conditioning.

    The conditioned column MUST be called 'label'.

    Arguments:
        frame: Standard ML dataframe, contains feature columns and rows as samples.
        mode: The mode of statistic computation.

    Returns:
        A dataframe that contains features as rows and conditioned column split as the columns. Entries are the
        statistics.
    """
    label_grouped = frame.groupby('label')

    if mode == 'mean_std':
        mean, std = label_grouped.mean().T, label_grouped.std().T
        mean.columns = ["{}.mean".format(x) for x in mean.columns]
        std.columns = ["{}.std".format(x) for x in std.columns]

        # Make the frame
        frame_stat = pd.concat((mean, std), axis=1)
        frame_stat = frame_stat[sorted(frame_stat.columns)]
    elif mode == 'proba':
        raise NotImplementedError("Need to decide how to handle continuous variables before this works.")
    else:
        raise NotImplementedError("Statistical tables are implemented only for ('mean_std', 'proba') methods.")

    return frame_stat


if __name__ == "__main__":
    # Load in the raw dataframe
    frame = pd.read_csv(DATA_FILE, index_col=0)

    # Save the column names so they can be viewed easily
    with open(DATA_DIR / "column_names.txt", "w") as file:
        file.write("\n".join(frame.columns))

    # Take a subsample of the columns that might be of use, we can modify this later
    # Note that there are A LOT (972 total features), the list of considered features should be continuously improved
    subframe = frame[COLUMN_SUBSET]

    # Make some labels. Here we use influenza.
    subframe = create_influenza_labels(subframe)

    # Save the data as reduced (as less columns) influenza data
    subframe.to_csv(DATA_DIR / "reduced_influenza.csv")

    # Split the data in to three splits: statistic data, train data, test data
    frame_stat, frame_train, frame_test = create_splits(subframe, stratify=True)

    # Save the information to the processed directory
    group_with_name = [('frame_stat', frame_stat), ('frame_train', frame_train), ('frame_test', frame_test)]
    for name, f in group_with_name:
        f.to_csv(PROCESSED_DIR / '{}.csv'.format(name))

    # Finally create a statistical table from the frame_stat data
    stat_table = create_statistical_table(frame_stat)
    stat_table.to_csv(PROCESSED_DIR / 'frame_stat_table.csv')


