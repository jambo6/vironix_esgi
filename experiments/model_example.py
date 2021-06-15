"""
model_example.py
==========================
Here we give an example of generating data according to the statistical distribution table. We then build a model on
this generated data and compare its performance to one built using all the training data.

Note that this currently uses the most basic method of data generation, that is, assume independent normals for each.
"""
import pandas as pd

import utils
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def generate_data(
    stat_table: pd.DataFrame, num_negative_samples: int = 5000, num_positive_samples: int = 5000
) -> pd.DataFrame:
    """Generates data according to a normal distributional assumption given a class stratified statistical table.

    Currently we rigidly require the table to have column names ['0.mean', '0.std', '1.mean', '1.std']. This should of
    course be made more generalisable in future.

    Arguments:
        stat_table: Dataframe with column names ['0.mean', '0.std', '1.mean', '1.std'] and rows as features.
        num_negative_samples: Integer representing the number of negative samples to generate.
        num_positive_samples: Integer representing the number of positive samples to generate.
    """
    assert list(stat_table.columns) == ["0.mean", "0.std", "1.mean", "1.std"]

    # Generate the data
    data = []
    for label_value, num_samples in [(0, num_negative_samples), (1, num_positive_samples)]:
        # Generate as mutivariate normal with no covariances
        means = stat_table["{}.mean".format(label_value)].values
        stds = np.diag(stat_table["{}.std".format(label_value)])
        generated_features = np.random.multivariate_normal(means, stds, size=int(num_samples))

        # Add some labels
        labels = np.ones(shape=(num_samples, 1)) * label_value
        generated_data = np.concatenate((generated_features, labels), axis=1)

        # Append
        data.append(generated_data)

    # Merge the positive and negative cases together
    data = np.concatenate(data)

    # Make it a dataframe so it is standardised format
    frame = pd.DataFrame(data=data, columns=list(stat_table.index) + ['label'])

    return frame


if __name__ == "__main__":
    # Load data
    stat_table, frame_train, frame_test = utils.load_dataframes()

    # Generate data according to the distributions from stat_table
    generated_data = generate_data(stat_table)

    # Fill the train and test frames with the population means
    frame_train = frame_train.fillna(frame_train.mean(axis=0))
    frame_test = frame_test.fillna(frame_train.mean(axis=0))

    # Train two basic models
    for name, data in [('Generated', generated_data), ('Training', frame_train)]:
        model = LogisticRegression()
        model.fit(data.drop('label', axis=1), data['label'])

        # Predict
        predictions = model.predict_proba(frame_test.drop('label', axis=1))[:, 1]
        auc = roc_auc_score(frame_test['label'], predictions)

        # Report metrics
        print('{} data AUC on test: {:.3f}'.format(name, auc))
