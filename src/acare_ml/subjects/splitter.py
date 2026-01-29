import numpy as np
import pandas as pd


def subject_train_test_split(
    X, y, subject_ids, test_size: float = 0.2, random_state: int = 42, stratify_by_label: bool = True
):
    unique_subjects = np.unique(subject_ids)
    np.random.seed(random_state)

    if stratify_by_label and y is not None:
        subject_labels = pd.DataFrame({"subject": subject_ids, "label": y}).groupby("subject")["label"].first()
        positive_subjects = subject_labels[subject_labels == 1].index.values
        negative_subjects = subject_labels[subject_labels == 0].index.values

        n_pos_test = int(len(positive_subjects) * test_size)
        n_neg_test = int(len(negative_subjects) * test_size)

        np.random.shuffle(positive_subjects)
        np.random.shuffle(negative_subjects)

        test_subjects = np.concatenate([positive_subjects[:n_pos_test], negative_subjects[:n_neg_test]])
    else:
        np.random.shuffle(unique_subjects)
        n_test = int(len(unique_subjects) * test_size)
        test_subjects = unique_subjects[:n_test]

    test_mask = np.isin(subject_ids, test_subjects)
    train_mask = ~test_mask

    if isinstance(X, pd.DataFrame):
        X_train, X_test = X[train_mask], X[test_mask]
    else:
        X_train, X_test = X[train_mask], X[test_mask]

    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test
