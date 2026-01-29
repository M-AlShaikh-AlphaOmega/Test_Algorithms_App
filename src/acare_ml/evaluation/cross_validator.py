from sklearn.model_selection import BaseCrossValidator
import numpy as np


class SubjectAwareCV(BaseCrossValidator):
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups (subject IDs) must be provided")

        unique_subjects = np.unique(groups)
        np.random.shuffle(unique_subjects)
        fold_size = len(unique_subjects) // self.n_splits

        for i in range(self.n_splits):
            test_subjects = unique_subjects[i * fold_size : (i + 1) * fold_size]
            test_idx = np.isin(groups, test_subjects)
            train_idx = ~test_idx
            yield np.where(train_idx)[0], np.where(test_idx)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
