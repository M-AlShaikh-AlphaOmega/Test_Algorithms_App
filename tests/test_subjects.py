import numpy as np
from acare_ml.subjects.splitter import subject_train_test_split


def test_subject_train_test_split():
    X = np.arange(100).reshape(-1, 1)
    y = np.array([0] * 50 + [1] * 50)
    subject_ids = np.array([f"S{i//10}" for i in range(100)])

    X_train, X_test, y_train, y_test = subject_train_test_split(
        X, y, subject_ids, test_size=0.2, random_state=42
    )

    train_subjects = set([subject_ids[i] for i in range(len(X)) if X[i] in X_train])
    test_subjects = set([subject_ids[i] for i in range(len(X)) if X[i] in X_test])

    assert len(train_subjects & test_subjects) == 0
