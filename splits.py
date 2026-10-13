import numpy as np


class FixedSplits:
    """
    Precomputed outer-CV splitter, so every CPM model (and every sbatch job)
    shares the same subjects and train/test splits.

    fold_ids[r, i] = test fold of subject i in repeat r
    subject_ids[i] = subject id for row i (the row order X, y and covariates must follow)

    Folds are yielded repeat by repeat, matching how CPMAnalysis derives the
    repeat index from ``n_repeats``.
    """
    def __init__(self, fold_ids, subject_ids):
        self.fold_ids = np.atleast_2d(np.array(fold_ids, dtype=np.int64))
        self.subject_ids = np.asarray(subject_ids).astype(str)
        self.n_repeats = self.fold_ids.shape[0]
        self.n_splits = len(np.unique(self.fold_ids[0]))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_repeats * self.n_splits

    def split(self, X, y=None, groups=None):
        for row in self.fold_ids:
            for fold in np.unique(row):
                yield np.flatnonzero(row != fold), np.flatnonzero(row == fold)

    def save(self, path):
        np.savez(path, fold_ids=self.fold_ids, subject_ids=self.subject_ids)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data['fold_ids'], data['subject_ids'])

    def __repr__(self):
        return f"FixedSplits(n_repeats={self.n_repeats}, n_splits={self.n_splits})"