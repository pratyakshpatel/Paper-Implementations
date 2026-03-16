"""
rotation forest builds an ensemble of decision trees where each tree sees a rotated feature
space. these rotations are designed to preserve information while changing the axis alignment
that trees rely on for splits.

feature rotations help because standard decision trees split one feature at a time, which can
struggle with oblique class boundaries. after rotation, those boundaries can become easier for
axis-aligned splits to approximate.

in this script, rotation forest uses pca-based block rotations learned from random feature
subsets, while random rotation forest uses random orthogonal rotations from qr decomposition.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


GLOBAL_SEED = 42


# utility functions

def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    # set numpy global seed for reproducibility
    np.random.seed(seed)


def make_main_dataset(seed: int = GLOBAL_SEED) -> Tuple[np.ndarray, np.ndarray]:
    # create a synthetic dataset with correlated features
    x, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=8,
        n_repeated=2,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.01,
        random_state=seed,
    )
    return x, y


def partition_features(
    n_features: int, n_subsets: int, rng: np.random.RandomState
) -> List[np.ndarray]:
    # randomly partition feature indices into near-equal subsets
    indices = np.arange(n_features)
    rng.shuffle(indices)
    subsets = np.array_split(indices, n_subsets)
    return [subset.astype(int) for subset in subsets if subset.size > 0]


def mean_upper_triangle_correlation(matrix: np.ndarray) -> float:
    # compute mean pairwise correlation excluding diagonal
    if matrix.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(matrix)
    upper_idx = np.triu_indices_from(corr, k=1)
    return float(np.mean(corr[upper_idx]))


def rotation_tree_correlation_rotation_forest(
    model: "RotationForestClassifier", x: np.ndarray
) -> float:
    # estimate average correlation between trees using class-1 probabilities
    x_scaled = model.scaler_.transform(x)
    probs = []
    for tree, rotation in zip(model.trees_, model.rotation_matrices_):
        tree_prob = tree.predict_proba(x_scaled @ rotation)[:, 1]
        probs.append(tree_prob)
    return mean_upper_triangle_correlation(np.array(probs))


def rotation_tree_correlation_random_rotation_forest(
    model: "RandomRotationForestClassifier", x: np.ndarray
) -> float:
    # estimate average correlation between trees using class-1 probabilities
    x_scaled = model.scaler_.transform(x)
    probs = []
    for tree, rotation in zip(model.trees_, model.rotation_matrices_):
        tree_prob = tree.predict_proba(x_scaled @ rotation)[:, 1]
        probs.append(tree_prob)
    return mean_upper_triangle_correlation(np.array(probs))


def rotation_tree_correlation_random_forest(
    model: RandomForestClassifier, x: np.ndarray
) -> float:
    # estimate average correlation between trees using class-1 probabilities
    probs = [tree.predict_proba(x)[:, 1] for tree in model.estimators_]
    return mean_upper_triangle_correlation(np.array(probs))


# rotation forest classifier

class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 30,
        n_feature_subsets: int = 5,
        sample_fraction: float = 0.75,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_feature_subsets = n_feature_subsets
        self.sample_fraction = sample_fraction
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _build_rotation_matrix(
        self, x_scaled: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        # create a block-diagonal rotation matrix from pca on random feature subsets
        n_samples, n_features = x_scaled.shape
        rotation = np.eye(n_features)
        subsets = partition_features(n_features, self.n_feature_subsets, rng)

        for subset in subsets:
            block_dim = subset.size
            sample_size = max(2 * block_dim, int(self.sample_fraction * n_samples))
            sample_size = min(sample_size, n_samples)
            row_idx = rng.choice(n_samples, size=sample_size, replace=False)

            x_block = x_scaled[row_idx][:, subset]
            pca = PCA(n_components=block_dim, svd_solver="full", random_state=rng.randint(1_000_000))
            pca.fit(x_block)

            # pca.components_ has shape (block_dim, block_dim)
            # using transpose gives an orthonormal block for right-multiplication
            block_rotation = pca.components_.T
            rotation[np.ix_(subset, subset)] = block_rotation

        return rotation

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RotationForestClassifier":
        # fit scaler, then fit each tree on a pca-rotated feature space
        rng = np.random.RandomState(self.random_state)

        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x)

        self.trees_: List[DecisionTreeClassifier] = []
        self.rotation_matrices_: List[np.ndarray] = []

        for _ in range(self.n_estimators):
            rotation = self._build_rotation_matrix(x_scaled, rng)
            x_rot = x_scaled @ rotation

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(1_000_000),
            )
            tree.fit(x_rot, y)

            self.trees_.append(tree)
            self.rotation_matrices_.append(rotation)

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # average probabilities from all rotated trees
        x_scaled = self.scaler_.transform(x)
        all_probs = []
        for tree, rotation in zip(self.trees_, self.rotation_matrices_):
            x_rot = x_scaled @ rotation
            all_probs.append(tree.predict_proba(x_rot))
        return np.mean(np.array(all_probs), axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # predict class by argmax over averaged probabilities
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


# random rotation forest classifier

class RandomRotationForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    @staticmethod
    def _random_orthogonal_matrix(d: int, rng: np.random.RandomState) -> np.ndarray:
        # generate a random orthogonal matrix via qr decomposition
        a = rng.randn(d, d)
        q, r = np.linalg.qr(a)

        # fix sign to make the draw deterministic for a given seed
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1
        q = q * signs
        return q

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomRotationForestClassifier":
        # fit scaler, then fit each tree on a random orthogonally-rotated feature space
        rng = np.random.RandomState(self.random_state)

        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x)

        n_features = x_scaled.shape[1]
        self.trees_: List[DecisionTreeClassifier] = []
        self.rotation_matrices_: List[np.ndarray] = []

        for _ in range(self.n_estimators):
            rotation = self._random_orthogonal_matrix(n_features, rng)
            x_rot = x_scaled @ rotation

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(1_000_000),
            )
            tree.fit(x_rot, y)

            self.trees_.append(tree)
            self.rotation_matrices_.append(rotation)

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # average probabilities from all rotated trees
        x_scaled = self.scaler_.transform(x)
        all_probs = []
        for tree, rotation in zip(self.trees_, self.rotation_matrices_):
            x_rot = x_scaled @ rotation
            all_probs.append(tree.predict_proba(x_rot))
        return np.mean(np.array(all_probs), axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # predict class by argmax over averaged probabilities
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


# experiment runner

@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    train_time_seconds: float
    tree_correlation: float


def run_experiment() -> List[ExperimentResult]:
    # build reproducible dataset and split into train/test
    x, y = make_main_dataset(seed=GLOBAL_SEED)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=GLOBAL_SEED, stratify=y
    )

    models = [
        (
            "random forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=GLOBAL_SEED,
                n_jobs=-1,
            ),
        ),
        (
            "rotation forest (pca)",
            RotationForestClassifier(
                n_estimators=30,
                n_feature_subsets=5,
                sample_fraction=0.75,
                random_state=GLOBAL_SEED,
            ),
        ),
        (
            "random rotation forest",
            RandomRotationForestClassifier(
                n_estimators=30,
                random_state=GLOBAL_SEED,
            ),
        ),
    ]

    results: List[ExperimentResult] = []

    for name, model in models:
        start = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)

        if isinstance(model, RotationForestClassifier):
            corr = rotation_tree_correlation_rotation_forest(model, x_test)
        elif isinstance(model, RandomRotationForestClassifier):
            corr = rotation_tree_correlation_random_rotation_forest(model, x_test)
        else:
            corr = rotation_tree_correlation_random_forest(model, x_test)

        results.append(
            ExperimentResult(
                name=name,
                accuracy=acc,
                train_time_seconds=elapsed,
                tree_correlation=corr,
            )
        )

    return results


def print_results(results: Sequence[ExperimentResult]) -> None:
    # print experiment results in a clear table
    print("\nmodel comparison on synthetic classification dataset")
    print("-" * 78)
    print(f"{'model':30s} {'accuracy':>12s} {'train time (s)':>16s} {'tree corr':>12s}")
    print("-" * 78)
    for r in results:
        print(
            f"{r.name:30s} {r.accuracy:12.4f} {r.train_time_seconds:16.4f} {r.tree_correlation:12.4f}"
        )
    print("-" * 78)


# visualization

def plot_decision_boundaries(seed: int = GLOBAL_SEED) -> None:
    import matplotlib.pyplot as plt

    # create a 2d dataset where oblique boundaries are useful
    x, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )

    models = [
        (
            "random forest",
            RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        ),
        (
            "rotation forest (pca)",
            RotationForestClassifier(n_estimators=30, n_feature_subsets=2, random_state=seed),
        ),
        (
            "random rotation forest",
            RandomRotationForestClassifier(n_estimators=30, random_state=seed),
        ),
    ]

    x_min, x_max = x[:, 0].min() - 1.0, x[:, 0].max() + 1.0
    y_min, y_max = x[:, 1].min() - 1.0, x[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for ax, (name, model) in zip(axes, models):
        model.fit(x_train, y_train)
        z = model.predict(grid).reshape(xx.shape)
        test_acc = accuracy_score(y_test, model.predict(x_test))

        ax.contourf(xx, yy, z, alpha=0.35, levels=2)
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=12, edgecolor="k", linewidth=0.2)
        ax.set_title(f"{name}\nacc={test_acc:.3f}")
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")

    fig.suptitle("decision boundaries on 2d data")
    plt.show()


# main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rotation forest experiment")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="disable decision boundary visualization",
    )
    return parser.parse_args()


def main() -> None:
    set_global_seed(GLOBAL_SEED)
    args = parse_args()

    results = run_experiment()
    print_results(results)

    if not args.no_plot:
        plot_decision_boundaries(seed=GLOBAL_SEED)


if __name__ == "__main__":
    main()
