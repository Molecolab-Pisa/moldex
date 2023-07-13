from __future__ import annotations
from typing import Callable, Tuple

import time

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import random
from jax import Array


# here only to provide a default without creating a dedicated submodule
def _euclidean_distance(a: ArrayLike, b: ArrayLike) -> Array:
    return ((a - b) ** 2).sum(-1) ** 0.5


def _initialize(
    n_samples: int, n_feats: int, n_centers: int
) -> Tuple[Array, Array, Array]:
    centers = jnp.full(n_centers, -1, dtype=int)
    landmarks = jnp.full((n_centers, n_feats), jnp.inf)
    distances = jnp.full(n_samples, jnp.inf)
    return centers, landmarks, distances


def _uniform_start(seed: int, n_samples: int) -> int:
    return random.randint(random.PRNGKey(seed), shape=(), minval=0, maxval=n_samples)


def _greedy_k_centers(
    X: ArrayLike, n_centers: int, metric: Callable, seed: int
) -> Tuple[Array, Array]:
    n_samples, n_feats = X.shape

    centers, landmarks, distances = _initialize(
        n_samples=n_samples, n_feats=n_feats, n_centers=n_centers
    )

    rint = _uniform_start(seed=seed, n_samples=n_samples)
    centers = centers.at[0].set(rint)
    landmarks = landmarks.at[0].set(X[rint])
    last_landmark = landmarks[0]

    @jax.jit
    def update_fun(i, val):
        centers, landmarks, distances, last_landmark = val
        dist_new = metric(X, last_landmark)
        distances = jnp.minimum(distances, dist_new)
        idx_max = jnp.argmax(distances)
        centers = centers.at[i].set(idx_max)
        landmarks = landmarks.at[i].set(X[idx_max])
        last_landmark = X[idx_max]
        return (centers, landmarks, distances, last_landmark)

    res = jax.lax.fori_loop(
        1,
        n_centers,
        update_fun,
        init_val=(centers, landmarks, distances, last_landmark),
    )
    centers, landmarks, _, _ = res
    return centers, landmarks


def k_centers(
    X: ArrayLike,
    n_centers: int,
    metric: Callable = _euclidean_distance,
    seed: int = None,
) -> Tuple[Array, Array]:
    """
    Simple greedy K-center algorithm, without cluster assignment.

    This implementation scales as O(kN), where k is the number of centers
    and N is the number of input data points. Instead of computing the every
    distance at each iteration, computes the distances to the last added
    landmark and compares with the shortest paths computed previously.

    In sampling contexts, this algorithm is also known as farthest point
    sampling (FPS).

    Args:
        X: input array, shape (n_samples, n_features)
        n_centers: number of rows / landmarks to select
        metric: callable computing the dissimilarity between
                X and a point in X, should return an array of
                shape (n_samples,)
        seed: seed for the pseudo random number generator

    Returns:
        centers: indices of the landmarks
        landmarks: selected rows of X

    References
    ----------
    [1] Har-Peled, Sariel.
        Geometric approximation algorithms.
        No. 173. American Mathematical Soc., 2011.
    [2] Vazirani, Vijay V.
        Approximation algorithms.
        Springer Science & Business Media, 2013.
    [3] Hochbaum, Dorit S., and David B. Shmoys.
        "A best possible heuristic for the k-center problem."
        Mathematics of operations research 10.2 (1985): 180-184.
    """
    n_samples = X.shape[0]

    if n_centers == n_samples:
        return X
    elif n_centers >= n_samples:
        raise ValueError("Number of centers exceeds the number of data points")

    if seed is None:
        seed = int(time.time())

    X = jnp.asarray(X)

    return _greedy_k_centers(X=X, n_centers=n_centers, metric=metric, seed=seed)


farthest_point_sampling = k_centers


class FarthestPointSampling:
    def __init__(self, n_centers: int, metric: Callable = _euclidean_distance) -> None:
        """
        Args:
            n_centers: number of rows / landmarks to select
            metric: callable computing the dissimilarity between
                    X and a point in X, should return an array of
                    shape (n_samples,)
        """
        self.n_centers = n_centers
        self.metric = metric

    def sample(self, X: ArrayLike, seed: int = None) -> Tuple[Array, Array]:
        """
        Args:
            X: input array, shape (n_samples, n_features)
            seed: seed for the pseudo random number generator

        Returns:
            centers: indices of the landmarks
            landmarks: selected rows of X
        """
        centers, landmarks = k_centers(
            X=X, n_centers=self.n_centers, metric=self.metric, seed=seed
        )
        # do not store the landmarks, avoid occupying memory
        self.centers_ = centers
        return centers, landmarks


FarthestPointSampling.__doc__ = farthest_point_sampling.__doc__

# Alias
FPS = FarthestPointSampling
