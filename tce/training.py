r"""
This module defines some training convenience wrappers for easily training and serializing a cluster expansion model
from a list of configurations, encoded as `ase.Atoms` objects.
"""

from abc import abstractmethod
from typing import Union, Protocol, runtime_checkable
import logging

import numpy as np
from numpy.typing import NDArray
from ase import Atoms


LOGGER = logging.getLogger(__name__)


def get_type_map(configurations: list[Atoms]) -> NDArray[np.str_]:

    r"""
    function that generates a species ordering for a list of configurations. this grabs all chemical types available
    within the list of configurations, and then sorts them in lexicographic order

    Args:
        configurations (list[Atoms]):
            list of atomic configurations
    """

    # not all configurations need to have the same number of types, calculate the union of types
    all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
    LOGGER.debug(f"{' '.join(all_types)} types detected in configurations")
    return np.array(sorted(list(all_types)))


@runtime_checkable
class Model(Protocol):

    r"""
    Model protocol which defines the contract of how a model should behave. This closely follows the `scikit-learn`
    contract, i.e., an object with a `.fit` method and a `.predict` method.
    """

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "Model":

        r"""
        fit with a data matrix $X$ and a target matrix $y$

        Args:
            X (NDArray[np.floating]):
                data matrix
            y (NDArray[np.floating]):
                target matrix
        """

        pass

    @abstractmethod
    def predict(self, x: NDArray[np.floating]) -> Union[NDArray[np.floating], float]:

        r"""
        predict for a particular data vector $x$

        Args:
            x (NDArray[np.floating]):
                data vector
        """

        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:

        r"""
        score a model

        Args:
            X (NDArray[np.floating]):
                data matrix
            y (NDArray[np.floating]):
                target matrix
        """


class LimitingRidge:

    r"""
    train by minimizing the limiting ridge loss:

    $$L(\beta \; | \; \lambda) = \|X\beta - y\|_2^2 + \lambda \|\beta\|_2^2$$

    $$\hat{\beta} = \lim_{\lambda\to 0^+} L(\beta\;|\;\lambda) $$
    """

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "Model":

        r"""
        fit using the Moore penrose inverse, i.e., $\hat{\beta} = X^+y$, and store the coefficients

        Args:
            X (NDArray[np.floating]):
                data matrix
            y (NDArray[np.floating]):
                target matrix
        """

        self.coef_ = np.linalg.pinv(X) @ y
        return self

    def predict(self, x: NDArray[np.floating]) -> Union[NDArray[np.floating], float]:

        r"""
        predict for a particular data vector $x$, i.e. $\hat{y} = x^\intercal \hat{\beta}

        Args:
            x (NDArray[np.floating]):
                data vector
        """

        if not hasattr(self, "coef_"):
            raise ValueError(f"need to fit {self.__class__.__name__} first!")

        return x @ self.coef_

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:

        r"""
        score a linear model with $R^2$

        Args:
            X (NDArray[np.floating]):
                data matrix
            y (NDArray[np.floating]):
                target matrix
        """

        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        return 1.0 - ss_res / ss_tot
