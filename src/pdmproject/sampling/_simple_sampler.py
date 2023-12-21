from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy import dtype, ndarray

from .base import SamplerBase


class SimpleSampler(SamplerBase):
    def __init__(
        self,
        lower_bound: npt.ArrayLike = (
            -5,
            -5,
            -np.pi,
            -np.pi / 2,
            -np.pi / 2,
            -np.pi / 2,
            -np.pi,
        ),
        upper_bound: npt.ArrayLike = (
            5,
            5,
            np.pi,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi,
        ),
    ) -> None:
        """Create a uniform SimpleSampler

        Args:
            lower_bound (npt.ArrayLike(len == 7), optional): The lower bound of this Sampler. Defaults to ( -5, -5, -np.pi, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi, ).
            upper_bound (npt.ArrayLike(len == 7), optional): The upper bound of this Sampler. Defaults to ( 5, 5, np.pi, np.pi / 2, np.pi / 2, np.pi / 2, np.pi, ).
        """
        lower_bound = np.asarray(lower_bound, dtype=np.float64)
        upper_bound = np.asarray(upper_bound, dtype=np.float64)

        assert len(lower_bound) == 7, "7 lower bounds must be specified"
        assert len(upper_bound) == 7, "7 upper bounds must be specified"

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def get_sample(
        self, sample_count: int = 1
    ) -> ndarray[tuple[int, Literal[7]], dtype[np.float64]]:
        return np.random.uniform(
            self._lower_bound, self._upper_bound, size=(sample_count, 7)
        )
