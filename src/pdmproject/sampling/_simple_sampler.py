from typing import Literal, Optional

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
        self, sample_count: Optional[int] = None
    ) -> ndarray[tuple[int, Literal[7]] | tuple[Literal[7]], dtype[np.float64]]:
        if sample_count is None:
            size = None
        else:
            size = (sample_count, 7)
        return np.random.uniform(self._lower_bound, self._upper_bound, size=size)

    @SamplerBase.lower_bound.getter
    def lower_bound(self):
        return self._lower_bound

    @SamplerBase.upper_bound.getter
    def upper_bound(self):
        return self._upper_bound
