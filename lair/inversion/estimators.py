"""
Inversion estimators.

This module contains various inversion estimators for solving inverse problems.
"""

from functools import cached_property
import numpy as np
from numpy.linalg import inv as invert

from lair.inversion.core import Estimator, ESTIMATOR_REGISTRY

# TODO
# - implement bayesian regularization factor usage


@ESTIMATOR_REGISTRY.register('bayesian')
class BayesianSolver(Estimator):
    """
    Bayesian inversion estimator class
    This class implements a Bayesian inversion framework for solving inverse problems,
    also known as the batch method.
    """

    def __init__(self,
                 z: np.ndarray,
                 x_0: np.ndarray,
                 H: np.ndarray,
                 S_0: np.ndarray,
                 S_z: np.ndarray,
                 c: np.ndarray | float | None = None,
                 rf: float = 1.0,
                 ):
        """
        Initialize inversion object

        Parameters
        ----------
        z : np.ndarray
            Observed data
        x_0 : np.ndarray
            Prior model estimate
        H : np.ndarray
            Forward operator
        S_0 : np.ndarray
            Prior error covariance
        S_z : np.ndarray
            Model-data mismatch covariance
        c : np.ndarray | float, optional
            Constant data, defaults to 0.0
        rf : float, optional
            Regularization factor, by default 1.0
        """
        super().__init__(z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, c=c)
        self.rf = rf  # TOOD implement usage of regularization factor

    def cost(self, x):
        """
        Cost function

        .. math::
            J(x) = \\frac{1}{2}(x - x_0)^T S_0^{-1}(x - x_0) + \\frac{1}{2}(z - Hx - c)^T S_z^{-1}(z - Hx - c)
        """
        print('Performing cost calculation...')
        diff_model = x - self.x_0
        diff_data = self.residual(x)
        cost_model = diff_model.T @ self._S_0_inv @ diff_model
        cost_data = diff_data.T @ self._S_z_inv @ diff_data
        return 0.5 * (cost_model + cost_data)

    @cached_property
    def x_hat(self):
        """
        Posterior Mean Model Estimate (solution)

        .. math::
            \\hat{x} = x_0 + K(z - Hx_0 - c)
        """
        print('Calculating Posterior Mean Model Estimate...')
        return self.x_0 + self.K @ self.residual(self.x_0)

    @cached_property
    def S_hat(self):
        """
        Posterior Error Covariance Matrix

        .. math::
            \\hat{S} = (H^T S_z^{-1} H + S_0^{-1})^{-1}
                = S_0 - (H S_0)^T(H S_0 H^T + S_z)^{-1}(H S_0)
        """
        print('Calculating Posterior Error Covariance Matrix...')
        # Both methods return the same result
        # return invert(self.H_T @ self.S_z_inv @ self.H + self.S_0_inv)
        return self.S_0 - self._HS_0.T @ invert(self._HS_0H + self.S_z) @ self._HS_0  # this one only has one invert call
