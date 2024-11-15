"""
Bayesian Inversion
"""

from functools import cached_property
import numpy as np


class Inversion:
    """
    Base batch inversion class

    Attributes
    ----------
    z : np.ndarray
        Observed data
    c : np.ndarray
        Constant (background) data
    x_0 : np.ndarray
        Prior state estimate
    H : np.ndarray
        Jacobian matrix
    S_0 : np.ndarray
        Prior error covariance
    S_z : np.ndarray
        Model-data mismatch covariance
    """

    def __init__(self,
                 z: np.ndarray,
                 c: np.ndarray,
                 x_0: np.ndarray,
                 H: np.ndarray,
                 S_0: np.ndarray,
                 S_z: np.ndarray,
                 rf: float = 1.0,
                 verbose: bool = True
                 ):
        """
        Initialize inversion object

        Parameters
        ----------
        z : np.ndarray
            Observed data
        c : np.ndarray
            Constant (background) data
        x_0 : np.ndarray
            Prior state estimate
        H : np.ndarray
            Jacobian matrix
        S_0 : np.ndarray
            Prior error covariance
        S_z : np.ndarray
            Model-data mismatch covariance
        """
        self.z = z
        self.c = c
        self.x_0 = x_0
        self.H = H
        self.S_0 = S_0
        self.S_z = S_z

        self.n_obs = S_z.shape[0]
        self.n_state = S_0.shape[0]

    def forward(self, x):
        """
        Forward model

        .. math::
            y = Hx + c
        """
        print('Performing forward calculation...')
        return self.H @ x + self.c

    def residual(self, x):
        """
        Forward model residual

        .. math::
            r = z - (Hx + c)
        """
        print('Performing residual calculation...')
        return self.z - self.forward(x)

    def cost(self, x):
        """
        Cost function

        .. math::
            J(x) = \\frac{1}{2}(x - x_0)^T S_0^{-1}(x - x_0) + \\frac{1}{2}(z - Hx - c)^T S_z^{-1}(z - Hx - c)
        """
        print('Performing cost calculation...')
        cost_model = (x - self.x_0).T @ np.linalg.pinv(self.S_0) @ (x - self.x_0)
        cost_data = (self.z - self.forward(x)).T @ np.linalg.pinv(self.S_z) @ (self.z - self.forward(x))
        return 0.5 * (cost_model + cost_data)

    @cached_property
    def K(self):
        """
        Kalman Gain Matrix

        .. math::
            K = (H S_0)^T (H S_0 H^T + S_z)^{-1}
        """
        print('Calculating Kalman Gain Matrix...')
        return (self.H @ self.S_0).T @ np.linalg.pinv(self.H @ self.S_0 @ self.H.T + self.S_z)  # TODO auto-completed - not sure if correct

    @cached_property
    def S_hat(self):
        """
        Posterior Error Covariance Matrix

        .. math::
            \\hat{S} = (H^T S_z^{-1} H + S_0^{-1})^{-1}
                = S_0 - (H S_0)^T(H S_0 H^T + S_z)^{-1}(H S_0)
        """
        print('Calculating Posterior Error Covariance Matrix...')
        return np.linalg.pinv(self.H.T @ np.linalg.pinv(self.S_z) @ self.H + np.linalg.pinv(self.S_0))
        # return self.S_0 - (self.H @ self.S_0).T @ np.linalg.pinv(self.H @ self.S_0 @ self.H.T + self.S_z) @ (self.H @ self.S_0)

    @cached_property
    def x_hat(self):
        """
        Posterior Mean State Estimate (solution)

        .. math::
            \\hat{x} = x_0 + K(z - Hx_0 - c)
        """
        print('Calculating Posterior Mean State Estimate...')
        return self.x_0 + self.K @ self.residual(self.x_0)

    @cached_property
    def A(self):
        """
        Average Kernel Matrix

        .. math::
            A = KH
        """
        print('Calculating Average Kernel Matrix...')
        return self.K @ self.H

    @cached_property
    def y_hat(self):
        """
        Posterior Mean Data Estimate

        .. math::
            \\hat{y} = H \\hat{x} + c
        """
        print('Calculating Posterior Mean Data Estimate...')
        return self.forward(self.x_hat)
