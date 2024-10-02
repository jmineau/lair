"""
Bayesian Inversion
"""

from functools import cached_property
from pandas import DataFrame
import xarray as xr
from xarray import DataArray


class Inversion:
    """
    Base batch inversion class

    Attributes
    ----------
    z : xarray.DataArray
        Observed data
    c : xarray.DataArray
        Constant (background) data
    x_0 : xarray.DataArray
        Prior state estimate
    H : xarray.DataArray
        Jacobian matrix
    S_0 : xarray.DataArray
        Prior error covariance
    S_z : xarray.DataArray
        Model-data mismatch covariance
    """

    def __init__(self,
                 z: DataArray | DataFrame,
                 c: DataArray | DataFrame,
                 x_0: DataArray,
                 H: DataArray,
                 S_0: DataArray,
                 S_z: DataArray,
                 rf: float = 1.0,
                 verbose: bool = True
                 ):
        """
        Initialize inversion object

        Parameters
        ----------
        z : xarray.DataArray | pandas.DataFrame
            Observed data
        c : xarray.DataArray | pandas.DataFrame
            Constant (background) data
        x_0 : xarray.DataArray
            Prior state estimate
        H : xarray.DataArray
            Jacobian matrix
        S_0 : xarray.DataArray
            Prior error covariance
        S_z : xarray.DataArray
            Model-data mismatch covariance
        """
        self.z = z if isinstance(z, DataArray) else DataArray(z)
        self.c = c if isinstance(c, DataArray) else DataArray(c)
        self.x_0 = x_0
        self.H = H
        self.S_0 = S_0
        self.S_z = S_z

        self.n_obs = len(z)
        self.n_state = H.shape[1]  # FIXME: check if this is correct

    def forward(self, x):
        """
        Forward model

        .. math::
            y = Hx + c
        """
        return self.H @ x + self.c

    def residual(self, x):
        """
        Forward model residual

        .. math::
            r = z - (Hx + c)
        """
        return self.z - self.forward(x)

    def cost(self, x):
        """
        Cost function

        .. math::
            J(x) = \\frac{1}{2}(x - x_0)^T S_0^{-1}(x - x_0) + \\frac{1}{2}(z - Hx - c)^T S_z^{-1}(z - Hx - c)
        """
        cost_model = (x - self.x_0).T @ xr.linalg.pinv(self.S_0) @ (x - self.x_0)
        cost_data = (self.z - self.forward(x)).T @ xr.linalg.pinv(self.S_z) @ (self.z - self.forward(x))
        return 0.5 * (cost_model + cost_data)

    @cached_property
    def K(self):
        """
        Kalman Gain Matrix

        .. math::
            K = (H S_0)^T (H S_0 H^T + S_z)^{-1}
        """
        return (self.H @ self.S_0).T @ xr.linalg.pinv(self.H @ self.S_0 @ self.H.T + self.S_z)  # TODO auto-completed - not sure if correct

    @cached_property
    def S_hat(self):
        """
        Posterior Error Covariance Matrix

        .. math::
            \\hat{S} = (H^T S_z^{-1} H + S_0^{-1})^{-1}
                = S_0 - (H S_0)^T(H S_0 H^T + S_z)^{-1}(H S_0)
        """
        return xr.linalg.pinv(H.T @ xr.linalg.pinv(S_z) @ H + xr.linalg.pinv(S_0))
        # return self.S_0 - (self.H @ self.S_0).T @ xr.linalg.pinv(self.H @ self.S_0 @ self.H.T + self.S_z) @ (self.H @ self.S_0)

    @cached_property
    def x_hat(self):
        """
        Posterior Mean State Estimate (solution)

        .. math::
            \\hat{x} = x_0 + K(z - Hx_0 - c)
        """
        return self.x_0 + self.K @ self.residual(self.x_0)

    @cached_property
    def A(self):
        """
        Average Kernel Matrix

        .. math::
            A = KH
        """
        return self.K @ self.H

    @cached_property
    def y_hat(self):
        """
        Posterior Mean Data Estimate

        .. math::
            \\hat{y} = H \\hat{x} + c
        """
        return self.forward(self.x_hat)
