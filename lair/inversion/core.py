"""
Core inversion classes and functions.

Inversion is a mathematical technique used to estimate unknown model parameters or states from observed data, typically by solving an optimization problem that combines prior knowledge, a forward model, and error statistics. The goal of inversion is to find the most probable model state (the "posterior") that explains the observations, given uncertainties in both the model and the data.
This module provides core classes and utilities for formulating and solving inverse problems, including:
- Abstract base classes for inversion estimators.
- Registry for estimator implementations.
- Forward operator and covariance matrix wrappers with index-aware functionality.
- Utilities for convolving state vectors with forward operators.
- The `InverseProblem` class, which orchestrates the alignment of data, prior information, error covariances, and the solution process.
Typical inversion workflows involve:
    - Defining observed data and prior model state estimates.
    - Specifying a forward operator that maps model states to observations.
    - Providing error covariance matrices for both prior and observation uncertainties.
    - Solving for the posterior state estimate and its uncertainty using an estimator.
The module is designed to support flexible data structures (e.g., pandas, xarray), robust index alignment, and extensible estimator implementations for a variety of inversion methodologies.
"""

from abc import ABC, abstractmethod
from functools import cached_property, partial
from pathlib import Path
from typing_extensions import \
    Self  # requires python 3.11 to import from typing

import numpy as np
from numpy.linalg import inv as invert
import pandas as pd
import xarray as xr


from lair.inversion.utils import dataframe_matrix_to_xarray, round_index


# TODO
# - Obs aggregation
# - file io


class Estimator(ABC):
    """
    Base inversion estimator class.

    Attributes
    ----------
    z : np.ndarray
        Observed data.
    x_0 : np.ndarray
        Prior model state estimate.
    H : np.ndarray
        Forward operator.
    S_0 : np.ndarray
        Prior error covariance.
    S_z : np.ndarray
        Model-data mismatch covariance.
    c : np.ndarray or float, optional
        Constant data, defaults to 0.0.
    n_z : int
        Number of observations.
    n_x : int
        Number of state variables.
    x_hat : np.ndarray
        Posterior mean model state estimate (solution).
    S_hat : np.ndarray
        Posterior error covariance.
    y_hat : np.ndarray
        Posterior modeled observations.
    y_0 : np.ndarray
        Prior modeled observations.
    K : np.ndarray
        Kalman gain.
    A : np.ndarray
        Averaging kernel.
    chi2 : float
       Chi-squared statistic.
    R2 : float
       Coefficient of determination.
    RMSE : float
       Root mean square error.
    U_red : np.ndarray
       Reduced uncertainty.

    Methods
    -------
    cost(x: np.ndarray) -> float
        Cost/loss/misfit function.
    forward(x: np.ndarray) -> np.ndarray
        Forward model calculation.
    residual(x: np.ndarray) -> np.ndarray
        Forward model residual.
    leverage(x: np.ndarray) -> np.ndarray
        Calculate the leverage matrix.
    """

    def __init__(self,
                 z: np.ndarray,
                 x_0: np.ndarray,
                 H: np.ndarray,
                 S_0: np.ndarray,
                 S_z: np.ndarray,
                 c: np.ndarray | float | None = None,
                 ):
        """
        Initialize the Estimator object.

        Parameters
        ----------
        z : np.ndarray
            Observed data.
        x_0 : np.ndarray
            Prior model state estimate.
        H : np.ndarray
            Forward operator.
        S_0 : np.ndarray
            Prior error covariance.
        S_z : np.ndarray
            Model-data mismatch covariance.
        c : np.ndarray or float, optional
            Constant data, defaults to 0.0.
        """
        self.z = z
        self.x_0 = x_0
        self.H = H
        self.S_0 = S_0
        self.S_z = S_z
        self.c = c if c is not None else 0.0

        self.n_z = z.shape[0]
        self.n_x = x_0.shape[0]

    def forward(self, x) -> np.ndarray:
        """
        Forward model calculation.

        .. math::
            y = Hx + c

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Model output (Hx + c).
        """
        print('Performing forward calculation...')
        return self.H @ x + self.c

    def residual(self, x) -> np.ndarray:
        """
        Forward model residual.

        .. math::
            r = z - (Hx + c)

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Residual (z - (Hx + c)).
        """
        print('Performing residual calculation...')
        return self.z - self.forward(x)

    def leverage(self, x) -> np.ndarray:
        """
        Calculate the leverage matrix.

        Which observations are likely to have more impact on the solution.

        .. math::
            L = Hx ((Hx)^T (H S_0 H^T + S_z)^{-1} Hx)^{-1} (Hx)^T (H S_0 H^T + S_z)^{-1}

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Leverage matrix.
        """
        print('Calculating Leverage matrix...')
        Hx = self.forward(x)
        Hx_T = Hx.T
        HS_0H_Sz_inv = invert(self._HS_0H + self.S_z)
        return Hx @ invert(Hx_T @ HS_0H_Sz_inv @ Hx) @ Hx_T @ HS_0H_Sz_inv

    @abstractmethod
    def cost(self, x) -> float:
        """
        Cost/loss/misfit function.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        float
            Cost value.
        """
        print('Performing cost calculation...')
        raise NotImplementedError

    @property
    @abstractmethod
    def x_hat(self) -> np.ndarray:
        """
        Posterior mean model state estimate (solution).

        Returns
        -------
        np.ndarray
            Posterior state estimate.
        """
        print('Calculating Posterior Mean Model State Estimate...')
        raise NotImplementedError

    @property
    @abstractmethod
    def S_hat(self) -> np.ndarray:
        """
        Posterior error covariance matrix.

        Returns
        -------
        np.ndarray
            Posterior error covariance matrix.
        """
        print('Calculating Posterior Error Covariance Matrix...')
        raise NotImplementedError

    @cached_property
    def y_hat(self) -> np.ndarray:
        """
        Posterior mean observation estimate.

        .. math::
            \\hat{y} = H \\hat{x} + c

        Returns
        -------
        np.ndarray
            Posterior observation estimate.
        """
        print('Calculating Posterior Mean Observation Estimate...')
        return self.forward(self.x_hat)

    @cached_property
    def y_0(self) -> np.ndarray:
        """
        Prior mean data estimate.

        .. math::
            \\hat{y}_0 = H x_0 + c

        Returns
        -------
        np.ndarray
            Prior data estimate.
        """
        print('Calculating Prior Mean Data Estimate...')
        return self.forward(self.x_0)

    @cached_property
    def K(self):
        """
        Kalman gain matrix.

        .. math::
            K = (H S_0)^T (H S_0 H^T + S_z)^{-1}

        Returns
        -------
        np.ndarray
            Kalman gain matrix.
        """
        print('Calculating Kalman Gain Matrix...')
        return self._HS_0.T @ invert(self._HS_0H + self.S_z)

    @cached_property
    def A(self):
        """
        Averaging kernel matrix.

        .. math::
            A = KH = (H S_0)^T (H S_0 H^T + S_z)^{-1} H

        Returns
        -------
        np.ndarray
            Averaging kernel matrix.
        """
        print('Calculating Averaging Kernel Matrix...')
        return self.K @ self.H

    @cached_property
    def _H_T(self):
        """
        Transpose of the forward operator
        """
        return self.H.T

    @cached_property
    def _HS_0(self):
        """
        ... math::
            H S_0
        """
        return self.H @ self.S_0

    @cached_property
    def _HS_0H(self):
        """
        ... math::
            H S_0 H^T
        """
        return self._HS_0 @ self._H_T

    @cached_property
    def _S_0_inv(self):
        """
        Inverse of prior error covariance matrix
        """
        return invert(self.S_0)

    @cached_property
    def _S_z_inv(self):
        """
        Inverse of model-data mismatch covariance matrix
        """
        return invert(self.S_z)

    @cached_property
    def DOFS(self) -> float:
        """
        Degrees Of Freedom for Signal (DOFS).

        .. math::
            DOFS = Tr(A)

        Returns
        -------
        float
            Degrees of Freedom value.
        """
        return np.trace(self.A)

    @cached_property
    def chi2(self) -> float:
        """
        Reduced Chi-squared statistic.

        .. math::
            \\chi^2 = \\frac{1}{n_z} ((z - H\\hat{x})^T S_z^{-1} (z - H\\hat{x}) + (\\hat{x} - x_0)^T S_0^{-1} (\\hat{x} - x_0))

        Returns
        -------
        float
            Reduced Chi-squared value.
        """
        # TBH im not 100% sure this is right
        return (self.chi2_obs + self.chi2_state) / self.n_z

    @cached_property
    def chi2_obs(self) -> float:
        """
        Chi-squared statistic for observation params

        .. math::
            \\chi^2 = (z - H\\hat{x})^T S_z^{-1} (z - H\\hat{x})

        Returns
        -------
        float
            Chi-squared value.
        """
        r = self.residual(self.x_hat)
        return (r.T @ self._S_z_inv @ r) / self.n_z

    @cached_property
    def chi2_state(self) -> float:
        """
        Chi-squared statistic for state params

        .. math::
            \\chi^2 = (\\hat{x} - x_0)^T S_0^{-1} (\\hat{x} - x_0)
        """
        r = self.x_hat - self.x_0
        return (r.T @ self._S_0_inv @ r) / self.n_x

    @cached_property
    def R2(self) -> float:
        """
        Coefficient of determination (R-squared).

        .. math::
            R^2 = corr(z, H\\hat{x})^2

        Returns
        -------
        float
            R-squared value.
        """
        print('Calculating Coefficient of determination (R-squared)...')
        return np.corrcoef(self.z, self.y_hat)[0, 1] ** 2

    @cached_property
    def RMSE(self) -> float:
        """
        Root mean square error (RMSE).

        .. math::
            RMSE = \\sqrt{\\frac{(z - H\\hat{x})^2}{n_z}}
        Returns
        -------
        float
            RMSE value.
        """
        print('Calculating Root Mean Square Error (RMSE)...')
        r = self.residual(self.x_hat)
        return np.sqrt((r ** 2) / self.n_z)

    @cached_property
    def U_red(self):
        """
        Uncertainty reduction metric.

        .. math::
            U_{red} = 1 - \\frac{\\sqrt{trace(\\hat{S})}}{\\sqrt{trace(S_0)}}

        Returns
        -------
        float
            Uncertainty reduction value.
        """
        print('Calculating Uncertainty reduction metric...')
        return 1 - (np.sqrt(np.trace(self.S_hat)) / np.sqrt(np.trace(self.S_0)))


class EstimatorRegistry(dict):
    """
    Registry for estimator classes.
    """
    def register(self, name: str):
        """
        Register an estimator class under a given name.

        Parameters
        ----------
        name : str
            Name to register the estimator under.

        Returns
        -------
        decorator : function
            Decorator to register the class.
        """
        def decorator(cls: type[Estimator]) -> type[Estimator]:
            self[name] = cls
            return cls
        return decorator


ESTIMATOR_REGISTRY = EstimatorRegistry()


def convolve(forward_operator: pd.DataFrame, state: pd.Series,
             coord_decimals: int = 6) -> pd.Series:
    """
    Convolve a forward_operator with a state field to get modeled observations.

    Parameters
    ----------
    forward_operator : pd.DataFrame
        DataFrame with columns corresponding to the state index
        and rows corresponding to the observation index.
    state : pd.Series
        Series with rows corresponding to the state index.
    coord_decimals : int, optional
        Number of decimal places to round coordinates to when matching indices,
        by default 6.

    Returns
    -------
    pd.Series
        Series with the same index as the forward_operator,
        containing the modeled observations.
    """
    fo = forward_operator.copy()
    state = state.copy()

    # Round floating point coordinates to avoid precision issues
    fo.columns = round_index(fo.columns, decimals=coord_decimals)
    state.index = round_index(state.index, decimals=coord_decimals)

    # Ensure the state index matches the forward operator columns
    if isinstance(fo.columns, pd.MultiIndex):
        if not isinstance(state.index, pd.MultiIndex):
            raise ValueError("If forward operator columns are a MultiIndex, state index must also be a MultiIndex.")
        state.index = state.index.reorder_levels(fo.columns.names)
    common = fo.columns.union(state.index)
    fo = fo.reindex(columns=common)
    state = state.reindex(index=common)

    if np.isnan(fo).any().any():
        raise ValueError("Forward operator contains NaN values after reindexing.")

    if np.isnan(state).any():
        raise ValueError("state contains NaN values after reindexing.")

    # Perform the matrix multiplication to get modeled observations
    modeled_obs = fo @ state
    modeled_obs.name = f'{state.name}_obs'

    return modeled_obs


class ForwardOperator:
    """
    Forward operator class for modeling observations.

    Parameters
    ----------
    data : pd.DataFrame
        Forward operator matrix.

    Attributes
    ----------
    data : pd.DataFrame
        Underlying forward operator matrix.
    obs_index : pd.Index

        Observation index (row index).
    state_index : pd.Index
        State index (column index).
    obs_dims : tuple
        Observation dimension names.
    state_dims : tuple
        State dimension names.

    Methods
    -------
    convolve(state: pd.Series, coord_decimals: int = 6) -> pd.Series
        Convolve the forward operator with a state vector.
    to_xarray() -> xr.DataArray
        Convert the forward operator to an xarray DataArray.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the ForwardOperator.

        Parameters
        ----------
        data : pd.DataFrame
            Forward operator matrix.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the underlying data of the forward operator.

        Returns
        -------
        pd.DataFrame
            Forward operator matrix.
        """
        return self._data

    @property
    def obs_index(self) -> pd.Index:
        """
        Get the observation index (row index) of the forward operator.

        Returns
        -------
        pd.Index
            Observation index.
        """
        return self._data.index

    @property
    def state_index(self) -> pd.Index:
        """
        Get the state index (column index) of the forward operator.

        Returns
        -------
        pd.Index
            State index.
        """
        return self._data.columns

    @property
    def obs_dims(self) -> tuple:
        """
        Get the observation dimensions (names of the row index).

        Returns
        -------
        tuple
            Observation dimension names.
        """
        return tuple(self.obs_index.names)

    @property
    def state_dims(self) -> tuple:
        """
        Get the state dimensions (names of the column index).

        Returns
        -------
        tuple
            State dimension names.
        """
        return tuple(self.state_index.names)

    def convolve(self, state: pd.Series, coord_decimals: int = 6
                 ) -> pd.Series:
        """
        Convolve the forward operator with a state vector.

        Parameters
        ----------
        state : pd.Series
            State vector.
        coord_decimals : int, optional
            Number of decimal places to round coordinates to when matching indices,
            by default 6.

        Returns
        -------
        pd.Series
            Result of convolution.
        """
        return convolve(forward_operator=self._data, state=state,
                        coord_decimals=coord_decimals)

    def to_xarray(self) -> xr.DataArray:
        """
        Convert the forward operator to an xarray DataArray.

        Returns
        -------
        xr.DataArray
            Xarray representation of the forward operator.
        """
        """Convert the forward operator to an xarray DataArray."""
        return dataframe_matrix_to_xarray(self._data)


class SymmetricMatrix:
    """
    Symmetric matrix wrapper class for pandas DataFrames.

    Parameters
    ----------
    data : pd.DataFrame
        Symmetric matrix with identical row and column indices.

    Attributes
    ----------
    data : pd.DataFrame
        Symmetric matrix.
    index : pd.Index
        Index of the symmetric matrix.
    dims : tuple
        Dimension names of the symmetric matrix.
    values : np.ndarray
        Underlying data as a NumPy array.
    shape : tuple
        Dimensionality of the symmetric matrix.
    loc : SymmetricMatrix._Indexer
        Custom accessor for label-based selection and assignment.

    Methods
    -------
    from_numpy(array: np.ndarray, index: pd.Index) -> SymmetricMatrix
        Create a SymmetricMatrix from a NumPy array and an index.
    reindex(index: pd.Index, **kwargs) -> SymmetricMatrix
        Reindex the symmetric matrix, filling new entries with 0.
    reorder_levels(order) -> SymmetricMatrix
        Reorder the levels of a MultiIndex symmetric matrix.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the SymmetricMatrix with a square DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Square symmetric matrix.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not data.index.equals(data.columns):
            raise ValueError("Symmetric matrix must have identical row and column indices.")
        
        self._data = data
        self.loc = self.__class__._Indexer(self)

    @classmethod
    def from_numpy(cls, array: np.ndarray, index: pd.Index) -> Self:
        """
        Create a SymmetricMatrix from a NumPy array.

        Parameters
        ----------
        array : np.ndarray
            Symmetric matrix array.
        index : pd.Index
            Index for rows and columns.

        Returns
        -------
        SymmetricMatrix
            SymmetricMatrix instance.
        """
        return cls(pd.DataFrame(array, index=index, columns=index))

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the underlying data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Underlying symmetric matrix.
        """
        return self._data

    @property
    def dims(self) -> tuple:
        """
        Returns a tuple representing the dimension names of the matrix.

        Returns
        -------
        tuple
            Dimension names of the symmetric matrix.
        """
        return tuple(self.index.names)

    @property
    def index(self) -> pd.Index:
        """
        Returns the pandas Index of the matrix.

        Returns
        -------
        pd.Index
            Index of the symmetric matrix.
        """
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        """
        Sets a new index for the symmetric matrix, ensuring it remains square.

        Parameters
        ----------
        index : pd.Index
            New index for the symmetric matrix.

        Raises
        ------
        TypeError
            If the index is not a pandas Index.
        ValueError
            If the index length does not match the number of rows/columns in the matrix.
        """
        if not isinstance(index, pd.Index):
            raise TypeError("Index must be a pandas Index.")
        if len(index) != self.data.shape[0]:
            raise ValueError("Index length must match the number of rows/columns in the matrix.")
        self._data.index = index
        self._data.columns = index

    @property
    def values(self) -> np.ndarray:
        """
        Returns the underlying data as a NumPy array.

        Returns
        -------
        np.ndarray
            Underlying data array.
        """
        return self.data.values

    @property
    def shape(self) -> tuple:
        """
        Returns a tuple representing the dimensionality of the matrix.

        Returns
        -------
        tuple
            Dimensionality of the symmetric matrix.
        """
        return self.data.shape

    def reindex(self, index: pd.Index, **kwargs) -> 'SymmetricMatrix':
        """
        Reindex the symmetric matrix, filling new entries with 0.

        Parameters
        ----------
        index : pd.Index
            New index for the symmetric matrix.
        **kwargs : additional keyword arguments
            Passed to pandas' reindex method.

        Returns
        -------
        SymmetricMatrix
            Reindexed SymmetricMatrix instance.
        """
        reindexed_data = self.data.reindex(index=index, columns=index, **kwargs).fillna(0.0)
        return self.__class__(data=reindexed_data)

    def reorder_levels(self, order) -> 'SymmetricMatrix':
        """
        Reorder the levels of a MultiIndex symmetric matrix.

        Parameters
        ----------
        order : list
            New order for the levels.

        Returns
        -------
        SymmetricMatrix
            SymmetricMatrix instance with reordered levels.

        Raises
        ------
        TypeError
            If the index is not a MultiIndex.
        """
        if not isinstance(self.index, pd.MultiIndex):
            raise TypeError("Index must be a MultiIndex to reorder levels.")
        data = self.data.copy()
        data = data.reorder_levels(order, axis='index')
        data = data.reorder_levels(order, axis='columns')
        return self.__class__(data=data)

    class _Indexer:
        """
        A custom accessor object for the SymmetricMatrix class, similar to
        pandas' .loc. It enables label-based selection and assignment while
        enforcing the symmetrical nature of a symmetric matrix.
        """
        def __init__(self, matrix_obj: 'SymmetricMatrix'):
            self._obj = matrix_obj

        def __getitem__(self, key):
            """
            Get data from the symmetric matrix.

            Parameters
            ----------
            key : scalar or array-like
                Row and column labels.

            Returns
            -------
            pd.DataFrame
                Selected data.
            """
            return self._obj.data.loc[key, key]

        def __setitem__(self, key, value):
            """
            Set data in the symmetric matrix, enforcing symmetry.

            Parameters
            ----------
            key : scalar or array-like
                Row and column labels.
            value : scalar or array-like
                Value to set.

            Notes
            -----
            This method automatically enforces symmetry and supports advanced indexing
            like slices and lists (e.g., cov.loc[:, 'a'] = some_values).
            """
            rows = cols = key

            # Set the primary value
            self._obj.data.loc[rows, cols] = value

            # Determine the value for the symmetric assignment
            symmetric_value = value
            if hasattr(value, 'T'):
                # For DataFrames, Series, and numpy arrays, we need the transpose
                # for the symmetric assignment. Scalars do not have .T.
                symmetric_value = value.T

            # Set the symmetric value
            self._obj.data.loc[cols, rows] = symmetric_value


class CovarianceMatrix(SymmetricMatrix):
    """
    Covariance matrix class wrapping pandas DataFrames.

    Attributes
    ----------
    variance : pd.Series
        Series containing the variances (diagonal elements).
    """

    @property
    def variance(self) -> pd.Series:
        """
        Returns the diagonal of the covariance matrix (the variances).

        Returns
        -------
        pd.Series
            Series containing the variances.
        """
        return pd.Series(np.diag(self.data), index=self.index, name='variance')


class InverseProblem:
    """
    Inverse problem class for estimating model states from observations.

    Represents a statistical inverse problem for estimating model states from observed data
    using Bayesian inference and linear forward operators.

    An inverse problem seeks to infer unknown model parameters (the "state") from observed data,
    given prior knowledge and a mathematical relationship (the forward operator) that links the state
    to the observations. This class provides a flexible interface for formulating and solving such
    problems using various estimators.

    Parameters
    ----------
    estimator : str or type[Estimator]
        The estimator to use for solving the inverse problem. Can be the name of a registered estimator
        or an Estimator class.
    obs : pd.Series
        Observed data as a pandas Series, indexed by observation dimensions.
    prior : pd.Series
        Prior estimate of the model state as a pandas Series, indexed by state dimensions.
    forward_operator : ForwardOperator or pd.DataFrame
        Linear operator mapping model state to observations. Can be a ForwardOperator instance or a
        pandas DataFrame with appropriate indices and columns.
    prior_error : CovarianceMatrix
        Covariance matrix representing uncertainty in the prior state estimate.
    modeldata_mismatch : CovarianceMatrix
        Covariance matrix representing uncertainty in the observed data (model-data mismatch).
    constant : float or pd.Series or None, optional
        Optional constant term added to the forward model output. If not provided, defaults to zero.
    state_index : pd.Index or None, optional
        Index for the state variables. If None, uses the index from the prior.
    estimator_kwargs : dict, optional
        Additional keyword arguments to pass to the estimator.
    coord_decimals : int, optional
        Number of decimal places to round coordinate values for alignment (default is 6).

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input data dimensions are incompatible or indices do not align.

    Attributes
    ----------
    obs_index : pd.Index
        Index of the observations used in the problem.
    state_index : pd.Index
        Index of the state variables used in the problem.
    obs_dims : tuple
        Names of the observation dimensions.
    state_dims : tuple
        Names of the state dimensions.
    n_obs : int
        Number of observations.
    n_state : int
        Number of state variables.
    posterior : pd.Series
        Posterior mean estimate of the model state.
    posterior_error : CovarianceMatrix
        Posterior error covariance matrix.
    posterior_obs : pd.Series
        Posterior mean estimate of the observations.
    prior_obs : pd.Series
        Prior mean estimate of the observations.
    xr : InverseProblem._XR
        Xarray interface for accessing inversion results as xarray DataArrays.

    Methods
    -------
    solve() -> dict[str, pd.Series | CovarianceMatrix | pd.Series]
        Solves the inverse problem and returns a dictionary with posterior state, posterior error
        covariance, and posterior observation estimates.

    Notes
    -----
    This class is designed for linear inverse problems with Gaussian error models, commonly encountered
    in geosciences, remote sensing, and other fields where model parameters are inferred from indirect
    measurements. It supports flexible input formats and provides robust alignment and validation of
    input data.
    """

    def __init__(self,
                 estimator: str | type[Estimator],
                 obs: pd.Series,
                 prior: pd.Series,
                 forward_operator: ForwardOperator | pd.DataFrame,
                 prior_error: SymmetricMatrix,
                 modeldata_mismatch: SymmetricMatrix,
                 constant: float | pd.Series | None = None,
                 state_index: pd.Index | None = None,
                 estimator_kwargs: dict = {},
                 coord_decimals: int = 6,
                 ) -> None:
        """
        Initialize the InverseProblem.

        Parameters
        ----------
        estimator : str or type[Estimator]
            Estimator class or its name as a string.
        obs : pd.Series
            Observed data.
        prior : pd.Series
            Prior model state estimate.
        forward_operator : pd.DataFrame
            Forward operator matrix.
        prior_error : CovarianceMatrix
            Prior error covariance matrix.
        modeldata_mismatch : CovarianceMatrix
            Model-data mismatch covariance matrix.
        constant : float or pd.Series, optional
            Constant data, defaults to 0.0.
        state_index : pd.Index, optional
            Index for the state variables.
        estimator_kwargs : dict, optional
            Additional keyword arguments for the estimator.
        obs_aggregation : optional
            Observation aggregation method.
        coord_decimals : int, optional
            Number of decimal places for rounding coordinates.

        Raises
        ------
        TypeError
            If any of the inputs are of the wrong type.
        ValueError
            If there are issues with the input data (e.g., incompatible dimensions).
        """
        # Validate state_index
        if state_index is None:
            state_index = prior.index
        if not isinstance(state_index, pd.Index):
            raise TypeError("state_index must be a pandas Index.")

        # Set problem dimensions
        self.obs_dims = tuple(obs.index.names)
        self.state_dims = tuple(prior.index.names)

        # Handle forward operator
        if isinstance(forward_operator, ForwardOperator):
            forward_operator = forward_operator.data

        # Handle constant data
        if not isinstance(constant, pd.Series):
            constant_series = obs.copy(deep=True)
            constant_series[:] = constant if constant is not None else 0.0
            constant = constant_series

        # Assert dimensions are in indices
        if not all(dim in forward_operator.index.names for dim in self.obs_dims):
            raise ValueError("Observation dimensions must be in the forward operator index.")
        if  not all(dim in constant.index.names for dim in self.obs_dims):
            raise ValueError("Observation dimensions must be in the constant index.")
        if not all(dim in forward_operator.columns.names for dim in self.state_dims):
            raise ValueError("State dimensions must be in the forward operator columns.")
        if not all(dim in state_index.names for dim in self.state_dims):
            raise ValueError("State dimensions must be in the state index.")
        
        # Order levels if indexes are MultiIndex
        if isinstance(forward_operator.index, pd.MultiIndex):
            forward_operator = forward_operator.reorder_levels(self.obs_dims,
                                                                axis='index')
            obs = obs.reorder_levels(self.obs_dims)
            modeldata_mismatch = modeldata_mismatch.reorder_levels(self.obs_dims)
            constant = constant.reorder_levels(self.obs_dims)
        if isinstance(forward_operator.columns, pd.MultiIndex):
            forward_operator = forward_operator.reorder_levels(self.state_dims,
                                                               axis='columns')
            prior = prior.reorder_levels(self.state_dims)
            prior_error = prior_error.reorder_levels(self.state_dims)

        # Round index coordinates to avoid floating point issues during alignment
        round_coords = partial(round_index, decimals=coord_decimals)
        state_index = round_coords(state_index)
        obs.index = round_coords(obs.index)
        prior.index = round_coords(prior.index)
        forward_operator.index = round_coords(forward_operator.index)
        forward_operator.columns = round_coords(forward_operator.columns)
        prior_error.index = round_coords(prior_error.index)
        modeldata_mismatch.index = round_coords(modeldata_mismatch.index)
        constant.index = round_coords(constant.index)

        # Define the obs index as the intersection of the observation and forward operator obs indices
        obs_index = obs.index.intersection(forward_operator.index)
        if obs_index.empty:
            raise ValueError("No overlapping indices between observations and forward operator.")

        # Align inputs
        self.obs = obs.reindex(obs_index).dropna()
        self.prior = prior.reindex(state_index).dropna()
        self.forward_operator = forward_operator.reindex(index=obs_index, columns=state_index).fillna(0.0)
        self.prior_error = prior_error.reindex(state_index)
        self.modeldata_mismatch = modeldata_mismatch.reindex(obs_index)
        self.constant = constant.reindex(obs_index).fillna(0.0)

        # Store the problem indices
        self.obs_index = obs_index
        self.state_index = state_index

        # Initialize the estimator
        estimator_input = {
            'z': self.obs.values,
            'x_0': self.prior.values,
            'H': self.forward_operator.values,
            'S_0': self.prior_error.values,
            'S_z': self.modeldata_mismatch.values,
            'c': self.constant.values
        }

        self.estimator = self._init_estimator(estimator, estimator_input=estimator_input, **estimator_kwargs)

        # Build xarray interface
        self.xr = self._XR(self)

    def _init_estimator(self, estimator: str | type[Estimator], 
                        estimator_input: dict,
                        **kwargs) -> Estimator:
        """
        Initialize the estimator.

        Parameters
        ----------
        estimator : str or type[Estimator]
            The estimator class or its name as a string.
        estimator_input : dict
            Input parameters for the estimator, including:
            - 'z': Observed data
            - 'x_0': Prior state estimate
            - 'H': Forward operator
            - 'S_0': Prior error covariance
            - 'S_z': Model-data mismatch covariance
            - 'c': Constant data (optional)
        kwargs : dict
            Additional keyword arguments to pass to the estimator constructor.

        Returns
        -------
        Estimator
            An instance of the specified estimator class.
        """
        if isinstance(estimator, str):
            if estimator not in ESTIMATOR_REGISTRY:
                raise ValueError(f"Estimator '{estimator}' is not registered.")
            estimator_cls = ESTIMATOR_REGISTRY[estimator]
        elif isinstance(estimator, type) and issubclass(estimator, Estimator):
            estimator_cls = estimator
        else:
            raise TypeError("Estimator must be a string or a subclass of Estimator.")

        z = estimator_input['z']
        x_0 = estimator_input['x_0']
        H = estimator_input['H']
        S_0 = estimator_input['S_0']
        S_z = estimator_input['S_z']
        c = estimator_input.get('c')

        return estimator_cls(z=z, x_0=x_0, H=H, S_0=S_0, S_z=S_z, c=c, **kwargs)

    def solve(self) -> dict[str, pd.Series | SymmetricMatrix | pd.Series]:
        """
        Solve the inversion problem using the configured estimator.

        Returns
        -------
        dict[str, State | Covariance | Data]
            A dictionary containing the posterior estimates:
            - 'posterior': Pandas series with the posterior mean model estimate.
            - 'posterior_error': Covariance object with the posterior error covariance matrix.
            - 'posterior_obs': Pandas series with the posterior observation estimates.
        """
        return {
            'posterior': self.posterior,
            'posterior_error': self.posterior_error,
            'posterior_obs': self.posterior_obs,
        }

    @property
    def n_obs(self) -> int:
        """
        Number of observations.

        Returns
        -------
        int
            Number of observations.
        """
        return self.estimator.n_z

    @property
    def n_state(self) -> int:
        """
        Number of state variables.

        Returns
        -------
        int
            Number of state variables.
        """
        return self.estimator.n_x

    @cached_property
    def posterior(self) -> pd.Series:
        """
        Posterior state estimate.

        Returns
        -------
        pd.Series
            Pandas series with the posterior mean model estimate.
        """
        x_hat = self.estimator.x_hat
        return pd.Series(x_hat, index=self.state_index, name='posterior')

    @cached_property
    def posterior_obs(self) -> pd.Series:
        """
        Posterior observation estimates.

        Returns
        -------
        pd.Series
            Pandas series with the posterior observation estimates.
        """
        y_hat = self.estimator.y_hat
        return pd.Series(y_hat, index=self.obs_index, name='posterior_obs')

    @cached_property
    def posterior_error(self) -> SymmetricMatrix:
        """
        Posterior error covariance matrix.

        Returns
        -------
        CovarianceMatrix
            CovarianceMatrix instance with the posterior error covariance matrix.
        """
        S_hat = self.estimator.S_hat
        return SymmetricMatrix(pd.DataFrame(S_hat, index=self.state_index, columns=self.state_index))

    @cached_property
    def prior_obs(self) -> pd.Series:
        """
        Prior observation estimates.

        Returns
        -------
        pd.Series
            Pandas series with the prior observation estimates.
        """
        y_0 = self.estimator.y_0
        return pd.Series(y_0, index=self.obs_index, name='prior_obs')

    class _XR:
        """
        Xarray interface for Inversion data.
        """
        def __init__(self, inversion: 'InverseProblem'):
            self._inversion = inversion

        def __getattr__(self, attr):
            """
            Get an xarray representation of an attribute from the inversion object.

            Parameters
            ----------
            attr : str
                Attribute name.

            Returns
            -------
            xr.DataArray
                Xarray representation of the attribute.

            Raises
            ------
            AttributeError
                If the attribute does not exist.
            TypeError
                If the attribute type is not supported.
            """
            if attr == '_inversion':
                return self._inversion
            if hasattr(self._inversion, attr):
                obj = getattr(self._inversion, attr)
                if isinstance(obj, pd.Series):
                    return self._series_to_xarray(series=obj, attr=attr)
                elif isinstance(obj, pd.DataFrame):
                    return self._dataframe_to_xarray(df=obj, attr=attr)
                else:
                    raise TypeError(f"Unable to represent {type(obj)} as Xarray.")
            else:
                raise AttributeError(f"'{type(self._inversion).__name__}' object has no attribute '{attr}'")

        def __setattr__(self, *args):
            """
            Prevent setting attributes on the Xarray interface.

            Parameters
            ----------
            *args : tuple
                Attribute name and value.

            Raises
            ------
            AttributeError
                If attempting to set an attribute.
            """
            if args[0] == '_inversion':
                super().__setattr__(*args)
            else:
                raise AttributeError(f"Cannot set attribute '{args[0]}' on Xarray interface.")

        @staticmethod
        def _series_to_xarray(series: pd.Series, attr) -> xr.DataArray:
            """
            Convert a Pandas Series to an Xarray DataArray.

            Parameters
            ----------
            series : pd.Series
                Pandas Series to convert.
            attr : str
                Attribute name.

            Returns
            -------
            xr.DataArray
                Xarray DataArray representation of the series.
            """
            series = series.copy()
            series.name = attr
            return series.to_xarray()

        @staticmethod
        def _dataframe_to_xarray(df: pd.DataFrame, attr) -> xr.DataArray:
            """
            Convert a Pandas DataFrame to an Xarray DataArray.

            Parameters
            ----------
            df : pd.DataFrame
                Pandas DataFrame to convert.
            attr : str
                Attribute name.

            Returns
            -------
            xr.DataArray
                Xarray DataArray representation of the DataFrame.
            """
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                # Stack all levels of the columns MultiIndex into the index
                n_levels = len(df.columns.levels)
                s = df.stack(list(range(n_levels)), future_stack=True)
            else:
                s = df.stack(future_stack=True)
            return InverseProblem._XR._series_to_xarray(series=s, attr=attr)
