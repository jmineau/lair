from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing_extensions import \
    Self  # requires python 3.11 to import from typing

import numpy as np
import pandas as pd
import xarray as xr


class Estimator(ABC):
    """
    Base inversion solver class

    Attributes
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
    c : np.ndarray | float
        Constant data
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
        """
        self.z = z
        self.x_0 = x_0
        self.H = H
        self.S_0 = S_0
        self.S_z = S_z
        self.c = c if c is not None else 0.0

    def forward(self, x) -> np.ndarray:
        """
        Forward model

        .. math::
            y = Hx + c
        """
        print('Performing forward calculation...')
        return self.H @ x + self.c

    def residual(self, x) -> np.ndarray:
        """
        Forward model residual

        .. math::
            r = z - (Hx + c)
        """
        print('Performing residual calculation...')
        return self.z - self.forward(x)

    @abstractmethod
    def cost(self, x) -> float:
        """
        Cost/loss/misfit function
        """
        print('Performing cost calculation...')
        pass

    @property
    @abstractmethod
    def S_hat(self) -> np.ndarray:
        """
        Posterior Error Covariance Matrix
        """
        print('Calculating Posterior Error Covariance Matrix...')
        pass

    @property
    @abstractmethod
    def x_hat(self) -> np.ndarray:
        """
        Posterior Mean Model Estimate (solution)
        """
        print('Calculating Posterior Mean Model Estimate...')
        pass

    @cached_property
    def y_hat(self) -> np.ndarray:
        """
        Posterior Mean Data Estimate

        .. math::
            \\hat{y} = H \\hat{x} + c
        """
        print('Calculating Posterior Mean Data Estimate...')
        return self.forward(self.x_hat)


class EstimatorRegistry(dict):
    def register(self, name: str):
        def decorator(cls: type[Estimator]) -> type[Estimator]:
            self[name] = cls
            return cls
        return decorator

ESTIMATOR_REGISTRY = EstimatorRegistry()


class Space:
    """Base class for mathematical spaces."""

    def __init__(self, name: str, coords: xr.Coordinates):
        self.name = name
        self.coords = coords
        self.dims = coords.dims
        self.n = len(self.coords.to_index())

    def __repr__(self):
        """Return a string representation of the Space object."""
        return f"Space(name={self.name}, dims={self.dims}, n={self.n})"

    def stack(self, data: xr.DataArray) -> xr.DataArray:
        """Stack the data into a single dimension."""
        return data.stack({self.name: self.dims})

    def unstack(self, data: xr.DataArray) -> xr.DataArray:
        """Unstack data back to original dimensions."""
        return data.unstack(self.name)

class Data:
    """
    Base class for space-aware data.

    Stores the original xarray.DataArray.
    """

    def __init__(self, data: np.ndarray | pd.Series | xr.DataArray,
                 space: Space | str):
        """
        Initialize Data object.
        
        Parameters
        ----------
        data : np.ndarray | pd.Series | xr.DataArray
            The data to be stored, which can be a NumPy array, pandas Series, or
            xarray DataArray.
        space : Space | str
            The space in which the data resides. A Space object must be passed
            for NumPy arrays. A string must be passed for pandas Series or xarray DataArray
            data which will be used as the name of the Space object created from
            the data's index or coordinates.
        """
        if isinstance(data, pd.Series):
            # Convert pandas Series to xarray DataArray
            data = data.to_xarray()

        if isinstance(data, xr.DataArray):
            assert isinstance(space, str), "Space must be a string for xarray DataArrays."  
            space = Space(name=space, coords=data.coords)

        elif isinstance(data, np.ndarray):
            assert isinstance(space, Space), "Space must be a Space object for NumPy arrays."
            if data.ndim != 1:  # TODO I feel this is a bit too strict probably can use this with posterior as well
                raise ValueError("NumPy arrays must be 1D.")
            data = xr.DataArray(data, dims=[space.dims[0]], coords=space.coords)

        else:
            raise TypeError("Data must be a NumPy array, pandas Series, or xarray DataArray.")

        # data_len = len(data)  # FIXME
        # if data_len != space.n:
        #     raise ValueError(f"Data length {data_len} does not match space size {space.n}.")

        self.data = data
        self.space = space

    def __repr__(self):
        """Return a string representation of the Data object."""
        return self.stacked.__repr__()

    def __getitem__(self, key):
        """Allows for intuitive dictionary-like access to the data."""
        return self.data.__getitem__(key)

    @property
    def coords(self) -> xr.Coordinates:
        return self.space.coords

    def sel(self, **kwargs) -> Self:
        """
        Select a subset of the data based on coordinates.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for selection, where keys are coordinate names
            and values are the desired values or slices.

        Returns
        -------
        Data
            A new Data object containing the selected data.
        """
        return self.__class__(data=self.data.sel(**kwargs), space=self.space.name)

    def align(self, out_space: Space) -> Self:
        """Align the data to a new space and return a new Data object."""
        assert isinstance(out_space, Space), "out_space must be a Space object."
        assert out_space.name == self.space.name, "Output space must have the same name as the input space."

        aligned_data = xr.align(xr.Dataset(coords=out_space.coords), self.data)

        return self.__class__(aligned_data, space=out_space.name)

    def merge(self, other: Self) -> Self:
        """
        Merge another Data object into this one.

        Parameters
        ----------
        other : Data
            Another Data object to merge.

        Returns
        -------
        Data
            A new Data object containing the merged data.
        """
        assert self.space.name == other.space.name, "Spaces must have the same name to merge."
        
        merged_data = xr.concat([self.data, other.data], dim=self.space.name)
        return self.__class__(data=merged_data, space=self.space)

    @property
    def stacked(self) -> xr.DataArray:
        """
        Return the data stacked into a single dimension according to the space.
        """
        return self.space.stack(self.data)


class Observation(Data):
    """Represents the observations z."""

    @property
    def z(self) -> np.ndarray:
        return self.stacked.values


class Constant(Data):
    """Represents the constant c."""

    @property
    def c(self) -> np.ndarray:
        return self.stacked.values


class State(Data):
    """Represents the model state x."""

    @property
    def x(self) -> np.ndarray:
        return self.stacked.values


class ForwardOperator:
    """Represents the forward operator H."""
    
    def __init__(self, data, in_space: Space | str, out_space: Space | str,
                 in_dims: list[str] | None = None, out_dims: list[str] | None = None):
        """
        Initialize the forward operator.

        Parameters
        ----------
        data :
            The forward operator matrix.
        in_space : Space | str
            The input space, either as a Space object or a string for the name.
        out_space : Space | str
            The output space, either as a Space object or a string for the name.
        in_dims : list[str] | None
            The input dimensions.
        out_dims : list[str] | None
            The output dimensions.
        """
        assert isinstance(in_space, (Space, str)), "Input space must be a Space object or a string."
        assert isinstance(out_space, (Space, str)), "Output space must be a Space object or a string."

        if isinstance(data, pd.Series):
            # Convert pandas Series to xarray DataArray
            data = data.to_xarray()

        if isinstance(data, xr.DataArray):
            assert isinstance(in_space, str), "Input space must be a string for xarray DataArrays."
            assert isinstance(out_space, str), "Output space must be a string for xarray DataArrays."
            assert in_dims is not None, "Input dimensions must be provided for xarray DataArrays."
            assert out_dims is not None, "Output dimensions must be provided for xarray DataArrays."

            # Create space objects from the data coordinates
            in_coords = xr.Coordinates({dim: data.coords[dim] for dim in in_dims})
            out_coords = xr.Coordinates({dim: data.coords[dim] for dim in out_dims})
            in_space = Space(name=in_space, coords=in_coords)
            out_space = Space(name=out_space, coords=out_coords)

        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("Forward operator data must be a 2D NumPy array.")
            assert isinstance(in_space, Space), "Input space must be a Space object for NumPy arrays."
            assert isinstance(out_space, Space), "Output space must be a Space object for NumPy arrays."
            assert len(in_space.dims) == 1, "Input space must have exactly one dimension."
            assert len(out_space.dims) == 1, "Output space must have exactly one dimension."

            in_dim = in_space.dims[0]
            out_dim = out_space.dims[0]

            data = xr.DataArray(data, dims=[in_dim, out_dim], coords={
                in_dim: in_space.coords[in_dim].coords,
                out_dim: out_space.coords[out_dim].coords}
            )
        else:
            raise TypeError("Data must be a 2D NumPy array or an xarray DataArray.")

        self.data = data
        self._in_space = in_space
        self._out_space = out_space

    def __repr__(self):
        """
        Return a string representation of the ForwardOperator object.
        """
        return repr(self.stacked)

    @property
    def spaces(self) -> dict[str, Space]:
        """
        Returns the input and output spaces of the forward operator.
        """
        return {
            self._in_space.name: self._in_space,
            self._out_space.name: self._out_space,
        }

    @property
    def coords(self) -> xr.Coordinates:
        """
        Returns the coordinates of the forward operator.
        """
        return self.data.coords

    def sel(self, **kwargs) -> Self:
        """
        Select a subset of the forward operator based on coordinates.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for selection, where keys are coordinate names
            and values are the desired values or slices.

        Returns
        -------
        ForwardOperator
            A new ForwardOperator object containing the selected data.
        """
        selected_data = self.data.sel(**kwargs)
        return self.__class__(data=selected_data,
                              in_space=self._in_space.name,
                              out_space=self._out_space.name,
                              in_dims=self._in_space.dims,
                              out_dims=self._out_space.dims)

    def align(self, in_space: Space, out_space: Space) -> Self:
        in_coords = in_space.coords
        out_coords = out_space.coords
        merged_coords_ds = in_coords.merge(out_coords)

        aligned_data = xr.align(merged_coords_ds, self.data)

        return self.__class__(data=aligned_data,
                              in_space=in_space,
                              out_space=out_space)

    @property
    def stacked(self) -> xr.DataArray:
        # order is important here, we need to stack the input space first
        return self.spaces[self._out_space.name].stack(self.spaces[self._in_space.name].stack(self.data))

    @property
    def H(self) -> np.ndarray: 
        """Return the forward operator matrix."""
        return self.stacked.values


class CovarianceMatrix:
    """
    A class representing a square covariance matrix in a specific space.
    """
    def __init__(self, data: np.ndarray, coords: xr.Coordinates):
        self.coords = coords
        self.index = coords.to_index()
        self._data = pd.DataFrame(data, index=self.index, columns=self.index)

    def __repr__(self):
        return self._data.__repr__()

    @property
    def data(self) -> np.ndarray:
        return self._data.values

    @data.setter
    def data(self, value: np.ndarray):
        if value.shape != self._data.shape:
            raise ValueError("New data must have the same shape as the existing data.")
        self._data.values[:] = value

    def sel(self, **kwargs) -> Self | float:
        if isinstance(self.index, pd.MultiIndex):
            slicers = []
            for name in self.index.names:
                if name in kwargs:
                    slicers.append(kwargs[name])
                else:
                    slicers.append(slice(None))
            indexer = tuple(slicers)
        else:
            indexer = kwargs[self.index.name]

        selected_data = self._data.loc[indexer, indexer]

        if not isinstance(selected_data, (pd.DataFrame, pd.Series)):
            return selected_data

        sel_coords = selected_data.index.to_series().to_xarray().coords

        return CovarianceMatrix(data=selected_data.values, coords=sel_coords)

    def align(self, space: Space) -> Self:
        """
        Align the covariance matrix to the provided xarray coordinates.
        All coordinates must already exist in the covariance matrix.
        """
        coords = space.coords
        target_index = coords.to_index()

        if not target_index.isin(self.index).all():
            missing = target_index[~target_index.isin(self.index)]
            raise ValueError(f"Some coordinates to align to are missing in the covariance matrix: {missing}")

        aligned_data = self._data.loc[target_index, target_index]
        return self.__class__(data=aligned_data.values, coords=coords)

    @property
    def variance(self) -> xr.DataArray:
        """
        Returns the diagonal of the covariance matrix (the variances).
        """
        return xr.DataArray(np.diag(self.data), coords=self.coords)

    @property
    def S(self) -> np.ndarray:
        """
        Returns the covariance matrix as a NumPy array.
        """
        return self.data


class InverseProblem:
    def __init__(self,
                 project: str | Path,
                 estimator: str | type[Estimator],
                 output_space: Space,
                 obs: Observation,
                 prior: State,
                 forward_operator: ForwardOperator,
                 prior_error: CovarianceMatrix,
                 modeldata_mismatch: CovarianceMatrix,
                 constant: float | Constant | None = None,
                 estimator_kwargs: dict = {},
                 data_aggregation = None,
                 ) -> None:

        # Set project directory
        self.path = Path(project)
        self.project = self.path.name
        self.path.mkdir(exist_ok=True, parents=True)  # create project directory if it doesn't exist

        self._data_space_name = obs.space.name
        self._model_space_name = output_space.name

        has_constant_class = isinstance(constant, Constant)

        # Validate inputs
        assert all(name == self._model_space_name for name in
                   [prior.space.name, forward_operator._out_space.name]), \
            "Output space name must match prior and forward operator output space names."
        assert obs.space.name == forward_operator._in_space.name, \
            "Observation space name must match forward operator input space name."
        if has_constant_class:
            assert constant.space.name == obs.space.name, \
                "Constant space name must match observation space name."

        # Define the data space as the intersection of the observation and forward operator data spaces
        intersected_index = obs.coords.to_index().intersection(
            forward_operator.spaces[forward_operator._in_space.name].coords.to_index())
        intersected_coords = intersected_index.to_series().to_xarray().coords
        data_space = Space(name=self._data_space_name, coords=intersected_coords)

        # Align inputs with problem spaces
        self.obs = obs.align(data_space)
        self.prior = prior.align(output_space)
        self.forward_operator = forward_operator.align(in_space=data_space, out_space=output_space)
        self.prior_error = prior_error.align(output_space)
        self.modeldata_mismatch = modeldata_mismatch.align(data_space)
        self.constant = constant.align(data_space) if has_constant_class else constant

        # Store the problem spaces
        self.spaces = {
            self._data_space_name: data_space,
            self._model_space_name: output_space,
        }

        # Set model and data dimensions
        self.n_z = self.spaces[self._data_space_name].n
        self.n_x = self.spaces[self._model_space_name].n

        # Initialize the estimator
        estimator_input = {
            'z': self.obs.z,
            'x_0': self.prior.x,
            'H': self.forward_operator.H,
            'S_0': self.prior_error.S,
            'S_z': self.modeldata_mismatch.S,
            'c': self.constant.c if has_constant_class else self.constant
        }

        self.estimator = self._init_estimator(estimator, estimator_input=estimator_input, **estimator_kwargs)

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

    def solve(self) -> dict[str, State | CovarianceMatrix | Data]:
        """
        Solve the inversion problem using the configured estimator.

        Returns
        -------
        dict[str, State | Covariance | Data]
            A dictionary containing the posterior estimates:
            - 'posterior': State object with the posterior model estimate.
            - 'posterior_error': Covariance object with the posterior error covariance matrix.
            - 'posterior_data': Data object with the posterior data estimate.
        """
        posterior = self.posterior
        posterior_data = self.posterior_data
        posterior_error = self.posterior_error
        return {
            'posterior': posterior,
            'posterior_error': posterior_error,
            'posterior_data': posterior_data
        }

    @cached_property
    def posterior(self) -> State:
        """
        Posterior state estimate.
        """
        x_hat = self.estimator.x_hat
        model_space = self.spaces[self._model_space_name]
        posterior = State(xr.DataArray(x_hat, coords={model_space.name: model_space.index}, dims=[model_space.name]),
                          space=model_space.name)
        return posterior

    @cached_property
    def posterior_data(self) -> Data:
        """
        Posterior data estimate.
        """
        y_hat = self.estimator.y_hat
        data_space = self.spaces[self._data_space_name]
        posterior_data = Data(data=y_hat, space=data_space)
        return posterior_data

    @cached_property
    def posterior_error(self) -> CovarianceMatrix:
        """
        Posterior error covariance matrix.
        """
        S_hat = self.estimator.S_hat
        model_space = self.spaces[self._model_space_name]
        posterior_error = CovarianceMatrix(S_hat, coords=model_space.coords)
        return posterior_error