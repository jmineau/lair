.. currentmodule:: lair.inversion

Inversion
=========

Inversion is a mathematical technique used to estimate unknown model parameters 
or states from observed data, typically by solving an optimization problem that 
combines prior knowledge, a forward model, and error statistics. The goal of inversion 
is to find the most probable model state (the "posterior") that explains the observations, 
given uncertainties in both the model and the data.

Typical inversion workflows involve:
    - Defining observed data and prior model state estimates.
    - Specifying a forward operator that maps model states to observations.
    - Providing error covariance matrices for both prior and observation uncertainties.
    - Solving for the posterior state estimate and its uncertainty using an estimator.
The module is designed to support flexible data structures (e.g., pandas, xarray), 
robust index alignment, and extensible estimator implementations for a variety of inversion methodologies.

.. rubric:: Modules

.. autosummary::
   :toctree: api
   :recursive:

   core
   estimators
   flux
   utils
