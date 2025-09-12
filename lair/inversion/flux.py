
from pathlib import Path

from lair.inversion.core import (
    Space,
    Data,
    Observation,
    State,
    ForwardOperator,
    CovarianceMatrix,
    Constant as Background,
    Estimator,
    InverseProblem
)


class Concentration(Observation):

    def __init__(self, data, **kwargs):
        if 'space' not in kwargs:
            kwargs['space'] = 'obs'
        super().__init__(data=data, **kwargs)


class Flux(State):

    def __init__(self, data, **kwargs):
        if 'space' not in kwargs:
            kwargs['space'] = 'state'
        super().__init__(data=data, **kwargs)


class Jacobian(ForwardOperator):
    def __init__(self, data,
                 obs_dims=['obs_location', 'obs_time'],
                 state_dims=['lon', 'lat', 'flux_time'],
                 **kwargs):
        obs_space = kwargs.pop('obs_space', 'obs')
        state_space = kwargs.pop('state_space', 'state')
        kwargs['obs_dims'] = obs_dims
        kwargs['state_dims'] = state_dims
        super().__init__(data=data,
                         obs_space=obs_space,
                         state_space=state_space,
                         **kwargs)


class FluxInversion(InverseProblem):
    def __init__(self,
                 project: str | Path,
                 estimator: str | type[Estimator],
                 obs: Concentration,
                 prior: Flux,
                 jacobian: Jacobian,
                 prior_error: CovarianceMatrix,
                 modeldata_mismatch: CovarianceMatrix,
                 background: Background | float | None = None,
                 state_space: Space | None = None,
                 estimator_kwargs: dict = {},
                 ) -> None:

        super().__init__(
            project=project,
            estimator=estimator,
            obs=obs,
            prior=prior,
            forward_operator=jacobian,
            prior_error=prior_error,
            modeldata_mismatch=modeldata_mismatch,
            constant=background,
            state_space=state_space,
            estimator_kwargs=estimator_kwargs
        )