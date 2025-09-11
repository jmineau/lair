
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
                 obs_dims=['obs_time', 'obs_location'],
                 state_dims=['lon', 'lat', 'flux_time'],
                 **kwargs):
        in_space = kwargs.pop('in_space', 'obs')
        out_space = kwargs.pop('out_space', 'state')
        kwargs['in_dims'] = obs_dims
        kwargs['out_dims'] = state_dims
        super().__init__(data=data,
                         in_space=in_space,
                         out_space=out_space,
                         **kwargs)


class FluxInversion(InverseProblem):
    def __init__(self,
                 project: str | Path,
                 estimator: str | type[Estimator],
                 output_space: Space,
                 obs: Data,
                 prior: State,
                 jacobian: Jacobian,
                 prior_error: CovarianceMatrix,
                 modeldata_mismatch: CovarianceMatrix,
                 background: Background | None = None,
                 estimator_kwargs: dict = {}
                 ) -> None:

        super().__init__(
            project=project,
            estimator=estimator,
            output_space=output_space,
            obs=obs,
            prior=prior,
            forward_operator=jacobian,
            prior_error=prior_error,
            modeldata_mismatch=modeldata_mismatch,
            constant=background,
            estimator_kwargs=estimator_kwargs
        )