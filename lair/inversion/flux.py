
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
            space = 'obs'
        super().__init__(data=data, **kwargs)


class Flux(State):

    def __init__(self, data, **kwargs):
        if 'space' not in kwargs:
            space = 'state'
        super().__init__(data=data, **kwargs)


class Jacobian(ForwardOperator):
    def __init__(self, data,
                 obs_dims=['time'],
                 state_dims=['lon', 'lat', 'state_time'],
                 **kwargs):
        in_space = kwargs.get('in_space', 'obs')
        out_space = kwargs.get('out_space', 'state')
        in_dims = kwargs.get('in_dims', obs_dims)
        out_dims = kwargs.get('out_dims', state_dims)
        super().__init__(data=data,
                         in_space=in_space,
                         out_space=out_space,
                         in_dims=in_dims,
                         out_dims=out_dims,
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