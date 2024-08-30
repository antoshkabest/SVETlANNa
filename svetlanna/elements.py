from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from .simulation_parameters import SimulationParameters
from .specs import ReprRepr, ParameterSpecs
from typing import Iterable
from .parameters import BoundedParameter, Parameter


INNER_PARAMETER_SUFFIX = '_svtlnn_inner_parameter'


class Element(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        simulation_parameters: SimulationParameters
    ) -> None:
        super().__init__()
        self.simulation_parameters = simulation_parameters

    @abstractmethod
    def forward(self, Ein: Tensor) -> Tensor:
        """Forward propagation through the optical element"""

    def to_specs(self) -> Iterable[ParameterSpecs]:
        """Create specs"""
        for (name, parameter) in self.named_parameters():

            # BoundedParameter and Parameter support
            if name.endswith(INNER_PARAMETER_SUFFIX):
                name = name.removesuffix(INNER_PARAMETER_SUFFIX)
                parameter = self.__getattribute__(name)

            yield ParameterSpecs(
                name=name,
                representations=(ReprRepr(value=parameter),)
            )

    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:

        # BoundedParameter and Parameter are handled by pointing
        # auxiliary attribute on them with a name plus INNER_PARAMETER_SUFFIX
        if isinstance(value, (BoundedParameter, Parameter)):
            super().__setattr__(
                name + INNER_PARAMETER_SUFFIX, value.inner_parameter
            )

        return super().__setattr__(name, value)
