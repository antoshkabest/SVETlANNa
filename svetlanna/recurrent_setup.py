from typing import Iterable
from .elements import Element
from . import Wavefront, SimulationParameters

import torch
from torch import nn


class RecurrentOpticalSetup:
    """
    A recurrent optical setup
    """
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        forward_elements: Iterable[Element],
        backward_elements: Iterable[Element],
        init_hidden: Wavefront | None = None,
        device: str | torch.device = torch.get_default_device(),
    ) -> None:
        """
        Parameters
        ----------
        forward_elements : Iterable[Element]
            A set of optical elements...
        backward_elements : Iterable[Element]
            A set of optical elements for a feedback (to get a hidden state).
        """
        self.simulation_parameters = simulation_parameters

        self.__device = device
        if self.simulation_parameters.device is not self.__device:
            # TODO: right way to compare devices? check how it works?
            self.simulation_parameters.__device = self.simulation_parameters.to(self.__device)

        self.forward_elements = forward_elements
        self.backward_elements = backward_elements

        self.forward_net = nn.Sequential(*self.forward_elements).to(self.__device)  # torch network for forward
        self.backward_net = nn.Sequential(*self.backward_elements).to(self.__device)  # torch network for backward

        if init_hidden is None:
            self.hidden_state = self.reset_hidden
        else:
            self.hidden_state = init_hidden.to(self.__device)

    def reset_hidden(self):
        """
        Resets a "hidden" wavefront.

        Returns
        -------
        Wavefront
            A zero wavefront as an initial "hidden" wavefront.
        """
        return Wavefront(
            torch.zeros(
                # TODO: right way to get a wavefront size?
                size=self.simulation_parameters.axes_size(
                    self.simulation_parameters.axes.__dir__()
                )
            )
        ).to(self.__device)

    def parameters(self):
        """
        Parameters of the network to optimize.

        Returns
        -------
        list()
            List of parameters for a forward and a backward networks together.
        """
        return list(self.forward_net.parameters()) + list(self.backward_net.parameters())

    def forward(self, input_wavefront: Wavefront) -> Wavefront:
        """
        A forward function for a network assembled from elements.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        torch.Tensor
            An output wavefront after propagation of an input wavefront (+ hidden wavefront)
            through a forward net (output of the network).
        """
        output_wavefront = self.forward_net(input_wavefront + self.hidden_state)
        self.hidden_state = self.backward_net(output_wavefront)  # update hidden state

        return output_wavefront

    def to(self, device: str | torch.device | int) -> 'RecurrentOpticalSetup':
        if self.__device == torch.device(device):
            return self

        return RecurrentOpticalSetup(
            simulation_parameters=self.simulation_parameters,
            forward_elements=self.forward_elements,
            backward_elements=self.backward_elements,
            init_hidden=self.hidden_state,
            device=device
        )

    @property
    def device(self) -> str | torch.device | int:
        return self.__device
