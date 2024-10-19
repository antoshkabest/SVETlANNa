from typing import Literal
from typing import overload
from typing import Protocol

import torch

from svetlanna import LinearOpticalSetup
from svetlanna.phase_retrieval_problem import algorithms


class SetupLike(Protocol):
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        ...

    def reverse(self, transmission_field: torch.Tensor) -> torch.Tensor:
        ...


@overload
def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    initial_phase: torch.Tensor = None,
    method: Literal['GS', 'HIO'] = 'GS',
    maxiter: int = 500,
    tol: float = 1e-3
):
    ...


@overload
def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    target_phase: torch.Tensor,
    target_region: torch.Tensor,
    initial_phase: torch.Tensor = None,
    method: Literal['GS', 'HIO'] = 'GS',
    maxiter: int = 500,
    tol: float = 1e-3
):
    ...


# TODO: fix docstrings
def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor = None,
    initial_phase: torch.Tensor = None,
    method: Literal['GS', 'HIO', 'Pendulum'] = 'GS',
    maxiter: int = 300,
    tol: float = 1e-3,
    constant_factor: float = 0.7,
    constant_amplitude_factor: float = 2.,
    constant_phase_factor: float = 2.
) -> torch.Tensor:
    """Function for solving phase retrieval problem: generating target
    intensity profile or reconstructing the phase profile of the field

    Parameters
    ----------
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup
    optical_setup : LinearOpticalSetup | SetupLike
        Optical system through which the beam is propagated
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    target_phase : torch.Tensor | None, optional
        Phase profile on the Fourier plane(optional for the generating
        target intensity profile problem)
    target_region : torch.Tensor, optional
        Region to preserve phase and amplitude profiles in the Fourier plane(
        optional for the generating target intensity profile problem)
    initial_phase : torch.Tensor, optional
        Initial approximation for the phase profile, by default None
    method : Literal[&#39;GS&#39;, &#39;HIO&#39;], optional
        Algorithms for phase retrieval problem, by default 'GS'
    maxiter : int, optional
        Maximum number of iterations, by default 100
    tol : float, optional
        Tolerance, by default 1e-3
    constant_factor : float, optional
        Learning rate parameter for the HIO method, by default 0.5

    Returns
    -------
    torch.Tensor
        Optimized phase profile from 0 to 2pi

    Raises
    ------
    ValueError
        Unknown optimization method
    """

    forward_propagation = optical_setup.forward
    reverse_propagation = optical_setup.reverse

    if initial_phase is None:
        initial_phase = 2 * torch.pi * torch.rand_like(source_intensity)

    if method == 'GS':

        phase_distribution = algorithms.gerchberg_saxton_algorithm(
            target_intensity=target_intensity,
            source_intensity=source_intensity,
            forward=forward_propagation,
            reverse=reverse_propagation,
            initial_approximation=initial_phase,
            tol=tol,
            maxiter=maxiter,
            target_phase=target_phase,
            target_region=target_region
        )
    elif method == 'HIO':
        phase_distribution = algorithms.hybrid_input_output(
            target_intensity=target_intensity,
            source_intensity=source_intensity,
            forward=forward_propagation,
            reverse=reverse_propagation,
            initial_approximation=initial_phase,
            tol=tol,
            maxiter=maxiter,
            target_phase=target_phase,
            target_region=target_region,
            constant_factor=constant_factor
        )

    elif method == 'Pendulum':
        phase_distribution = algorithms.pendulum(
            target_intensity=target_intensity,
            source_intensity=source_intensity,
            forward=forward_propagation,
            reverse=reverse_propagation,
            initial_approximation=initial_phase,
            tol=tol,
            maxiter=maxiter,
            target_phase=target_phase,
            target_region=target_region,
            constant_amplitude_factor=constant_amplitude_factor,
            constant_phase_factor=constant_phase_factor
        )

    else:
        raise ValueError('Unknown optimization method')

    return phase_distribution
