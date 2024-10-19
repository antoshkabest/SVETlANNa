import torch

from svetlanna.phase_retrieval_problem import phase_retrieval_result as prr


def gerchberg_saxton_algorithm(
    target_intensity: torch.Tensor,
    source_intensity: torch.Tensor,
    forward,
    reverse,
    initial_approximation: torch.Tensor,
    tol: float,
    maxiter: int,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
) -> prr.PhaseRetrievalResult:
    """Gerchberg-Saxton algorithm(GS) for solving the phase retrieval problem

    Parameters
    ----------
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup(in the image plane)
    forward : _type_
        Function which describes forward propagation through the optical system
    reverse : _type_
        Function which describes reverse propagation through the optical system
    initial_approximation : torch.Tensor
        Initial approximation for the phase profile
    tol : float
        Accuracy for the algorithm
    maxiter : int
        Maximum number of iterations
    target_phase : torch.Tensor | None, optional
        Phase profile on the Fourier plane(optional for the generating
        target intensity profile problem) for reconstructing phase profile
        problem, by default None
    target_region : torch.Tensor | None, optional
        Region to preserve phase and amplitude profiles in the Fourier plane
        for reconstructing phase profile problem(optional for the generating
        target intensity profile problem), by default None

    Returns
    -------
    torch.Tensor
        Phase profile from 0 to 2pi
    """

    cost_func_evolution: list = []

    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    incident_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0
    current_error = 100.

    y_nodes, x_nodes = source_amplitude.shape
    number_of_pixels = x_nodes * y_nodes

    while True:

        output_field = forward(incident_field)

        if (target_phase is not None) and (target_region is not None):

            # TODO: ask about surface
            current_phase_target = torch.angle(output_field)
            current_phase_target = current_phase_target + (
                2 * torch.pi
            ) * (current_phase_target < 0).float()

            # TODO: check product
            target_phase_distribution = target_phase * target_region + (
                1. - target_region
            ) * current_phase_target

            target_field = target_amplitude * torch.exp(
                1j * target_phase_distribution
            ) / torch.abs(output_field)

        else:

            target_field = target_amplitude * output_field / torch.abs(
                output_field
            )

        current_target_intensity = torch.pow(torch.abs(output_field), 2)

        source_field = reverse(target_field)

        error = torch.sqrt(
            torch.sum(
                torch.pow(
                    torch.sqrt(current_target_intensity) - target_amplitude, 2
                )
            ) / (torch.sum(target_intensity) * number_of_pixels)
        )

        if (torch.abs(current_error - error) <= tol) or (
            number_of_iterations >= maxiter
        ):

            phase_function = torch.angle(incident_field)
            phase_function = phase_function + (
                2 * torch.pi
            ) * (phase_function < 0.).float()

            break

        else:

            current_phase_source = torch.angle(source_field)
            current_phase_source = current_phase_source + (
                2 * torch.pi
            ) * (current_phase_source < 0.)

            incident_field = source_amplitude * torch.exp(
                1j * current_phase_source
            )

            number_of_iterations += 1
            current_error = error
            cost_func_evolution.append(error)

    phase_retrieval_result = prr.PhaseRetrievalResult(
        solution=phase_function,
        cost_func=error,
        cost_func_evolution=cost_func_evolution,
        number_of_iterations=number_of_iterations
    )
    return phase_retrieval_result


def hybrid_input_output(
    target_intensity: torch.Tensor,
    source_intensity: torch.Tensor,
    forward,
    reverse,
    initial_approximation: torch.Tensor,
    tol: float,
    maxiter: int,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
    constant_factor: float = 0.9
) -> prr.PhaseRetrievalResult:
    """Hybrid Input-Output algorithm for for solving the phase retrieval
    problem

    Parameters
    ----------
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup(in the image plane)
    forward : _type_
        Function which describes forward propagation through the optical system
    reverse : _type_
        Function which describes reverse propagation through the optical system
    initial_approximation : torch.Tensor
        Initial approximation for the phase profile
    tol : float
        Accuracy for the algorithm
    maxiter : int
        Maximum number of iterations
    target_phase : torch.Tensor | None, optional
        Phase profile on the Fourier plane(optional for the generating
        target intensity profile problem) for reconstructing phase profile
        problem, by default None
    target_region : torch.Tensor | None, optional
        Region to preserve phase and amplitude profiles in the Fourier plane
        for reconstructing phase profile problem(optional for the generating
        target intensity profile problem), by default None
    constant_factor: float
        Learning rate value for the HIO algorithm, by default 0.9

    Returns
    -------
    torch.Tensor
        Phase profile from 0 to 2pi
    """
    cost_func_evolution: list = []

    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    input_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0
    # current_error = 10000.

    y_nodes, x_nodes = source_amplitude.shape
    number_of_pixels = x_nodes * y_nodes

    support_constrain = (
        target_amplitude > torch.max(target_amplitude) / 5
    ).float()

    while True:

        output_field = forward(input_field)

        # TODO: ask about surface
        output_field_phase = torch.angle(output_field)
        output_field_phase = output_field_phase + (
            2 * torch.pi
        ) * (output_field_phase < 0).float()

        if (target_phase is not None) and (target_region is not None):

            # TODO: check product
            updated_phase = target_phase * target_region + (
                1. - target_region
            ) * output_field_phase

            updated_output_field = target_amplitude * torch.exp(
                1j * updated_phase
            ) - (1. - support_constrain) * constant_factor * output_field

        else:

            updated_output_field = target_amplitude * torch.exp(
                1j * output_field_phase
            ) - (1. - support_constrain) * constant_factor * output_field

        updated_input_field = reverse(updated_output_field)

        current_source_intensity = torch.pow(torch.abs(updated_input_field), 2)

        error = torch.sqrt(
            torch.sum(
                current_source_intensity * (1. - support_constrain)
            ) / (
                torch.sum(current_source_intensity * support_constrain) * number_of_pixels  # noqa: E501
            )
        )

        phase_function = torch.angle(updated_input_field)
        phase_function = phase_function + (
            2 * torch.pi
        ) * (phase_function < 0.).float()

        # TODO: back current_error - error
        if (torch.abs(error) <= tol) or (
            number_of_iterations >= maxiter
        ):
            break

        else:

            input_field = source_amplitude * torch.exp(1j * phase_function)

            number_of_iterations += 1
            # current_error = error
            cost_func_evolution.append(error)

    phase_retrieval_result = prr.PhaseRetrievalResult(
        solution=phase_function,
        cost_func=error,
        cost_func_evolution=cost_func_evolution,
        number_of_iterations=number_of_iterations
    )

    return phase_retrieval_result


def pendulum(
    target_intensity: torch.Tensor,
    source_intensity: torch.Tensor,
    forward,
    reverse,
    initial_approximation: torch.Tensor,
    tol: float,
    maxiter: int,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
    constant_amplitude_factor: float = 2,
    constant_phase_factor: float = 2
) -> prr.PhaseRetrievalResult:
    """Updated Hybrid Input-Output algorithm(Pendulum algorithm) for solving
    the phase retrieval problem. It is assumed that the phase is independent
    of the complex amplitude modulus

    Parameters
    ----------
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup(in the image plane)
    forward : _type_
        Function which describes forward propagation through the optical system
    reverse : _type_
        Function which describes reverse propagation through the optical system
    initial_approximation : torch.Tensor
        Initial approximation for the phase profile
    tol : float
        Accuracy for the algorithm
    maxiter : int
        Maximum number of iterations
    target_phase : torch.Tensor | None, optional
        Phase profile on the Fourier plane(optional for the generating
        target intensity profile problem) for reconstructing phase profile
        problem, by default None
    target_region : torch.Tensor | None, optional
        Region to preserve phase and amplitude profiles in the Fourier plane
        for reconstructing phase profile problem(optional for the generating
        target intensity profile problem), by default None
    constant_amplitude_factor : float, optional
        Learning rate value for the updating amplitudes, by default 2
    constant_phase_factor : float, optional
        Learning rate value for the updating phase distribution, by default 2

    Returns
    -------
    prr.PhaseRetrievalResult
        Result of solving the phase retrieval problem
    """
    cost_func_evolution: list = []

    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    input_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0

    y_nodes, x_nodes = source_amplitude.shape
    number_of_pixels = x_nodes * y_nodes

    support_constrain = (
        target_amplitude > torch.max(target_amplitude) / 5
    ).float()

    while True:

        output_field = forward(input_field)

        output_field_phase = torch.angle(output_field)
        output_field_phase = output_field_phase + (
            2 * torch.pi
        ) * (output_field_phase < 0).float()

        updated_amplitude = (1. - constant_amplitude_factor) * (
            target_amplitude
        ) + constant_amplitude_factor * torch.abs(output_field)

        if (target_phase is not None) and (target_region is not None):

            new_phase = target_phase * target_region + (
                1. - target_region
            ) * output_field_phase

            new_output_field = target_amplitude * torch.exp(
                1j * new_phase
            ) - (1. - support_constrain) * constant_amplitude_factor * output_field     # noqa:E501

            new_phase = torch.angle(new_output_field)

            new_phase = new_phase + (
                2 * torch.pi
            ) * (new_phase < 0).float()

            updated_phase = (1. + constant_phase_factor) * output_field_phase - constant_phase_factor * new_phase  # noqa:E501

            updated_output_field = updated_amplitude * torch.exp(
                1j * updated_phase
            )

        else:

            new_output_field = target_amplitude * torch.exp(
                1j * output_field_phase
            ) - (1. - support_constrain) * constant_amplitude_factor * output_field  # noqa:E501

            new_phase = torch.angle(new_output_field)

            new_phase = new_phase + (
                2 * torch.pi
            ) * (new_phase < 0).float()

            updated_phase = (1. + constant_phase_factor) * output_field_phase - constant_phase_factor * new_phase   # noqa:E501

            updated_output_field = updated_amplitude * torch.exp(
                1j * updated_phase
            )

        updated_input_field = reverse(updated_output_field)

        current_source_intensity = torch.pow(torch.abs(updated_input_field), 2)

        error = torch.sqrt(
            torch.sum(
                current_source_intensity * (1. - support_constrain)
            ) / (
                torch.sum(current_source_intensity * support_constrain) * number_of_pixels  # noqa: E501
            )
        )

        phase_function = torch.angle(updated_input_field)
        phase_function = phase_function + (
            2 * torch.pi
        ) * (phase_function < 0.).float()

        # TODO: back current_error - error
        if (torch.abs(error) <= tol) or (
            number_of_iterations >= maxiter
        ):
            break

        else:

            input_field = source_amplitude * torch.exp(1j * phase_function)

            number_of_iterations += 1
            # current_error = error
            cost_func_evolution.append(error)

    phase_retrieval_result = prr.PhaseRetrievalResult(
        solution=phase_function,
        cost_func=error,
        cost_func_evolution=cost_func_evolution,
        number_of_iterations=number_of_iterations
    )

    return phase_retrieval_result
