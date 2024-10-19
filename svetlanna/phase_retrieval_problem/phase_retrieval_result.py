from dataclasses import dataclass
import torch


# TODO: ask for message and status code
@dataclass(frozen=True, slots=True)
class PhaseRetrievalResult:
    """Represents the phase retrieval result
    """

    solution: torch.Tensor
    cost_func: float
    cost_func_evolution: list
    number_of_iterations: int
