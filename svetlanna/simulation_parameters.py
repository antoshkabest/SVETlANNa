from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationParameters:
    x_size: float
    y_size: float
    x_nodes: int
    y_nodes: int
