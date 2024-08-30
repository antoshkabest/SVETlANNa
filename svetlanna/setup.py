from typing import Iterable
from .elements import Element
from torch import nn
from torch import Tensor


class LinearOpticalSetup:
    def __init__(self, elements: Iterable[Element]) -> None:
        self.elements = elements
        self.net = nn.Sequential(*elements)

    def forward(self, Ein: Tensor) -> Tensor:
        return self.net(Ein)
