import torch
from typing import Callable, Any


class Parameter(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super(cls, Parameter).__new__(cls)

    def __init__(
        self,
        data: Any,
        requires_grad: bool = True
    ):
        super().__init__()

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        self.inner_parameter = torch.nn.Parameter(
            data=data,
            requires_grad=requires_grad
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs = {k: v.inner_parameter if isinstance(v, cls) else v for k, v in kwargs.items()}
        args = (a.inner_parameter if isinstance(a, cls) else a for a in args)
        return func(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self.inner_parameter)


def sigmoid_inv(x):
    return torch.log(x/(1-x))


class BoundedParameter(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super(cls, BoundedParameter).__new__(cls)

    def __init__(
        self,
        data: Any,
        min_value: Any,
        max_value: Any,
        bound_func: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        inv_bound_func: Callable[[torch.Tensor], torch.Tensor] = sigmoid_inv,
        requires_grad: bool = True
    ):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        if not isinstance(min_value, torch.Tensor):
            min_value = torch.tensor(min_value)

        if not isinstance(max_value, torch.Tensor):
            max_value = torch.tensor(max_value)

        self.min_value = min_value
        self.max_value = max_value

        self.__a = self.max_value-self.min_value
        self.__b = self.min_value

        # initial inner parameter value
        initial_value = inv_bound_func((data - self.__b) / self.__a)

        self.inner_parameter = torch.nn.Parameter(
            data=initial_value,
            requires_grad=requires_grad
        )

        self.bound_func = bound_func

    @property
    def value(self):
        return self.__a * self.bound_func(self.inner_parameter) + self.__b

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs = {k: v.value if isinstance(v, cls) else v for k, v in kwargs.items()}
        args = (a.value if isinstance(a, cls) else a for a in args)
        return func(*args, **kwargs)

    def __repr__(self) -> str:
        return f'Bounded parameter containing:\n{repr(self.value)}'
