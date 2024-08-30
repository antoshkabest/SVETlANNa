from typing import Iterable, Any, Generator, TextIO
from abc import ABC, abstractmethod
from io import BufferedWriter
from contextlib import contextmanager
from pathlib import Path
from sys import stdout
from numpy.typing import ArrayLike
import numpy as np


class SaveContexts:
    def __init__(
        self,
        parameter_name: str,
        directory: str
    ):
        self.parameter_name = parameter_name
        self._directory = directory
        self._generated_files: list[Path] = []

    def get_new_filepath(self, filetype) -> Path:
        suffix = '.' + filetype

        total_files = len(
            list(
                filter(
                    lambda f: f.suffix == suffix,
                    self._generated_files
                    )
                )
            )
        file_name = self.parameter_name + f'_{total_files}'

        return Path(self._directory,  file_name).with_suffix(suffix)

    @contextmanager
    def file(self, filetype: str) -> Generator[BufferedWriter, Any, None]:
        filepath = self.get_new_filepath(filetype=filetype)
        with open(filepath, mode='wb') as file:
            yield file
        self._generated_files.append(filepath)

    @contextmanager
    def stdout(self, stream: TextIO = stdout) -> Generator[TextIO, Any, None]:
        yield stream


class ParameterRepr(ABC):
    @abstractmethod
    def save(self, context: SaveContexts):
        """save to file"""


class ImageRepr(ParameterRepr):
    def __init__(
        self,
        value: Any,
        mpl_kwargs: dict[str, Any] | None = None,
        filetype: str = 'png'
    ):
        super().__init__()
        self.value = value
        self.filetype = filetype
        self.mpl_kwargs = mpl_kwargs if mpl_kwargs is not None else {}

    def save(self, context: SaveContexts):
        import matplotlib.pyplot as plt

        with context.file(filetype=self.filetype) as f:
            plt.imshow(self.value, **self.mpl_kwargs)
            plt.savefig(f)
            plt.close()


class ReprRepr(ParameterRepr):
    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def save(self, context: SaveContexts):
        with context.stdout(stdout) as f:
            f.write(context.parameter_name + ':\n')
            f.write(repr(self.value) + '\n')


class NpyFileRepr(ParameterRepr):
    def __init__(self, value: ArrayLike):
        super().__init__()
        self.value = value

    def save(self, context: SaveContexts):
        with context.file(filetype='npy') as f:
            np.save(f, self.value)


class ParameterSpecs:
    def __init__(
        self,
        name: str,
        representations: Iterable[ParameterRepr]
    ) -> None:
        self.name = name
        self.representations = representations

    def save(
        self,
        directory: str,
        context_type: type[SaveContexts] = SaveContexts
    ):
        context = context_type(
            parameter_name=self.name,
            directory=directory
        )
        for representation in self.representations:
            representation.save(context=context)
