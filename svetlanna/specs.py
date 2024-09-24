from typing import Iterable, Any, Generator, TextIO
from abc import ABC, abstractmethod
from io import BufferedWriter
from contextlib import contextmanager
from pathlib import Path
from sys import stdout
from numpy.typing import ArrayLike
import numpy as np


class SaveContexts:
    """Generates different context managers that can be used
    to write a parameter value to stdout or file.
    """
    def __init__(
        self,
        parameter_name: str,
        directory: str
    ):
        """
        Parameters
        ----------
        parameter_name : str
            the human-readable name for the parameter
        directory : str
            the directory where the generated file will be saved, if any
        """
        self.parameter_name = parameter_name
        self._directory = directory
        self._generated_files: list[Path] = []  # paths of all generated files

    def get_new_filepath(self, extension: str) -> Path:
        """Create a new filepath for a specific extension.
        The generated filename of a specific extension will have a unique name
        ending with `_<n>.<extension>`, where `<n>` is auto-incrementing index.

        Parameters
        ----------
        extension : str
            filename extension

        Returns
        -------
        Path
            relative path to the file
        """
        suffix = '.' + extension

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
    def file(self, extension: str) -> Generator[BufferedWriter, Any, None]:
        filepath = self.get_new_filepath(extension=extension)
        with open(filepath, mode='wb') as file:
            yield file
        self._generated_files.append(filepath)

    @contextmanager
    def stdout(self, stream: TextIO = stdout) -> Generator[TextIO, Any, None]:
        yield stream


class ParameterRepr(ABC):
    """Base class of the parameter representation.
    """
    @abstractmethod
    def save(self, context: SaveContexts):
        """Save the parameter, using save contexts.

        Parameters
        ----------
        context : SaveContexts
            save contexts that can be used to write the parameter data.
        """


class ImageRepr(ParameterRepr):
    """Representation of the parameter as an image.
    Image generation is based on the `matplotlib` package.
    """
    def __init__(
        self,
        value: Any,
        mpl_kwargs: dict[str, Any] | None = None,
        format: str = 'png'
    ):
        """
        Parameters
        ----------
        value : Any
            The image data. See `matplotlib.pyplot.imshow` docs.
        mpl_kwargs : dict[str, Any] | None, optional
            kwargs, that will be passed to `matplotlib.pyplot.imshow`,
            by default None
        format : str, optional
            the image format, by default 'png'
        """
        super().__init__()
        self.value = value
        self.format = format
        self.mpl_kwargs = mpl_kwargs if mpl_kwargs is not None else {}

    def save(self, context: SaveContexts):
        import matplotlib.pyplot as plt

        with context.file(extension=self.format) as f:
            plt.imshow(self.value, **self.mpl_kwargs)
            plt.savefig(f)
            plt.close()


class ReprRepr(ParameterRepr):
    """Representation of the parameter as a plain text.
    """
    def __init__(self, value: Any):
        """
        Parameters
        ----------
        value : Any
            object with defined `__repr__` method that will be used
            to generate plain text.
        """
        super().__init__()
        self.value = value

    def save(self, context: SaveContexts):
        with context.stdout(stdout) as f:
            f.write(context.parameter_name + ':\n')
            f.write(repr(self.value) + '\n')


class NpyFileRepr(ParameterRepr):
    """Representation of the parameter as a `.npy` file.
    """
    def __init__(self, value: ArrayLike):
        """
        Parameters
        ----------
        value : ArrayLike
            parameter data.
        """
        super().__init__()
        self.value = value

    def save(self, context: SaveContexts):
        with context.file(extension='npy') as f:
            np.save(f, self.value)


class ParameterSpecs:
    """Container with all representations for the parameter.
    """
    def __init__(
        self,
        name: str,
        representations: Iterable[ParameterRepr]
    ) -> None:
        """
        Parameters
        ----------
        name : str
            the parameter's name.
        representations : Iterable[ParameterRepr]
            all representations of the parameter.
        """
        self.name = name
        self.representations = representations

    def save(
        self,
        directory: str,
        context_type: type[SaveContexts] = SaveContexts
    ):
        """Save all representations to a specific directory.

        Parameters
        ----------
        directory : str
            directory where all representations will be saved.
        context_type : type[SaveContexts], optional
            type of the save contexts, by default SaveContexts
        """
        context = context_type(
            parameter_name=self.name,
            directory=directory
        )
        for representation in self.representations:
            representation.save(context=context)
