from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


class IModel(ABC):
    """Common interface for all project models.

    Runtime options such as hyperparameters, dropout, and early stopping
    must be resolved from files under the project-level
    ``Config/`` directory rather than from these method signatures.
    """

    @abstractmethod
    def Train(self, DataSetPath: PathLike, ResultSavePath: PathLike) -> None:
        """Train the model and save logs, metrics, and network files."""

    @abstractmethod
    def Load(self, NetworkFilePath: PathLike) -> None:
        """Load a saved network file into the model."""

    @abstractmethod
    def Inference(self, DataPath: PathLike, ResultSavePath: PathLike) -> None:
        """Run inference on input data and save the prediction results."""
