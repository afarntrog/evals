import json
import os

from .types.evaluation import EvaluationData


class LocalFileStore:
    """Stores task execution results as JSON on the local filesystem.

    Args:
        path: File path for reading/writing evaluation data.
    """

    def __init__(self, path: str):
        self._path = path

    def load(self) -> list[EvaluationData] | None:
        """Load evaluation data from the file.

        Returns:
            A list of EvaluationData if the file exists, None otherwise.
        """
        if not os.path.exists(self._path):
            return None

        with open(self._path) as f:
            raw = json.load(f)

        return [EvaluationData.model_validate(item) for item in raw]

    def save(self, data: list[EvaluationData]) -> None:
        """Save evaluation data to the file as JSON.

        Args:
            data: The evaluation data to persist.
        """
        with open(self._path, "w") as f:
            json.dump([item.model_dump() for item in data], f)
