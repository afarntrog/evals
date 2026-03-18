from typing_extensions import Protocol, runtime_checkable

from .evaluation import EvaluationData


@runtime_checkable
class TaskStore(Protocol):
    """Protocol for storing and loading task execution results.

    Implementations handle persistence of EvaluationData to any backing store
    (local files, S3, databases, etc.), enabling reuse of task results across
    multiple evaluation runs.
    """

    def load(self) -> list[EvaluationData] | None:
        """Load previously stored evaluation data.

        Returns:
            A list of EvaluationData if data exists, None otherwise.
        """
        ...

    def save(self, data: list[EvaluationData]) -> None:
        """Save evaluation data to the store.

        Args:
            data: The evaluation data to persist.
        """
        ...
