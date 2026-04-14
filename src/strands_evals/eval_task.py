"""Decorator and handlers for wrapping task functions with evaluation behavior."""

import functools
from collections.abc import Callable
from typing import Any

from .case import Case
from .mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from .telemetry import StrandsEvalsTelemetry


class EvalTaskHandler:
    """Base handler that normalizes task function return values.

    Subclass to add behavior before/after task execution (e.g., telemetry collection).
    """

    def before(self, case: Case) -> None:
        """Called before the task function runs. Override to add setup logic."""
        pass

    def after(self, case: Case, result: Any) -> dict[str, Any]:
        """Called after the task function runs. Normalizes the result to a dict.

        Args:
            case: The test case that was executed.
            result: The raw return value from the task function.

        Returns:
            A dict compatible with Experiment (must have at least "output" key).
        """
        if isinstance(result, dict):
            return result
        return {"output": str(result)}


class TracedHandler(EvalTaskHandler):
    """Handler that collects OpenTelemetry spans and maps them to a Session.

    Use with @eval_task when your evaluators need trajectory data.

    Args:
        mapper: Session mapper to use. Defaults to StrandsInMemorySessionMapper.

    Example:
        @eval_task(TracedHandler())
        def my_task(case):
            return str(my_agent(case.input))

        @eval_task(TracedHandler(mapper=MyCustomMapper()))
        def my_task(case):
            return str(my_agent(case.input))
    """

    def __init__(self, mapper=None):
        self._telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
        self._mapper = mapper or StrandsInMemorySessionMapper()

    def before(self, case: Case) -> None:
        self._telemetry.in_memory_exporter.clear()

    def after(self, case: Case, result: Any) -> dict[str, Any]:
        processed = super().after(case, result)

        spans = list(self._telemetry.in_memory_exporter.get_finished_spans())
        session = self._mapper.map_to_session(spans, case.session_id)
        processed.setdefault("trajectory", session)

        return processed


def eval_task(handler: EvalTaskHandler | None = None) -> Callable:
    """Decorator that wraps a task function with evaluation behavior.

    Args:
        handler: Handler that runs before/after the task function.
            Defaults to EvalTaskHandler (normalizes return values only).

    Example:
        # Simple — just normalize output
        @eval_task()
        def my_task(case):
            return str(my_agent(case.input))

        # With telemetry collection
        @eval_task(TracedHandler())
        def my_task(case):
            return str(my_agent(case.input))
    """
    if handler is None:
        handler = EvalTaskHandler()

    def decorator(fn: Callable[[Case], Any]) -> Callable[[Case], dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(case: Case) -> dict[str, Any]:
            handler.before(case)
            result = fn(case)
            return handler.after(case, result)

        return wrapper

    return decorator
