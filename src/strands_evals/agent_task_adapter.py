"""Adapts agent factories into task callables compatible with Experiment.run_evaluations."""

from collections.abc import Callable
from typing import Any

from .case import Case
from .mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from .telemetry import StrandsEvalsTelemetry


def create_agent_task(agent_factory: Callable[[], Any]) -> Callable[[Case], dict[str, Any]]:
    """Wrap an agent factory into a task function for use with Experiment.run_evaluations.

    Per invocation, this:
    1. Clears the shared in-memory span exporter
    2. Creates a fresh Agent from the factory
    3. Calls agent(case.input)
    4. Collects finished spans and maps them to a Session
    5. Returns {"output": ..., "trajectory": session}

    Args:
        agent_factory: A no-arg callable that returns a strands Agent instance.

    Returns:
        A task callable that takes a Case and returns a structured dict.
    """
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    mapper = StrandsInMemorySessionMapper()

    def task(case: Case) -> dict[str, Any]:
        telemetry.in_memory_exporter.clear()

        agent = agent_factory()
        result = agent(case.input)

        spans = list(telemetry.in_memory_exporter.get_finished_spans())
        session = mapper.map_to_session(spans, case.session_id)

        return {"output": str(result), "trajectory": session}

    return task
