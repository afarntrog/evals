"""Adapts agent factories into task callables compatible with Experiment.run_evaluations."""

from collections.abc import Callable
from typing import Any

from .case import Case
from .mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from .telemetry import StrandsEvalsTelemetry


class AgentTask:
    """A task that creates a fresh Agent per case and invokes it with case.input.

    Args:
        agent_factory: A no-arg callable that returns a strands Agent instance.

    Example:
        task = AgentTask(lambda: Agent(model="...", tools=[calculator]))
        experiment.run_evaluations(task=task)
    """

    def __init__(self, agent_factory: Callable[[], Any]):
        self._agent_factory = agent_factory

    def __call__(self, case: Case) -> dict[str, Any]:
        agent = self._agent_factory()
        result = agent(case.input)
        return {"output": str(result)}


class TracedAgentTask(AgentTask):
    """An AgentTask that also collects OpenTelemetry spans and maps them to a Session.

    Use this when your evaluators need trajectory data (e.g., TrajectoryEvaluator).

    Args:
        agent_factory: A no-arg callable that returns a strands Agent instance.

    Example:
        task = TracedAgentTask(lambda: Agent(model="...", tools=[calculator]))
        experiment.run_evaluations(task=task)
    """

    def __init__(self, agent_factory: Callable[[], Any]):
        super().__init__(agent_factory)
        self._telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
        self._mapper = StrandsInMemorySessionMapper()

    def __call__(self, case: Case) -> dict[str, Any]:
        self._telemetry.in_memory_exporter.clear()

        agent = self._agent_factory()
        result = agent(case.input)

        spans = list(self._telemetry.in_memory_exporter.get_finished_spans())
        session = self._mapper.map_to_session(spans, case.session_id)

        return {"output": str(result), "trajectory": session}
