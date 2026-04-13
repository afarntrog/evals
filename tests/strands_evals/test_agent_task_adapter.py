"""Tests for AgentTask and TracedAgentTask."""

from unittest.mock import MagicMock, patch

from strands_evals.agent_task_adapter import AgentTask, TracedAgentTask
from strands_evals.case import Case


class TestAgentTask:
    """Tests for the base AgentTask class."""

    def test_is_callable(self):
        """AgentTask instances are callable."""
        task = AgentTask(MagicMock())
        assert callable(task)

    def test_calls_factory_to_create_agent(self):
        """Calls the factory to get a fresh agent per case."""
        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="result")
        factory = MagicMock(return_value=mock_agent)

        task = AgentTask(factory)
        task(Case(name="test", input="hello"))

        factory.assert_called_once()

    def test_calls_agent_with_case_input(self):
        """Invokes the agent with case.input."""
        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="answer")
        factory = MagicMock(return_value=mock_agent)

        task = AgentTask(factory)
        task(Case(name="test", input="What is 2+2?"))

        mock_agent.assert_called_once_with("What is 2+2?")

    def test_returns_dict_with_output(self):
        """Returns a dict with 'output' key."""
        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="42")
        factory = MagicMock(return_value=mock_agent)

        task = AgentTask(factory)
        result = task(Case(name="test", input="What is 2+2?"))

        assert isinstance(result, dict)
        assert result["output"] == "42"
        assert "trajectory" not in result

    def test_works_with_experiment_run_evaluations(self):
        """AgentTask works when passed as the task parameter to Experiment."""
        from strands_evals.evaluators.evaluator import Evaluator
        from strands_evals.experiment import Experiment
        from strands_evals.types.evaluation import EvaluationOutput

        class PassingEvaluator(Evaluator):
            def evaluate(self, evaluation_case):
                return [EvaluationOutput(score=1.0, test_pass=True, reason="pass")]

        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="output")
        factory = MagicMock(return_value=mock_agent)

        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )
        reports = experiment.run_evaluations(task=AgentTask(factory))

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0


class TestTracedAgentTask:
    """Tests for TracedAgentTask that adds telemetry collection."""

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_calls_agent_with_case_input(self, mock_mapper_cls, mock_telemetry_cls):
        """Invokes the agent with case.input."""
        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="answer")
        factory = MagicMock(return_value=mock_agent)

        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry
        mock_mapper_cls.return_value.map_to_session.return_value = MagicMock()

        task = TracedAgentTask(factory)
        task(Case(name="test", input="What is 2+2?"))

        mock_agent.assert_called_once_with("What is 2+2?")

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_returns_dict_with_output_and_trajectory(self, mock_mapper_cls, mock_telemetry_cls):
        """Returns a dict with both 'output' and 'trajectory' keys."""
        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="42")
        factory = MagicMock(return_value=mock_agent)

        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry

        mock_session = MagicMock()
        mock_mapper_cls.return_value.map_to_session.return_value = mock_session

        task = TracedAgentTask(factory)
        result = task(Case(name="test", input="What is 2+2?"))

        assert result["output"] == "42"
        assert result["trajectory"] is mock_session

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_collects_spans_and_maps_to_session(self, mock_mapper_cls, mock_telemetry_cls):
        """Collects spans from telemetry and maps them to a Session."""
        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="res")
        factory = MagicMock(return_value=mock_agent)

        mock_spans = [MagicMock(), MagicMock()]
        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = mock_spans
        mock_telemetry_cls.return_value = mock_telemetry

        mock_session = MagicMock()
        mock_mapper = MagicMock()
        mock_mapper.map_to_session.return_value = mock_session
        mock_mapper_cls.return_value = mock_mapper

        task = TracedAgentTask(factory)
        result = task(Case(name="test", input="hi", session_id="sess-123"))

        mock_mapper.map_to_session.assert_called_once_with(list(mock_spans), "sess-123")
        assert result["trajectory"] is mock_session

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_clears_spans_between_calls(self, mock_mapper_cls, mock_telemetry_cls):
        """Each call clears the in-memory exporter before running the agent."""
        mock_agent = MagicMock()
        mock_agent.return_value.__str__ = MagicMock(return_value="res")
        factory = MagicMock(return_value=mock_agent)

        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry
        mock_mapper_cls.return_value = MagicMock()

        task = TracedAgentTask(factory)
        task(Case(name="test1", input="a"))
        task(Case(name="test2", input="b"))

        assert mock_telemetry.in_memory_exporter.clear.call_count == 2
