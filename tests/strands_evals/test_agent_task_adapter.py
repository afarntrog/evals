"""Tests for the agent task adapter that wraps agent factories into task callables."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.agent_task_adapter import create_agent_task
from strands_evals.case import Case


class TestCreateAgentTask:
    """Tests for create_agent_task function."""

    def test_returns_callable(self):
        """create_agent_task returns a callable."""
        factory = MagicMock()
        task = create_agent_task(factory)
        assert callable(task)

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_calls_factory_to_create_agent(self, mock_mapper_cls, mock_telemetry_cls):
        """The task calls the factory to get a fresh agent per case."""
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(message="result")
        mock_agent.return_value.__str__ = MagicMock(return_value="result")

        factory = MagicMock(return_value=mock_agent)

        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry

        mock_mapper = MagicMock()
        mock_mapper.map_to_session.return_value = MagicMock()
        mock_mapper_cls.return_value = mock_mapper

        case = Case(name="test", input="hello")
        task = create_agent_task(factory)
        task(case)

        factory.assert_called_once()

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_calls_agent_with_case_input(self, mock_mapper_cls, mock_telemetry_cls):
        """The task invokes the agent with case.input."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.__str__ = MagicMock(return_value="the answer")
        mock_agent.return_value = mock_result

        factory = MagicMock(return_value=mock_agent)

        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry

        mock_mapper = MagicMock()
        mock_mapper.map_to_session.return_value = MagicMock()
        mock_mapper_cls.return_value = mock_mapper

        case = Case(name="test", input="What is 2+2?")
        task = create_agent_task(factory)
        task(case)

        mock_agent.assert_called_once_with("What is 2+2?")

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_returns_dict_with_output_and_trajectory(self, mock_mapper_cls, mock_telemetry_cls):
        """The task returns a dict with 'output' and 'trajectory' keys."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.__str__ = MagicMock(return_value="42")
        mock_agent.return_value = mock_result

        factory = MagicMock(return_value=mock_agent)

        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry

        mock_session = MagicMock()
        mock_mapper = MagicMock()
        mock_mapper.map_to_session.return_value = mock_session
        mock_mapper_cls.return_value = mock_mapper

        case = Case(name="test", input="What is 2+2?")
        task = create_agent_task(factory)
        result = task(case)

        assert isinstance(result, dict)
        assert "output" in result
        assert "trajectory" in result
        assert result["output"] == "42"
        assert result["trajectory"] is mock_session

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_collects_spans_and_maps_to_session(self, mock_mapper_cls, mock_telemetry_cls):
        """The task collects spans from telemetry and maps them to a Session."""
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(__str__=MagicMock(return_value="res"))

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

        case = Case(name="test", input="hi", session_id="sess-123")
        task = create_agent_task(factory)
        result = task(case)

        mock_mapper.map_to_session.assert_called_once_with(list(mock_spans), "sess-123")
        assert result["trajectory"] is mock_session

    @patch("strands_evals.agent_task_adapter.StrandsEvalsTelemetry")
    @patch("strands_evals.agent_task_adapter.StrandsInMemorySessionMapper")
    def test_clears_spans_between_calls(self, mock_mapper_cls, mock_telemetry_cls):
        """Each call clears the in-memory exporter before running the agent."""
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(__str__=MagicMock(return_value="res"))

        factory = MagicMock(return_value=mock_agent)

        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry

        mock_mapper_cls.return_value = MagicMock()

        task = create_agent_task(factory)

        case1 = Case(name="test1", input="a")
        case2 = Case(name="test2", input="b")
        task(case1)
        task(case2)

        assert mock_telemetry.in_memory_exporter.clear.call_count == 2
