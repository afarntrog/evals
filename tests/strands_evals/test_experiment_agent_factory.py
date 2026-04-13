"""Tests for Experiment.run_evaluations with agent_factory parameter."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.case import Case
from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.experiment import Experiment
from strands_evals.types.evaluation import EvaluationOutput


class PassingEvaluator(Evaluator):
    """Evaluator that always passes for testing."""

    def evaluate(self, evaluation_case):
        return [EvaluationOutput(score=1.0, test_pass=True, reason="pass")]


class TestRunEvaluationsWithAgentFactory:
    """Tests for the agent_factory parameter on run_evaluations."""

    def test_rejects_both_task_and_agent_factory(self):
        """Passing both task and agent_factory raises ValueError."""
        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )

        def my_task(case):
            return "output"

        def my_factory():
            return MagicMock()

        with pytest.raises(ValueError, match="Cannot specify both"):
            experiment.run_evaluations(task=my_task, agent_factory=my_factory)

    def test_rejects_neither_task_nor_agent_factory(self):
        """Passing neither task nor agent_factory raises ValueError."""
        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )

        with pytest.raises(ValueError, match="Must specify either"):
            experiment.run_evaluations()

    @patch("strands_evals.experiment.create_agent_task")
    def test_agent_factory_creates_task_via_adapter(self, mock_create_agent_task):
        """When agent_factory is passed, it's wrapped via create_agent_task."""
        mock_task = MagicMock(return_value="output")
        mock_create_agent_task.return_value = mock_task

        factory = MagicMock()
        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )
        experiment.run_evaluations(agent_factory=factory)

        mock_create_agent_task.assert_called_once_with(factory)

    @patch("strands_evals.experiment.create_agent_task")
    def test_agent_factory_results_flow_to_evaluators(self, mock_create_agent_task):
        """Results from the adapted agent task are evaluated normally."""
        mock_task = MagicMock(return_value={"output": "42", "trajectory": None})
        mock_create_agent_task.return_value = mock_task

        factory = MagicMock()
        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )
        reports = experiment.run_evaluations(agent_factory=factory)

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0
        assert reports[0].test_passes[0] is True

    def test_existing_task_parameter_still_works(self):
        """Existing task= parameter continues to work unchanged."""
        def my_task(case):
            return "output"

        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )
        reports = experiment.run_evaluations(task=my_task)

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0


class TestRunEvaluationsAsyncWithAgentFactory:
    """Tests for the agent_factory parameter on run_evaluations_async."""

    def test_rejects_both_task_and_agent_factory(self):
        """Passing both task and agent_factory raises ValueError."""
        import asyncio

        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            asyncio.run(experiment.run_evaluations_async(task=MagicMock(), agent_factory=MagicMock()))

    def test_rejects_neither_task_nor_agent_factory(self):
        """Passing neither task nor agent_factory raises ValueError."""
        import asyncio

        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )

        with pytest.raises(ValueError, match="Must specify either"):
            asyncio.run(experiment.run_evaluations_async())

    @patch("strands_evals.experiment.create_agent_task")
    def test_agent_factory_works_async(self, mock_create_agent_task):
        """agent_factory works with run_evaluations_async."""
        import asyncio

        mock_task = MagicMock(return_value="output")
        mock_create_agent_task.return_value = mock_task

        factory = MagicMock()
        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )
        reports = asyncio.run(experiment.run_evaluations_async(agent_factory=factory))

        mock_create_agent_task.assert_called_once_with(factory)
        assert len(reports) == 1
