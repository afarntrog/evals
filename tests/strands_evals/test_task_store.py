import json
import os

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Equals
from strands_evals.local_file_store import LocalFileStore
from strands_evals.types import EvaluationData


class TestLocalFileStore:
    def test_save_and_load(self, tmp_path):
        """Save EvaluationData, then load it back and verify contents."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        data = [
            EvaluationData(input="hello", actual_output="world", name="case_0"),
            EvaluationData(input="foo", actual_output="bar", name="case_1"),
        ]
        store.save(data)
        loaded = store.load()

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].input == "hello"
        assert loaded[0].actual_output == "world"
        assert loaded[1].input == "foo"
        assert loaded[1].actual_output == "bar"

    def test_load_returns_none_when_file_missing(self, tmp_path):
        """Load from a non-existent file returns None."""
        store = LocalFileStore(str(tmp_path / "nonexistent.json"))
        assert store.load() is None

    def test_save_creates_file(self, tmp_path):
        """Save creates the file on disk."""
        path = str(tmp_path / "results.json")
        store = LocalFileStore(path)
        store.save([EvaluationData(input="test", actual_output="result")])
        assert os.path.exists(path)

    def test_save_produces_valid_json(self, tmp_path):
        """Saved file contains valid JSON that can be parsed independently."""
        path = str(tmp_path / "results.json")
        store = LocalFileStore(path)
        store.save([EvaluationData(input="test", actual_output="result", name="case_0")])

        with open(path) as f:
            raw = json.load(f)

        assert isinstance(raw, list)
        assert len(raw) == 1
        assert raw[0]["input"] == "test"
        assert raw[0]["actual_output"] == "result"

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        """All EvaluationData fields survive a save/load roundtrip."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        data = [
            EvaluationData(
                input="query",
                actual_output="response",
                expected_output="expected",
                name="full_case",
                metadata={"session_id": "abc123", "key": "value"},
                actual_trajectory=["tool_a", "tool_b"],
                expected_trajectory=["tool_a"],
            )
        ]
        store.save(data)
        loaded = store.load()

        assert loaded is not None
        result = loaded[0]
        assert result.input == "query"
        assert result.actual_output == "response"
        assert result.expected_output == "expected"
        assert result.name == "full_case"
        assert result.metadata == {"session_id": "abc123", "key": "value"}
        assert result.actual_trajectory == ["tool_a", "tool_b"]
        assert result.expected_trajectory == ["tool_a"]

    def test_save_overwrites_existing(self, tmp_path):
        """Saving to the same path overwrites previous data."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        store.save([EvaluationData(input="first", actual_output="a")])
        store.save([EvaluationData(input="second", actual_output="b")])

        loaded = store.load()
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].input == "second"


class TestRunEvaluationsWithTaskStore:
    def test_saves_task_output_when_no_stored_data(self, tmp_path):
        """When store has no data, run_evaluations runs tasks and saves."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        case = Case(name="test", input="hello", expected_output="hello")
        experiment = Experiment(cases=[case], evaluators=[Equals()])

        experiment.run_evaluations(lambda c: c.input, task_store=store)

        loaded = store.load()
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].actual_output == "hello"

    def test_skips_task_when_stored_data_exists(self, tmp_path):
        """When store has data, run_evaluations skips task execution."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        store.save([EvaluationData(input="hello", actual_output="hello", expected_output="hello", name="test")])

        case = Case(name="test", input="hello", expected_output="hello")
        task_call_count = 0

        def task(c):
            nonlocal task_call_count
            task_call_count += 1
            return c.input

        experiment = Experiment(cases=[case], evaluators=[Equals()])
        reports = experiment.run_evaluations(task, task_store=store)

        assert task_call_count == 0
        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0

    def test_task_optional_when_store_has_data(self, tmp_path):
        """task parameter can be None when store has existing data."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        store.save([EvaluationData(input="hello", actual_output="hello", expected_output="hello", name="test")])

        case = Case(name="test", input="hello", expected_output="hello")
        experiment = Experiment(cases=[case], evaluators=[Equals()])
        reports = experiment.run_evaluations(task=None, task_store=store)

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0

    def test_raises_when_no_task_and_no_stored_data(self, tmp_path):
        """Raises ValueError when task is None and store has no data."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        case = Case(name="test", input="hello", expected_output="hello")
        experiment = Experiment(cases=[case], evaluators=[Equals()])

        with pytest.raises(ValueError, match="No stored data found and no task provided"):
            experiment.run_evaluations(task=None, task_store=store)

    def test_without_task_store_behaves_as_before(self):
        """run_evaluations without task_store works exactly as before."""
        case = Case(name="test", input="hello", expected_output="hello")
        experiment = Experiment(cases=[case], evaluators=[Equals()])
        reports = experiment.run_evaluations(lambda c: c.input)

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0


class TestRunEvaluationsAsyncWithTaskStore:
    @pytest.mark.asyncio
    async def test_saves_task_output_when_no_stored_data(self, tmp_path):
        """When store has no data, run_evaluations_async runs tasks and saves."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        case = Case(name="test", input="hello", expected_output="hello")
        experiment = Experiment(cases=[case], evaluators=[Equals()])

        await experiment.run_evaluations_async(lambda c: c.input, task_store=store)

        loaded = store.load()
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].actual_output == "hello"

    @pytest.mark.asyncio
    async def test_skips_task_when_stored_data_exists(self, tmp_path):
        """When store has data, run_evaluations_async skips task execution."""
        store = LocalFileStore(str(tmp_path / "results.json"))
        store.save([EvaluationData(input="hello", actual_output="hello", expected_output="hello", name="test")])

        case = Case(name="test", input="hello", expected_output="hello")
        task_call_count = 0

        def task(c):
            nonlocal task_call_count
            task_call_count += 1
            return c.input

        experiment = Experiment(cases=[case], evaluators=[Equals()])
        reports = await experiment.run_evaluations_async(task, task_store=store)

        assert task_call_count == 0
        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0
