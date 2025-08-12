import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from strands_evaluation.case import Case
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.evaluator import Evaluator
from strands_evaluation.evaluators.interactions_evaluator import InteractionsEvaluator
from strands_evaluation.evaluators.output_evaluator import OutputEvaluator
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from strands_evaluation.generators.dataset_generator import DatasetGenerator


class TestDatasetGenerator:
    def test_init(self):
        """Test initialization"""
        generator = DatasetGenerator(
            str,
            int,
            trajectory_type=str,
            include_expected_output=False,
            include_expected_trajectory=True,
            include_expected_interactions=True,
            include_metadata=True,
            model="test-model",
            max_parallel_num_cases=5,
        )
        assert generator.input_type == str
        assert generator.output_type == int
        assert generator.include_expected_output is False
        assert generator.include_expected_trajectory is True
        assert generator.include_expected_interactions is True
        assert generator.include_metadata is True
        assert generator.model == "test-model"
        assert generator.max_parallel_num_cases == 5

    @pytest.mark.asyncio
    async def test_case_worker(self):
        """Test case worker functionality"""
        generator = DatasetGenerator(str, str)
        queue = asyncio.Queue()
        queue.put_nowait(None)
        results = []

        mock_agent = AsyncMock()
        mock_case_data = MagicMock()
        mock_case_data.model_dump.return_value = {"name": "test", "input": "hello"}
        mock_agent.structured_output_async.return_value = mock_case_data

        with patch("strands_evaluation.generators.dataset_generator.Agent", return_value=mock_agent):
            await generator._case_worker(queue, "test prompt", [], results)

        assert len(results) == 1
        assert isinstance(results[0], Case)

    @pytest.mark.asyncio
    async def test_generate_cases_async(self):
        """Test async case generation"""
        generator = DatasetGenerator(str, str, max_parallel_num_cases=2)

        mock_agent = AsyncMock()
        mock_case_data = MagicMock()
        mock_case_data.model_dump.return_value = {"name": "test", "input": "hello"}
        mock_agent.structured_output_async.return_value = mock_case_data

        with patch("strands_evaluation.generators.dataset_generator.Agent", return_value=mock_agent):
            cases = await generator.generate_cases_async("test prompt", num_cases=3)

        assert len(cases) == 3
        assert all(isinstance(case, Case) for case in cases)

    @pytest.mark.asyncio
    async def test_construct_evaluator_async_output(self):
        """Test constructing OutputEvaluator"""
        generator = DatasetGenerator(str, str)

        mock_agent = AsyncMock()
        mock_agent.invoke_async.return_value = "Generated rubric"

        with patch("strands_evaluation.generators.dataset_generator.Agent", return_value=mock_agent):
            evaluator = await generator.construct_evaluator_async("test prompt", OutputEvaluator)

        assert isinstance(evaluator, OutputEvaluator)
        assert evaluator.rubric == "Generated rubric"

    @pytest.mark.asyncio
    async def test_construct_evaluator_async_trajectory(self):
        """Test constructing TrajectoryEvaluator"""
        generator = DatasetGenerator(str, str)

        mock_agent = AsyncMock()
        mock_agent.invoke_async.return_value = "Generated rubric"

        with patch("strands_evaluation.generators.dataset_generator.Agent", return_value=mock_agent):
            evaluator = await generator.construct_evaluator_async("test prompt", TrajectoryEvaluator)

        assert isinstance(evaluator, TrajectoryEvaluator)
        assert evaluator.rubric == "Generated rubric"

    @pytest.mark.asyncio
    async def test_construct_evaluator_async_interactions(self):
        """Test constructing InteractionsEvaluator"""
        generator = DatasetGenerator(str, str)

        mock_agent = AsyncMock()
        mock_agent.invoke_async.return_value = "Generated rubric"

        with patch("strands_evaluation.generators.dataset_generator.Agent", return_value=mock_agent):
            evaluator = await generator.construct_evaluator_async("test prompt", InteractionsEvaluator)

        assert isinstance(evaluator, InteractionsEvaluator)
        assert evaluator.rubric == "Generated rubric"

    @pytest.mark.asyncio
    async def test_construct_evaluator_async_invalid(self):
        """Test constructing evaluator with invalid type"""
        generator = DatasetGenerator(str, str)

        class CustomEvaluator(Evaluator):
            pass

        with pytest.raises(ValueError, match="is not a default evaluator"):
            await generator.construct_evaluator_async("test prompt", CustomEvaluator)

    @pytest.mark.asyncio
    async def test_from_scratch_async_no_evaluator(self):
        """Test generating dataset from scratch without evaluator"""
        generator = DatasetGenerator(str, str)

        mock_cases = [Case(name="test", input="hello")]

        with patch.object(generator, "generate_cases_async", return_value=mock_cases):
            dataset = await generator.from_scratch_async(["topic1"], "test task", num_cases=1)

        assert isinstance(dataset, Dataset)
        assert dataset.cases == mock_cases
        assert isinstance(dataset.evaluator, Evaluator)

    @pytest.mark.asyncio
    async def test_from_scratch_async_with_evaluator(self):
        """Test generating dataset from scratch with evaluator"""
        generator = DatasetGenerator(str, str)

        mock_cases = [Case(name="test", input="hello")]
        mock_evaluator = OutputEvaluator(rubric="test rubric")

        with (
            patch.object(generator, "generate_cases_async", return_value=mock_cases),
            patch.object(generator, "construct_evaluator_async", return_value=mock_evaluator),
        ):
            dataset = await generator.from_scratch_async(["topic1"], "test task", evaluator=OutputEvaluator)

        assert isinstance(dataset, Dataset)
        assert dataset.cases == mock_cases
        assert dataset.evaluator == mock_evaluator

    @pytest.mark.asyncio
    async def test_from_context_async_no_evaluator(self):
        """Test generating dataset from context without evaluator"""
        generator = DatasetGenerator(str, str)

        mock_cases = [Case(name="test", input="hello")]

        with patch.object(generator, "generate_cases_async", return_value=mock_cases):
            dataset = await generator.from_context_async("test context", "test task", num_cases=1)

        assert isinstance(dataset, Dataset)
        assert dataset.cases == mock_cases
        assert isinstance(dataset.evaluator, Evaluator)

    @pytest.mark.asyncio
    async def test_from_context_async_with_evaluator(self):
        """Test generating dataset from context with evaluator"""
        generator = DatasetGenerator(str, str)

        mock_cases = [Case(name="test", input="hello")]
        mock_evaluator = OutputEvaluator(rubric="test rubric")

        with (
            patch.object(generator, "generate_cases_async", return_value=mock_cases),
            patch.object(generator, "construct_evaluator_async", return_value=mock_evaluator),
        ):
            dataset = await generator.from_context_async("test context", "test task", evaluator=OutputEvaluator)

        assert isinstance(dataset, Dataset)
        assert dataset.cases == mock_cases
        assert dataset.evaluator == mock_evaluator

    @pytest.mark.asyncio
    async def test_from_dataset_async_generic_evaluator(self):
        """Test generating dataset from existing dataset with generic evaluator"""
        generator = DatasetGenerator(str, str)

        source_cases = [Case(name="source", input="source_input")]
        source_dataset = Dataset(cases=source_cases, evaluator=Evaluator())
        mock_new_cases = [Case(name="new", input="new_input")]

        with patch.object(generator, "generate_cases_async", return_value=mock_new_cases):
            dataset = await generator.from_dataset_async(source_dataset, "test task")

        assert isinstance(dataset, Dataset)
        assert dataset.cases == mock_new_cases
        assert isinstance(dataset.evaluator, Evaluator)

    @pytest.mark.asyncio
    async def test_from_dataset_async_default_evaluator(self):
        """Test generating dataset from existing dataset with default evaluator"""
        generator = DatasetGenerator(str, str)

        source_cases = [Case(name="source", input="source_input")]
        source_evaluator = OutputEvaluator(rubric="source rubric")
        source_dataset = Dataset(cases=source_cases, evaluator=source_evaluator)
        mock_new_cases = [Case(name="new", input="new_input")]
        mock_new_evaluator = OutputEvaluator(rubric="new rubric")

        with (
            patch.object(generator, "generate_cases_async", return_value=mock_new_cases),
            patch.object(generator, "construct_evaluator_async", return_value=mock_new_evaluator),
        ):
            dataset = await generator.from_dataset_async(source_dataset, "test task")

        assert isinstance(dataset, Dataset)
        assert dataset.cases == mock_new_cases
        assert dataset.evaluator == mock_new_evaluator

    @pytest.mark.asyncio
    async def test_update_current_dataset_async_add_cases_only(self):
        """Test updating dataset by adding new cases only"""
        generator = DatasetGenerator(str, str)

        source_cases = [Case(name="source", input="source_input")]
        source_evaluator = Evaluator()
        source_dataset = Dataset(cases=source_cases, evaluator=source_evaluator)
        mock_new_cases = [Case(name="new", input="new_input")]

        with patch.object(generator, "generate_cases_async", return_value=mock_new_cases):
            dataset = await generator.update_current_dataset_async(
                source_dataset, "test task", add_new_cases=True, add_new_rubric=False
            )

        assert len(dataset.cases) == 2
        assert dataset.cases == source_cases + mock_new_cases
        assert dataset.evaluator == source_evaluator

    @pytest.mark.asyncio
    async def test_update_current_dataset_async_add_rubric_only(self):
        """Test updating dataset by adding new rubric only"""
        generator = DatasetGenerator(str, str)

        source_cases = [Case(name="source", input="source_input")]
        source_evaluator = OutputEvaluator(rubric="source rubric")
        source_dataset = Dataset(cases=source_cases, evaluator=source_evaluator)
        mock_new_evaluator = OutputEvaluator(rubric="new rubric")

        with patch.object(generator, "construct_evaluator_async", return_value=mock_new_evaluator):
            dataset = await generator.update_current_dataset_async(
                source_dataset, "test task", add_new_cases=False, add_new_rubric=True
            )

        assert dataset.cases == source_cases
        print("here", dataset.evaluator)
        assert dataset.evaluator == mock_new_evaluator

    @pytest.mark.asyncio
    async def test_update_current_dataset_async_new_evaluator_type(self):
        """Test updating dataset with new evaluator type"""
        generator = DatasetGenerator(str, str)

        source_cases = [Case(name="source", input="source_input")]
        source_evaluator = OutputEvaluator(rubric="source rubric")
        source_dataset = Dataset(cases=source_cases, evaluator=source_evaluator)
        mock_new_evaluator = TrajectoryEvaluator(rubric="new rubric")

        with patch.object(generator, "construct_evaluator_async", return_value=mock_new_evaluator):
            dataset = await generator.update_current_dataset_async(
                source_dataset,
                "test task",
                add_new_cases=False,
                add_new_rubric=True,
                new_evaluator_type=TrajectoryEvaluator,
            )

        assert dataset.cases == source_cases
        assert isinstance(dataset.evaluator, TrajectoryEvaluator)

    @pytest.mark.asyncio
    async def test_update_current_dataset_async_unsupported_evaluator_type(self):
        """Test updating dataset with unsupported evaluator type falls back to original"""
        generator = DatasetGenerator(str, str)
        
        source_cases = [Case(name="source", input="source_input")]
        source_evaluator = OutputEvaluator(rubric="source rubric")
        source_dataset = Dataset(cases=source_cases, evaluator=source_evaluator)
        
        class UnsupportedEvaluator(Evaluator):
            pass
        
        dataset = await generator.update_current_dataset_async(
            source_dataset, "test task", add_new_cases=False, add_new_rubric=True,
            new_evaluator_type=UnsupportedEvaluator
        )
        
        assert dataset.cases == source_cases
        assert dataset.evaluator == source_evaluator

    def test_default_evaluators_mapping(self):
        """Test that default evaluators are properly mapped"""
        generator = DatasetGenerator(str, str)

        assert OutputEvaluator in generator._default_evaluators
        assert TrajectoryEvaluator in generator._default_evaluators
        assert InteractionsEvaluator in generator._default_evaluators
