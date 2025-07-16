import pytest
from unittest.mock import Mock, patch
from src.strands_evaluation.evaluators.llm_evaluator import LLMEvaluator
from src.strands_evaluation.types.evaluation import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    agent.structured_output.return_value = EvaluationOutput(
        score=0.8, 
        test_pass=True, 
        reason="Mock evaluation result"
    )
    return agent


@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="What is 2+2?",
        actual_output="4",
        expected_output="4",
        name="math_test"
    )


class TestLLMEvaluator:
    
    def test_init_with_defaults(self):
        """Test LLMEvaluator initialization with default values"""
        evaluator = LLMEvaluator(rubric="Test rubric")
        
        assert evaluator.rubric == "Test rubric"
        assert evaluator.model is None
        assert evaluator.include_inputs == True
        assert evaluator.system_prompt is not None  # Uses default template
    
    def test_init_with_custom_values(self):
        """Test LLMEvaluator initialization with custom values"""
        custom_prompt = "Custom system prompt"
        evaluator = LLMEvaluator(
            rubric="Custom rubric",
            model="gpt-4",
            system_prompt=custom_prompt,
            include_inputs=False
        )
        
        assert evaluator.rubric == "Custom rubric"
        assert evaluator.model == "gpt-4"
        assert evaluator.include_inputs == False
        assert evaluator.system_prompt == custom_prompt
    
    @patch('src.strands_evaluation.evaluators.llm_evaluator.Agent')
    def test_evaluate_with_inputs(self, mock_agent_class, evaluation_data, mock_agent):
        """Test evaluation with inputs included"""
        mock_agent_class.return_value = mock_agent
        evaluator = LLMEvaluator(rubric="Test rubric")
        
        result = evaluator.evaluate(evaluation_data)
        
        # Verify Agent was created with correct parameters
        mock_agent_class.assert_called_once_with(
            model=None, 
            system_prompt=evaluator.system_prompt, 
            callback_handler=None
        )
        
        # Verify structured_output was called
        mock_agent.structured_output.assert_called_once()
        call_args = mock_agent.structured_output.call_args
        assert call_args[0][0] == EvaluationOutput
        prompt = call_args[0][1]
        assert "<Input>What is 2+2?</Input>" in prompt
        assert "<Output>4</Output>" in prompt
        assert "<ExpectedOutput>4</ExpectedOutput>" in prompt
        assert "<Rubric>Test rubric</Rubric>" in prompt
        
        assert result.score == 0.8
        assert result.test_pass == True
    
    @patch('src.strands_evaluation.evaluators.llm_evaluator.Agent')
    def test_evaluate_without_inputs(self, mock_agent_class, evaluation_data, mock_agent):
        """Test evaluation without inputs included"""
        mock_agent_class.return_value = mock_agent
        evaluator = LLMEvaluator(rubric="Test rubric", include_inputs=False)
        
        result = evaluator.evaluate(evaluation_data)
        
        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        assert "<Input>" not in prompt
        assert "<Output>4</Output>" in prompt
        assert "<ExpectedOutput>4</ExpectedOutput>" in prompt
        assert "<Rubric>Test rubric</Rubric>" in prompt
        
        assert result.score == 0.8
        assert result.test_pass == True
    
    @patch('src.strands_evaluation.evaluators.llm_evaluator.Agent')
    def test_evaluate_without_expected_output(self, mock_agent_class, mock_agent):
        """Test evaluation without expected output"""
        mock_agent_class.return_value = mock_agent
        evaluator = LLMEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            actual_output="result",
        )
        
        evaluator.evaluate(evaluation_data)
        
        call_args = mock_agent.structured_output.call_args
        prompt = call_args[0][1]
        assert "<ExpectedOutput>" not in prompt
        assert "<Output>result</Output>" in prompt
    
    def test_evaluate_missing_actual_output(self):
        """Test evaluation raises exception when actual_output is missing"""
        evaluator = LLMEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input="test",
            expected_output="expected"
        )
        
        with pytest.raises(Exception, match="Please make sure the task function return the output"):
            evaluator.evaluate(evaluation_data)

    