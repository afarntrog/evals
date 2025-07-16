from typing_extensions import TypeVar

from .evaluator import Evaluator
from ..types.evaluation import EvaluationData, EvaluationOutput
from .utils.prompt_templates import judge_output_template as SYSTEM_PROMPT
from strands import Agent

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class LLMEvaluator(Evaluator[InputT, OutputT]):
    """
    An evaluator that is LLM-based. 

    Attributes:
        rubric: The user-specified criteria for evaluating a collection of test cases.
        model: A string representing the model-id for Bedrock to use.
                    Defaults to strands.models.BedrockModel if None.
        system_prompt: System prompt to guide model behavior.
                    If None, the evaluator will use one of the default template.
        include_inputs: Whether to include inputs to the task in the evaluation or not.
    """
    def __init__(self, rubric: str, model: str | None = None,
                system_prompt: str = SYSTEM_PROMPT, include_inputs: bool = True):
        super().__init__()
        self.rubric = rubric
        self.model = model
        self.include_inputs = include_inputs
        self.system_prompt = system_prompt

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.
        """
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        evaluation_prompt = "Evaluate this singular test case. THE FINAL SCORE MUST BE A DECIMAL BETWEEN 0.0 AND 1.0 (NOT 0 to 10 OR 0 to 100). \n"
        if self.include_inputs:   
            evaluation_prompt += f"<Input>{evaluation_case.input}</Input>\n"

        if not evaluation_case.actual_output:
            raise Exception("Please make sure the task function return the output or a dictionary with the key 'output'.")
        
        evaluation_prompt += f"<Output>{evaluation_case.actual_output}</Output>\n"
        if evaluation_case.expected_output:
            evaluation_prompt += f"<ExpectedOutput>{evaluation_case.expected_output}</ExpectedOutput>\n"
        evaluation_prompt += f"<Rubric>{self.rubric}</Rubric>"

        result = evaluator_agent.structured_output(EvaluationOutput, evaluation_prompt)
        return result
    
if __name__ == "__main__":
    pass

