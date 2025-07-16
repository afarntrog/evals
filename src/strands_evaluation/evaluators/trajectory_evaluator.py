from typing_extensions import TypeVar

from .evaluator import Evaluator
from ..types.evaluation import EvaluationData, EvaluationOutput
from .utils.prompt_templates import judge_trajectory_template_tools as SYSTEM_PROMPT
from strands import Agent
from .utils.evaluation_tools import exact_match_scorer, in_order_match_scorer, any_order_match_scorer

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class TrajectoryEvaluator(Evaluator[InputT, OutputT]):
    """
    An evaluator that is trajectory-based.

    Attributes:
        rubric: The user-specified criteria for evaluating a collection of test cases.
        model: A string representing the model-id for Bedrock to use.
                    Defaults to strands.models.BedrockModel if None.
        system_prompt: System prompt to guide model behavior.
                    If None, the evaluator will use one of the default template.
        include_inputs: Whether to include inputs to the task in the evaluation or not.
    """
    def __init__(self, rubric: str, model: str | None = None, system_prompt: str = SYSTEM_PROMPT,
                include_inputs: bool = True):
        super().__init__()
        self.rubric = rubric
        self.model = model
        self.include_inputs = include_inputs
        self._tools = [exact_match_scorer, in_order_match_scorer, any_order_match_scorer]
        self.system_prompt = system_prompt

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.
        """
        evaluator_agent = Agent(model=self.model,
                                system_prompt=self.system_prompt,
                                tools = self._tools,
                                callback_handler=None)
        evaluation_prompt = "Evaluate this singular test case. THE FINAL SCORE MUST BE A DECIMAL BETWEEN 0.0 AND 1.0 (NOT 0 to 10 OR 0 to 100). \n"
        if self.include_inputs:   
            evaluation_prompt += f"<Input>{evaluation_case.input}</Input>\n"

        if evaluation_case.actual_output:
            evaluation_prompt += f"<Output>{evaluation_case.actual_output}</Output>\n"

        if evaluation_case.expected_output:
            evaluation_prompt += f"<ExpectedOutput>{evaluation_case.expected_output}</ExpectedOutput>\n"

        if not evaluation_case.actual_trajectory:
            raise Exception("Please make sure the task function return a dictionary with the key 'trajectory'.")
        evaluation_prompt += f"<Trajectory>{evaluation_case.actual_trajectory}</Trajectory>\n"

        if evaluation_case.expected_trajectory:
            evaluation_prompt += f"<ExpectedTrajectory>{evaluation_case.expected_trajectory}</ExpectedTrajectory>\n"
        evaluation_prompt += f"<Rubric>{self.rubric}</Rubric>"

        result = evaluator_agent.structured_output(EvaluationOutput, evaluation_prompt)
        return result
    
if __name__ == "__main__":
    pass