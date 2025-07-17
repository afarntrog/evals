from pydantic import BaseModel
from typing_extensions import TypeVar, Generic, Any
from collections.abc import Callable

from .case import Case
from .types.evaluation import EvaluationData
from .types.evaluation_report import EvaluationReport
from .evaluators.evaluator import Evaluator
from .evaluators.trajectory_evaluator import TrajectoryEvaluator
from .evaluators.llm_evaluator import LLMEvaluator

import json
import os

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class Dataset(BaseModel, Generic[InputT, OutputT]):
    """
    A collection of test cases, representing a dataset.

    Dataset organizes a collection of test cases and evaluate them all with
    the defined evaluator on some task. 

    Attributes:
        cases: A list of test cases in the dataset.
        evaluator: The evaluator to be used on the test cases.

    Example:
        dataset = Dataset[str, str](
            cases=[
                Case(name="Simple Knowledge",
                        input="What is the capital of France?",
                        expected_output="The capital of France is Paris.",
                        expected_trajectory=[],
                        metadata={"category": "knowledge"}),
               Case(name="Simple Math",
                        input="What is 2x2?",
                        expected_output="2x2 is 4.",
                        expected_trajectory=["calculator],
                        metadata={"category": "math"})
            ],
            evaluator=Evaluator()
        )
    """
    cases: list[Case[InputT, OutputT]]
    evaluator: Evaluator[InputT, OutputT]

    def _run_task(self, task: Callable[[InputT], OutputT | dict[str, Any]], case: Case[InputT, OutputT]) -> EvaluationData[InputT, OutputT]:       
        """
        Run the task with the inputs from the test case.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.
            case: The test case containing neccessary information to run the task

        Return:
            An EvaluationData record containing the input and actual output, name, expected output, and metadata.
        """
        evaluation_context = EvaluationData(name=case.name,
                                            input = case.input,
                                            expected_output=case.expected_output,
                                            expected_trajectory=case.expected_trajectory,
                                            metadata=case.metadata)
        task_output = task(case.input)
        if isinstance(task_output, dict): # could be evaluating the trajectory as well
            evaluation_context.actual_output = task_output.get("output")
            evaluation_context.actual_trajectory = task_output.get("trajectory")
        else: # evaluating only the output
            evaluation_context.actual_output = task_output
        return evaluation_context

    def run_evaluations(self, task: Callable[[InputT], OutputT | dict[str, Any]]) -> EvaluationReport:
        """
        Run the evaluations for all of the test cases with the evaluator.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.
                                     
        Return:
            An EvaluationReport containing the overall score, individual case results, and basic feedback for each test case.
        """        
        scores = []
        test_passes = []
        cases = []         
        reasons = []              
        for case in self.cases:
            try:
                evaluation_context = self._run_task(task, case)
                evaluation_output = self.evaluator.evaluate(evaluation_context)
                cases.append(evaluation_context.model_dump())
                test_passes.append(evaluation_output.test_pass)
                scores.append(evaluation_output.score)
                if evaluation_output.reason:
                    reasons.append(evaluation_output.reason)
                else:
                    reasons.append("")
            except Exception as e:
                cases.append(case.model_dump())
                test_passes.append(False)
                scores.append(0)
                reasons.append(f"An error occured : {str(e)}")

        report = EvaluationReport(overall_score = sum(scores)/len(scores) if len(scores) else 0,
                                  scores = scores,
                                  test_passes = test_passes,
                                  cases = cases,
                                  reasons=reasons)

        return report
    
    def to_dict(self) -> dict:
        """
        Convert the dataset to a dictionary.
        
        Return:
            A dictionary representation of the dataset.
        """
        return {
            "cases": [case.model_dump() for case in self.cases],
            "evaluator": self.evaluator.to_dict()
        }
    
    def to_file(self, file_name: str, format: str, directory: str = "dataset_files"):
        """
        Write the dataset to a file.

        Args:
            file_name: Name of the file without extension.
            format: The format of the file to be saved.
            directory: Directory to save the file (default: "dataset_files").
        """
        os.makedirs(directory, exist_ok=True)      
        if format == "json":
            with open(f"{directory}/{file_name}.json", "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise Exception(f"Format {format} is not supported.")
    
    @classmethod
    def from_dict(cls, data: dict, custom_evaluators: list[Evaluator] = []):
        """
        Create a dataset from a dictionary.

        Args:
            data: A dictionary representation of the dataset.
            custom_evaluators: A list of relevant custom evaluators.

        Return:
            A Dataset object.
        """
        cases = [Case.model_validate(case_data) for case_data in data["cases"]]
        default_evaluators = {"Evaluator": Evaluator,
                            "LLMEvaluator": LLMEvaluator,
                            "TrajectoryEvaluator": TrajectoryEvaluator}
        all_evaluators = {**default_evaluators, **{v.get_type_name(): v for v in custom_evaluators}}

        evaluator_type = data["evaluator"]["evaluator_type"]
        evaluator_args = {k:v for k,v in data["evaluator"].items() if k != "evaluator_type"}
 
        if evaluator_type in all_evaluators:
            evaluator = all_evaluators[evaluator_type](**evaluator_args)
        else:
           raise Exception(f"Cannot find {evaluator_type}. Make sure the evaluator type is spelled correctly and all relevant custom evaluators are passed in.")
           
        return cls(cases=cases, evaluator=evaluator)
    
    @classmethod
    def from_file(cls, file_path: str, format: str, custom_evaluators: list[Evaluator] = []):
        """
        Create a dataset from a file.

        Args:
            file_path: Path to the file.
            format: The format of the file to be read.
            custom_evaluators: A list of relevant custom evaluators.

        Return:
            A Dataset object.
        """
        if format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            raise Exception(f"Format {format} is not supported.")

        return cls.from_dict(data, custom_evaluators)


if __name__ == "__main__":
    pass


