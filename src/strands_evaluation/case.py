from pydantic import BaseModel
from typing_extensions import TypeVar, Generic, Any

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class Case(BaseModel, Generic[InputT, OutputT]):
    """
    A single test case, representing a row in Dataset.
    
    Each test case represents a single test scenario with inputs to test.
    Optionally, a test case may contains a name, expected outputs, expected trajectory,
    and arbitrary metadata.
    
    Attributes:
        input: The input to the task. eg. the query to the agent
        name: The name of the test case. This will be used to identify the test in the summary report.
        expected_output: The expected response given the input. eg. the agent's response
        expected_trajectory: The expected trajectory of a task given the input. eg. sequence of tools
        metadata: Additional information about the test case.

    Example:
        case = Case[str,str](name="Simple Math",
                        input="What is 2x2?",
                        expected_output="2x2 is 4.",
                        expected_trajectory=["calculator],
                        metadata={"category": "math"})
        
        simple_test_case = Case(input="What is 2x2?")                
    """
    name: str | None = None
    input: InputT
    expected_output: OutputT | None = None
    expected_trajectory: list[Any] | None = None
    metadata: dict[str, Any] = {}