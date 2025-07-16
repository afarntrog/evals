from strands import Agent
from strands_tools import calculator

from strands_evaluation.dataset import Dataset 
from strands_evaluation.case import Case
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator

def tools_trajectory_example():
    """
    Demonstrates evaluating tool usage trajectories in agent responses.
    
    This example:
    1. Creates test cases with expected outputs and tool trajectories
    2. Creates a TrajectoryEvaluator to assess tool usage
    3. Creates a dataset with the test cases and evaluator
    4. Saves the dataset to a JSON file
    5. Defines a task function that uses an agent with calculator tool
       and returns both the response and the tools used
    6. Runs evaluations and returns the report
    
    Returns:
        EvaluationReport: The evaluation results
    """
    ### Step 1: Create test cases ###
    test_case1 = Case[str, str](name = "calculator-1",
                                    input="What is the square root of 9?",
                                    expected_output="The square root of 9 is 3.",
                                    expected_trajectory=["calculator"],
                                    metadata={"category": "math"})
                                    
    test_case2 = Case[str, str](name = "calculator-2",
                                    input="What is 2x2?",
                                    expected_output="4",
                                    metadata={"category": "math"})
                                    
    
    ### Step 2: Create evaluator ###   
    trajectory_evaluator = TrajectoryEvaluator(rubric = "The trajectory should represent a reasonable use of tools based on the input.",
                                            include_inputs = True)

    ### Step 3: Create dataset ###     
    dataset = Dataset[str, str](cases = [test_case1, test_case2], evaluator = trajectory_evaluator)

    ### Step 3.5: Save the dataset ###
    dataset.to_file("tools_trajectory_dataset", "json")

    ### Step 4: Define task ###  
    def get_response(query: str) -> dict:
        agent = Agent(tools = [calculator])
        response = agent(query)
       
        return {"output": str(response),
                "trajectory": list(response.metrics.tool_metrics.keys())}
    
    ### Step 5: Run evaluation ###                                                                                                                                                
    report = dataset.run_evaluations(get_response)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.tools_trajectory 
    report = tools_trajectory_example()
    report.display()
    report.to_file("tools_trajectory_report", "json")


