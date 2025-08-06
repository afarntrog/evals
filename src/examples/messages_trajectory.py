from strands import Agent
from strands_evaluation.case import Case
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from strands_tools import calculator


def messages_trajectory():
    """
    Demonstrates evaluating conversation trajectories with multiple inputs.

    This example:
    1. Creates test cases with sequences of inputs (conversation turns)
    2. Creates a TrajectoryEvaluator to assess conversation quality
    3. Creates a dataset with the test cases and evaluator
    4. Saves the dataset to a JSON file
    5. Defines a task function that processes multiple inputs and returns the conversation history
    6. Runs evaluations and returns the report

    Returns:
        EvaluationReport: The evaluation results
    """
    ### Step 1: Create test cases ###
    test_case1 = Case[list, str](name="intro", input=["Hello", "What tools do you have?", "How can I use them?"])

    test_case2 = Case[list, str](
        name="math teacher",
        input=["How do I take the derivatives?", "How do I take the derivative of 2x with respect to x?"],
    )

    ### Step 2: Create evaluator ###
    trajectory_evaluator = TrajectoryEvaluator(
        rubric="The trajectory should represent a reasonable sequence of messages based on the inputs.",
        include_inputs=True,
    )

    ### Step 3: Create dataset ###
    dataset = Dataset[list, str](cases=[test_case1, test_case2], evaluator=trajectory_evaluator)

    ### Step 3.5: Save the dataset ###
    dataset.to_file("messages_trajectory_dataset", "json")

    ### Step 4: Define task ###
    def get_response(inputs: list) -> dict:
        agent = Agent(tools=[calculator], callback_handler=None)
        for query in inputs:
            response = agent(query)

        return {"trajectory": list(agent.messages)}

    ### Step 5: Run evaluation ###
    report = dataset.run_evaluations(get_response)
    return report


def memory_messages_trajectory():
    """
    Demonstrates evaluating conversation memory and context retention.

    This example:
    1. Creates a test case with a sequence of inputs including a memory test
    2. Creates a trajectory evaluator to assess conversation quality
    3. Creates a dataset with the test case and evaluator
    4. Saves the dataset to a JSON file
    5. Defines a task function that processes multiple inputs and returns both
       the final response and the full conversation history
    6. Runs evaluations and returns the report

    Returns:
        EvaluationReport: The evaluation results
    """
    ### Step 1: Create test cases ###
    test_case1 = Case[list, str](
        name="intro",
        expected_output="""Looking at our conversation history, you've asked me:
                        1. **"Hello"** - You greeted me first
                        2. **"What tools do you have?"** - You asked about my available tools (which I just provided a comprehensive list for)
                        That's our complete conversation so far!""",
        input=["Hello", "What tools do you have?", "What did I asked you before?"],
    )

    ### Step 2: Create evaluator ###
    trajectory_evaluator = TrajectoryEvaluator(
        rubric="The trajectory should represent a reasonable sequence of messages based on the inputs.",
        include_inputs=True,
    )

    ### Step 3: Create dataset ###
    dataset = Dataset[list, str](cases=[test_case1], evaluator=trajectory_evaluator)

    ### Step 3.5: Save the dataset ###
    dataset.to_file("memory_messages_dataset", "json")

    ### Step 4: Define task ###
    def get_response(inputs: list) -> dict:
        agent = Agent(tools=[calculator], callback_handler=None)
        response = None
        for query in inputs:
            response = agent(query)

        return {"output": str(response), "trajectory": list(agent.messages)}

    ### Step 5: Run evaluation ###
    report = dataset.run_evaluations(get_response)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.messages_trajectory
    report = messages_trajectory()
    report.display(include_output=False)
    report.to_file("messages_trajectory_report", "json")

    report = memory_messages_trajectory()
    report.display()
    report.to_file("memory_messages_trajectory_report", "json")
