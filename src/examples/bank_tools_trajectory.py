import asyncio
import datetime

from strands import Agent, tool
from strands_evaluation.case import Case
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from strands_evaluation.evaluators.utils import helper_funcs

balances = {"Eric": -100, "Audrey": 800, "Brian": 300, "Hailey": 0}


@tool
def get_balance(person: str) -> int:
    """
    get the balance of a bank account.

    Args:
        person (str): The person to check the balance for.

    Returns:
        int: The balance of the bank account on the given day.
    """
    # Simple example, but real case could check the database etc.
    return balances.get(person, 0)


@tool
def modify_balance(person: str, amount: int) -> None:
    """
    Modify the balance of a bank account by a given amount.

    Args:
        person (str): The person to modify the balance for.
        amount (int): The amount to add to the balance.

    Returns:
        None
    """
    balances[person] += amount


@tool
def collect_debt() -> list[tuple]:
    """
    Check all of the bank accounts for any debt.

    Returns:
        list: A list of tuples, where each tuple contains the person and their debt.
    """
    debt = []
    for person in balances:
        if balances[person] < 0:
            debt.append((person, abs(balances[person])))

    return debt


async def async_descriptive_tools_trajectory_example():
    """
    Demonstrates evaluating tool usage trajectories in agent responses asynchronously.

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
    case1 = Case(
        name="Negative money",
        input="Eric wants to spend $100.",
        expected_output="Eric should not be able to spend money. We need to collect $100 from him.",
        expected_trajectory=["get_balance", "collect_debt"],
        metadata={"category": "banking"},
    )
    case2 = Case(
        name="Positive money",
        input="Audrey wants to spend $100.",
        expected_output="Audrey should be able to spend the money successfully. Her balance is now $700.",
        expected_trajectory=["get_balance", "modify_balance", "get_balance"],
        metadata={"category": "banking"},
    )
    case3 = Case(
        name="Exact spending",
        input="Brian wants to spend 300.",
        expected_output="Brian spends the money successfully. Brian's balance is now 0.",
        expected_trajectory=["get_balance", "modify_balance", "get_balance"],
    )
    case4 = Case(
        name="No money",
        input="Hailey wants to spend $1.",
        expected_output="Hailey should not be able to spend money.",
        expected_trajectory=["get_balance"],
        metadata={"category": "banking"},
    )

    ### Step 2: Create evaluator ###
    trajectory_evaluator = TrajectoryEvaluator(
        rubric="The trajectory should be in the correct order with all of the steps as the expected"
        " 0 if any step is missing.",
        include_inputs=True,
    )

    ### Step 3: Create dataset ###
    dataset = Dataset[str, str](cases=[case1, case2, case3, case4], evaluator=trajectory_evaluator)

    ### Step 3.5: Save the dataset ###
    dataset.to_file("async_bank_tools_trajectory_dataset", "json")

    ### Step 4: Define task ###
    async def get_response(query: str) -> dict:
        bank_prompt = (
            "You are a banker, ensure that only people with sufficient balance can spend them."
            " Collect debt from people with negative balance."
            " Be sure to report the current balance after all of the actions."
        )
        agent = Agent(
            tools=[get_balance, modify_balance, collect_debt], system_prompt=bank_prompt, callback_handler=None
        )
        response = await agent.invoke_async(query)
        trajectory_evaluator.update_trajectory_description(helper_funcs.extract_tools_description(agent))
        return {
            "output": str(response),
            "trajectory": helper_funcs.extract_agent_tools_used_from_messages(agent.messages),
        }

    ### Step 5: Run evaluation ###
    report = await dataset.run_evaluations_async(get_response)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.bank_tools_trajectory
    start = datetime.datetime.now()
    report = asyncio.run(async_descriptive_tools_trajectory_example())
    end = datetime.datetime.now()
    print("Async: ", end - start)
    report.to_file("async_bank_tools_trajectory_report", "json")
    report.run_display(include_actual_trajectory=True)
