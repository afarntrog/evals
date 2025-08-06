import asyncio
import datetime

from strands import Agent
from strands.multiagent import Swarm
from strands_evaluation.case import Case
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.interactions_evaluator import InteractionsEvaluator
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from strands_evaluation.evaluators.utils import helper_funcs


async def async_swarm_node_history_example():
    """
    Demonstrates evaluating node history trajectories in agent responses asynchronously.

    Returns:
        EvaluationReport: Report containing trajectory evaluation results
    """
    ### Step 1: Create test cases ###
    test1 = Case(
        input="Design and implement a simple Rest API for a todo app.",
        expected_trajectory=["researcher", "architect", "coder", "reviewer"],
    )

    ### Step 2: Create evaluator ###
    evaluator = TrajectoryEvaluator(rubric="The swarm should ultilized the agents as expected.")

    ### Step 3: Create dataset ###
    dataset = Dataset(cases=[test1], evaluator=evaluator)

    ### Step 4: Define task ###
    def sde_swarm(task: str):
        # Create specialized agents
        researcher = Agent(name="researcher", system_prompt="You are a research specialist...")
        coder = Agent(name="coder", system_prompt="You are a coding specialist...")
        reviewer = Agent(name="reviewer", system_prompt="You are a code review specialist...")
        architect = Agent(name="architect", system_prompt="You are a system architecture specialist...")

        # Create a swarm with these agents
        swarm = Swarm(
            [researcher, coder, reviewer, architect],
            max_handoffs=20,
            max_iterations=20,
            execution_timeout=900.0,  # 15 minutes
            node_timeout=300.0,  # 5 minutes per agent
            repetitive_handoff_detection_window=8,  # There must be >= 3 unique agents in the last 8 handoffs
            repetitive_handoff_min_unique_agents=3,
        )

        result = swarm(task)
        return {"trajectory": [node.node_id for node in result.node_history]}

    ### Step 5: Run evaluation ###
    report = await dataset.run_evaluations_async(sde_swarm)
    return report


async def async_swarm_interactions_example():
    """
    Demonstrates evaluating handoff interactions between agents asynchronously.

    Returns:
        EvaluationReport: Report containing interaction evaluation results
    """
    ### Step 1: Create test cases ###
    test1 = Case(input="Design and implement a simple Rest API for a todo app.")

    ### Step 2: Create evaluator ###
    evaluator = InteractionsEvaluator(
        rubric="The interaction sequence should represent a logical and optimal handoff of tasks" \
        " from one agent to another."
    )

    ### Step 3: Create dataset ###
    dataset = Dataset(cases=[test1], evaluator=evaluator)

    ### Step 4: Define task ###
    def sde_swarm(task: str):
        # Create specialized agents
        researcher = Agent(name="researcher", system_prompt="You are a research specialist...")
        coder = Agent(name="coder", system_prompt="You are a coding specialist...")
        reviewer = Agent(name="reviewer", system_prompt="You are a code review specialist...")
        architect = Agent(name="architect", system_prompt="You are a system architecture specialist...")

        # Create a swarm with these agents
        swarm = Swarm(
            [researcher, coder, reviewer, architect],
            max_handoffs=20,
            max_iterations=20,
            execution_timeout=900.0,  # 15 minutes
            node_timeout=300.0,  # 5 minutes per agent
            repetitive_handoff_detection_window=8,  # There must be >= 2 unique agents in the last 8 handoffs
            repetitive_handoff_min_unique_agents=2,
        )

        result = swarm(task)
        interaction_info = helper_funcs.extract_swarm_interactions(result)
        return {"interactions": interaction_info}

    ### Step 5: Run evaluation ###
    report = await dataset.run_evaluations_async(sde_swarm)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.evaluate_swarm
    start = datetime.datetime.now()
    report = asyncio.run(async_swarm_node_history_example())
    end = datetime.datetime.now()
    report.to_file("async_swarm_node_history_report", "json")
    report.run_display(include_actual_trajectory=True)

    start = datetime.datetime.now()
    report = asyncio.run(async_swarm_interactions_example())
    end = datetime.datetime.now()
    report.to_file("async_swarm_interactions_report", "json")
    report.run_display(include_actual_interactions=True, include_expected_interactions=True)
