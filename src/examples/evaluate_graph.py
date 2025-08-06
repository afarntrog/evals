import asyncio
import datetime

from strands import Agent
from strands.multiagent import GraphBuilder
from strands_evaluation.case import Case
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.interactions_evaluator import InteractionsEvaluator
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from strands_evaluation.evaluators.utils import helper_funcs


async def async_graph_node_history_example():
    """
    Demonstrates evaluating node execution order trajectories in graph responses asynchronously.

    Returns:
        EvaluationReport: Report containing trajectory evaluation results
    """
    ### Step 1: Create test cases ###
    test1 = Case(
        input="Research the impact of AI on healthcare and create a comprehensive report",
        expected_trajectory=["research", "fact_check", "analysis", "report"],
    )
    test2 = Case(
        input="Research the impact of robotics on healthcare and create a short report",
        expected_trajectory=["research", "fact_check", "analysis", "report"],
    )

    ### Step 2: Create evaluator ###
    evaluator = TrajectoryEvaluator(rubric="The graph pattern should ultilized the agents as expected.")

    ### Step 3: Create dataset ###
    dataset = Dataset(cases=[test1, test2], evaluator=evaluator)

    ### Step 4: Define task ###
    def research_graph(task: str):
        # Create specialized agents
        researcher = Agent(name="researcher", system_prompt="You are a research specialist...")
        analyst = Agent(name="analyst", system_prompt="You are a data analysis specialist...")
        fact_checker = Agent(name="fact_checker", system_prompt="You are a fact checking specialist...")
        report_writer = Agent(name="report_writer", system_prompt="You are a report writing specialist...")

        # Create a graph with these agents
        builder = GraphBuilder()
        # Add nodes
        builder.add_node(researcher, "research")
        builder.add_node(analyst, "analysis")
        builder.add_node(fact_checker, "fact_check")
        builder.add_node(report_writer, "report")

        # Add edges (dependencies)
        builder.add_edge("research", "analysis")
        builder.add_edge("research", "fact_check")
        builder.add_edge("analysis", "report")
        builder.add_edge("fact_check", "report")

        # Set entry points (optional - will be auto-detected if not specified)
        builder.set_entry_point("research")

        # Build the graph
        graph = builder.build()

        result = graph(task)

        return {"trajectory": [node.node_id for node in result.execution_order]}

    ### Step 5: Run evaluation ###
    report = await dataset.run_evaluations_async(research_graph)
    return report


async def async_graph_interaction_history_example():
    """
    Demonstrates evaluating interactions between graph nodes asynchronously.

    Returns:
        EvaluationReport: Report containing interaction evaluation results
    """
    ### Step 1: Create test cases ###
    test1 = Case(
        input="Research the impact of AI on healthcare and create a short report",
        expected_interactions=[
            {"node_name": "research", "dependencies": []},
            {"node_name": "fact_check", "dependencies": ["research"]},
            {"node_name": "analysis", "dependencies": ["research"]},
            {"node_name": "report", "dependencies": ["fact_check", "analysis"]},
        ],
    )
    test2 = Case(input="Research the impact of robotics on healthcare and create a short report")

    ### Step 2: Create evaluator ###
    rubric = {
        "research": "The research node should be the starting point and generate a query about the topic.",
        "fact_check": "The fact check node should come after research and verify the accuracy of the generated query.",
        "analysis": "The analysis node should come after research and generate a summary of the findings.",
        "report": "The report node should come after analysis"
        " and fact check and synthesize the information into a coherent report.",
    }
    # if want to use the same rubric
    basic_rubric = "The graph system should ultilized the agents as expected with relevant information." \
    " The actual interactions should include more information than expected."
    evaluator = InteractionsEvaluator(rubric=rubric)

    ### Step 3: Create dataset ###
    dataset = Dataset(cases=[test1, test2], evaluator=evaluator)

    ### Step 4: Define task ###
    def research_graph(task: str):
        # Create specialized agents
        researcher = Agent(name="researcher", system_prompt="You are a research specialist...")
        analyst = Agent(name="analyst", system_prompt="You are a data analysis specialist...")
        fact_checker = Agent(name="fact_checker", system_prompt="You are a fact checking specialist...")
        report_writer = Agent(name="report_writer", system_prompt="You are a report writing specialist...")

        # Create a graph with these agents
        builder = GraphBuilder()
        # Add nodes
        builder.add_node(researcher, "research")
        builder.add_node(analyst, "analysis")
        builder.add_node(fact_checker, "fact_check")
        builder.add_node(report_writer, "report")

        # Add edges (dependencies)
        builder.add_edge("research", "analysis")
        builder.add_edge("research", "fact_check")
        builder.add_edge("analysis", "report")
        builder.add_edge("fact_check", "report")

        # Set entry points (optional - will be auto-detected if not specified)
        builder.set_entry_point("research")

        # Build the graph
        graph = builder.build()

        result = graph(task)
        interactions = helper_funcs.extract_graph_interactions(result)

        return {"interactions": interactions}

    ### Step 5: Run evaluation ###
    report = await dataset.run_evaluations_async(research_graph)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.evaluate_graph

    # start = datetime.datetime.now()
    # report = asyncio.run(async_graph_node_history_example())
    # end = datetime.datetime.now()
    # print("Async node history", end - start)
    # report.to_file("async_graph_node_history_report", "json")
    # report.run_display(include_actual_trajectory=True)

    start = datetime.datetime.now()
    report = asyncio.run(async_graph_interaction_history_example())
    end = datetime.datetime.now()
    print("Async node interactions", end - start)
    report.to_file("async_graph_interaction_history_report", "json")
    report.run_display()
