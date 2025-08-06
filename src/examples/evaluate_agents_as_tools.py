import asyncio
import datetime

from strands import Agent, tool
from strands_evaluation.case import Case
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.interactions_evaluator import InteractionsEvaluator
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from strands_evaluation.evaluators.utils import helper_funcs
from strands_tools import http_request, retrieve


async def async_evaluate_tools_as_agents_example():
    """Evaluate an orchestrator agent that uses specialized agents as tools.

    Returns:
        EvaluationReport: Report containing trajectory evaluation results
    """
    ### Step 1: Create test cases ###
    test1 = Case(input="Plan a trip for Paris next weekend.")

    ### Step 2: Create evaluator ###
    evaluator = TrajectoryEvaluator(rubric="Are the tools used reasonably and logically?")

    ### Step 3: Create dataset ###
    dataset = Dataset(cases=[test1], evaluator=evaluator)

    ### Step 4: Define task ###
    def trip_planner(task: str):
        # Define a specialized system prompt
        RESEARCH_ASSISTANT_PROMPT = """
        You are a specialized research assistant. Focus only on providing
        factual, well-sourced information in response to research questions.
        Always cite your sources when possible.
        """

        @tool
        def research_assistant(query: str) -> str:
            """
            Process and respond to research-related queries.

            Args:
                query: A research question requiring factual information

            Returns:
                A detailed research answer with citations
            """
            try:
                # Strands Agents SDK makes it easy to create a specialized agent
                research_agent = Agent(
                    system_prompt=RESEARCH_ASSISTANT_PROMPT,
                    tools=[retrieve, http_request],  # Research-specific tools
                )

                # Call the agent and return its response
                response = research_agent(query)
                return str(response)
            except Exception as e:
                return f"Error in research assistant: {str(e)}"

        @tool
        def trip_planning_assistant(query: str) -> str:
            """
            Create travel itineraries and provide travel advice.

            Args:
                query: A travel planning request with destination and preferences

            Returns:
                A detailed travel itinerary or travel advice
            """
            try:
                travel_agent = Agent(
                    system_prompt="""You are a specialized travel planning assistant.
                    Create detailed travel itineraries based on user preferences.""",
                    tools=[retrieve, http_request],  # Travel information tools
                )
                response = travel_agent(query)
                return str(response)
            except Exception as e:
                return f"Error in trip planning: {str(e)}"

        # Define the orchestrator system prompt with clear tool selection guidance
        MAIN_SYSTEM_PROMPT = """
        You are an assistant that routes queries to specialized agents:
        - For research questions and factual information → Use the research_assistant tool
        - For product recommendations and shopping advice → Use the product_recommendation_assistant tool
        - For travel planning and itineraries → Use the trip_planning_assistant tool
        - For simple questions not requiring specialized knowledge → Answer directly

        Always select the most appropriate tool based on the user's query.
        """

        # Strands Agents SDK allows easy integration of agent tools
        orchestrator = Agent(
            system_prompt=MAIN_SYSTEM_PROMPT, callback_handler=None, tools=[research_assistant, trip_planning_assistant]
        )

        response = orchestrator(task)

        # This helper function does not include message
        # return {
        #     "output": str(response),
        #     "trajectory": helper_funcs.extract_agent_tools_used_from_metrics(response)}

        # This helper function does not include message
        return {
            "output": str(response),
            "trajectory": helper_funcs.extract_agent_tools_used_from_messages(orchestrator.messages),
        }

    report = await dataset.run_evaluations_async(trip_planner)
    return report


async def async_agents_as_tools_interaction_example():
    """Evaluate an orchestrator agent that uses specialized agents as tools using InteractionEvaluator.

    Returns:
        EvaluationReport: Report containing the interaction evaluation results
    """
    ### Step 1: Create test cases ###
    test1 = Case(input="Plan a trip for Paris next weekend.")

    ### Step 2: Create evaluator ###
    evaluator = InteractionsEvaluator(rubric="Are the tools used reasonably and logically?")

    ### Step 3: Create dataset ###
    dataset = Dataset(cases=[test1], evaluator=evaluator)

    ### Step 4: Define task ###
    def trip_planner(task: str):
        # Define a specialized system prompt
        RESEARCH_ASSISTANT_PROMPT = """
        You are a specialized research assistant. Focus only on providing
        factual, well-sourced information in response to research questions.
        Always cite your sources when possible.
        """

        @tool
        def research_assistant(query: str) -> str:
            """
            Process and respond to research-related queries.

            Args:
                query: A research question requiring factual information

            Returns:
                A detailed research answer with citations
            """
            try:
                # Strands Agents SDK makes it easy to create a specialized agent
                research_agent = Agent(
                    system_prompt=RESEARCH_ASSISTANT_PROMPT,
                    tools=[retrieve, http_request],  # Research-specific tools
                    callback_handler=None,
                )

                # Call the agent and return its response
                response = research_agent(query)
                return str(response)
            except Exception as e:
                return f"Error in research assistant: {str(e)}"

        @tool
        def trip_planning_assistant(query: str) -> str:
            """
            Create travel itineraries and provide travel advice.

            Args:
                query: A travel planning request with destination and preferences

            Returns:
                A detailed travel itinerary or travel advice
            """
            try:
                travel_agent = Agent(
                    system_prompt="""You are a specialized travel planning assistant.
                    Create detailed travel itineraries based on user preferences.""",
                    tools=[retrieve, http_request],  # Travel information tools
                    callback_handler=None,
                )
                response = travel_agent(query)
                return str(response)
            except Exception as e:
                return f"Error in trip planning: {str(e)}"

        # Define the orchestrator system prompt with clear tool selection guidance
        MAIN_SYSTEM_PROMPT = """
        You are an assistant that routes queries to specialized agents:
        - For research questions and factual information → Use the research_assistant tool
        - For product recommendations and shopping advice → Use the product_recommendation_assistant tool
        - For travel planning and itineraries → Use the trip_planning_assistant tool
        - For simple questions not requiring specialized knowledge → Answer directly

        Always select the most appropriate tool based on the user's query.
        """

        # Strands Agents SDK allows easy integration of agent tools
        orchestrator = Agent(
            system_prompt=MAIN_SYSTEM_PROMPT, callback_handler=None, tools=[research_assistant, trip_planning_assistant]
        )
        response = orchestrator(task)

        evaluator.update_interaction_description(helper_funcs.extract_tools_description(orchestrator))
        tools_used = helper_funcs.extract_agent_tools_used_from_messages(orchestrator.messages)
        interactions = []
        for tool_used in tools_used:
            interactions.append(
                {"node_name": tool_used.get("name"), "dependencies": [], "messages": tool_used.get("message")}
            )

        # This helper function does not include message
        return {"output": str(response), "interactions": interactions}

    report = await dataset.run_evaluations_async(trip_planner)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.evaluate_agents_as_tools
    # start = datetime.datetime.now()
    # report = asyncio.run(async_evaluate_tools_as_agents_example())
    # end = datetime.datetime.now()
    # print("Time: ", end - start)
    # report.to_file("async_tools_as_agents_report_w_output", "json")
    # report.run_display(include_actual_trajectory=True)

    start = datetime.datetime.now()
    report = asyncio.run(async_agents_as_tools_interaction_example())
    end = datetime.datetime.now()
    print("Time: ", end - start)
    report.to_file("async_agent_as_tools_interaction", "json")
    report.run_display(include_actual_interactions=True)
