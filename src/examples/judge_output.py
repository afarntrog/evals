from strands import Agent
from strands_evaluation.dataset import Dataset
from strands_evaluation.case import Case
from strands_evaluation.evaluators.llm_evaluator import LLMEvaluator

def output_judge_example():
   """
   Demonstrates using an LLM-based evaluator to judge agent outputs.
   
   This example:
   1. Creates test cases
   2. Creates an LLMEvaluator with a specified rubric
   3. Creates a dataset with the test cases and evaluator
   4. Saves the dataset to a JSON file
   5. Defines a task function that uses an agent to generate responses
   6. Runs evaluations and returns the report
   
   Returns:
      EvaluationReport: The evaluation results
   """
   ### Step 1: Create test cases ###
   test_case1 = Case[str, str](name = "knowledge-1",
                                 input="What is the capital of France?",
                                 expected_output="The capital of France is Paris.",
                                 metadata={"category": "knowledge"})
                                 
   test_case2 = Case[str, str](name = "knowledge-2",
                                 input="What color is the ocean?",
                                 metadata={"category": "knowledge"})

   ### Step 2: Create evaluator ###   
   LLM_judge = LLMEvaluator(rubric="The output should represent a reasonable answer to the input.",
                           include_inputs = True)
   ## or 
   LLM_judge_w_prompt = LLMEvaluator(rubric="The output should represent a reasonable answer to the input.",
                                 system_prompt = "You are an expert AI evaluator. Your job is to assess the quality of the response based according to a user-specified rubric. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}",
                                 include_inputs = True)

   ### Step 3: Create dataset ###                                                                
   dataset = Dataset[str, str](cases = [test_case1, test_case2],
                                             evaluator = LLM_judge)
   ### Step 3.5: Save the dataset ###
   dataset.to_file("judge_output_dataset", "json")

   ### Step 4: Define task ###                                      
   # simple example here but could be more complex depending on the user's needs
   def get_response(query: str) -> str:
      agent = Agent(callback_handler=None)
      return str(agent(query))

   ### Step 5: Run evaluation ###                                                                                                                                                
   report = dataset.run_evaluations(get_response)
   return report

if __name__ == "__main__":
   # run the file as a module: eg. python -m examples.judge_output 
   report = output_judge_example()
   report.display()
   report.to_file("judge_output_report", "json")