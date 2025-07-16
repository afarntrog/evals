from pydantic import BaseModel
from typing_extensions import TypeVar, Generic, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json
import os

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class EvaluationData(BaseModel, Generic[InputT, OutputT]):
    """
    A record of all of the context for the evaluator to evaluate a test case.

    Attributes:
        input: The input to the task. eg. the query to the agent
        actual_output: The actual response given the input.
        expected_output: The expected response given the input.
        actual_trajectory: The actual trajectory of a task given the input. 
        expected_trajectory: The expected trajectory of a task given the input. 
        name: The name of the test case. This will be used to identify the test in the summary report.
        metadata: Additional information about the test case.
    """
    input: InputT
    actual_output: OutputT | None = None
    name: str | None = None
    expected_output: OutputT | None = None
    expected_trajectory: list[Any] | None = None
    actual_trajectory: list[Any] | None = None
    metadata: dict = {}

class EvaluationReport(BaseModel):
    """
    A report of the evaluation of a task.

    Attributes:
        overall_score: The overall score of the task.
        scores: A list of the score for each test case in order.
        cases: A list of records for each test case.
        test_passes: A list of booleans indicating whether the test pass or fail.
        reasons: A list of reason for each test case.
    """
    overall_score: float
    scores: list[float]
    cases: list[dict]
    test_passes: list[bool]
    reasons: list[str] | None = []

    def __str__(self):
        """
        Returns a string representation of the report in its simplest forms.
        """
        display_text = "\n ### REPORT ### \n Name, input, score, test_pass, reason \n"
        for i in range(len(self.scores)):
            display_text += f"Test case {i+1}, {self.cases[i]["input"]}, {self.scores[i]}, {self.test_passes[i]}, {self.reasons[i]} \n\n"
        return display_text
    
    def display(self, include_input = True, include_output = True):
        """
        Render a beautiful string representation of the report with as much details as configured.
        """
        console = Console()
        console.print(Panel(f"[bold blue]Overall Score: {self.overall_score:.2f}[/bold blue]", title="📊 Evaluation Report"))
        
        table = Table(title="Test Case Results", show_lines=True, padding=(1, 1))
        table.add_column("#", style="cyan")
        table.add_column("Name", style="magenta", max_width=15)
        table.add_column("Score", style="green")
        table.add_column("Pass", justify="center")
        if include_input:
            table.add_column("Input", style="yellow")
        if include_output:
            table.add_column("Output", style="yellow")
        table.add_column("Reason", style="yellow")
        
        for i in range(len(self.scores)):
            name = self.cases[i].get("name", f"Test {i+1}")
            pass_status = "✅" if self.test_passes[i] else "❌"
            reason = self.reasons[i] if i < len(self.reasons) else "N/A"
            
            renderables = [str(i+1), 
                            name, 
                            f"{self.scores[i]:.2f}", 
                            pass_status]
            if include_input:
                renderables.append(str(self.cases[i].get("input")))
            if include_output:
                renderables.append(str(self.cases[i].get("actual_output")))
            
            renderables.append(reason)
            table.add_row(*renderables)
        
        console.print(table)
    
    def to_dict(self):
        """
        Returns a dictionary representation of the report.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an EvaluationReport instance from a dictionary.

        Args:
            data: A dictionary containing the report data.
        """
        return cls.model_validate(data)

    def to_file(self, file_name: str, format: str, directory: str = "report_files"):
        """
        Write the report to a file.

        Args:
            file_name: Name of the file without extension.
            format: The format of the file to be saved.
            directory: Directory to save the file (default: "report_files").
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if format == "json":
            with open(f"{directory}/{file_name}.json", "w") as f:
                json.dump(self.to_dict(), f, indent = 2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def from_file(cls, file_path: str, format: str):
        """
        Create an EvaluationReport instance from a file.

        Args:
            file_path: Path to the file.
            format: The format of the file to be read.

        Return:
            An EvaluationReport object.
        """
        if format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return cls.from_dict(data)

class EvaluationOutput(BaseModel):
    """
    Structured output for LLM-based judge.

    Attributes:
        score: The score of the test case.
        test_pass: Whether the test pass or fail.
        reason: The reason for the score for each test case.
    """
    score: float
    test_pass: bool
    reason: str | None = None
