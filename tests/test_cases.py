import pytest

from src.strands_evaluation.case import Case


class TestCase:
    def test_create_simple_case(self):
        """Test creating a basic Case with only input"""
        case = Case(input="What is 2+2?")
        assert case.input == "What is 2+2?"
        assert case.name is None
        assert case.expected_output is None
        assert case.expected_trajectory is None
        assert case.expected_interactions is None
        assert case.metadata == {}

    def test_create_full_case(self):
        """Test creating a Case with all fields"""
        case = Case[str, str](
            name="Math Test",
            input="What is 2+2?",
            expected_output="4",
            expected_trajectory=["calculator"],
            expected_interactions=[{"node_name": "math_agent", "message": "2x2 is 4.", "dependencies": []}],
            metadata={"category": "math", "difficulty": "easy"},
        )

        assert case.name == "Math Test"
        assert case.input == "What is 2+2?"
        assert case.expected_output == "4"
        assert case.expected_trajectory == ["calculator"]
        assert case.expected_interactions == [{"node_name": "math_agent", "message": "2x2 is 4.", "dependencies": []}]
        assert case.metadata == {"category": "math", "difficulty": "easy"}

    def test_case_with_different_types(self):
        """Test Case with different input/output types"""
        case = Case[int, float](input=42, expected_output=42.0)

        assert case.input == 42
        assert case.expected_output == 42.0

    def test_case_with_interactions(self):
        """Test Case with expected_interactions"""
        interactions = [
            {"agent": "planner", "message": "plan", "dependencies": []},
            {"agent": "executor", "message": "execute", "dependencies": []},
        ]
        case = Case[str, str](input="Complex task", expected_interactions=interactions)

        assert case.expected_interactions == interactions

    def test_case_required_input(self):
        """Test that input is required"""
        with pytest.raises(ValueError):
            Case()
