"""Tests for computational provenance tracking."""

import pandas as pd

from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.schema import Variable, VariableType
from statqa.qa.generator import QAGenerator


class TestComputationalProvenance:
    """Test computational provenance tracking functionality."""

    def test_numeric_computation_log(self):
        """Test computation log for numeric variables."""
        # Create test data
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="test_var")

        variable = Variable(
            name="test_var",
            label="Test Variable",
            var_type=VariableType.NUMERIC_CONTINUOUS,
        )

        # Run analysis
        analyzer = UnivariateAnalyzer()
        result = analyzer.analyze(data, variable)

        # Check that computation_log exists
        assert "computation_log" in result
        assert isinstance(result["computation_log"], list)
        assert len(result["computation_log"]) > 0

        # Check specific computations are logged
        log_str = " ".join(result["computation_log"])
        assert "valid_data = data.dropna()" in log_str
        assert "valid_data.mean()" in log_str
        assert "valid_data.median()" in log_str
        assert "valid_data.std()" in log_str
        assert "scipy.stats.skew" in log_str
        assert "scipy.stats.shapiro" in log_str or "scipy.stats.anderson" in log_str

    def test_categorical_computation_log(self):
        """Test computation log for categorical variables."""
        # Create categorical test data
        data = pd.Series([1, 2, 1, 3, 2, 1], name="category_var")

        variable = Variable(
            name="category_var",
            label="Category Variable",
            var_type=VariableType.CATEGORICAL_NOMINAL,
            valid_values={1: "Option A", 2: "Option B", 3: "Option C"},
        )

        # Run analysis
        analyzer = UnivariateAnalyzer()
        result = analyzer.analyze(data, variable)

        # Check computation log
        assert "computation_log" in result
        assert isinstance(result["computation_log"], list)

        # Check specific categorical computations are logged
        log_str = " ".join(result["computation_log"])
        assert "valid_data = data.dropna()" in log_str
        assert "value_counts()" in log_str
        assert "idxmax()" in log_str
        assert "np.sum" in log_str  # For entropy/diversity calculations

    def test_qa_provenance_includes_python_commands(self):
        """Test that Q/A provenance includes computational commands."""
        # Create test data
        data = pd.Series([4.3, 5.8, 6.1, 5.2, 7.9], name="sepal_length")

        variable = Variable(
            name="sepal_length",
            label="Sepal Length",
            var_type=VariableType.NUMERIC_CONTINUOUS,
        )

        # Run full pipeline
        analyzer = UnivariateAnalyzer()
        result = analyzer.analyze(data, variable)

        formatter = InsightFormatter()
        insight = formatter.format_univariate(result)

        qa_gen = QAGenerator()
        qa_pairs = qa_gen.generate_qa_pairs(result, insight, ["sepal_length"])

        # Check Q/A pairs have enhanced provenance
        assert len(qa_pairs) > 0
        qa = qa_pairs[0]

        assert "provenance" in qa
        prov = qa["provenance"]

        # Check python_commands field exists
        assert "python_commands" in prov
        assert isinstance(prov["python_commands"], list)
        assert len(prov["python_commands"]) > 0

        # Check some expected commands are present
        commands_str = " ".join(prov["python_commands"])
        assert "valid_data.mean()" in commands_str
        assert "valid_data.median()" in commands_str

    def test_computation_results_in_log(self):
        """Test that computation results are included in log comments."""
        data = pd.Series([10.0, 20.0, 30.0], name="test")

        variable = Variable(name="test", label="Test", var_type=VariableType.NUMERIC_CONTINUOUS)

        analyzer = UnivariateAnalyzer()
        result = analyzer.analyze(data, variable)

        # Check that results are included as comments
        log_str = " ".join(result["computation_log"])

        # Mean should be 20.0
        assert "Result: 20.0" in log_str
        # Should have other result comments
        assert "Result:" in log_str

    def test_normality_test_tracking(self):
        """Test that normality tests are properly tracked."""
        # Create data that will trigger Shapiro-Wilk test (< 5000 samples)
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="test")

        variable = Variable(name="test", label="Test", var_type=VariableType.NUMERIC_CONTINUOUS)

        analyzer = UnivariateAnalyzer()
        result = analyzer.analyze(data, variable)

        log_str = " ".join(result["computation_log"])
        # Should have normality test
        assert "scipy.stats.shapiro" in log_str or "scipy.stats.anderson" in log_str

    def test_provenance_format_consistency(self):
        """Test that provenance format is consistent with documentation."""
        data = pd.Series([1.0, 2.0, 3.0], name="test")

        variable = Variable(
            name="test", label="Test Variable", var_type=VariableType.NUMERIC_CONTINUOUS
        )

        analyzer = UnivariateAnalyzer()
        result = analyzer.analyze(data, variable)

        formatter = InsightFormatter()
        insight = formatter.format_univariate(result)

        qa_gen = QAGenerator()
        qa_pairs = qa_gen.generate_qa_pairs(result, insight, ["test"])

        qa = qa_pairs[0]
        prov = qa["provenance"]

        # Check required provenance fields
        required_fields = [
            "generated_at",
            "tool",
            "tool_version",
            "generation_method",
            "variables",
            "python_commands",
        ]

        for field in required_fields:
            assert field in prov, f"Required field '{field}' missing from provenance"

        # Check python_commands is properly formatted
        assert isinstance(prov["python_commands"], list)
        for cmd in prov["python_commands"]:
            assert isinstance(cmd, str)
            # Commands should either be function calls or assignments
            assert any(
                pattern in cmd for pattern in ["=", "(", "Result:"]
            ), f"Invalid command format: {cmd}"
