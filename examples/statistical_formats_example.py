"""
Statistical formats usage example.

This example demonstrates how to parse SPSS, Stata, and SAS files
and extract metadata for analysis.
"""

import tempfile
from pathlib import Path

import pandas as pd

from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.parsers import StatisticalFormatParser

# Optional: only run if pyreadstat is available
try:
    import pyreadstat
except ImportError:
    print(
        "This example requires pyreadstat. Install with: pip install statqa[statistical-formats]"
    )
    exit(1)


def create_sample_spss_file():
    """Create a sample SPSS file for demonstration."""
    # Sample survey data
    data = {
        "respondent_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [25, 34, 45, 23, 56, 67, 29, 33, 41, 52],
        "gender": [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        "education": [3, 4, 2, 4, 1, 2, 3, 4, 3, 2],
        "satisfaction": [4, 5, 3, 2, 5, 4, 3, 4, 2, 5],
        "income": [
            45000,
            65000,
            55000,
            35000,
            85000,
            95000,
            42000,
            58000,
            48000,
            72000,
        ],
    }

    df = pd.DataFrame(data)

    # Rich metadata for the SPSS file
    variable_value_labels = {
        "gender": {1: "Male", 2: "Female"},
        "education": {
            1: "High School or Less",
            2: "Some College",
            3: "Bachelor's Degree",
            4: "Graduate Degree",
        },
        "satisfaction": {
            1: "Very Dissatisfied",
            2: "Dissatisfied",
            3: "Neutral",
            4: "Satisfied",
            5: "Very Satisfied",
        },
    }

    column_labels = {
        "respondent_id": "Respondent ID",
        "age": "Age in Years",
        "gender": "Gender Identity",
        "education": "Highest Education Level",
        "satisfaction": "Job Satisfaction Level",
        "income": "Annual Household Income (USD)",
    }

    # Create temporary SPSS file
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as temp_file:
        temp_path = temp_file.name

    pyreadstat.write_sav(
        df,
        temp_path,
        variable_value_labels=variable_value_labels,
        column_labels=column_labels,
        file_label="Sample Employee Satisfaction Survey",
    )

    return Path(temp_path)


def main():
    """Main demonstration function."""
    print("🎯 StatQA Statistical Formats Example")
    print("=" * 50)

    # 1. Create sample SPSS file
    print("\n📊 Creating sample SPSS file...")
    spss_file = create_sample_spss_file()
    print(f"✓ Created: {spss_file}")

    try:
        # 2. Parse with StatisticalFormatParser
        print("\n📋 Parsing SPSS metadata...")
        parser = StatisticalFormatParser()

        # Validate file
        if parser.validate(spss_file):
            print("✓ File validation passed")
        else:
            print("❌ File validation failed")
            return

        # Parse the file
        codebook = parser.parse(spss_file)
        print(f"✓ Parsed codebook: {codebook.name}")
        print(f"  Variables: {len(codebook.variables)}")
        print(f"  Dataset info: {len(codebook.dataset_info)} metadata fields")

        # 3. Explore extracted metadata
        print("\n📝 Variable Metadata:")
        print("-" * 30)

        for var_name, variable in codebook.variables.items():
            print(f"\n{var_name.upper()}")
            print(f"  Label: {variable.label}")
            print(f"  Type: {variable.var_type}")

            if variable.valid_values:
                print("  Values:")
                for code, label in variable.valid_values.items():
                    print(f"    {code}: {label}")

        # 4. Show dataset-level metadata
        print("\n📊 Dataset Information:")
        print("-" * 30)
        print(f"Rows: {codebook.dataset_info.get('number_rows')}")
        print(f"Columns: {codebook.dataset_info.get('number_columns')}")
        print(f"Encoding: {codebook.dataset_info.get('file_encoding')}")
        print(f"Created: {codebook.dataset_info.get('creation_time')}")

        # 5. Demonstrate integration with analysis pipeline
        print("\n🔬 Running Analysis Pipeline:")
        print("-" * 30)

        # Load the actual data for analysis
        df, _metadata = pyreadstat.read_sav(spss_file)

        # Analyze a variable
        analyzer = UnivariateAnalyzer()
        formatter = InsightFormatter()

        # Analyze age variable
        age_results = analyzer.analyze(df["age"], codebook.variables["age"])
        age_insight = formatter.format_univariate(age_results)

        print(f"Age Analysis: {age_insight}")

        # Analyze satisfaction variable
        satisfaction_results = analyzer.analyze(
            df["satisfaction"], codebook.variables["satisfaction"]
        )
        satisfaction_insight = formatter.format_univariate(satisfaction_results)

        print(f"Satisfaction Analysis: {satisfaction_insight}")

        # 6. Show CLI equivalent
        print("\n⚡ CLI Equivalent:")
        print("-" * 30)
        print(f"statqa parse-codebook {spss_file.name} --output codebook.json")
        print(f"statqa analyze {spss_file.name} codebook.json --output-dir results/")

        print("\n✅ Example completed successfully!")

    finally:
        # Cleanup
        spss_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
