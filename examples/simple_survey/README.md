# Simple Survey Example: Quick Start

A minimal example demonstrating statqa with synthetic survey data. Perfect for learning the basics without downloading large datasets.

## Quick Start

Run the complete pipeline in one command:

```bash
python quick_start.py
```

This will:
1. Generate synthetic survey data (1,000 respondents)
2. Create a codebook with variable metadata
3. Run statistical analyses (univariate and bivariate)
4. Generate natural language insights
5. Create visualizations
6. Save results to `output/`

## Example Output

```
Generated synthetic survey data: 1000 rows, 8 columns

Variables:
- age: Respondent age (18-80 years)
- gender: Gender (Male/Female)
- education: Education level (High School/Bachelor/Master/PhD)
- income: Annual income ($20k-$200k)
- satisfaction: Job satisfaction (1-5 scale)
- political_interest: Political interest (1-5 scale)
- region: Geographic region (North/South/East/West)
- year: Survey year (2020-2023)

Running analyses...
✓ Univariate: 8 variables analyzed
✓ Bivariate: 28 pairs analyzed
✓ Saved 36 insights to output/insights.json

Example insights:
- **Respondent age**: mean=49.2, median=49.0, std=17.8. N=1000
- **Income** by **Education**: PhD=$142k, Master=$98k, Bachelor=$72k, High School=$51k
- **Satisfaction** ↔ **Income**: r=0.34, p<0.001
- **Political interest** by **Region**: North=3.2, South=2.8, East=3.1, West=3.4
```

## What You'll Learn

This example demonstrates:

1. **Data preparation**: Creating codebooks and loading data
2. **Univariate analysis**: Descriptive statistics for individual variables
3. **Bivariate analysis**: Relationships between variables
4. **Visualization**: Automatic plot generation
5. **Interpretation**: Converting stats to natural language
6. **Export**: Saving results in structured formats

## Customization

### Modify the Survey

Edit `quick_start.py` to customize:

```python
# Change sample size
n = 5000  # instead of 1000

# Add new variables
data["new_variable"] = np.random.choice(["A", "B", "C"], n)

# Modify distributions
data["age"] = np.random.normal(45, 15, n).clip(18, 80)  # normal instead of uniform
```

### Run Custom Analyses

```python
from statqa.analysis.temporal import TemporalAnalyzer
from statqa.analysis.causal import CausalAnalyzer

# Analyze trends over time
temporal = TemporalAnalyzer()
trends = temporal.analyze(data, time_var="year", target_var="satisfaction")

# Estimate treatment effects
causal = CausalAnalyzer()
effect = causal.analyze(
    data, treatment="education", outcome="income", confounders=["age", "region"]
)
```

### Use Your Own Data

Replace the synthetic data generation with your own CSV:

```python
import pandas as pd
from statqa.utils.io import load_data

# Load your data
data = load_data("my_survey.csv")

# Create codebook manually or from CSV
from statqa.metadata.parsers import CSVParser

parser = CSVParser()
codebook = parser.parse("my_codebook.csv")
```

## File Structure

```
simple_survey/
├── README.md          # This file
├── quick_start.py     # Main example script
└── output/            # Generated outputs (created on first run)
    ├── survey_data.csv
    ├── codebook.json
    ├── insights.json
    └── *.png
```

## Next Steps

After running this example:

1. **Explore the code**: Open `quick_start.py` to see how it works
2. **Check the outputs**: Browse `output/` to see generated files
3. **Try ANES example**: See `../anes_example/` for a real-world dataset
4. **Read the docs**: Visit https://gojiplus.github.io/statqa
5. **Build your own**: Adapt this example to your dataset

## Comparison with ANES Example

| Feature | Simple Survey | ANES Example |
|---------|--------------|--------------|
| Data size | 1K rows, 8 vars | 70K rows, 1K vars |
| Setup time | < 1 minute | ~10 minutes (download) |
| Run time | < 30 seconds | ~5-10 minutes |
| Complexity | Beginner | Advanced |
| Use case | Learning | Real research |

## Troubleshooting

**Problem**: Import errors
**Solution**: Install statqa: `pip install statqa`

**Problem**: No output directory
**Solution**: Directory is created automatically; check file permissions

**Problem**: Empty plots
**Solution**: Ensure matplotlib backend is working; try `export MPLBACKEND=Agg`

## Support

For questions or issues:
- **statqa package**: https://github.com/gojiplus/statqa/issues
- **Documentation**: https://gojiplus.github.io/statqa
