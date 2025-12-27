# Visualization

Create publication-ready plots from your statistical analyses.

## Overview

StatQA's visualization system automatically generates appropriate plots based on variable types and analysis results, with customizable themes and export options.

## Basic Plotting

Generate plots from analysis results:

```python
from statqa.visualization.plots import PlotFactory

factory = PlotFactory(theme="publication", figsize=(10, 6))

# Univariate plots
fig = factory.plot_univariate(data, variable_metadata, result)

# Bivariate plots
fig = factory.plot_bivariate(data, var1, var2, result)

# Temporal plots
fig = factory.plot_temporal(data, time_var, value_var, result)
```

## CLI Usage

```bash
# Generate plots during analysis
statqa analyze data.csv codebook.json --output-dir results/ --plots

# Complete pipeline with visualizations
statqa pipeline data.csv codebook.csv --output-dir output/ --plots
```

## Plot Types

### Univariate Plots

Automatic plot selection based on variable type:

#### Numeric Variables
- **Histograms**: Distribution visualization
- **Box plots**: Outlier detection and quartiles
- **Q-Q plots**: Normality assessment

```python
# Numeric variable plotting
fig = factory.plot_univariate(data['age'], age_variable, analysis_result)
```

#### Categorical Variables
- **Bar charts**: Frequency distributions
- **Pie charts**: Proportions (when appropriate)

```python
# Categorical variable plotting
fig = factory.plot_univariate(data['gender'], gender_variable, analysis_result)
```

### Bivariate Plots

#### Numeric × Numeric
- **Scatter plots**: With regression lines
- **Correlation heatmaps**: For multiple variables

```python
fig = factory.plot_bivariate(data, age_var, income_var, correlation_result)
```

#### Categorical × Numeric
- **Box plots**: Group comparisons
- **Violin plots**: Distribution shapes by group

```python
fig = factory.plot_bivariate(data, gender_var, salary_var, anova_result)
```

#### Categorical × Categorical
- **Stacked bar charts**: Association patterns
- **Heatmaps**: Cross-tabulation visualization

```python
fig = factory.plot_bivariate(data, education_var, income_bracket_var, chi_square_result)
```

### Temporal Plots

#### Time Series
- **Line plots**: Trends over time
- **Trend lines**: With confidence intervals
- **Change point markers**: Structural breaks

```python
fig = factory.plot_temporal(data, year_var, gdp_var, trend_result)
```

#### Grouped Time Series
- **Multi-line plots**: Trends by group
- **Faceted plots**: Separate panels per group

```python
fig = factory.plot_temporal(data, year_var, gdp_var, grouped_result, group_var)
```

## Themes

### Built-in Themes

```python
# Publication theme (default)
factory = PlotFactory(theme="publication")

# Minimal theme
factory = PlotFactory(theme="minimal")

# Dark theme
factory = PlotFactory(theme="dark")

# Custom theme
custom_theme = {
    "figure.figsize": (12, 8),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "serif"
}
factory = PlotFactory(theme=custom_theme)
```

### Color Palettes

```python
# Colorblind-friendly palette (default)
factory = PlotFactory(palette="colorblind")

# Sequential palette for continuous data
factory = PlotFactory(palette="viridis")

# Qualitative palette for categorical data
factory = PlotFactory(palette="Set2")
```

## Customization

### Plot Options

```python
fig = factory.plot_univariate(
    data['age'],
    age_var,
    result,
    title="Age Distribution",
    xlabel="Age (years)",
    show_stats=True,     # Display statistics on plot
    show_outliers=True,  # Highlight outliers
    alpha=0.7           # Transparency
)
```

### Export Options

```python
# Save as PNG
factory.save_plot(fig, "age_distribution.png", dpi=300)

# Save as PDF (vector graphics)
factory.save_plot(fig, "age_distribution.pdf")

# Save as SVG
factory.save_plot(fig, "age_distribution.svg")
```

## Batch Plotting

Generate plots for all analyses:

```python
# Plot all univariate analyses
for variable, result in univariate_results.items():
    fig = factory.plot_univariate(data[variable], variables[variable], result)
    factory.save_plot(fig, f"univariate_{variable}.png")

# Plot all bivariate relationships
for pair, result in bivariate_results.items():
    var1, var2 = pair.split('_vs_')
    fig = factory.plot_bivariate(data, variables[var1], variables[var2], result)
    factory.save_plot(fig, f"bivariate_{var1}_{var2}.png")
```

## Interactive Plots

For exploration and presentations:

```python
# Enable interactive mode
factory = PlotFactory(interactive=True)

fig = factory.plot_bivariate(data, var1, var2, result)
fig.show()  # Opens in browser with zoom/pan capabilities
```

The visualization system ensures all plots are publication-ready with proper labels, legends, and statistical annotations.
