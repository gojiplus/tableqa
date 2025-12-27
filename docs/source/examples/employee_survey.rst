Employee Survey Analysis
========================

Analysis of employee satisfaction survey data with ordinal and categorical variables.

Dataset Overview
----------------

Synthetic employee survey with 1,000 responses covering:

- **Demographics**: Age, department, tenure
- **Satisfaction Measures**: Job satisfaction, work-life balance  
- **Performance**: Ratings and promotion history
- **Attitudes**: Engagement scores and turnover intent

Key Features Demonstrated
-------------------------

- **Ordinal Analysis**: Satisfaction scales (1-5 ratings)
- **Group Comparisons**: Satisfaction by department
- **Causal Analysis**: Factors affecting turnover intent
- **Missing Data**: Realistic patterns of non-response

Code Example
------------

.. code-block:: python

   # Bivariate analysis: Job satisfaction by department
   biv_analyzer = BivariateAnalyzer()
   result = biv_analyzer.analyze(
       data,
       codebook.variables['job_satisfaction'],
       codebook.variables['department']  
   )
   
   # Results include chi-square test and effect size
   print(f"Chi-square p-value: {result['p_value']:.4f}")
   print(f"Effect size (Cramer's V): {result['effect_size']:.3f}")

Generated Insights
------------------

::

   Q: How does job satisfaction vary by department?
   A: **Job Satisfaction by Department**: Engineering shows highest satisfaction (mean=4.2), 
      while Sales shows lowest (mean=3.1). Chi-square test indicates significant differences 
      (p<0.001, Cramer's V=0.34).

Files in Example
----------------

- ``employee_survey.csv``: 1,000 employee responses
- ``survey_codebook.csv``: Variable definitions with ordinal scales
- ``analysis.py``: Multi-level analysis script  
- ``results/``: Department comparisons and satisfaction plots