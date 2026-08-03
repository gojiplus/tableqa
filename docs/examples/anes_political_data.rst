ANES Political Data Analysis
============================

Analysis of American National Election Studies (ANES) survey data demonstrating complex political science workflows.

Dataset Overview
----------------

ANES 2020 Time Series Study subset focusing on:

- **Political Attitudes**: Party identification, ideology, candidate evaluations
- **Demographics**: Age, education, income, race/ethnicity  
- **Voting Behavior**: Turnout, vote choice, political engagement
- **Policy Preferences**: Issue positions and priorities

Advanced Features
-----------------

- **Complex Survey Weights**: Proper handling of sampling design
- **Temporal Analysis**: Change over election cycle
- **Multiple Imputation**: Missing data in sensitive topics
- **Polarization Metrics**: Political attitude distributions

Code Example
------------

.. code-block:: python

   # Temporal analysis: Political polarization over time  
   from statqa.analysis.temporal import TemporalAnalyzer
   
   temporal_analyzer = TemporalAnalyzer()
   result = temporal_analyzer.analyze(
       data,
       time_var=codebook.variables['interview_date'],
       outcome_var=codebook.variables['party_id_strength'],
       trend_method='mann_kendall'
   )
   
   # Test for increasing polarization
   print(f"Trend p-value: {result['trend_pvalue']:.4f}")
   print(f"Trend direction: {result['trend_direction']}")

Political Science Insights
--------------------------

::

   Q: How has partisan strength changed during the 2020 election cycle?
   A: **Partisan Identification Strength**: Significant strengthening trend detected 
      (Mann-Kendall τ=0.23, p<0.001). Strong partisans increased from 45% to 62% 
      over the pre-election period, indicating heightened polarization.

Files in Example
----------------

- ``anes_2020_subset.csv``: 3,000 respondent subset
- ``anes_codebook.csv``: Complex variable definitions with survey weights
- ``political_analysis.py``: Full political science workflow
- ``temporal_plots/``: Trend visualizations and change point detection