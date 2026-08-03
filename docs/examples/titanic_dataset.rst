Titanic Dataset Analysis
========================

Survival analysis of Titanic passenger data with categorical and survival outcomes.

Dataset Overview
----------------

Historic Titanic passenger manifest with survival outcomes:

- **Demographics**: Age, sex, passenger class
- **Ticket Information**: Fare, cabin, embarkation port
- **Family**: Number of siblings/spouses, parents/children
- **Outcome**: Survival (binary)

Analysis Focus
--------------

- **Survival Rates**: By passenger class, sex, age groups
- **Effect Sizes**: Quantifying factors affecting survival
- **Missing Data**: Age and cabin information patterns
- **Causal Inference**: Controlling for confounding factors

Code Example
------------

.. code-block:: python

   # Causal analysis: Effect of passenger class on survival
   from statqa.analysis.causal import CausalAnalyzer
   
   causal_analyzer = CausalAnalyzer()
   result = causal_analyzer.analyze(
       data,
       treatment_var=codebook.variables['pclass'],
       outcome_var=codebook.variables['survived'],
       confounders=['age', 'sex']
   )
   
   # Adjusted effect with confidence intervals
   print(f"Adjusted OR: {result['adjusted_effect']:.2f}")
   print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")

Key Findings
------------

::

   Q: How did passenger class affect survival rates?
   A: **Survival by Passenger Class**: First class had 62% survival vs 24% for third class. 
      After adjusting for age and sex, first class passengers had 3.2x higher odds of survival 
      (95% CI: 2.1-4.8, p<0.001).

Files in Example
----------------

- ``titanic.csv``: 891 passenger records
- ``titanic_codebook.txt``: Variable metadata with survival coding
- ``survival_analysis.py``: Causal inference workflow
- ``plots/``: Survival curves and demographic breakdowns