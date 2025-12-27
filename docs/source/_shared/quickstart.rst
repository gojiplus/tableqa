Quick Start
===========

1. Create a Codebook
--------------------

.. code-block:: python

   from statqa.metadata.parsers import TextParser

   codebook_text = """
   # Variable: age
   Label: Respondent Age
   Type: numeric_continuous
   Units: years
   Range: 18-99
   Missing: -1, 999

   # Variable: satisfaction
   Label: Job Satisfaction
   Type: categorical_ordinal
   Values:
     1: Very Dissatisfied
     2: Dissatisfied
     3: Neutral
     4: Satisfied
     5: Very Satisfied
   """

   parser = TextParser()
   codebook = parser.parse(codebook_text)

2. Run Statistical Analyses
---------------------------

.. code-block:: python

   import pandas as pd
   from statqa.analysis import UnivariateAnalyzer, BivariateAnalyzer

   # Load your data
   data = pd.read_csv("survey_data.csv")

   # Univariate analysis
   analyzer = UnivariateAnalyzer()
   result = analyzer.analyze(data["age"], codebook.variables["age"])

   print(result)
   # Output: {'mean': 42.5, 'median': 41.0, 'std': 12.3, ...}

   # Bivariate analysis
   biv_analyzer = BivariateAnalyzer()
   result = biv_analyzer.analyze(
       data,
       codebook.variables["age"],
       codebook.variables["satisfaction"]
   )

3. Generate Natural Language Insights
-------------------------------------

.. code-block:: python

   from statqa.interpretation import InsightFormatter

   formatter = InsightFormatter()
   insight = formatter.format_univariate(result)

   print(insight)
   # Output: "**Respondent Age**: mean=42.5, median=41.0, std=12.3, range=[18, 95]. N=1,000 [2.3% outliers]."

4. Create Multimodal Q/A Pairs for LLM Training
-----------------------------------------------

.. code-block:: python

   from statqa.qa import QAGenerator
   from statqa.visualization import PlotFactory

   qa_gen = QAGenerator(use_llm=False)  # Template-based

   # Generate Q/A pairs with visual metadata
   plot_data = {
       "data": data,
       "variables": codebook.variables,
       "output_path": "plots/univariate_age.png"
   }
   visual_metadata = qa_gen.generate_visual_metadata(result, variables=["age"], plot_data=plot_data)
   qa_pairs = qa_gen.generate_qa_pairs(result, insight, variables=["age"], visual_data=visual_metadata)

   for qa in qa_pairs:
       print(f"Q: {qa['question']}")
       print(f"A: {qa['answer']}")
       print(f"Plot: {qa['visual']['primary_plot']}")
       print(f"Caption: {qa['visual']['caption']}")
       print(f"Provenance: {qa['provenance']}\n")
