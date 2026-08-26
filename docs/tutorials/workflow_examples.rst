Workflow Examples
=================

Common analysis workflows and patterns.

Complete Analysis Pipeline
--------------------------

Survey Data Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from statqa.cli.main import app

   # Complete pipeline via CLI. `app` is the Typer application the `statqa`
   # entry point exposes; there is no `main()`.
   app(
       [
           'pipeline',
           'survey_data.csv',
           'survey_codebook.csv',
           '--output-dir', 'results/',
           '--enrich', '--qa',
       ],
       standalone_mode=False,
   )

Custom Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from statqa.metadata.parsers.csv import CSVParser
   from statqa.analysis.univariate import UnivariateAnalyzer
   from statqa.analysis.bivariate import BivariateAnalyzer
   from statqa.qa.generator import QAGenerator
   import pandas as pd

   # 1. Parse metadata
   parser = CSVParser()
   codebook = parser.parse("codebook.csv")

   # 2. Load data
   data = pd.read_csv("data.csv")

   # 3. Run analyses
   uni_analyzer = UnivariateAnalyzer()
   bi_analyzer = BivariateAnalyzer()

   results = []
   for var_name, variable in codebook.variables.items():
       if var_name in data.columns:
           # Univariate analysis
           result = uni_analyzer.analyze(data[var_name], variable)
           results.append(result)

           # Bivariate with other numeric variables
           for other_name, other_var in codebook.variables.items():
               if other_name != var_name and other_var.is_numeric():
                   bi_result = bi_analyzer.analyze(data, variable, other_var)
                   results.append(bi_result)

   # 4. Generate Q/A pairs
   generator = QAGenerator()
   all_qa = []
   for result in results:
       qa_pairs = generator.generate_qa_pairs(result, "Statistical analysis")
       all_qa.extend(qa_pairs)

   print(f"Generated {len(all_qa)} Q/A pairs")

For More Examples
-----------------

See the :doc:`../examples/index` section for real-world dataset examples.
