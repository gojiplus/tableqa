Quickstart Tutorial
===================

Learn the basic StatQA workflow in 5 minutes.

Installation
------------

.. code-block:: bash

   pip install statqa

Basic Usage
-----------

1. Parse a Codebook
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from statqa.metadata.parsers.csv import CSVParser

   parser = CSVParser()
   codebook = parser.parse("your_codebook.csv")

2. Run Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   from statqa.analysis.univariate import UnivariateAnalyzer
   import pandas as pd

   # Load your data
   data = pd.read_csv("your_data.csv")

   # Analyze a variable
   analyzer = UnivariateAnalyzer()
   results = analyzer.analyze(data['age'], codebook.get_variable('age'))

3. Generate Q/A Pairs
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from statqa.qa.generator import QAGenerator

   generator = QAGenerator()
   qa_pairs = generator.generate_qa_pairs(results, "Age distribution analysis")

Next Steps
----------

- See :doc:`../api_reference/index` for complete API documentation
- Check out :doc:`../examples/index` for detailed examples
- Read :doc:`workflow_examples` for common use cases
