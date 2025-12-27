Iris Dataset Analysis
=====================

Complete analysis of the classic Iris dataset demonstrating StatQA's capabilities.

Dataset Overview
----------------

The Iris dataset contains measurements of 150 iris flowers across 3 species:

- **Sepal Length/Width**: Continuous measurements in cm
- **Petal Length/Width**: Continuous measurements in cm  
- **Species**: Categorical (setosa, versicolor, virginica)

Code Example
------------

.. code-block:: python

   from statqa.metadata.parsers.text import TextParser
   from statqa.analysis.univariate import UnivariateAnalyzer
   from statqa.analysis.bivariate import BivariateAnalyzer
   from statqa.qa.generator import QAGenerator
   import pandas as pd

   # Load the Iris dataset
   data = pd.read_csv("examples/iris/iris.csv")
   
   # Parse codebook
   parser = TextParser()
   with open("examples/iris/codebook.txt") as f:
       codebook = parser.parse(f.read())

   # Analyze sepal length distribution
   analyzer = UnivariateAnalyzer()
   result = analyzer.analyze(
       data['sepal_length'], 
       codebook.variables['sepal_length']
   )
   
   # Generate Q/A pairs
   qa_gen = QAGenerator()
   qa_pairs = qa_gen.generate_qa_pairs(result, "Iris sepal length analysis")
   
   for qa in qa_pairs[:3]:
       print(f"Q: {qa['question']}")
       print(f"A: {qa['answer']}\n")

Expected Output
---------------

::

   Q: What is the distribution of Sepal Length?
   A: **Sepal Length**: mean=5.84, median=5.80, std=0.83, range=[4.30, 7.90]. N=150 [non-normal distribution].

   Q: What are the key statistics for Sepal Length?
   A: Sepal Length has a mean of 5.84 cm with standard deviation 0.83 cm. The distribution spans from 4.30 to 7.90 cm across 150 observations.

   Q: How is Sepal Length distributed?
   A: Sepal Length shows a slightly right-skewed distribution with most values between 5.0-6.5 cm. Normality tests indicate departure from normal distribution.

Files in Example
----------------

- ``iris.csv``: Dataset with 150 observations
- ``codebook.txt``: Variable metadata definitions  
- ``run_analysis.py``: Complete analysis script
- ``results/``: Generated outputs and visualizations