Provenance Tracking
===================

StatQA provides comprehensive provenance metadata for all generated content, ensuring full reproducibility and traceability.

Metadata Structure
------------------

Every Q/A pair includes detailed provenance information:

.. code-block:: python

   {
     "provenance": {
       "generated_at": "2025-11-19T19:21:28+00:00",
       "tool": "statqa", 
       "tool_version": "0.2.0",
       "generation_method": "template",  # or "llm_paraphrase"
       "analysis_type": "univariate",    # univariate/bivariate/temporal/causal
       "analyzer": "UnivariateAnalyzer",
       "variables": ["age"],
       "statistical_tests": ["shapiro_wilk", "jarque_bera"],
       "python_commands": [
         "data['age'].mean()  # Result: 42.5",
         "data['age'].std()   # Result: 12.3"
       ],
       "llm_model": "gpt-4-turbo",       # if LLM was used
       "template_id": "distribution_summary"  # if template-based
     }
   }

Benefits of Provenance Tracking
--------------------------------

1. **Reproducibility**: Exact commands and parameters used
2. **Quality Control**: Track generation methods and models
3. **Audit Trails**: Full history of analysis decisions
4. **Version Management**: Tool and model versions recorded
5. **Research Integrity**: Transparent methodology documentation