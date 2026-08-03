Multimodal Q/A Generation
==========================

StatQA creates **CLIP-style multimodal databases** where each statistical question is paired with both textual answers AND rich visual metadata. This enables training of multimodal AI systems that understand both statistical text and visual representations.

Enhanced Q/A Format
--------------------

Each Q/A pair includes comprehensive metadata:

.. code-block:: json

   {
     "question": "What is the distribution of Sepal Length?",
     "answer": "**Sepal Length**: mean=5.84, median=5.80, std=0.83, range=[4.30, 7.90]. N=150 [non-normal distribution].",
     "type": "distributional",
     "provenance": {
       "generated_at": "2025-11-19T19:21:28+00:00",
       "tool": "statqa",
       "tool_version": "0.2.0",
       "generation_method": "template",
       "analysis_type": "univariate",
       "variables": ["sepal_length"],
       "python_commands": ["valid_data.mean()  # Result: 5.84"]
     },
     "visual": {
       "plot_type": "histogram",
       "caption": "Histogram showing sepal length distribution with mean=5.84 and std=0.83 (N=150).",
       "alt_text": "Histogram chart with sepal length values on x-axis and frequency density on y-axis.",
       "visual_elements": {
         "chart_type": "histogram",
         "x_axis": "Sepal Length",
         "y_axis": "Density",
         "key_features": ["distribution shape", "mean line"],
         "colors": ["blue bars", "red mean line"],
         "annotations": ["Mean: 5.84"]
       },
       "primary_plot": "/path/to/univariate_sepal_length.png",
       "generation_code": "plot_factory.plot_univariate(data['sepal_length'], sepal_length_var, 'plot.png')"
     }
   }

Question-Plot Association Mapping
----------------------------------

StatQA automatically associates relevant visualizations with each statistical insight:

- **Distribution questions** → Histograms for numeric data, bar charts for categorical
- **Correlation questions** → Scatter plots with regression lines
- **Group comparison questions** → Box plots showing group differences
- **Categorical relationships** → Heatmaps with frequency counts

Accessibility & Multimodal Features
------------------------------------

Every visualization includes:

- **Descriptive captions** with statistical context and interpretation
- **Alt-text** for screen readers and accessibility compliance
- **Visual elements extraction** for computer vision training (colors, features, annotations)
- **Reproducible generation code** for programmatic recreation