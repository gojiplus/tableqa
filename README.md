# tableqa

<tableqa> is a Python framework for **automatically extracting structured facts** (descriptive, comparative, temporal, and causal) from large tabular datasets. It converts raw columns and values into clear, human-readable statements and Q/A pairs, enabling rapid knowledge discovery, RAG corpus construction, and downstream training of language models.

## 🎯 Objective

* **Automate fact extraction** from rectangular (tabular) data without manual templating.
* Support **descriptive** (univariate/bivariate), **comparative**, **temporal**, and **causal** analyses.
* Output carefully worded facts suitable for reporting, dashboards, or fine‑tuning LLMs.

## 🛠️ Problem Statement

Given a tabular dataset, current LLM‑driven approaches include:

1. **Row embeddings** – embedding each row as a monolithic chunk (inefficient and loses structure).
2. **NL→SQL** – generating queries from natural language (requires schema familiarity, starts from zero knowledge).
3. **Existing RAG** – indexing raw rows/columns without domain insight.

**Goal:** introduce a new pipeline that injects structured analyses and domain‑aware facts into a Retrieval‑Augmented Generation system, enabling direct, high‑quality querying of tabular data.

## 🧩 Approach

We break the pipeline into three clear phases:

1. **Metadata Understanding**

   * Parse the codebook or schema to map variable codes to labels and capture type, missingness, and experimental notes.
   * Use prior domain knowledge or an LLM to verify and enrich metadata interpretations.

2. **Insight Extraction**

   * **Descriptive**: run univariate (mean, median, mode), bivariate (correlation, group means), and temporal trend analyses; handle categorical, numeric, missing, and experimental variables appropriately.
   * **Causal Proxies**: fit regression models with optional controls to surface associations in causal language.
   * **Interpretation**: transform numbers into careful natural‑language statements with statistical context and source details.
   * **Visualization Prep**: prepare charts or tables for each insight, so that in a RAG architecture, the system can retrieve and render plots on demand.

3. **Q/A Generation**

   * Wrap each fact into multiple question/answer pairs, using templates and LLM paraphrasing, creating a rich QA database for LLM fine‑tuning or retrieval.

## 📊 ANES Application Example

1. **Metadata Loading**: Parse a codebook (e.g., ANES PDF) to map variable codes (e.g., `VCF0101`) to human-friendly labels (e.g., "Respondent Age").
2. **Descriptive Analysis**:

   * **Univariate**: Means, medians, standard deviations for numeric columns; frequency breakdowns for categoricals.
   * **Bivariate**: Pairwise correlations (numeric × numeric); group mean comparisons (categorical × numeric).
3. **Temporal Trends**: Detect changes in averages or proportions across a year (or time) variable.
4. **Causal Proxies**: Fit simple regression models (with optional controls) to surface associations in causal language.
5. **Fact Formatting**: Convert statistical outputs into **clear natural-language statements**.
6. **Question Suggestion**: Leverage an LLM, using variable metadata from the codebook, to brainstorm insightful and domain-relevant questions (e.g., "How does political ideology influence trust in government across age cohorts?") that guide the fact extraction pipeline.

## 📊 ANES Application Example

* **Dataset**: American National Election Studies (ANES) Time Series (1948–2020).
* **Codebook**: `anes_timeseries_cdf_codebook_var_20220916.pdf` maps 1,030 variables to labels.
* **Script**: Loads `anes_data.csv`, builds `var_map`, and runs the four extraction modules.

### Example Facts Extracted

* *On average, Respondent Age is 48.23 (median 47.00, SD 14.52).*
* *52.7% of respondents have Gender = "Female".*
* *Respondents with Education = "College degree" have higher average Political Interest (4.20) than those with "High school" (3.10).*
* *From 1948 to 2020, average Trust in Government changed by 1.85 points (from 2.45 to 4.30).*
* *Controlling for Age and Gender, a one-unit increase in Political Ideology is associated with a 0.75-unit change in Support for Gun Control.*

## ⚙️ Requirements

* Python 3.8+
* `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `pdfplumber`

## 🚀 Usage

```bash
git clone https://github.com/your-org/tableqa.git
cd tableqa
pip install -r requirements.txt
python extract_facts.py --data anes_data.csv --codebook anes_codebook.pdf --output anes_facts.txt
```

## 🤝 Contributing

1. Fork the repo and create a feature branch.
2. Write tests for new extraction methods.
3. Submit a pull request with detailed descriptions.

---

*This project was demonstrated on the ANES Time Series dataset. Adaptable to any rectangular data source.*
