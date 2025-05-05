import argparse
import pdfplumber
import pandas as pd
import json
import openai
import re
import logging
from tqdm.auto import tqdm
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_codebook(pdf_path):
    """
    Parse ANES codebook PDF to extract variable metadata:
    - varname: variable code
    - label: question label
    - valid_values: text after 'Valid'
    - missing_values: text after 'Missing'
    - notes: all other descriptive text
    Returns a DataFrame with columns [varname, label, valid_values, missing_values, notes].
    """
    records = []
    current = None
    # Pattern to detect start of variable entry
    var_pattern = re.compile(r'^(VCF\d{3,4}[a-z]?)\b\s*(.*)')
    valid_pattern = re.compile(r'^Valid\b[:\s]*(.*)')
    missing_pattern = re.compile(r'^(Missing|INAP\.)\b[:\s]*(.*)')

    logging.info(f"Opening PDF codebook: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        # Locate start of "VARIABLE DESCRIPTION" section
        start_idx = 0
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            if 'VARIABLE DESCRIPTION' in text:
                start_idx = i + 1
                logging.info(f"Found start on page {start_idx + 1}")
                break

        for page in tqdm(pdf.pages[start_idx:], desc="Parsing pages", unit="page"):
            text = page.extract_text() or ''
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                m_var = var_pattern.match(line)
                if m_var:
                    # Save previous
                    if current:
                        records.append(current)
                    # New record
                    current = {
                        'varname': m_var.group(1),
                        'label': m_var.group(2).strip(),
                        'valid_values': '',
                        'missing_values': '',
                        'notes': ''
                    }
                elif current:
                    m_valid = valid_pattern.match(line)
                    if m_valid:
                        current['valid_values'] = m_valid.group(1).strip()
                        continue
                    m_miss = missing_pattern.match(line)
                    if m_miss:
                        current['missing_values'] += m_miss.group(2).strip() + '; '
                        continue
                    # Otherwise accumulate to notes
                    current['notes'] += line + ' '
        # Append last
        if current:
            records.append(current)

    logging.info(f"Parsed {len(records)} variables from codebook.")
    return pd.DataFrame(records)


def generate_question_templates_chunked(metadata_df, api_key=None, chunk_size=20, max_questions_per_chunk=20):
    """
    Chunk variables to stay within token limits and generate questions for each batch.
    Returns a flat list of question strings.
    """
    if api_key:
        openai.api_key = api_key
        logging.info("Using provided OpenAI API key.")
    else:
        logging.info("No API key provided; using env variable if set.")

    templates = []
    total_vars = len(metadata_df)
    num_chunks = math.ceil(total_vars / chunk_size)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_vars)
        chunk_df = metadata_df.iloc[start:end]

        # Build context string for this chunk
        context_lines = []
        for _, row in chunk_df.iterrows():
            var = row['varname']
            label = row['label']
            valid = row['valid_values']
            miss = row['missing_values']
            ctx = f"- {var} ({label})"
            if valid:
                ctx += f"; Valid: {valid}"
            if miss:
                ctx += f"; Missing: {miss}"
            context_lines.append(ctx)
        context = "\n".join(context_lines)

        prompt = (
            "You are a social science analyst working with the ANES dataset. "
            "Given the following variables and their value coding, generate up to "
            f"{max_questions_per_chunk} insightful analysis questions covering: \n"
            "* Univariate descriptives (distribution, summary stats)\n"
            "* Bivariate relationships (correlations, group comparisons)\n"
            "* Temporal trends (if year variable present)\n"
            "* Simple causal hypotheses\n"
            "Each question should reference one or more of the listed variables and respect their coding.\n"
            "Variables:\n" + context + "\n" +
            "List questions as a numbered list."
        )

        logging.info(f"Generating questions for chunk {chunk_idx+1}/{num_chunks} (vars {start}-{end-1})")
        try:
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You help scientists write research questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600
            )
            text = resp.choices[0].message.content.strip()
            for line in text.split('\n'):
                m = re.match(r'^\d+\.\s*(.+)', line)
                if m:
                    templates.append(m.group(1).strip())
            logging.info(f"Chunk {chunk_idx+1}: extracted questions count now {len(templates)}")
        except Exception as e:
            logging.error(f"LLM error in chunk {chunk_idx+1}: {e}")

    return templates


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata and generate question templates chunked for ANES variables."  )
    parser.add_argument('--codebook', required=True, help='Path to ANES codebook PDF')
    parser.add_argument('--output-metadata', default='anes_metadata.csv',
                        help='Output CSV file for parsed metadata')
    parser.add_argument('--output-templates', default='question_templates.txt',
                        help='Output text file for question templates')
    parser.add_argument('--api-key', default=None, help='OpenAI API key or set OPENAI_API_KEY')
    parser.add_argument('--chunk-size', type=int, default=20,
                        help='Number of variables per LLM prompt')
    parser.add_argument('--max-questions', type=int, default=20,
                        help='Max questions to generate per chunk')
    args = parser.parse_args()

    # Parse metadata
    logging.info("Starting metadata parsing...")
    meta_df = parse_codebook(args.codebook)
    meta_df.to_csv(args.output_metadata, index=False)
    logging.info(f"Saved metadata ({len(meta_df)} vars) to {args.output_metadata}")

    # Generate chunked templates
    logging.info("Starting chunked question generation...")
    templates = generate_question_templates_chunked(
        meta_df,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        max_questions_per_chunk=args.max_questions
    )

    with open(args.output_templates, 'w') as ft:
        for i, q in enumerate(templates, 1):
            ft.write(f"{i}. {q}\n")
    logging.info(f"Saved {len(templates)} question templates to {args.output_templates}")

if __name__ == '__main__':
    main()
