#!/usr/bin/env python3
"""
ANES Metadata Parser

Parses the ANES codebook PDF and extracts variable metadata using the tableqa framework.
This script demonstrates how to:
1. Parse ANES-specific codebook format (PDF)
2. Convert to tableqa metadata schema
3. Generate research questions using LLM

Usage:
    python 01_parse_metadata.py \
        --codebook ../data/raw/anes_timeseries_cdf_codebook_var_20220916.pdf \
        --output-metadata ../data/anes_metadata.csv \
        --output-templates ../templates/question_templates.txt \
        --api-key YOUR_API_KEY
"""

import argparse
import math
import re
from pathlib import Path

import openai
import pandas as pd
from tqdm.auto import tqdm


try:
    import pdfplumber
except ImportError:
    print("pdfplumber not installed. Install with: pip install tableqa[pdf]")
    raise


# Configure logging
from statqa.utils.logging import setup_logging


logger = setup_logging(__name__, level="INFO")


def parse_anes_codebook(pdf_path: str) -> pd.DataFrame:
    """
    Parse ANES codebook PDF to extract variable metadata.

    This is ANES-specific parsing logic that extracts:
    - varname: variable code (e.g., VCF0001)
    - label: question/variable label
    - valid_values: valid value ranges/codes
    - missing_values: missing data codes
    - notes: additional descriptive text

    Args:
        pdf_path: Path to ANES codebook PDF

    Returns:
        DataFrame with columns [varname, label, valid_values, missing_values, notes]
    """
    records = []
    current = None

    # Patterns for parsing ANES codebook structure
    var_pattern = re.compile(r"^(VCF\d{3,4}[a-z]?)\b\s*(.*)")
    valid_pattern = re.compile(r"^Valid\b[:\s]*(.*)")
    missing_pattern = re.compile(r"^(Missing|INAP\.)\b[:\s]*(.*)")

    logger.info(f"Opening ANES codebook PDF: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        # Locate start of "VARIABLE DESCRIPTION" section
        start_idx = 0
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if "VARIABLE DESCRIPTION" in text:
                start_idx = i + 1
                logger.info(f"Found VARIABLE DESCRIPTION section on page {start_idx + 1}")
                break

        # Parse all pages starting from variable descriptions
        for page in tqdm(pdf.pages[start_idx:], desc="Parsing pages", unit="page"):
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check if this is a new variable entry
                m_var = var_pattern.match(line)
                if m_var:
                    # Save previous variable
                    if current:
                        records.append(current)
                    # Start new variable record
                    current = {
                        "varname": m_var.group(1),
                        "label": m_var.group(2).strip(),
                        "valid_values": "",
                        "missing_values": "",
                        "notes": "",
                    }
                elif current:
                    # Parse valid values
                    m_valid = valid_pattern.match(line)
                    if m_valid:
                        current["valid_values"] = m_valid.group(1).strip()
                        continue

                    # Parse missing values
                    m_miss = missing_pattern.match(line)
                    if m_miss:
                        current["missing_values"] += m_miss.group(2).strip() + "; "
                        continue

                    # Accumulate other text as notes
                    current["notes"] += line + " "

        # Save the last variable
        if current:
            records.append(current)

    logger.info(f"Parsed {len(records)} variables from ANES codebook")
    return pd.DataFrame(records)


def generate_research_questions(
    metadata_df: pd.DataFrame,
    api_key: str | None = None,
    chunk_size: int = 20,
    max_questions_per_chunk: int = 20,
) -> list[str]:
    """
    Generate research questions using LLM based on variable metadata.

    Uses chunking to stay within API token limits for large codebooks.

    Args:
        metadata_df: DataFrame with variable metadata
        api_key: OpenAI API key (optional if OPENAI_API_KEY env var is set)
        chunk_size: Number of variables to include per LLM prompt
        max_questions_per_chunk: Maximum questions to generate per chunk

    Returns:
        List of research question strings
    """
    if api_key:
        openai.api_key = api_key
        logger.info("Using provided OpenAI API key")
    else:
        logger.info("Using OPENAI_API_KEY environment variable")

    questions = []
    total_vars = len(metadata_df)
    num_chunks = math.ceil(total_vars / chunk_size)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_vars)
        chunk_df = metadata_df.iloc[start:end]

        # Build context string for this chunk
        context_lines = []
        for _, row in chunk_df.iterrows():
            var = row["varname"]
            label = row["label"]
            valid = row["valid_values"]
            miss = row["missing_values"]

            ctx = f"- {var} ({label})"
            if valid:
                ctx += f"; Valid: {valid}"
            if miss:
                ctx += f"; Missing: {miss}"
            context_lines.append(ctx)

        context = "\n".join(context_lines)

        # Create prompt for LLM
        prompt = (
            "You are a social science analyst working with the ANES (American National Election Studies) dataset. "
            f"Given the following variables and their value coding, generate up to {max_questions_per_chunk} "
            "insightful analysis questions covering:\n"
            "* Univariate descriptives (distribution, summary statistics)\n"
            "* Bivariate relationships (correlations, group comparisons)\n"
            "* Temporal trends (if year variable present)\n"
            "* Simple causal hypotheses\n\n"
            "Each question should reference one or more of the listed variables and respect their coding.\n\n"
            "Variables:\n" + context + "\n\n"
            "List questions as a numbered list."
        )

        logger.info(
            f"Generating questions for chunk {chunk_idx + 1}/{num_chunks} "
            f"(variables {start}-{end - 1})"
        )

        try:
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You help scientists write research questions."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
            )

            text = resp.choices[0].message.content.strip()

            # Extract numbered questions
            for line in text.split("\n"):
                m = re.match(r"^\d+\.\s*(.+)", line)
                if m:
                    questions.append(m.group(1).strip())

            logger.info(f"Chunk {chunk_idx + 1}: Total questions collected: {len(questions)}")

        except Exception as e:
            logger.error(f"LLM error in chunk {chunk_idx + 1}: {e}")

    return questions


def main():
    """Main entry point for ANES metadata parsing."""
    parser = argparse.ArgumentParser(
        description="Parse ANES codebook and generate research questions using tableqa framework"
    )
    parser.add_argument(
        "--codebook",
        required=True,
        help="Path to ANES codebook PDF (e.g., anes_timeseries_cdf_codebook_var_20220916.pdf)",
    )
    parser.add_argument(
        "--output-metadata",
        default="../data/anes_metadata.csv",
        help="Output CSV file for parsed metadata (default: ../data/anes_metadata.csv)",
    )
    parser.add_argument(
        "--output-templates",
        default="../templates/question_templates.txt",
        help="Output text file for question templates (default: ../templates/question_templates.txt)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of variables per LLM prompt (default: 20)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=20,
        help="Maximum questions to generate per chunk (default: 20)",
    )
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help="Skip question generation (only parse metadata)",
    )

    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.output_metadata).parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_questions:
        Path(args.output_templates).parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse ANES codebook
    logger.info("=" * 60)
    logger.info("Step 1: Parsing ANES codebook metadata")
    logger.info("=" * 60)

    metadata_df = parse_anes_codebook(args.codebook)
    metadata_df.to_csv(args.output_metadata, index=False)
    logger.info(f"✓ Saved metadata for {len(metadata_df)} variables to {args.output_metadata}")

    # Step 2: Generate research questions (optional)
    if not args.skip_questions:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Generating research questions with LLM")
        logger.info("=" * 60)

        questions = generate_research_questions(
            metadata_df,
            api_key=args.api_key,
            chunk_size=args.chunk_size,
            max_questions_per_chunk=args.max_questions,
        )

        with open(args.output_templates, "w") as f:
            for i, q in enumerate(questions, 1):
                f.write(f"{i}. {q}\n")

        logger.info(f"✓ Saved {len(questions)} research questions to {args.output_templates}")
    else:
        logger.info("\n✓ Skipped question generation (--skip-questions flag used)")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Metadata parsing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
