# Installation

## Basic Installation

```bash
pip install statqa
```

## With Optional Features

```bash
# Include LLM support (OpenAI/Anthropic)
pip install statqa[llm]

# Include PDF parsing
pip install statqa[pdf]

# Include statistical formats (SPSS/Stata/SAS)
pip install statqa[statistical-formats]

# Development installation
pip install statqa[dev]

# Complete installation
pip install statqa[all]
```

## From Source

```bash
git clone https://github.com/gojiplus/statqa.git
cd statqa
uv pip install -e ".[dev]"
```

## Development Environment

For development, we recommend using `uv` for faster dependency management:

```bash
# Install uv first
pip install uv

# Clone and setup
git clone https://github.com/gojiplus/statqa.git
cd statqa
uv sync --all-extras
```
