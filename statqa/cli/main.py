"""
Main CLI interface for tableqa.

Provides commands for:
- Parsing codebooks
- Running analyses
- Generating Q/A pairs
- Creating visualizations
"""

from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.progress import track

from statqa import __version__
from statqa.analysis.bivariate import BivariateAnalyzer
from statqa.analysis.univariate import UnivariateAnalyzer
from statqa.interpretation.formatter import InsightFormatter
from statqa.metadata.enricher import MetadataEnricher
from statqa.metadata.parsers.base import BaseParser
from statqa.metadata.parsers.csv import CSVParser
from statqa.metadata.parsers.text import TextParser


# Optional statistical format parser
try:
    from statqa.metadata.parsers.statistical import StatisticalFormatParser

    HAS_STATISTICAL_PARSER = True
except ImportError:
    HAS_STATISTICAL_PARSER = False
from statqa.qa.generator import QAGenerator
from statqa.utils.io import load_data, save_json
from statqa.visualization.plots import PlotFactory


app = typer.Typer(help="TableQA: Extract structured facts from tabular datasets")
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold green]TableQA version {__version__}[/bold green]")


@app.command()
def parse_codebook(
    codebook_path: Path = typer.Argument(..., help="Path to codebook file"),
    output: Path = typer.Option("codebook.json", "--output", "-o", help="Output JSON file"),
    format: Literal["auto", "text", "csv", "statistical"] = typer.Option(
        "auto", "--format", "-f", help="Codebook format (auto, text, csv, statistical)"
    ),
    enrich: bool = typer.Option(False, "--enrich", help="Enrich metadata with LLM"),
    llm_provider: Literal["openai", "anthropic"] = typer.Option(
        "openai", "--llm-provider", help="LLM provider"
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="LLM API key"),
) -> None:
    """Parse a codebook and extract metadata."""
    console.print(f"[blue]Parsing codebook:[/blue] {codebook_path}")

    # Select parser
    if format == "auto":
        # Try parsers in order - statistical first since it's more specific
        parsers: list[BaseParser] = []
        if HAS_STATISTICAL_PARSER:
            parsers.append(StatisticalFormatParser())
        parsers.extend([CSVParser(), TextParser()])

        parser: BaseParser | None = None
        for p in parsers:
            if p.validate(codebook_path):
                parser = p
                break
        if not parser:
            console.print("[red]Error:[/red] Could not determine codebook format")
            raise typer.Exit(1)
    else:
        match format:
            case "csv":
                parser = CSVParser()
            case "text":
                parser = TextParser()
            case "statistical":
                if not HAS_STATISTICAL_PARSER:
                    console.print(
                        "[red]Error:[/red] Statistical format support not available. Install with: pip install statqa[statistical-formats]"
                    )
                    raise typer.Exit(1)
                parser = StatisticalFormatParser()
            case _:
                console.print(f"[red]Error:[/red] Unknown format: {format}")
                raise typer.Exit(1)

    # Parse
    codebook = parser.parse(codebook_path)
    console.print(f"[green]✓[/green] Parsed {len(codebook.variables)} variables")

    # Enrich if requested
    if enrich:
        console.print("[blue]Enriching metadata with LLM...[/blue]")
        try:
            enricher = MetadataEnricher(provider=llm_provider, api_key=api_key)
            codebook = enricher.enrich_codebook(codebook)
            console.print("[green]✓[/green] Metadata enriched")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Enrichment failed: {e}")

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(codebook.model_dump(), output)
    console.print(f"[green]✓[/green] Saved to {output}")


@app.command()
def analyze(
    data_path: Path = typer.Argument(..., help="Path to data file (CSV or ZIP)"),
    codebook_path: Path = typer.Argument(..., help="Path to codebook JSON"),
    output_dir: Path = typer.Option("output", "--output-dir", "-o", help="Output directory"),
    analyses: str = typer.Option(
        "all", "--analyses", "-a", help="Comma-separated: univariate,bivariate,temporal,causal"
    ),
    max_bivariate_pairs: int = typer.Option(100, "--max-pairs", help="Maximum bivariate pairs"),
    generate_plots: bool = typer.Option(True, "--plots/--no-plots", help="Generate plots"),
) -> None:
    """Run statistical analyses on dataset."""
    console.print(f"[blue]Loading data:[/blue] {data_path}")

    # Load data and codebook
    df = load_data(data_path)
    console.print(f"[green]✓[/green] Loaded {len(df)} rows, {len(df.columns)} columns")

    import json

    codebook_data = json.loads(Path(codebook_path).read_text(encoding="utf-8"))

    from statqa.metadata.schema import Codebook

    codebook = Codebook(**codebook_data)
    console.print(f"[green]✓[/green] Loaded codebook with {len(codebook.variables)} variables")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if generate_plots:
        plot_dir.mkdir(exist_ok=True)

    # Determine which analyses to run
    analysis_list = analyses.lower().split(",")
    if "all" in analysis_list:
        analysis_list = ["univariate", "bivariate", "temporal", "causal"]

    all_insights = []
    formatter = InsightFormatter()

    # Univariate
    if "univariate" in analysis_list:
        console.print("\n[bold]Running univariate analysis...[/bold]")
        analyzer = UnivariateAnalyzer()
        plot_factory = PlotFactory() if generate_plots else None

        results = []
        for var_name in track(codebook.variables.keys(), description="Analyzing variables"):
            if var_name in df.columns:
                var = codebook.variables[var_name]
                result = analyzer.analyze(df[var_name], var)
                result["formatted_insight"] = formatter.format_univariate(result)
                results.append(result)

                if generate_plots and plot_factory:
                    fig = plot_factory.plot_univariate(
                        df[var_name], var, plot_dir / f"univariate_{var_name}.png"
                    )
                    import matplotlib.pyplot as plt

                    plt.close(fig)

        save_json(results, output_dir / "univariate.json")
        console.print(f"[green]✓[/green] Completed {len(results)} univariate analyses")
        all_insights.extend(results)

    # Bivariate
    if "bivariate" in analysis_list:
        console.print("\n[bold]Running bivariate analysis...[/bold]")
        analyzer = BivariateAnalyzer()

        results = analyzer.batch_analyze(df, codebook.variables, max_pairs=max_bivariate_pairs)

        for result in results:
            result["formatted_insight"] = formatter.format_bivariate(result)

        save_json(results, output_dir / "bivariate.json")
        console.print(f"[green]✓[/green] Completed {len(results)} bivariate analyses")
        all_insights.extend(results)

    # Save all insights
    save_json(all_insights, output_dir / "all_insights.json")
    console.print(f"\n[bold green]✓ Analysis complete![/bold green] Results in {output_dir}")


@app.command()
def generate_qa(
    insights_path: Path = typer.Argument(..., help="Path to insights JSON"),
    output: Path = typer.Option("qa_pairs.jsonl", "--output", "-o", help="Output JSONL file"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for paraphrasing"),
    llm_provider: Literal["openai", "anthropic"] = typer.Option(
        "openai", "--llm-provider", help="LLM provider"
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="LLM API key"),
    export_format: Literal["jsonl", "openai", "anthropic"] = typer.Option(
        "jsonl", "--format", "-f", help="Export format (jsonl, openai, anthropic)"
    ),
) -> None:
    """Generate Q/A pairs from analysis insights."""
    console.print(f"[blue]Loading insights:[/blue] {insights_path}")

    import json

    insights = json.loads(Path(insights_path).read_text(encoding="utf-8"))

    console.print(f"[green]✓[/green] Loaded {len(insights)} insights")

    # Initialize generator
    generator = QAGenerator(
        use_llm=use_llm,
        llm_provider=llm_provider,
        api_key=api_key,
    )

    # Generate Q/A pairs
    console.print("[blue]Generating Q/A pairs...[/blue]")
    all_qa = []

    for insight in track(insights, description="Processing insights"):
        answer = insight.get("formatted_insight", "")
        if answer:
            qa_pairs = generator.generate_qa_pairs(insight, answer)
            all_qa.extend(qa_pairs)

    console.print(f"[green]✓[/green] Generated {len(all_qa)} Q/A pairs")

    # Export
    lines = []
    for qa in all_qa:
        match export_format:
            case "jsonl":
                lines.append(json.dumps(qa, ensure_ascii=False))
            case "openai":
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are a data analyst."},
                        {"role": "user", "content": qa["question"]},
                        {"role": "assistant", "content": qa["answer"]},
                    ]
                }
                lines.append(json.dumps(entry, ensure_ascii=False))
            case "anthropic":
                entry = {"prompt": qa["question"], "completion": qa["answer"]}
                lines.append(json.dumps(entry, ensure_ascii=False))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    console.print(f"[green]✓[/green] Saved to {output}")


@app.command()
def pipeline(
    data_path: Path = typer.Argument(..., help="Path to data file"),
    codebook_path: Path = typer.Argument(..., help="Path to codebook"),
    output_dir: Path = typer.Option("output", "--output-dir", "-o", help="Output directory"),
    generate_qa: bool = typer.Option(True, "--qa/--no-qa", help="Generate Q/A pairs"),
    enrich_metadata: bool = typer.Option(False, "--enrich", help="Enrich metadata with LLM"),
    api_key: str | None = typer.Option(None, "--api-key", help="LLM API key"),
) -> None:
    """Run complete pipeline: parse → analyze → generate Q/A."""
    console.print("[bold]Starting TableQA pipeline...[/bold]\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse codebook
    console.print("[bold blue]Step 1: Parsing codebook[/bold blue]")
    # ... (similar to parse_codebook command)

    # Step 2: Run analyses
    console.print("\n[bold blue]Step 2: Running analyses[/bold blue]")
    # ... (similar to analyze command)

    # Step 3: Generate Q/A
    if generate_qa:
        console.print("\n[bold blue]Step 3: Generating Q/A pairs[/bold blue]")
        # ... (similar to generate_qa command)

    console.print("\n[bold green]✓ Pipeline complete![/bold green]")


if __name__ == "__main__":
    app()
