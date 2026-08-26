"""Main CLI interface for tableqa.

Provides commands for:
- Parsing codebooks
- Running analyses
- Generating Q/A pairs
- Creating visualizations
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer
from pydantic import ValidationError
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

# Optional statistical format parser. Bind to None rather than only tracking a
# boolean: narrowing on `is None` is what lets a type checker see it is bound.
#
# Importing the class is not enough to know it is usable. The module keeps
# working without pyreadstat -- that is the point of the optional dependency --
# so the import always succeeds and only the constructor raises. Availability
# has to be read from HAS_PYREADSTAT.
try:
    from statqa.metadata.parsers.statistical import (
        HAS_PYREADSTAT,
        StatisticalFormatParser,
    )
except ImportError:
    StatisticalFormatParser = None
    HAS_PYREADSTAT = False

from statqa.qa.generator import QAGenerator
from statqa.utils.io import load_data, save_json
from statqa.visualization.plots import PlotFactory

if TYPE_CHECKING:
    import pandas as pd

    from statqa.metadata.schema import Codebook

#: True only when the parser can actually be constructed.
HAS_STATISTICAL_PARSER = StatisticalFormatParser is not None and HAS_PYREADSTAT

app = typer.Typer(help="TableQA: Extract structured facts from tabular datasets")
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold green]TableQA version {__version__}[/bold green]")


def _load_codebook(
    codebook_path: Path,
    format: str = "auto",
    enrich: bool = False,
    llm_provider: Literal["openai", "anthropic"] = "openai",
    api_key: str | None = None,
) -> "Codebook":
    """Parse a codebook file, optionally enriching it with an LLM.

    Shared by the `parse-codebook` and `pipeline` commands. `pipeline` cannot
    call the command function directly: its parameters carry `typer.Option`
    defaults, so an omitted argument arrives as an `OptionInfo` sentinel rather
    than a value.

    Args:
        codebook_path: Path to the codebook file.
        format: One of auto, csv, text, statistical.
        enrich: Whether to enrich metadata with an LLM.
        llm_provider: Provider to enrich with.
        api_key: API key for the provider.

    Returns:
        The parsed codebook.

    Raises:
        typer.Exit: If the format cannot be determined, is unknown, or names
            statistical support that is not installed.
    """
    console.print(f"[blue]Parsing codebook:[/blue] {codebook_path}")

    # Select parser
    if format == "auto":
        # Try parsers in order - statistical first since it's more specific
        parsers: list[BaseParser] = []
        if StatisticalFormatParser is not None and HAS_STATISTICAL_PARSER:
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
                if StatisticalFormatParser is None or not HAS_STATISTICAL_PARSER:
                    # The bracket is escaped because rich reads [...] as markup
                    # and would otherwise drop the extra, leaving the message
                    # telling the reader to run plain `pip install statqa`.
                    console.print(
                        "[red]Error:[/red] Statistical format support not "
                        "available. Install with: "
                        r"pip install 'statqa\[statistical-formats]'"
                    )
                    raise typer.Exit(1)
                parser = StatisticalFormatParser()
            case _:
                console.print(f"[red]Error:[/red] Unknown format: {format}")
                raise typer.Exit(1)

    codebook = parser.parse(codebook_path)
    console.print(f"[green]✓[/green] Parsed {len(codebook.variables)} variables")

    if enrich:
        console.print("[blue]Enriching metadata with LLM...[/blue]")
        try:
            enricher = MetadataEnricher(provider=llm_provider, api_key=api_key)
            codebook = enricher.enrich_codebook(codebook)
            console.print("[green]✓[/green] Metadata enriched")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Enrichment failed: {e}")

    return codebook


@app.command()
def parse_codebook(
    codebook_path: Path = typer.Argument(..., help="Path to codebook file"),
    output: Path = typer.Option(
        "codebook.json", "--output", "-o", help="Output JSON file"
    ),
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
    codebook = _load_codebook(codebook_path, format, enrich, llm_provider, api_key)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(codebook.model_dump(), output)
    console.print(f"[green]✓[/green] Saved to {output}")


#: Analyses `--analyses all` expands to. `causal` is deliberately absent: it
#: needs a treatment and an outcome, which no option supplies and no codebook
#: implies, so including it here would silently request something unrunnable.
ALL_ANALYSES = ("univariate", "bivariate", "temporal")


def _run_analyses(
    df: "pd.DataFrame",
    codebook: "Codebook",
    output_dir: Path,
    analyses: str,
    max_bivariate_pairs: int,
    generate_plots: bool,
) -> list[dict[str, Any]]:
    """Run the requested analyses and write their results.

    Shared by the `analyze` and `pipeline` commands.

    Args:
        df: The dataset.
        codebook: Variable metadata.
        output_dir: Directory to write per-analysis and combined JSON into.
        analyses: Comma-separated analysis names, or ``all``.
        max_bivariate_pairs: Cap on bivariate and temporal pairs.
        generate_plots: Whether to write univariate plots.

    Returns:
        Every insight produced, in the order the analyses ran.

    Raises:
        typer.Exit: If an analysis is requested that this command cannot run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if generate_plots:
        plot_dir.mkdir(exist_ok=True)

    analysis_list = [name.strip() for name in analyses.lower().split(",")]
    if "all" in analysis_list:
        analysis_list = list(ALL_ANALYSES)

    # Refuse rather than ignore. `causal` and any typo used to be accepted and
    # then silently produce nothing, which reads exactly like an analysis that
    # found nothing to say.
    unknown = [name for name in analysis_list if name not in ALL_ANALYSES]
    if unknown:
        console.print(
            f"[red]Error:[/red] Cannot run: {', '.join(unknown)}. "
            f"Available: {', '.join(ALL_ANALYSES)}."
        )
        if "causal" in unknown:
            console.print(
                "Causal analysis needs a treatment and an outcome variable, "
                "which the codebook does not imply. Use "
                "statqa.analysis.causal.CausalAnalyzer directly."
            )
        raise typer.Exit(1)

    all_insights: list[dict[str, Any]] = []
    formatter = InsightFormatter()

    if "univariate" in analysis_list:
        console.print("\n[bold]Running univariate analysis...[/bold]")
        analyzer = UnivariateAnalyzer()
        plot_factory = PlotFactory() if generate_plots else None

        results = []
        for var_name in track(
            codebook.variables.keys(), description="Analyzing variables"
        ):
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

    if "bivariate" in analysis_list:
        console.print("\n[bold]Running bivariate analysis...[/bold]")
        bivariate = BivariateAnalyzer()

        results = bivariate.batch_analyze(
            df, codebook.variables, max_pairs=max_bivariate_pairs
        )

        for result in results:
            result["formatted_insight"] = formatter.format_bivariate(result)

        save_json(results, output_dir / "bivariate.json")
        console.print(f"[green]✓[/green] Completed {len(results)} bivariate analyses")
        all_insights.extend(results)

    if "temporal" in analysis_list:
        results = _run_temporal(df, codebook, formatter, max_bivariate_pairs)
        save_json(results, output_dir / "temporal.json")
        all_insights.extend(results)

    save_json(all_insights, output_dir / "all_insights.json")
    return all_insights


def _run_temporal(
    df: "pd.DataFrame",
    codebook: "Codebook",
    formatter: InsightFormatter,
    max_pairs: int,
) -> list[dict[str, Any]]:
    """Trend every numeric variable against every time variable.

    The pairing is inferable, unlike causal analysis: `Variable.is_temporal()`
    identifies the time axis and `is_numeric()` the series to trend along it.
    A dataset with no datetime variable simply produces nothing, and says so.

    Args:
        df: The dataset.
        codebook: Variable metadata.
        formatter: Formatter for the natural-language insight.
        max_pairs: Cap on the number of pairs analysed.

    Returns:
        One insight per analysed pair.
    """
    console.print("\n[bold]Running temporal analysis...[/bold]")
    from statqa.analysis.temporal import TemporalAnalyzer

    present = [name for name in codebook.variables if name in df.columns]
    time_vars = [n for n in present if codebook.variables[n].is_temporal()]
    value_vars = [n for n in present if codebook.variables[n].is_numeric()]

    if not time_vars:
        console.print(
            "[yellow]No datetime variables in the codebook; "
            "nothing to trend against.[/yellow]"
        )
        return []

    analyzer = TemporalAnalyzer()
    pairs = [(t, v) for t in time_vars for v in value_vars if t != v][:max_pairs]

    results = []
    for time_name, value_name in track(pairs, description="Analyzing trends"):
        try:
            result = analyzer.analyze_trend(
                df, codebook.variables[time_name], codebook.variables[value_name]
            )
        except (ValueError, KeyError, TypeError) as exc:
            # One unusable pair -- an unparseable date column, too few points --
            # is not a reason to abandon the rest.
            console.print(
                f"[yellow]Skipped {value_name} over {time_name}:[/yellow] {exc}"
            )
            continue
        result["formatted_insight"] = formatter.format_temporal(result)
        results.append(result)

    console.print(f"[green]✓[/green] Completed {len(results)} temporal analyses")
    return results


@app.command()
def analyze(
    data_path: Path = typer.Argument(..., help="Path to data file (CSV or ZIP)"),
    codebook_path: Path = typer.Argument(..., help="Path to codebook JSON"),
    output_dir: Path = typer.Option(
        "output", "--output-dir", "-o", help="Output directory"
    ),
    analyses: str = typer.Option(
        "all",
        "--analyses",
        "-a",
        help="Comma-separated: univariate,bivariate,temporal,causal",
    ),
    max_bivariate_pairs: int = typer.Option(
        100, "--max-pairs", help="Maximum bivariate pairs"
    ),
    generate_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Generate plots"
    ),
) -> None:
    """Run statistical analyses on dataset."""
    console.print(f"[blue]Loading data:[/blue] {data_path}")

    df = load_data(data_path)
    console.print(f"[green]✓[/green] Loaded {len(df)} rows, {len(df.columns)} columns")

    import json

    codebook_data = json.loads(Path(codebook_path).read_text(encoding="utf-8"))

    from statqa.metadata.schema import Codebook

    try:
        codebook = Codebook.from_dict(codebook_data, name=Path(codebook_path).stem)
    except (ValidationError, ValueError) as exc:
        console.print(f"[red]Error:[/red] Could not read {codebook_path}: {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"[green]✓[/green] Loaded codebook with {len(codebook.variables)} variables"
    )

    _run_analyses(
        df, codebook, output_dir, analyses, max_bivariate_pairs, generate_plots
    )
    console.print(
        f"\n[bold green]✓ Analysis complete![/bold green] Results in {output_dir}"
    )


def _qa_lines(
    insights: list[dict[str, Any]],
    use_llm: bool = False,
    llm_provider: Literal["openai", "anthropic"] = "openai",
    api_key: str | None = None,
    export_format: Literal["jsonl", "openai", "anthropic"] = "jsonl",
) -> list[str]:
    """Turn insights into serialized Q/A lines.

    Shared by the `generate-qa` and `pipeline` commands.

    Args:
        insights: Analysis results carrying a `formatted_insight`.
        use_llm: Whether to paraphrase with an LLM.
        llm_provider: Provider to paraphrase with.
        api_key: API key for the provider.
        export_format: One of jsonl, openai, anthropic.

    Returns:
        One serialized line per Q/A pair.
    """
    import json

    generator = QAGenerator(
        use_llm=use_llm,
        llm_provider=llm_provider,
        api_key=api_key,
    )

    console.print("[blue]Generating Q/A pairs...[/blue]")
    all_qa = []

    for insight in track(insights, description="Processing insights"):
        answer = insight.get("formatted_insight", "")
        if answer:
            qa_pairs = generator.generate_qa_pairs(insight, answer)
            all_qa.extend(qa_pairs)

    console.print(f"[green]✓[/green] Generated {len(all_qa)} Q/A pairs")

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
    return lines


@app.command()
def generate_qa(
    insights_path: Path = typer.Argument(..., help="Path to insights JSON"),
    output: Path = typer.Option(
        "qa_pairs.jsonl", "--output", "-o", help="Output JSONL file"
    ),
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

    lines = _qa_lines(insights, use_llm, llm_provider, api_key, export_format)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    console.print(f"[green]✓[/green] Saved to {output}")


def _codebook_from_any(
    codebook_path: Path,
    enrich: bool = False,
    llm_provider: Literal["openai", "anthropic"] = "openai",
    api_key: str | None = None,
) -> "Codebook":
    """Load a codebook given either a raw codebook file or codebook JSON.

    `parse-codebook` converts raw to JSON and `analyze` consumes JSON, so a
    pipeline spanning both has to accept whichever the caller has. The README
    passes a CSV; the bundled examples ship JSON. Both work.

    Args:
        codebook_path: Path to a codebook in any supported form.
        enrich: Whether to enrich metadata with an LLM.
        llm_provider: Provider to enrich with.
        api_key: API key for the provider.

    Returns:
        The codebook.
    """
    import json

    from statqa.metadata.schema import Codebook

    try:
        data = json.loads(codebook_path.read_text(encoding="utf-8"))
        codebook = Codebook.from_dict(data, name=codebook_path.stem)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ):
        # Not codebook JSON; hand it to the format parsers, which report their
        # own error if they cannot read it either.
        return _load_codebook(codebook_path, "auto", enrich, llm_provider, api_key)

    console.print(
        f"[green]✓[/green] Loaded codebook with {len(codebook.variables)} variables"
    )
    if enrich:
        console.print("[blue]Enriching metadata with LLM...[/blue]")
        try:
            enricher = MetadataEnricher(provider=llm_provider, api_key=api_key)
            codebook = enricher.enrich_codebook(codebook)
            console.print("[green]✓[/green] Metadata enriched")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Enrichment failed: {e}")
    return codebook


@app.command()
def pipeline(
    data_path: Path = typer.Argument(..., help="Path to data file"),
    codebook_path: Path = typer.Argument(..., help="Path to codebook"),
    output_dir: Path = typer.Option(
        "output", "--output-dir", "-o", help="Output directory"
    ),
    # Named `make_qa` rather than `generate_qa`: that name is the command
    # function in this module, and a parameter shadowing it would turn a call
    # to the command into a call to a bool.
    make_qa: bool = typer.Option(True, "--qa/--no-qa", help="Generate Q/A pairs"),
    analyses: str = typer.Option(
        "all",
        "--analyses",
        "-a",
        help=f"Comma-separated: {','.join(ALL_ANALYSES)}",
    ),
    max_bivariate_pairs: int = typer.Option(
        100, "--max-pairs", help="Maximum bivariate pairs"
    ),
    generate_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Generate plots"
    ),
    enrich_metadata: bool = typer.Option(
        False, "--enrich", help="Enrich metadata with LLM"
    ),
    llm_provider: Literal["openai", "anthropic"] = typer.Option(
        "openai", "--llm-provider", help="LLM provider"
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="LLM API key"),
) -> None:
    """Run complete pipeline: parse → analyze → generate Q/A."""
    console.print("[bold]Starting TableQA pipeline...[/bold]\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse codebook
    console.print("[bold blue]Step 1: Parsing codebook[/bold blue]")
    codebook = _codebook_from_any(codebook_path, enrich_metadata, llm_provider, api_key)
    codebook_out = output_dir / "codebook.json"
    save_json(codebook.model_dump(), codebook_out)
    console.print(f"[green]✓[/green] Saved to {codebook_out}")

    # Step 2: Run analyses
    console.print("\n[bold blue]Step 2: Running analyses[/bold blue]")
    console.print(f"[blue]Loading data:[/blue] {data_path}")
    df = load_data(data_path)
    console.print(f"[green]✓[/green] Loaded {len(df)} rows, {len(df.columns)} columns")

    insights = _run_analyses(
        df, codebook, output_dir, analyses, max_bivariate_pairs, generate_plots
    )

    # Step 3: Generate Q/A
    if make_qa:
        console.print("\n[bold blue]Step 3: Generating Q/A pairs[/bold blue]")
        lines = _qa_lines(insights, api_key=api_key, llm_provider=llm_provider)
        qa_out = output_dir / "qa_pairs.jsonl"
        qa_out.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]✓[/green] Saved to {qa_out}")

    console.print(
        f"\n[bold green]✓ Pipeline complete![/bold green] Results in {output_dir}"
    )


if __name__ == "__main__":
    app()
