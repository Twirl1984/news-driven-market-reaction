"""
Command-line interface for Market Event AI trading system.

This module provides a comprehensive CLI for managing the entire ML trading pipeline,
from data acquisition to backtesting and reporting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from market_event_ai.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Context object to pass configuration between commands
class Context:
    """CLI context object."""

    def __init__(self):
        self.config = None
        self.settings = settings


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file (YAML)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (DEBUG level)",
)
@click.pass_context
def main(ctx: click.Context, config: Optional[Path], verbose: bool):
    """
    Market Event AI - Trading system correlating political events with ETF movements.

    A comprehensive CLI for managing the entire machine learning trading pipeline,
    from data acquisition to backtesting and performance reporting.
    """
    ctx.obj = Context()

    # Load custom configuration if provided
    if config:
        logger.info(f"Loading configuration from {config}")
        with open(config, "r") as f:
            ctx.obj.config = yaml.safe_load(f)

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Ensure directories exist
    settings.paths.create_dirs()


@main.command()
@click.option(
    "--source",
    "-s",
    type=click.Choice(["trump_tweets", "gdelt", "financial", "all"]),
    default="all",
    help="Data source to download",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for data download (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for data download (YYYY-MM-DD)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-download even if data exists",
)
@click.pass_context
def download(
    ctx: click.Context,
    source: str,
    start_date: Optional[str],
    end_date: Optional[str],
    force: bool,
):
    """
    Download raw data from various sources.

    Downloads Trump tweets, GDELT events, and financial market data.
    Data is saved to the configured raw data directory.

    Examples:
        market-event-ai download --source trump_tweets
        market-event-ai download --source financial --start-date 2016-01-01
        market-event-ai download --force
    """
    from market_event_ai.data import downloaders

    logger.info(f"Starting download for source: {source}")

    try:
        if source == "all" or source == "trump_tweets":
            logger.info("Downloading Trump tweets...")
            downloaders.download_trump_tweets(
                output_dir=settings.paths.data_raw,
                start_date=start_date,
                end_date=end_date,
                force=force,
            )

        if source == "all" or source == "gdelt":
            logger.info("Downloading GDELT events...")
            downloaders.download_gdelt_events(
                output_dir=settings.paths.data_raw,
                start_date=start_date,
                end_date=end_date,
                force=force,
            )

        if source == "all" or source == "financial":
            logger.info("Downloading financial data...")
            downloaders.download_financial_data(
                output_dir=settings.paths.data_raw,
                assets_config=settings.load_assets_config(),
                start_date=start_date,
                end_date=end_date,
                force=force,
            )

        click.echo(click.style(f"✓ Download complete for {source}", fg="green"))
        logger.info(f"Data saved to {settings.paths.data_raw}")

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Download failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Input directory with raw data (defaults to configured raw data path)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for preprocessed data (defaults to configured processed path)",
)
@click.option(
    "--clean-text/--no-clean-text",
    default=True,
    help="Apply text cleaning and normalization",
)
@click.option(
    "--remove-duplicates/--keep-duplicates",
    default=True,
    help="Remove duplicate entries",
)
@click.pass_context
def preprocess(
    ctx: click.Context,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    clean_text: bool,
    remove_duplicates: bool,
):
    """
    Preprocess raw data for feature engineering.

    Performs text cleaning, normalization, deduplication, and timestamp alignment.
    Outputs cleaned and structured data ready for feature extraction.

    Examples:
        market-event-ai preprocess
        market-event-ai preprocess --no-clean-text
        market-event-ai preprocess --input-dir data/raw --output-dir data/processed
    """
    from market_event_ai.preprocess import preprocessors

    input_dir = input_dir or settings.paths.data_raw
    output_dir = output_dir or settings.paths.data_processed

    logger.info(f"Starting preprocessing from {input_dir}")

    try:
        preprocessors.preprocess_tweets(
            input_dir=input_dir,
            output_dir=output_dir,
            clean_text=clean_text,
            remove_duplicates=remove_duplicates,
        )

        preprocessors.preprocess_gdelt(
            input_dir=input_dir,
            output_dir=output_dir,
            remove_duplicates=remove_duplicates,
        )

        preprocessors.preprocess_financial_data(
            input_dir=input_dir,
            output_dir=output_dir,
        )

        click.echo(click.style("✓ Preprocessing complete", fg="green"))
        logger.info(f"Preprocessed data saved to {output_dir}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Preprocessing failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Input directory with preprocessed data",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for features",
)
@click.option(
    "--feature-set",
    "-f",
    type=click.Choice(["text", "sentiment", "embeddings", "technical", "all"]),
    default="all",
    help="Feature set to generate",
)
@click.option(
    "--embedding-model",
    type=str,
    help="Model name for text embeddings (e.g., distilbert-base-uncased)",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for embedding generation",
)
@click.pass_context
def features(
    ctx: click.Context,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    feature_set: str,
    embedding_model: Optional[str],
    batch_size: int,
):
    """
    Generate features from preprocessed data.

    Extracts various features including text embeddings, sentiment scores,
    technical indicators, and event aggregations.

    Examples:
        market-event-ai features
        market-event-ai features --feature-set sentiment
        market-event-ai features --embedding-model roberta-base --batch-size 64
    """
    from market_event_ai.features import extractors

    input_dir = input_dir or settings.paths.data_processed
    output_dir = output_dir or settings.paths.data_features
    embedding_model = embedding_model or settings.model.embedding_model

    logger.info(f"Generating features from {input_dir}")

    try:
        if feature_set == "all" or feature_set == "text":
            logger.info("Generating text features...")
            extractors.generate_text_features(
                input_dir=input_dir,
                output_dir=output_dir,
            )

        if feature_set == "all" or feature_set == "sentiment":
            logger.info("Generating sentiment features...")
            extractors.generate_sentiment_features(
                input_dir=input_dir,
                output_dir=output_dir,
            )

        if feature_set == "all" or feature_set == "embeddings":
            logger.info(f"Generating embeddings using {embedding_model}...")
            extractors.generate_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                model_name=embedding_model,
                batch_size=batch_size,
            )

        if feature_set == "all" or feature_set == "technical":
            logger.info("Generating technical indicators...")
            extractors.generate_technical_features(
                input_dir=input_dir,
                output_dir=output_dir,
            )

        click.echo(click.style(f"✓ Feature generation complete", fg="green"))
        logger.info(f"Features saved to {output_dir}")

    except Exception as e:
        logger.error(f"Feature generation failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Feature generation failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Input directory with features",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for labels",
)
@click.option(
    "--horizon",
    "-h",
    type=int,
    multiple=True,
    default=[1, 5, 20],
    help="Prediction horizon in days (can specify multiple)",
)
@click.option(
    "--label-type",
    type=click.Choice(["returns", "direction", "volatility", "all"]),
    default="all",
    help="Type of labels to generate",
)
@click.option(
    "--min-threshold",
    type=float,
    help="Minimum threshold for binary labels",
)
@click.pass_context
def label(
    ctx: click.Context,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    horizon: tuple,
    label_type: str,
    min_threshold: Optional[float],
):
    """
    Generate labels/targets for model training.

    Creates prediction targets based on forward returns, direction changes,
    or volatility at specified time horizons.

    Examples:
        market-event-ai label
        market-event-ai label --horizon 1 --horizon 5 --horizon 20
        market-event-ai label --label-type direction --min-threshold 0.02
    """
    from market_event_ai.labels import generators

    input_dir = input_dir or settings.paths.data_features
    output_dir = output_dir or settings.paths.data_labels
    min_threshold = min_threshold or settings.trading.signal_threshold

    logger.info(f"Generating labels from {input_dir}")
    logger.info(f"Horizons: {list(horizon)}")

    try:
        if label_type == "all" or label_type == "returns":
            logger.info("Generating return labels...")
            generators.generate_return_labels(
                input_dir=input_dir,
                output_dir=output_dir,
                horizons=list(horizon),
            )

        if label_type == "all" or label_type == "direction":
            logger.info("Generating direction labels...")
            generators.generate_direction_labels(
                input_dir=input_dir,
                output_dir=output_dir,
                horizons=list(horizon),
                threshold=min_threshold,
            )

        if label_type == "all" or label_type == "volatility":
            logger.info("Generating volatility labels...")
            generators.generate_volatility_labels(
                input_dir=input_dir,
                output_dir=output_dir,
                horizons=list(horizon),
            )

        click.echo(click.style("✓ Label generation complete", fg="green"))
        logger.info(f"Labels saved to {output_dir}")

    except Exception as e:
        logger.error(f"Label generation failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Label generation failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--model-type",
    "-m",
    type=click.Choice(["xgboost", "lightgbm", "random_forest", "neural_net", "transformer"]),
    help="Type of model to train",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Input directory with features and labels",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for trained models",
)
@click.option(
    "--target",
    "-t",
    type=str,
    default="return_1d",
    help="Target variable to predict",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    help="Number of cross-validation folds",
)
@click.option(
    "--hyperopt/--no-hyperopt",
    default=False,
    help="Enable hyperparameter optimization",
)
@click.option(
    "--n-trials",
    type=int,
    default=100,
    help="Number of hyperparameter optimization trials",
)
@click.pass_context
def train(
    ctx: click.Context,
    model_type: Optional[str],
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    target: str,
    cv_folds: int,
    hyperopt: bool,
    n_trials: int,
):
    """
    Train machine learning models.

    Trains models on prepared features and labels with optional cross-validation
    and hyperparameter optimization.

    Examples:
        market-event-ai train --model-type xgboost
        market-event-ai train --model-type neural_net --hyperopt --n-trials 50
        market-event-ai train --target return_5d --cv-folds 10
    """
    from market_event_ai.models import trainers

    model_type = model_type or settings.model.model_type
    input_dir = input_dir or settings.paths.data_labels
    output_dir = output_dir or settings.paths.models

    logger.info(f"Training {model_type} model")
    logger.info(f"Target variable: {target}")

    try:
        trainer = trainers.get_trainer(model_type)

        model = trainer.train(
            data_dir=input_dir,
            target=target,
            cv_folds=cv_folds,
            hyperopt=hyperopt,
            n_trials=n_trials if hyperopt else None,
            random_seed=settings.model.random_seed,
        )

        # Save model
        model_path = output_dir / f"{model_type}_{target}.joblib"
        trainer.save_model(model, model_path)

        click.echo(click.style(f"✓ Model training complete", fg="green"))
        logger.info(f"Model saved to {model_path}")

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Model training failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model",
)
@click.option(
    "--test-data",
    type=click.Path(exists=True, path_type=Path),
    help="Path to test data (optional, uses train/test split if not provided)",
)
@click.option(
    "--metrics",
    "-m",
    type=click.Choice(["all", "regression", "classification"]),
    default="all",
    help="Metrics to compute",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for evaluation results",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    model_path: Path,
    test_data: Optional[Path],
    metrics: str,
    output_dir: Optional[Path],
):
    """
    Evaluate trained models on test data.

    Computes various performance metrics and generates evaluation reports.

    Examples:
        market-event-ai evaluate --model-path models/xgboost_return_1d.joblib
        market-event-ai evaluate --model-path models/model.joblib --test-data data/test.parquet
        market-event-ai evaluate --model-path models/model.joblib --metrics classification
    """
    from market_event_ai.evaluation import evaluators

    output_dir = output_dir or settings.paths.reports / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating model: {model_path}")

    try:
        evaluator = evaluators.ModelEvaluator.from_path(model_path)

        results = evaluator.evaluate(
            test_data_path=test_data,
            metrics=metrics,
            output_dir=output_dir,
        )

        # Display summary metrics
        click.echo(click.style("\n=== Evaluation Results ===", fg="cyan", bold=True))
        for metric_name, metric_value in results.get("metrics", {}).items():
            click.echo(f"{metric_name}: {metric_value:.4f}")

        click.echo(click.style(f"\n✓ Evaluation complete", fg="green"))
        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Evaluation failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backtest start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backtest end date (YYYY-MM-DD)",
)
@click.option(
    "--initial-capital",
    type=float,
    help="Initial capital for backtesting",
)
@click.option(
    "--strategy",
    type=click.Choice(["long_short", "long_only", "market_neutral"]),
    default="long_short",
    help="Trading strategy to use",
)
@click.option(
    "--rebalance-freq",
    type=click.Choice(["daily", "weekly", "monthly"]),
    default="daily",
    help="Portfolio rebalancing frequency",
)
@click.option(
    "--walk-forward/--no-walk-forward",
    default=True,
    help="Enable walk-forward analysis",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for backtest results",
)
@click.pass_context
def backtest(
    ctx: click.Context,
    model_path: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    initial_capital: Optional[float],
    strategy: str,
    rebalance_freq: str,
    walk_forward: bool,
    output_dir: Optional[Path],
):
    """
    Run backtesting on trained models.

    Simulates trading strategy performance using historical data with realistic
    transaction costs and slippage.

    Examples:
        market-event-ai backtest --model-path models/xgboost_return_1d.joblib
        market-event-ai backtest --model-path models/model.joblib --strategy long_only
        market-event-ai backtest --model-path models/model.joblib --start-date 2018-01-01 --end-date 2020-12-31
    """
    from market_event_ai.portfolio import backtesters

    output_dir = output_dir or settings.paths.backtests
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = start_date or settings.trading.backtest_start_date
    end_date = end_date or settings.trading.backtest_end_date
    initial_capital = initial_capital or settings.trading.initial_capital

    logger.info(f"Running backtest with model: {model_path}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Strategy: {strategy}, Initial capital: ${initial_capital:,.2f}")

    try:
        backtester = backtesters.Backtester(
            model_path=model_path,
            strategy=strategy,
            initial_capital=initial_capital,
            trading_cost_bps=settings.trading.trading_cost_bps,
            slippage_bps=settings.trading.slippage_bps,
        )

        results = backtester.run(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            walk_forward=walk_forward,
            walk_forward_window=settings.trading.walk_forward_window_days,
        )

        # Display summary statistics
        click.echo(click.style("\n=== Backtest Results ===", fg="cyan", bold=True))
        click.echo(f"Total Return: {results['total_return']:.2%}")
        click.echo(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        click.echo(f"Max Drawdown: {results['max_drawdown']:.2%}")
        click.echo(f"Win Rate: {results['win_rate']:.2%}")
        click.echo(f"Num Trades: {results['num_trades']}")

        # Save results
        results_path = output_dir / f"backtest_{model_path.stem}_{start_date}_{end_date}.parquet"
        backtester.save_results(results, results_path)

        click.echo(click.style(f"\n✓ Backtest complete", fg="green"))
        logger.info(f"Results saved to {results_path}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Backtest failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Input directory with results (evaluation or backtest)",
)
@click.option(
    "--report-type",
    "-t",
    type=click.Choice(["evaluation", "backtest", "full"]),
    default="full",
    help="Type of report to generate",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["html", "pdf", "markdown"]),
    default="html",
    help="Output format for report",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for report",
)
@click.option(
    "--include-plots/--no-plots",
    default=True,
    help="Include visualization plots in report",
)
@click.pass_context
def report(
    ctx: click.Context,
    input_dir: Optional[Path],
    report_type: str,
    output_format: str,
    output: Optional[Path],
    include_plots: bool,
):
    """
    Generate comprehensive reports.

    Creates detailed reports with visualizations and performance analysis.

    Examples:
        market-event-ai report --report-type backtest
        market-event-ai report --input-dir data/backtests --format pdf
        market-event-ai report --report-type full --output reports/analysis.html
    """
    from market_event_ai.reports import generators

    input_dir = input_dir or settings.paths.backtests
    output_dir = settings.paths.reports
    output_dir.mkdir(parents=True, exist_ok=True)

    if not output:
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        output = output_dir / f"report_{report_type}_{timestamp}.{output_format}"

    logger.info(f"Generating {report_type} report from {input_dir}")

    try:
        generator = generators.ReportGenerator(
            input_dir=input_dir,
            output_path=output,
            format=output_format,
        )

        generator.generate(
            report_type=report_type,
            include_plots=include_plots,
        )

        click.echo(click.style(f"✓ Report generated successfully", fg="green"))
        click.echo(f"Report saved to: {output}")
        logger.info(f"Report saved to {output}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        click.echo(click.style(f"✗ Report generation failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def info(ctx: click.Context):
    """
    Display system configuration and status.

    Shows current settings, paths, and available resources.
    """
    click.echo(click.style("\n=== Market Event AI Configuration ===", fg="cyan", bold=True))

    click.echo(click.style("\nPaths:", fg="yellow"))
    click.echo(f"  Raw Data:       {settings.paths.data_raw}")
    click.echo(f"  Processed Data: {settings.paths.data_processed}")
    click.echo(f"  Features:       {settings.paths.data_features}")
    click.echo(f"  Labels:         {settings.paths.data_labels}")
    click.echo(f"  Models:         {settings.paths.models}")
    click.echo(f"  Backtests:      {settings.paths.backtests}")
    click.echo(f"  Reports:        {settings.paths.reports}")
    click.echo(f"  Database:       {settings.paths.sqlite_db}")

    click.echo(click.style("\nModel Configuration:", fg="yellow"))
    click.echo(f"  Model Type:      {settings.model.model_type}")
    click.echo(f"  Embedding Model: {settings.model.embedding_model}")
    click.echo(f"  Random Seed:     {settings.model.random_seed}")

    click.echo(click.style("\nTrading Configuration:", fg="yellow"))
    click.echo(f"  Initial Capital:   ${settings.trading.initial_capital:,.2f}")
    click.echo(f"  Trading Cost:      {settings.trading.trading_cost_bps} bps")
    click.echo(f"  Slippage:          {settings.trading.slippage_bps} bps")
    click.echo(f"  Signal Threshold:  {settings.trading.signal_threshold}")

    click.echo(click.style("\nBacktest Period:", fg="yellow"))
    click.echo(f"  Start Date: {settings.trading.backtest_start_date}")
    click.echo(f"  End Date:   {settings.trading.backtest_end_date}")
    click.echo(f"  Walk-Forward Window: {settings.trading.walk_forward_window_days} days")

    click.echo(click.style("\nLogging:", fg="yellow"))
    click.echo(f"  Level: {settings.log_level}")

    click.echo()


if __name__ == "__main__":
    main()
