"""Report generation modules."""
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class ReportGenerator:
    """Generate backtest reports."""
    
    def __init__(self, backtest_dir: Path, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            backtest_dir: Directory with backtest results
            output_dir: Directory for output reports
        """
        self.backtest_dir = backtest_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self) -> Path:
        """Generate markdown summary report."""
        logger.info("Generating markdown report...")
        
        # Load results
        with open(self.backtest_dir / "metrics.json", 'r') as f:
            metrics = json.load(f)
        
        equity_df = pd.read_csv(self.backtest_dir / "equity_curve.csv")
        trades_df = pd.read_csv(self.backtest_dir / "trades.csv")
        
        # Generate report
        report = f"""# Trading Strategy Backtest Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

### Key Metrics

| Metric | Value |
|--------|-------|
| CAGR | {metrics.get('cagr', 0):.2%} |
| Total Return | {metrics.get('total_return', 0):.2%} |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {metrics.get('max_drawdown', 0):.2%} |
| Number of Trades | {metrics.get('num_trades', 0)} |
| Turnover | {metrics.get('turnover', 0):.2f}x |
| Final Equity | ${metrics.get('final_equity', 0):,.2f} |

### Benchmark Comparison

| Strategy | Return |
|----------|--------|
| Our Strategy | {metrics.get('total_return', 0):.2%} |
| Buy & Hold | {metrics.get('buy_hold_return', 0):.2%} |
| Outperformance | {(metrics.get('total_return', 0) - metrics.get('buy_hold_return', 0)):.2%} |

## Trading Activity

- **Total Trades**: {len(trades_df)}
- **Average Trade Size**: ${trades_df['value'].mean():,.2f}
- **Buy Trades**: {len(trades_df[trades_df['action'] == 'BUY'])}
- **Sell Trades**: {len(trades_df[trades_df['action'] == 'SELL'])}

## Equity Curve

![Equity Curve](equity_curve.png)

## Conclusions

The strategy {'outperformed' if metrics.get('total_return', 0) > metrics.get('buy_hold_return', 0) else 'underperformed'} 
the buy-and-hold benchmark by {abs(metrics.get('total_return', 0) - metrics.get('buy_hold_return', 0)):.2%}.

Key observations:
- Sharpe ratio of {metrics.get('sharpe_ratio', 0):.2f} indicates {'strong' if metrics.get('sharpe_ratio', 0) > 1 else 'moderate' if metrics.get('sharpe_ratio', 0) > 0.5 else 'weak'} risk-adjusted returns
- Maximum drawdown of {metrics.get('max_drawdown', 0):.2%} shows portfolio volatility
- Generated {metrics.get('num_trades', 0)} trades over the backtest period

## Risk Considerations

- This is a historical simulation and past performance does not guarantee future results
- Transaction costs and slippage are modeled but may differ in live trading
- Market conditions may change significantly from the backtest period
"""
        
        output_file = self.output_dir / "summary.md"
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Markdown report saved to {output_file}")
        return output_file
    
    def generate_plots(self):
        """Generate visualization plots."""
        logger.info("Generating plots...")
        
        # Load data
        equity_df = pd.read_csv(self.backtest_dir / "equity_curve.csv")
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        # Equity curve plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Equity curve
        axes[0].plot(equity_df['date'], equity_df['equity'], label='Strategy Equity', linewidth=2)
        axes[0].axhline(y=equity_df['equity'].iloc[0], color='gray', linestyle='--', label='Initial Capital')
        axes[0].set_title('Portfolio Equity Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        
        axes[1].fill_between(equity_df['date'], equity_df['drawdown'], 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "equity_curve.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {output_file}")
        
        # Trade distribution plot
        if (self.backtest_dir / "trades.csv").exists():
            trades_df = pd.read_csv(self.backtest_dir / "trades.csv")
            
            if len(trades_df) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Action distribution
                action_counts = trades_df['action'].value_counts()
                axes[0].bar(action_counts.index, action_counts.values)
                axes[0].set_title('Trade Actions Distribution')
                axes[0].set_ylabel('Count')
                
                # Trade value distribution
                axes[1].hist(trades_df['value'], bins=20, edgecolor='black')
                axes[1].set_title('Trade Value Distribution')
                axes[1].set_xlabel('Trade Value ($)')
                axes[1].set_ylabel('Frequency')
                
                plt.tight_layout()
                
                output_file = self.output_dir / "trades_analysis.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Trade analysis plots saved to {output_file}")


def generate_report(model_type: str = None):
    """
    Main report generation pipeline.
    
    Args:
        model_type: Type of model to generate report for
    """
    logger.info("Starting report generation...")
    
    # Use model type from settings if not provided
    if model_type is None:
        model_type = settings.model.model_type
    
    # Set paths
    backtest_dir = settings.paths.backtests / model_type
    if not backtest_dir.exists():
        raise FileNotFoundError(f"Backtest results not found: {backtest_dir}")
    
    output_dir = settings.paths.reports / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    generator = ReportGenerator(backtest_dir, output_dir)
    
    # Generate plots
    generator.generate_plots()
    
    # Generate markdown report
    report_file = generator.generate_markdown_report()
    
    logger.info(f"Report generated successfully: {report_file}")
    
    return report_file
