"""Event study analysis for market reaction to events."""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class EventStudy:
    """Perform event study analysis."""
    
    def __init__(self, window_pre: int = 5, window_post: int = 5):
        """
        Initialize event study.
        
        Args:
            window_pre: Days before event to analyze
            window_post: Days after event to analyze
        """
        self.window_pre = window_pre
        self.window_post = window_post
    
    def calculate_abnormal_returns(
        self,
        events_df: pd.DataFrame,
        financial_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate abnormal returns around events.
        
        Args:
            events_df: DataFrame with events (must have 'date' column)
            financial_df: DataFrame with financial data
        
        Returns:
            DataFrame with abnormal returns
        """
        logger.info("Calculating abnormal returns...")
        
        # Ensure dates are datetime
        events_df['date'] = pd.to_datetime(events_df['date'])
        financial_df['date'] = pd.to_datetime(financial_df['date'])
        
        # Calculate market returns (average across all tickers)
        market_returns = financial_df.groupby('date')['log_return'].mean().reset_index()
        market_returns.columns = ['date', 'market_return']
        
        # Merge with financial data
        financial_df = financial_df.merge(market_returns, on='date', how='left')
        
        # Calculate abnormal returns (stock return - market return)
        financial_df['abnormal_return'] = financial_df['log_return'] - financial_df['market_return']
        
        results = []
        
        # For each event
        for _, event in events_df.iterrows():
            event_date = event['date']
            
            # Get window around event
            for ticker in financial_df['ticker'].unique():
                ticker_data = financial_df[financial_df['ticker'] == ticker].copy()
                ticker_data = ticker_data.sort_values('date')
                
                # Find event date index
                event_idx = ticker_data[ticker_data['date'] == event_date].index
                
                if len(event_idx) == 0:
                    continue
                
                event_idx = event_idx[0]
                
                # Get window data
                start_idx = max(0, event_idx - self.window_pre)
                end_idx = min(len(ticker_data), event_idx + self.window_post + 1)
                
                window_data = ticker_data.iloc[start_idx:end_idx].copy()
                
                # Calculate relative days from event
                window_data['days_from_event'] = (
                    (window_data['date'] - event_date).dt.days
                )
                
                # Add event info
                window_data['event_id'] = event.get('event_id', 'unknown')
                window_data['event_polarity'] = event.get('polarity_mean', 0)
                
                results.append(window_data)
        
        if not results:
            logger.warning("No abnormal returns calculated")
            return pd.DataFrame()
        
        # Combine all results
        abnormal_returns_df = pd.concat(results, ignore_index=True)
        
        logger.info(f"Calculated abnormal returns for {len(abnormal_returns_df)} observations")
        
        return abnormal_returns_df
    
    def aggregate_event_impact(self, abnormal_returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate abnormal returns by days from event.
        
        Args:
            abnormal_returns_df: DataFrame with abnormal returns
        
        Returns:
            Aggregated results
        """
        logger.info("Aggregating event impact...")
        
        # Group by days from event
        aggregated = abnormal_returns_df.groupby('days_from_event').agg({
            'abnormal_return': ['mean', 'std', 'count'],
            'log_return': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            'days_from_event',
            'avg_abnormal_return', 'std_abnormal_return', 'count',
            'avg_return', 'std_return'
        ]
        
        # Calculate t-statistic
        aggregated['t_stat'] = (
            aggregated['avg_abnormal_return'] /
            (aggregated['std_abnormal_return'] / np.sqrt(aggregated['count']))
        )
        
        # Calculate p-value
        aggregated['p_value'] = 2 * (1 - stats.t.cdf(
            np.abs(aggregated['t_stat']),
            df=aggregated['count'] - 1
        ))
        
        # Mark significant
        aggregated['significant'] = aggregated['p_value'] < 0.05
        
        logger.info(f"Aggregated {len(aggregated)} time windows")
        
        return aggregated


def run_event_study():
    """Run event study analysis."""
    logger.info("Starting event study analysis...")
    
    # Load data
    events_file = settings.paths.data_processed / "tweets_processed.csv"
    if not events_file.exists():
        events_file = settings.paths.data_processed / "gdelt_processed.csv"
    
    financial_file = settings.paths.data_processed / "financial_processed.parquet"
    
    if not events_file.exists() or not financial_file.exists():
        raise FileNotFoundError("Missing preprocessed data files")
    
    # Load
    events_df = pd.read_csv(events_file) if events_file.suffix == '.csv' else pd.read_parquet(events_file)
    financial_df = pd.read_parquet(financial_file)
    
    # Aggregate events by date
    events_df['date'] = pd.to_datetime(events_df['timestamp_utc']).dt.date
    daily_events = events_df.groupby('date').agg({
        'event_id': 'first',
        'polarity': 'mean'
    }).reset_index()
    daily_events.columns = ['date', 'event_id', 'polarity_mean']
    
    # Run event study
    study = EventStudy(window_pre=5, window_post=5)
    abnormal_returns = study.calculate_abnormal_returns(daily_events, financial_df)
    
    if len(abnormal_returns) == 0:
        logger.warning("No abnormal returns to analyze")
        return
    
    # Aggregate results
    aggregated = study.aggregate_event_impact(abnormal_returns)
    
    # Save results
    output_dir = settings.paths.reports / "event_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    abnormal_returns.to_csv(output_dir / "abnormal_returns.csv", index=False)
    aggregated.to_csv(output_dir / "aggregated_impact.csv", index=False)
    
    # Generate summary
    summary = f"""# Event Study Results

## Summary Statistics

- Total events analyzed: {len(daily_events)}
- Total observations: {len(abnormal_returns)}
- Analysis window: [{-study.window_pre}, +{study.window_post}] days

## Significant Impacts

{aggregated[aggregated['significant']].to_markdown(index=False)}

## Average Abnormal Returns by Day

{aggregated[['days_from_event', 'avg_abnormal_return', 'p_value']].to_markdown(index=False)}
"""
    
    with open(output_dir / "summary.md", 'w') as f:
        f.write(summary)
    
    logger.info(f"Event study results saved to {output_dir}")
    
    return aggregated
