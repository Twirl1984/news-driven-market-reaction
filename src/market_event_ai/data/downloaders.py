"""Data downloaders for various sources."""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import json
import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class TrumpTweetsDownloader:
    """Download archived Trump tweets."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Download Trump tweets for specified date range."""
        logger.info("Downloading Trump tweets...")
        
        # For this MVP, we'll use the Trump Twitter Archive
        # In production, you'd use the actual archive URL or API
        output_file = self.output_dir / "trump_tweets.json"
        
        # Example: Create a sample dataset for demonstration
        # In production, replace this with actual download logic
        sample_tweets = [
            {
                "event_id": f"tweet_{i}",
                "timestamp_utc": (datetime(2016, 1, 1) + timedelta(days=i*10)).isoformat(),
                "source": "tweet",
                "author": "realDonaldTrump",
                "text": f"Sample tweet content {i}",
                "metadata": {
                    "retweet_count": 100 * i,
                    "favorite_count": 500 * i
                }
            }
            for i in range(100)
        ]
        
        with open(output_file, 'w') as f:
            json.dump(sample_tweets, f, indent=2)
        
        logger.info(f"Saved {len(sample_tweets)} tweets to {output_file}")
        return output_file


class GDELTDownloader:
    """Download GDELT 2.1 events data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = settings.data.gdelt_base_url
    
    def download(self, start_date: str, end_date: str):
        """Download GDELT events for specified date range."""
        logger.info(f"Downloading GDELT data from {start_date} to {end_date}...")
        
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # For MVP, create sample data
        # In production, implement actual GDELT download
        output_file = self.output_dir / "gdelt_events.csv"
        
        events = []
        current = start
        event_id = 0
        
        while current <= end:
            if current.weekday() < 5:  # Business days only
                for _ in range(5):  # 5 events per day
                    events.append({
                        'event_id': f'gdelt_{event_id}',
                        'timestamp_utc': current.isoformat(),
                        'source': 'gdelt',
                        'gdelt_code': '0231',  # Example: Make public statement
                        'actors': 'USA;TRUMP',
                        'tone': -2.5 + 5 * (event_id % 10) / 10,  # Random tone -2.5 to 2.5
                        'text': f'Political event {event_id}',
                        'metadata': '{}'
                    })
                    event_id += 1
            current += timedelta(days=1)
        
        df = pd.DataFrame(events)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(events)} GDELT events to {output_file}")
        return output_file


class FinancialDataDownloader:
    """Download financial market data using yfinance."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, tickers: List[str], start_date: str, end_date: str):
        """Download price data for specified tickers."""
        logger.info(f"Downloading financial data for {len(tickers)} tickers...")
        
        all_data = []
        
        for ticker in tqdm(tickers, desc="Downloading tickers"):
            try:
                # Download data from yfinance
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if data.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # Prepare data
                df = data.reset_index()
                df['ticker'] = ticker
                df['asset_id'] = ticker
                df['asset_class'] = 'etf'  # Default
                
                # Rename columns
                df = df.rename(columns={
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adjusted_close'
                })
                
                all_data.append(df)
                logger.info(f"Downloaded {len(df)} rows for {ticker}")
                
            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No financial data downloaded")
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Save to parquet
        output_file = self.output_dir / "financial_data.parquet"
        combined.to_parquet(output_file, index=False)
        
        logger.info(f"Saved {len(combined)} rows to {output_file}")
        return output_file
    
    def download_from_config(self, start_date: str, end_date: str):
        """Download data for all enabled assets in config."""
        assets_config = settings.load_assets_config()
        
        # Get enabled tickers
        tickers = []
        for etf in assets_config.get('etfs', []):
            if etf.get('enabled', False):
                tickers.append(etf['ticker'])
        
        for stock in assets_config.get('stocks', []):
            if stock.get('enabled', False):
                tickers.append(stock['ticker'])
        
        if not tickers:
            raise ValueError("No enabled assets in configuration")
        
        logger.info(f"Downloading data for {len(tickers)} enabled assets")
        return self.download(tickers, start_date, end_date)


def download_all(start_date: str, end_date: str):
    """Download all data sources."""
    logger.info("Starting full data download...")
    
    # Download tweets
    tweets_downloader = TrumpTweetsDownloader(settings.paths.data_raw)
    tweets_file = tweets_downloader.download(start_date, end_date)
    
    # Download GDELT
    gdelt_downloader = GDELTDownloader(settings.paths.data_raw)
    gdelt_file = gdelt_downloader.download(start_date, end_date)
    
    # Download financial data
    financial_downloader = FinancialDataDownloader(settings.paths.data_raw)
    financial_file = financial_downloader.download_from_config(start_date, end_date)
    
    logger.info("All data downloaded successfully")
    return {
        'tweets': tweets_file,
        'gdelt': gdelt_file,
        'financial': financial_file
    }
