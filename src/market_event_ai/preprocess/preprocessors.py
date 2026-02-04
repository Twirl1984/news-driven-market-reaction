"""Data preprocessing modules."""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from textblob import TextBlob

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize text data."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing URLs, mentions, special characters."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def extract_sentiment(text: str) -> Dict[str, float]:
        """Extract sentiment scores using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}


class EventPreprocessor:
    """Preprocess event data (tweets, GDELT, etc.)."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_cleaner = TextCleaner()
    
    def preprocess_tweets(self, input_file: Path) -> Path:
        """Preprocess Trump tweets."""
        logger.info(f"Preprocessing tweets from {input_file}")
        
        with open(input_file, 'r') as f:
            tweets = json.load(f)
        
        processed = []
        for tweet in tweets:
            # Clean text
            clean_text = self.text_cleaner.clean_text(tweet['text'])
            
            # Extract sentiment
            sentiment = self.text_cleaner.extract_sentiment(clean_text)
            
            # Create processed event
            processed.append({
                'event_id': tweet['event_id'],
                'timestamp_utc': tweet['timestamp_utc'],
                'source': tweet['source'],
                'author': tweet['author'],
                'text_raw': tweet['text'],
                'text_clean': clean_text,
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity'],
                'metadata': json.dumps(tweet['metadata'])
            })
        
        # Save to CSV
        df = pd.DataFrame(processed)
        output_file = self.output_dir / "tweets_processed.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Processed {len(processed)} tweets to {output_file}")
        return output_file
    
    def preprocess_gdelt(self, input_file: Path) -> Path:
        """Preprocess GDELT events."""
        logger.info(f"Preprocessing GDELT events from {input_file}")
        
        df = pd.read_csv(input_file)
        
        # Clean text
        df['text_clean'] = df['text'].apply(self.text_cleaner.clean_text)
        
        # Extract sentiment
        sentiments = df['text_clean'].apply(self.text_cleaner.extract_sentiment)
        df['polarity'] = sentiments.apply(lambda x: x['polarity'])
        df['subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])
        
        # Parse timestamp
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        df['date'] = df['timestamp_utc'].dt.date
        
        output_file = self.output_dir / "gdelt_processed.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Processed {len(df)} GDELT events to {output_file}")
        return output_file


class FinancialDataPreprocessor:
    """Preprocess financial market data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess(self, input_file: Path) -> Path:
        """Preprocess financial data."""
        logger.info(f"Preprocessing financial data from {input_file}")
        
        df = pd.read_parquet(input_file)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Calculate returns
        df = df.sort_values(['ticker', 'timestamp'])
        df['return'] = df.groupby('ticker')['close'].pct_change()
        df['log_return'] = df.groupby('ticker')['close'].apply(
            lambda x: np.log(x / x.shift(1))
        )
        
        # Calculate volatility (20-day rolling)
        df['volatility_20d'] = df.groupby('ticker')['log_return'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        
        # Calculate moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df.groupby('ticker')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Drop NaN in critical columns
        df = df.dropna(subset=['close', 'volume'])
        
        output_file = self.output_dir / "financial_processed.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Processed {len(df)} financial records to {output_file}")
        return output_file


def preprocess_all():
    """Preprocess all data sources."""
    logger.info("Starting preprocessing of all data sources...")
    
    # Preprocess events
    event_processor = EventPreprocessor(settings.paths.data_processed)
    
    tweets_file = settings.paths.data_raw / "trump_tweets.json"
    if tweets_file.exists():
        event_processor.preprocess_tweets(tweets_file)
    
    gdelt_file = settings.paths.data_raw / "gdelt_events.csv"
    if gdelt_file.exists():
        event_processor.preprocess_gdelt(gdelt_file)
    
    # Preprocess financial data
    financial_processor = FinancialDataPreprocessor(settings.paths.data_processed)
    financial_file = settings.paths.data_raw / "financial_data.parquet"
    if financial_file.exists():
        financial_processor.preprocess(financial_file)
    
    logger.info("All preprocessing completed")


# Wrapper functions for CLI
def preprocess_tweets(input_dir: Path, output_dir: Path, clean_text: bool = True, remove_duplicates: bool = True):
    """Preprocess tweets (CLI wrapper)."""
    processor = EventPreprocessor(output_dir)
    input_file = input_dir / "trump_tweets.json"
    if input_file.exists():
        return processor.preprocess_tweets(input_file)
    else:
        logger.warning(f"Tweets file not found: {input_file}")
        return None


def preprocess_gdelt(input_dir: Path, output_dir: Path, remove_duplicates: bool = True):
    """Preprocess GDELT events (CLI wrapper)."""
    processor = EventPreprocessor(output_dir)
    input_file = input_dir / "gdelt_events.csv"
    if input_file.exists():
        return processor.preprocess_gdelt(input_file)
    else:
        logger.warning(f"GDELT file not found: {input_file}")
        return None


def preprocess_financial_data(input_dir: Path, output_dir: Path):
    """Preprocess financial data (CLI wrapper)."""
    processor = FinancialDataPreprocessor(output_dir)
    input_file = input_dir / "financial_data.parquet"
    if input_file.exists():
        return processor.preprocess(input_file)
    else:
        logger.warning(f"Financial data file not found: {input_file}")
        return None
