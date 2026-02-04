"""Test data downloaders."""
import pytest
from pathlib import Path
from market_event_ai.data.downloaders import TrumpTweetsDownloader, GDELTDownloader


def test_trump_tweets_downloader(tmp_path):
    """Test Trump tweets downloader."""
    downloader = TrumpTweetsDownloader(tmp_path)
    output_file = downloader.download()
    
    assert output_file.exists()
    assert output_file.suffix == '.json'


def test_gdelt_downloader(tmp_path):
    """Test GDELT downloader."""
    downloader = GDELTDownloader(tmp_path)
    output_file = downloader.download('2016-01-01', '2016-01-31')
    
    assert output_file.exists()
    assert output_file.suffix == '.csv'
