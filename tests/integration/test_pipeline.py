"""Integration tests for full pipeline."""
import pytest
from pathlib import Path
from market_event_ai.data.downloaders import TrumpTweetsDownloader, GDELTDownloader


@pytest.mark.integration
def test_data_download_integration(tmp_path):
    """Test that downloaders work together."""
    # Download tweets
    tweets_downloader = TrumpTweetsDownloader(tmp_path)
    tweets_file = tweets_downloader.download()
    assert tweets_file.exists()
    
    # Download GDELT
    gdelt_downloader = GDELTDownloader(tmp_path)
    gdelt_file = gdelt_downloader.download('2016-01-01', '2016-01-31')
    assert gdelt_file.exists()
