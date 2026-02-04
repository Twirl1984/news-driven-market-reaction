"""Test preprocessors."""
import pytest
from market_event_ai.preprocess.preprocessors import TextCleaner


def test_text_cleaner():
    """Test text cleaning."""
    cleaner = TextCleaner()
    
    # Test URL removal
    text = "Check this out http://example.com great!"
    cleaned = cleaner.clean_text(text)
    assert 'http' not in cleaned
    assert 'example.com' not in cleaned
    
    # Test mention removal
    text = "Hey @user this is cool"
    cleaned = cleaner.clean_text(text)
    assert '@' not in cleaned
    
    # Test hashtag
    text = "This is #awesome"
    cleaned = cleaner.clean_text(text)
    assert 'awesome' in cleaned


def test_sentiment_extraction():
    """Test sentiment extraction."""
    cleaner = TextCleaner()
    
    # Positive text
    sentiment = cleaner.extract_sentiment("This is wonderful and amazing!")
    assert sentiment['polarity'] > 0
    
    # Negative text
    sentiment = cleaner.extract_sentiment("This is terrible and awful!")
    assert sentiment['polarity'] < 0
