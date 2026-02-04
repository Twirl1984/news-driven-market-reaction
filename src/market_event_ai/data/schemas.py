"""Data schemas for Market Event AI."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Event:
    """Political event schema."""
    event_id: str
    timestamp_utc: datetime
    source: str  # 'tweet', 'gdelt', 'executive_order', 'media'
    author: Optional[str]
    text: str
    metadata: Dict[str, Any]
    
    # GDELT specific
    gdelt_code: Optional[str] = None
    actors: Optional[str] = None
    tone: Optional[float] = None


@dataclass
class FinancialData:
    """Financial market data schema."""
    asset_id: str
    ticker: str
    asset_class: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
