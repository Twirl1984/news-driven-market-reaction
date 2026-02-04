"""Backtesting engine for trading strategy."""
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import json
import joblib

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class Portfolio:
    """Portfolio simulator with realistic costs."""
    
    def __init__(
        self,
        initial_capital: float,
        transaction_cost_bps: float,
        slippage_bps: float
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            transaction_cost_bps: Transaction cost in basis points
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps / 10000  # Convert to decimal
        self.slippage_bps = slippage_bps / 10000
        
        self.cash = initial_capital
        self.positions = {}  # ticker -> shares
        self.equity_curve = []
        self.trades = []
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            shares * prices.get(ticker, 0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + position_value
    
    def execute_trade(
        self,
        date: pd.Timestamp,
        ticker: str,
        signal: str,
        price: float,
        target_pct: float = 1.0
    ):
        """
        Execute a trade based on signal.
        
        Args:
            date: Trade date
            ticker: Ticker symbol
            signal: 'LONG' or 'FLAT'
            price: Current price
            target_pct: Target position as % of portfolio (0-1)
        """
        current_value = self.get_total_value({ticker: price})
        current_shares = self.positions.get(ticker, 0)
        current_position_value = current_shares * price
        
        if signal == 'LONG':
            # Target position
            target_value = current_value * target_pct
            trade_value = target_value - current_position_value
            
            if abs(trade_value) < 100:  # Minimum trade size
                return
            
            # Calculate shares to trade
            if trade_value > 0:  # Buy
                # Apply costs
                effective_price = price * (1 + self.slippage_bps)
                cost = trade_value * self.transaction_cost_bps
                shares_to_buy = (trade_value - cost) / effective_price
                
                if shares_to_buy * effective_price > self.cash:
                    shares_to_buy = self.cash / effective_price * 0.99  # Leave some cash
                
                if shares_to_buy > 0:
                    total_cost = shares_to_buy * effective_price + cost
                    self.cash -= total_cost
                    self.positions[ticker] = self.positions.get(ticker, 0) + shares_to_buy
                    
                    self.trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': effective_price,
                        'value': total_cost,
                        'signal': signal
                    })
            
            elif trade_value < 0:  # Sell some
                effective_price = price * (1 - self.slippage_bps)
                shares_to_sell = min(abs(trade_value) / effective_price, current_shares)
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * effective_price
                    cost = proceeds * self.transaction_cost_bps
                    self.cash += proceeds - cost
                    self.positions[ticker] -= shares_to_sell
                    
                    if self.positions[ticker] < 0.01:  # Close position if too small
                        self.positions.pop(ticker, None)
                    
                    self.trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': effective_price,
                        'value': proceeds,
                        'signal': signal
                    })
        
        elif signal == 'FLAT':
            # Close position if exists
            if ticker in self.positions and current_shares > 0:
                effective_price = price * (1 - self.slippage_bps)
                proceeds = current_shares * effective_price
                cost = proceeds * self.transaction_cost_bps
                self.cash += proceeds - cost
                shares_sold = self.positions.pop(ticker)
                
                self.trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_sold,
                    'price': effective_price,
                    'value': proceeds,
                    'signal': signal
                })


class Backtester:
    """Backtest trading strategy."""
    
    def __init__(
        self,
        model_path: Path,
        initial_capital: float = None,
        transaction_cost_bps: float = None,
        slippage_bps: float = None
    ):
        """
        Initialize backtester.
        
        Args:
            model_path: Path to trained model directory
            initial_capital: Starting capital
            transaction_cost_bps: Transaction cost in basis points
            slippage_bps: Slippage in basis points
        """
        self.model = joblib.load(model_path / "model.joblib")
        
        with open(model_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        self.initial_capital = initial_capital or settings.trading.initial_capital
        self.transaction_cost_bps = transaction_cost_bps or settings.trading.trading_cost_bps
        self.slippage_bps = slippage_bps or settings.trading.slippage_bps
    
    def run(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on data.
        
        Args:
            data: Labeled data with features
        
        Returns:
            Dict with results
        """
        logger.info(f"Running backtest on {len(data)} samples...")
        
        # Sort by date
        data = data.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Prepare features
        feature_names = self.metadata['feature_names']
        X = data[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Generate signals
        predictions = self.model.predict(X)
        data['predicted_label'] = predictions
        data['predicted_signal'] = data['predicted_label'].map({1: 'LONG', 0: 'FLAT'})
        
        # Initialize portfolio
        portfolio = Portfolio(
            self.initial_capital,
            self.transaction_cost_bps,
            self.slippage_bps
        )
        
        # Simulate trading
        unique_dates = data['date'].unique()
        
        for date in unique_dates:
            day_data = data[data['date'] == date]
            
            # Get current prices
            prices = dict(zip(day_data['ticker'], day_data['close']))
            
            # Execute trades for each ticker
            for _, row in day_data.iterrows():
                portfolio.execute_trade(
                    date=row['date'],
                    ticker=row['ticker'],
                    signal=row['predicted_signal'],
                    price=row['close'],
                    target_pct=1.0  # 100% in single asset (can be modified for multi-asset)
                )
            
            # Record equity
            total_value = portfolio.get_total_value(prices)
            portfolio.equity_curve.append({
                'date': date,
                'equity': total_value,
                'cash': portfolio.cash,
                'positions_value': total_value - portfolio.cash
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics(portfolio, data)
        
        return {
            'portfolio': portfolio,
            'metrics': metrics,
            'predictions': data[['date', 'ticker', 'predicted_signal', 'close', 'future_return']]
        }
    
    def calculate_metrics(self, portfolio: Portfolio, data: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        logger.info("Calculating performance metrics...")
        
        equity_df = pd.DataFrame(portfolio.equity_curve)
        
        if len(equity_df) == 0:
            return {}
        
        # Returns
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # CAGR
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        total_return = (equity_df['equity'].iloc[-1] / portfolio.initial_capital) - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe Ratio (assuming 252 trading days)
        returns = equity_df['return'].dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Max Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Hit Rate
        winning_trades = [t for t in portfolio.trades if t['action'] == 'SELL']
        # Simplified hit rate calculation
        hit_rate = 0.5  # Placeholder
        
        # Turnover
        total_traded = sum(abs(t['value']) for t in portfolio.trades)
        avg_equity = equity_df['equity'].mean()
        turnover = total_traded / avg_equity if avg_equity > 0 else 0
        
        # Buy & Hold benchmark
        buy_hold_return = self.calculate_buy_hold(data)
        
        metrics = {
            'cagr': cagr,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'turnover': turnover,
            'num_trades': len(portfolio.trades),
            'final_equity': equity_df['equity'].iloc[-1],
            'buy_hold_return': buy_hold_return
        }
        
        logger.info(f"Backtest metrics: {metrics}")
        
        return metrics
    
    def calculate_buy_hold(self, data: pd.DataFrame) -> float:
        """Calculate buy and hold return."""
        # Use first ticker as benchmark
        ticker = data['ticker'].iloc[0]
        ticker_data = data[data['ticker'] == ticker].sort_values('date')
        
        if len(ticker_data) == 0:
            return 0
        
        initial_price = ticker_data['close'].iloc[0]
        final_price = ticker_data['close'].iloc[-1]
        
        return (final_price - initial_price) / initial_price


def run_backtest(model_type: str = None):
    """
    Main backtesting pipeline.
    
    Args:
        model_type: Type of model to backtest
    """
    logger.info("Starting backtest...")
    
    # Use model type from settings if not provided
    if model_type is None:
        model_type = settings.model.model_type
    
    # Load model
    model_dir = settings.paths.models / model_type
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")
    
    # Load test data
    data_file = settings.paths.data_labels / "labeled_data_classification.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Labeled data not found: {data_file}")
    
    data = pd.read_parquet(data_file)
    
    # Run backtest
    backtester = Backtester(model_dir)
    results = backtester.run(data)
    
    # Save results
    output_dir = settings.paths.backtests / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save equity curve
    equity_df = pd.DataFrame(results['portfolio'].equity_curve)
    equity_df.to_csv(output_dir / "equity_curve.csv", index=False)
    
    # Save trades
    trades_df = pd.DataFrame(results['portfolio'].trades)
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    logger.info(f"Backtest results saved to {output_dir}")
    
    return results
