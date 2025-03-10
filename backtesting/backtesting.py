#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Type, Any
import importlib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add parent directory to Python path to import from strategies
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BacktestConfig:
    def __init__(self, **kwargs):
        self.initial_capital = float(kwargs.get('initial_capital', 10000))
        self.commission = float(kwargs.get('commission', 0.001))
        self.slippage = float(kwargs.get('slippage', 0.001))
        self.position_size = float(kwargs.get('position_size', 0.1))
        self.stop_loss = float(kwargs.get('stop_loss', 0.02))
        self.take_profit = float(kwargs.get('take_profit', 0.05))

class Trade:
    def __init__(self, entry_time: datetime, entry_price: float, position_size: float, side: str):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.position_size = position_size
        self.side = side  # 'long' or 'short'
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0
        self.status = 'open'

    def close(self, exit_time: datetime, exit_price: float):
        self.exit_time = exit_time
        self.exit_price = exit_price
        multiplier = 1 if self.side == 'long' else -1
        self.pnl = (self.exit_price - self.entry_price) * self.position_size * multiplier
        self.status = 'closed'

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(trades: List[Trade], equity_curve: pd.Series) -> Dict[str, float]:
        if not trades:
            return {}

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        total_pnl = sum(t.pnl for t in trades)
        
        # Returns
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        
        # Risk metrics
        daily_returns = returns * 100
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_trades': total_trades,
            'profit_factor': (
                sum(t.pnl for t in trades if t.pnl > 0) /
                abs(sum(t.pnl for t in trades if t.pnl < 0))
                if sum(t.pnl for t in trades if t.pnl < 0) != 0 else float('inf')
            ),
            'average_trade': total_pnl / total_trades if total_trades > 0 else 0,
        }

class Backtester:
    def __init__(self, config: BacktestConfig, strategy_class: Type):
        self.config = config
        self.strategy = strategy_class(config.__dict__)
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_equity = config.initial_capital
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate that the data contains all required columns."""
        required_columns = self.strategy.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the backtest on the provided data."""
        logger.info("Starting backtest...")
        
        # Validate data
        self._validate_data(data)
        
        # Validate strategy parameters
        if not self.strategy.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        try:
            # Initialize strategy with historical data
            self.strategy.initialize(data)
            
            for i in range(len(data)):
                current_bar = data.iloc[i]
                
                # Update strategy
                signal = self.strategy.generate_signal(current_bar)
                
                # Process open positions
                self._process_open_positions(current_bar)
                
                # Process new signals
                if signal != 0:
                    self._process_signal(signal, current_bar)
                
                # Update equity curve
                self.equity_curve.append(self.current_equity)
            
            # Calculate performance metrics
            equity_series = pd.Series(self.equity_curve, index=data.index)
            metrics = PerformanceMetrics.calculate_metrics(self.trades, equity_series)
            
            logger.info("Backtest completed successfully")
            return {
                'metrics': metrics,
                'equity_curve': equity_series,
                'trades': self.trades,
                'indicators': self.strategy.indicators
            }
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise

    def _process_signal(self, signal: int, bar: pd.Series):
        """Process a new trading signal."""
        position_value = self.current_equity * self.config.position_size
        shares = position_value / bar['close']
        
        if signal > 0:  # Long signal
            trade = Trade(bar.name, bar['close'], shares, 'long')
            self.trades.append(trade)
            logger.info(f"Opened long position: {shares} shares at {bar['close']}")
            
        elif signal < 0:  # Short signal
            trade = Trade(bar.name, bar['close'], shares, 'short')
            self.trades.append(trade)
            logger.info(f"Opened short position: {shares} shares at {bar['close']}")

    def _process_open_positions(self, bar: pd.Series):
        """Process all open positions for stop loss and take profit."""
        for trade in [t for t in self.trades if t.status == 'open']:
            # Calculate current P&L
            multiplier = 1 if trade.side == 'long' else -1
            unrealized_pnl = (bar['close'] - trade.entry_price) * trade.position_size * multiplier
            pnl_percentage = unrealized_pnl / (trade.entry_price * trade.position_size)
            
            # Check stop loss and take profit
            if (pnl_percentage <= -self.config.stop_loss or 
                pnl_percentage >= self.config.take_profit):
                trade.close(bar.name, bar['close'])
                self.current_equity += trade.pnl
                logger.info(f"Closed {trade.side} position: P&L = {trade.pnl:.2f}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Strategy Backtester')
    parser.add_argument('--strategy', type=str, required=True, help='Name of the strategy class to use')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-file', type=str, required=True, help='Path to historical data CSV file')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--position-size', type=float, default=0.1, help='Position size as fraction of equity')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)
        
    return args

def load_strategy(strategy_name: str) -> Type:
    """Dynamically load the strategy class."""
    try:
        module_path = f"strategies.{strategy_name.lower()}"
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, strategy_name)
        return strategy_class
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load strategy {strategy_name}: {e}")
        sys.exit(1)

def load_data(file_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load and validate historical data."""
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp and filter by date range
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        data = data.loc[start_date:end_date]
        
        if len(data) == 0:
            raise ValueError("No data found for the specified date range")
        
        return data
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def main():
    """Main function to run the backtester."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Load strategy
    strategy_class = load_strategy(args.strategy)
    
    # Load historical data
    data = load_data(args.data_file, args.start_date, args.end_date)
    
    # Create config
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission=args.commission,
        position_size=args.position_size
    )
    
    # Initialize and run backtester
    backtester = Backtester(config, strategy_class)
    results = backtester.run(data)
    
    # Print results
    print("\nBacktest Results:")
    print("================")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Save results to CSV
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"backtest_results_{timestamp}.csv"
    
    pd.DataFrame({
        'timestamp': results['equity_curve'].index,
        'equity': results['equity_curve'].values
    }).to_csv(results_file)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
