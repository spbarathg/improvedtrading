# Solana Trading Bot

A professional-grade trading bot for the Solana blockchain, featuring real-time market data analysis, automated trading strategies, and risk management.

## Features

- Real-time market data processing
- Automated trading strategies
- Risk management system
- Interactive terminal UI
- Structured logging
- Performance monitoring
- Configurable trading parameters

## Prerequisites

- Python 3.9+
- Solana wallet with SOL for trading
- Access to Solana RPC endpoint

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tradingbot.git
cd tradingbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
PRIVATE_KEY=your_solana_private_key
SOLANA_RPC_URL=your_rpc_endpoint
TRADING_PAIRS=["SOL/USDC"]
MAX_POSITION_SIZE=1.0
SLIPPAGE_BPS=50
STOP_LOSS_PCT=5.0
TAKE_PROFIT_PCT=10.0
MAX_DAILY_LOSS_PCT=2.0
MAX_OPEN_TRADES=5
```

## Usage

1. Start the trading bot:
```bash
python -m trading_bot.main
```

2. Use the interactive terminal UI to:
   - Start/stop trading
   - Monitor real-time metrics
   - Adjust risk management settings
   - Configure system parameters

## Configuration

The bot can be configured through:
- Environment variables
- `.env` file
- Interactive UI settings

Key configuration parameters:
- `TRADING_PAIRS`: List of trading pairs to monitor
- `MAX_POSITION_SIZE`: Maximum position size in SOL
- `SLIPPAGE_BPS`: Maximum allowed slippage in basis points
- `STOP_LOSS_PCT`: Stop loss percentage
- `TAKE_PROFIT_PCT`: Take profit percentage
- `MAX_DAILY_LOSS_PCT`: Maximum daily loss percentage
- `MAX_OPEN_TRADES`: Maximum number of open trades

## Logging

Logs are stored in `logs/trading_bot.log` with rotation enabled. The log level can be configured through the `LOG_LEVEL` environment variable.

## Safety Features

- Automatic stop-loss execution
- Daily loss limits
- Position size limits
- Slippage protection
- Connection monitoring
- Error recovery

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Directory Structure

```
trading_bot/
├── main.py              # Main entry point and terminal UI
├── config.py            # Configuration settings
├── data_pipeline/       # Data fetching and processing
│   ├── data_fetcher.py  # Market data collection
│   ├── data_storage.py  # Data persistence
│   └── feature_extractor.py  # Feature calculation
├── trading/            # Trading components
│   ├── exchange.py     # Solana exchange interface
│   ├── order_manager.py  # Order handling
│   └── risk_manager.py   # Risk management
└── utils/              # Utility functions
    ├── logger.py       # Logging setup
    ├── key_manager.py  # Key management
    └── helpers.py      # Helper functions
```

## Monitoring and Maintenance

1. **Logs**: Check `logs/trading_bot.log` for detailed activity logs
2. **Health Checks**: The bot performs automatic health checks every minute
3. **Performance Metrics**: Monitor trading performance through the terminal interface
4. **Error Handling**: The bot automatically retries failed operations with exponential backoff

## Security Considerations

1. **Private Key Management**:
   - Store your private key securely
   - Never share your private key
   - Use environment variables for sensitive data

2. **Risk Management**:
   - Start with small position sizes
   - Monitor stop-loss and take-profit levels
   - Regularly review trading parameters

3. **Network Security**:
   - Use secure RPC endpoints
   - Monitor for suspicious activity
   - Keep the bot updated with latest security patches

## Troubleshooting

1. **Connection Issues**:
   - Check your internet connection
   - Verify RPC endpoint availability
   - Review firewall settings

2. **Trading Errors**:
   - Check wallet balance
   - Verify transaction signatures
   - Review error logs

3. **Performance Issues**:
   - Monitor system resources
   - Check for memory leaks
   - Review data pipeline efficiency

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. The developers are not responsible for any financial losses incurred through the use of this software. 