import asyncio
import signal
import sys
import curses
import logging
import random
from datetime import datetime
from typing import Dict, Set, Type, Optional
from trading_bot.config import config
from trading_bot.data_pipeline.data_fetcher import DataFetcher
from trading_bot.data_pipeline.data_storage import DataStorage
from trading_bot.data_pipeline.feature_extractor import FeatureExtractor
from trading_bot.trading.exchange import Exchange
from trading_bot.trading.order_manager import OrderManager
from trading_bot.trading.risk_manager import RiskManager
from trading_bot.logging_config import setup_logging
from trading_bot.utils.key_manager import KeyManager
from trading_bot.utils.helpers import setup_directories
from prometheus_client import Gauge
from ai_model.integration import AIModelIntegration
from collections import deque
from pathlib import Path
from trading_bot.utils.logger import setup_logger

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize logging configuration
setup_logging(
    log_level=config.LOG_LEVEL,
    log_file=config.LOG_FILE
)

class TerminalUI:
    def __init__(self, bot):
        self.bot = bot
        self.screen = None
        self.running = True
        self.current_menu = 'main'
        self.selected_option = 0
        self.input_timeout = 100  # ms
        self._cleanup_tasks: Set[asyncio.Task] = set()
        
        # Dynamic configuration binding
        self.menu_options = {
            'main': [
                'Start Trading Bot',
                'Real-time Metrics',
                'Risk Management Settings',
                'System Configuration',
                'Exit'
            ],
            'settings': [
                *list(config.SETTINGS_MAP.keys()),
                'Back to Main Menu'
            ]
        }
        
        # Config type validation mapping
        self.setting_types: Dict[str, Type] = {
            key: typ for key, (_, typ) in config.SETTINGS_MAP.items()
        }

    def _create_task(self, coro):
        """Safe task creation with automatic cleanup"""
        task = asyncio.create_task(coro)
        self._cleanup_tasks.add(task)
        task.add_done_callback(lambda t: self._cleanup_tasks.discard(t))
        return task

    def draw_interface(self):
        """Optimized interface rendering with layout templates"""
        try:
            self.screen.clear()
            # Draw border
            self.screen.border()
            # Draw header
            header = "Solana Trading Bot - Professional Trading Automation"
            self.screen.addstr(1, (curses.COLS - len(header)) // 2, header, curses.A_BOLD)
            # Draw status line
            status = f"▲ Uptime: {self.bot.uptime} | ▼ Latency: {self.bot.latency:.2f}ms"
            self.screen.addstr(curses.LINES-2, 2, status, curses.color_pair(2))
            
            if self.current_menu == 'main':
                self._draw_main_menu()
            elif self.current_menu == 'settings':
                self._draw_settings_menu()
                
            self.screen.refresh()
        except curses.error as e:
            logger.error(f"Rendering error: {e}")

    def _draw_main_menu(self):
        """Optimized main menu rendering"""
        start_y = 5
        for idx, option in enumerate(self.menu_options['main']):
            attr = curses.A_REVERSE if idx == self.selected_option else curses.A_NORMAL
            self.screen.addstr(start_y + idx, 4, f"{'▶' if idx == self.selected_option else ' '} {option}", attr)

    def _draw_settings_menu(self):
        """Dynamic settings menu with live values"""
        start_y = 5
        for idx, (setting, (attr_name, _)) in enumerate(config.SETTINGS_MAP.items()):
            current_value = getattr(config, attr_name)
            attr = curses.A_REVERSE if idx == self.selected_option else curses.A_NORMAL
            text = f"{'▶' if idx == self.selected_option else ' '} {setting}: {current_value}"
            self.screen.addstr(start_y + idx, 4, text, attr)

    async def handle_realtime_input(self):
        """Non-blocking input handling with debouncing"""
        try:
            key = self.screen.getch()
            if key != -1:
                if key in (curses.KEY_UP, ord('k')):
                    self.selected_option = max(0, self.selected_option - 1)
                elif key in (curses.KEY_DOWN, ord('j')):
                    self.selected_option = min(len(self.menu_options[self.current_menu])-1, self.selected_option + 1)
                elif key == ord('\n'):
                    self._handle_menu_selection()
                elif key == ord('q'):
                    self.running = False
        except curses.error as e:
            logger.error(f"Input error: {e}")

    def _handle_menu_selection(self):
        """Optimized menu selection handler"""
        if self.current_menu == 'main':
            menu_actions = [
                lambda: self._create_task(self.bot.start()),
                lambda: self._show_realtime_metrics(),
                lambda: setattr(self, 'current_menu', 'settings'),
                lambda: self._show_system_config(),
                lambda: setattr(self, 'running', False)
            ]
            menu_actions[self.selected_option]()
        elif self.current_menu == 'settings':
            if self.selected_option == len(config.SETTINGS_MAP):
                self.current_menu = 'main'
            else:
                self._modify_setting()

    def _modify_setting(self):
        """Type-safe setting modification with validation"""
        setting_name, (attr_name, typ) = list(config.SETTINGS_MAP.items())[self.selected_option]
        current_value = getattr(config, attr_name)
        
        self.screen.addstr(10, 4, f"Modify {setting_name} ({typ.__name__}): ", curses.A_BOLD)
        curses.echo()
        try:
            input_str = self.screen.getstr(11, 4).decode('utf-8')
            validated = self._validate_input(input_str, typ)
            setattr(config, attr_name, validated)
            logger.info(f"Updated setting {attr_name} to {validated}")
        except ValueError as e:
            self.screen.addstr(12, 4, f"Invalid input: {e}", curses.color_pair(3))
            self.screen.getch()
        finally:
            curses.noecho()

    def _validate_input(self, value: str, typ: Type):
        """Type validation with error handling"""
        try:
            if typ == bool:
                return value.lower() in ('true', '1', 'yes')
            return typ(value)
        except ValueError:
            raise ValueError(f"Invalid {typ.__name__} value: {value}")

    async def run_ui(self):
        """Optimized main UI loop"""
        try:
            self.screen = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.curs_set(0)
            self.screen.timeout(self.input_timeout)

            while self.running:
                self.draw_interface()
                await self.handle_realtime_input()
                await asyncio.sleep(0)  # Yield control
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Comprehensive resource cleanup"""
        try:
            if self.screen:
                curses.endwin()
            
            # Cancel all UI tasks
            for task in self._cleanup_tasks:
                task.cancel()
            
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            
            # Ensure bot shutdown
            if self.bot.running:
                await self.bot.stop()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

class TradingBot:
    def __init__(self, config_path: str):
        """Initialize trading bot with configuration."""
        self.config = config
        self.setup_logging()
        
        logger.info("Initializing trading bot")
        self.exchange = Exchange(self.config.EXCHANGE)
        self.order_manager = OrderManager(self.config.ORDER, self.exchange)
        self.risk_manager = RiskManager(self.config.RISK)
        self.data_fetcher = DataFetcher(self.config.DATA)
        
        self.running = False
        self.tasks = []
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("Trading bot initialized successfully")

    def setup_logging(self):
        """Configure logging system."""
        log_dir = Path(self.config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"trading_bot_{datetime.now():%Y%m%d_%H%M%S}.log"
        setup_logger(
            log_level=self.config.LOG_LEVEL,
            log_file=str(log_file)
        )
        logger.info("Logging system configured",
                   extra={
                       'log_level': self.config.LOG_LEVEL,
                       'log_file': str(log_file)
                   })

    async def start(self):
        """Start the trading bot."""
        try:
            logger.info("Starting trading bot")
            self.running = True
            
            # Connect to exchange
            if not await self.exchange.connect():
                logger.error("Failed to connect to exchange")
                return
            
            # Start background tasks
            self.tasks = [
                asyncio.create_task(self._market_data_loop()),
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._risk_check_loop())
            ]
            
            logger.info("Trading bot started successfully")
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error("Error starting trading bot",
                        exc_info=True,
                        extra={'error': str(e)})
            await self.stop()

    async def stop(self):
        """Stop the trading bot."""
        try:
            logger.info("Stopping trading bot")
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Cleanup connections
            await self.exchange.disconnect()
            
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping trading bot",
                        exc_info=True,
                        extra={'error': str(e)})

    def _handle_signal(self, signum, frame):
        """Handle system signals."""
        logger.info("Received shutdown signal",
                   extra={'signal': signum})
        asyncio.create_task(self.stop())

    async def _market_data_loop(self):
        """Background task for market data collection."""
        try:
            logger.info("Starting market data collection")
            while self.running:
                try:
                    for symbol in self.config.SYMBOLS:
                        data = await self.exchange.get_market_data(symbol)
                        if data:
                            await self.data_fetcher.store_market_data(data)
                    
                    await asyncio.sleep(self.config.DATA_INTERVAL)
                    
                except Exception as e:
                    logger.error("Error in market data loop",
                               exc_info=True,
                               extra={'error': str(e)})
                    await asyncio.sleep(self.config.ERROR_RETRY_DELAY)
                    
        except asyncio.CancelledError:
            logger.info("Market data collection stopped")

    async def _trading_loop(self):
        """Background task for trading logic."""
        try:
            logger.info("Starting trading loop")
            while self.running:
                try:
                    # Process each configured symbol
                    for symbol in self.config.SYMBOLS:
                        # Get latest market data
                        data = await self.data_fetcher.get_latest_market_data(symbol)
                        if not data:
                            continue
                        
                        # Generate trading signals
                        signal = await self._generate_trading_signal(data)
                        if signal:
                            # Execute trade
                            await self._execute_trade(signal)
                    
                    await asyncio.sleep(self.config.TRADING_INTERVAL)
                    
                except Exception as e:
                    logger.error("Error in trading loop",
                               exc_info=True,
                               extra={'error': str(e)})
                    await asyncio.sleep(self.config.ERROR_RETRY_DELAY)
                    
        except asyncio.CancelledError:
            logger.info("Trading loop stopped")

    async def _risk_check_loop(self):
        """Background task for risk monitoring."""
        try:
            logger.info("Starting risk monitoring")
            while self.running:
                try:
                    # Check risk limits
                    if not await self.risk_manager.check_risk_limits():
                        logger.warning("Risk limits breached, initiating risk mitigation")
                        await self._handle_risk_breach()
                    
                    await asyncio.sleep(self.config.RISK_CHECK_INTERVAL)
                    
                except Exception as e:
                    logger.error("Error in risk check loop",
                               exc_info=True,
                               extra={'error': str(e)})
                    await asyncio.sleep(self.config.ERROR_RETRY_DELAY)
                    
        except asyncio.CancelledError:
            logger.info("Risk monitoring stopped")

    async def _generate_trading_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal from market data."""
        try:
            # Implementation of trading signal generation
            return None
        except Exception as e:
            logger.error("Error generating trading signal",
                        exc_info=True,
                        extra={
                            'error': str(e),
                            'market_data': market_data
                        })
            return None

    async def _execute_trade(self, signal: Dict):
        """Execute a trade based on signal."""
        try:
            # Check risk limits
            if not await self.risk_manager.check_trade(signal):
                logger.warning("Trade rejected by risk manager",
                             extra={'signal': signal})
                return
            
            # Place order
            order_id = await self.order_manager.place_order(signal)
            if order_id:
                logger.info("Trade executed successfully",
                           extra={
                               'order_id': order_id,
                               'signal': signal
                           })
            
        except Exception as e:
            logger.error("Error executing trade",
                        exc_info=True,
                        extra={
                            'error': str(e),
                            'signal': signal
                        })

    async def _handle_risk_breach(self):
        """Handle risk limit breaches."""
        try:
            # Cancel all active orders
            active_orders = self.order_manager.get_active_orders()
            for order in active_orders:
                await self.order_manager.cancel_order(order['order_id'])
            
            logger.info("Risk breach handled",
                       extra={'cancelled_orders': len(active_orders)})
            
        except Exception as e:
            logger.error("Error handling risk breach",
                        exc_info=True,
                        extra={'error': str(e)})

def main():
    """Main entry point for the trading bot."""
    try:
        config_path = "config.yaml"  # or get from command line args
        bot = TradingBot(config_path)
        
        # Run the bot
        asyncio.run(bot.start())
        
    except Exception as e:
        logger.error("Fatal error in trading bot",
                    exc_info=True,
                    extra={'error': str(e)})
        raise

if __name__ == "__main__":
    main()