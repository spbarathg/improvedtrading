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
from trading_bot.utils.logger import setup_logger
from trading_bot.utils.key_manager import KeyManager
from trading_bot.utils.helpers import setup_directories
from prometheus_client import Gauge
from ai_model.integration import AIModelIntegration
from collections import deque

# Initialize logger with proper configuration
logger = setup_logger(
    log_level=config.LOG_LEVEL,
    log_file=config.LOG_FILE,
    json_output=True,
    async_logging=True,
    colorize_console=True
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
    def __init__(self):
        self.config = config
        self.running = False
        self.tasks: Set[asyncio.Task] = set()
        self.start_time: Optional[datetime] = None
        self.latency: float = 0.0
        self.uptime: str = "0:00:00"
        
        # Initialize components with proper dependency injection and connection pooling
        self.exchange = Exchange(config)
        self.data_storage = DataStorage(config)
        self.data_fetcher = DataFetcher(config, self.data_storage)
        self.risk_manager = RiskManager(config)
        self.order_manager = OrderManager(config, self.exchange, self.risk_manager)
        
        # Initialize AI model integration with performance tracking
        self.ai_model = AIModelIntegration(config)
        self.model_performance = {
            'accuracy': deque(maxlen=100),
            'latency': deque(maxlen=100),
            'predictions': deque(maxlen=1000)
        }
        
        # Initialize metrics collection with more detailed tracking
        self.metrics = {
            'uptime': Gauge('bot_uptime', 'Bot uptime in seconds'),
            'latency': Gauge('bot_latency', 'Current latency in ms'),
            'active_tasks': Gauge('bot_active_tasks', 'Number of active tasks'),
            'memory_usage': Gauge('bot_memory_usage', 'Memory usage in MB'),
            'model_accuracy': Gauge('model_accuracy', 'Current model accuracy'),
            'model_latency': Gauge('model_latency', 'Model prediction latency'),
            'api_calls': Gauge('api_calls', 'Number of API calls'),
            'trade_success_rate': Gauge('trade_success_rate', 'Trade success rate'),
            'profit_loss': Gauge('profit_loss', 'Current profit/loss')
        }
        
        # Resource management
        self.resource_limits = {
            'max_concurrent_tasks': self._calculate_optimal_tasks(),
            'max_memory_usage': self._calculate_max_memory(),
            'max_api_calls_per_minute': config.MAX_API_CALLS_PER_MINUTE
        }
        
        # Performance monitoring
        self.performance_stats = {
            'api_calls': deque(maxlen=60),  # Track last minute
            'execution_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=60)
        }
        
        # Initialize connection pools
        self._init_connection_pools()

    def _init_connection_pools(self):
        """Initialize connection pools for better resource management"""
        import aiohttp
        self.http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=self.resource_limits['max_concurrent_tasks'],
                ttl_dns_cache=300
            )
        )

    def _calculate_optimal_tasks(self):
        """Calculate optimal number of concurrent tasks based on system resources"""
        import psutil
        cpu_count = psutil.cpu_count()
        return min(cpu_count * 2, self.config.MAX_CONCURRENT_TASKS)

    def _calculate_max_memory(self):
        """Calculate maximum allowed memory usage"""
        import psutil
        total_memory = psutil.virtual_memory().total
        return int(total_memory * 0.8)  # Use up to 80% of available memory

    async def _critical_health_check(self):
        """Enhanced real-time critical system monitoring"""
        while self.running:
            try:
                # System resource monitoring
                memory_usage = self._get_memory_usage()
                if memory_usage > self.resource_limits['max_memory_usage']:
                    logger.warning(f"High memory usage: {memory_usage/1024/1024:.2f}MB")
                    await self._optimize_memory_usage()
                
                # Exchange connectivity with circuit breaker
                if not await self._check_exchange_with_circuit_breaker():
                    continue
                
                # Data pipeline health
                if not await self.data_storage.check_connection():
                    logger.critical("Data storage connection failed!")
                    await self.stop()
                
                # API rate limiting check
                api_calls_last_minute = sum(self.performance_stats['api_calls'])
                if api_calls_last_minute >= self.resource_limits['max_api_calls_per_minute']:
                    logger.warning("API rate limit approaching, throttling requests")
                    await self._throttle_api_calls()
                
                # Performance monitoring
                await self._update_performance_metrics()
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Critical health check failed: {e}")

    async def _check_exchange_with_circuit_breaker(self):
        """Implement circuit breaker pattern for exchange connectivity"""
        MAX_FAILURES = 3
        RESET_TIMEOUT = 60
        
        if not hasattr(self, '_exchange_failures'):
            self._exchange_failures = 0
            self._last_failure = None
        
        try:
            if self._exchange_failures >= MAX_FAILURES:
                if (datetime.now() - self._last_failure).total_seconds() < RESET_TIMEOUT:
                    logger.error("Circuit breaker open, skipping exchange check")
                    return False
                self._exchange_failures = 0
            
            if not await self.exchange.check_connection():
                self._exchange_failures += 1
                self._last_failure = datetime.now()
                logger.warning(f"Exchange connection failed ({self._exchange_failures}/{MAX_FAILURES})")
                await self.exchange.reconnect()
                return False
            
            self._exchange_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"Exchange circuit breaker error: {e}")
            return False

    async def _optimize_memory_usage(self):
        """Optimize memory usage when approaching limits"""
        try:
            # Clear unnecessary caches
            self.model_performance['predictions'].clear()
            
            # Trim historical data
            if len(self.performance_stats['execution_times']) > 100:
                self.performance_stats['execution_times'] = deque(
                    list(self.performance_stats['execution_times'])[-100:],
                    maxlen=1000
                )
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics for monitoring"""
        try:
            # Update memory usage
            self.performance_stats['memory_usage'].append(self._get_memory_usage())
            
            # Update metrics
            self.metrics['memory_usage'].set(self._get_memory_usage() / 1024 / 1024)
            self.metrics['active_tasks'].set(len(self.tasks))
            self.metrics['api_calls'].set(sum(self.performance_stats['api_calls']))
            
            # Calculate and update model metrics
            if self.model_performance['accuracy']:
                avg_accuracy = sum(self.model_performance['accuracy']) / len(self.model_performance['accuracy'])
                self.metrics['model_accuracy'].set(avg_accuracy)
            
            if self.model_performance['latency']:
                avg_latency = sum(self.model_performance['latency']) / len(self.model_performance['latency'])
                self.metrics['model_latency'].set(avg_latency)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _get_memory_usage(self):
        """Get current memory usage in bytes"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss

    async def _throttle_api_calls(self):
        """Implement API call throttling"""
        await asyncio.sleep(60)  # Basic throttling
        self.performance_stats['api_calls'].clear()

    async def _background_services(self):
        """Non-critical background tasks"""
        while self.running:
            try:
                # Update uptime
                if self.start_time:
                    delta = datetime.now() - self.start_time
                    self.uptime = f"{delta.seconds//3600}:{(delta.seconds//60)%60:02}:{delta.seconds%60:02}"
                
                # Latency monitoring
                self.latency = await self.exchange.measure_latency()
                
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Background service error: {e}")

    async def start(self):
        """Start the trading bot with AI model integration."""
        try:
            if self.running:
                logger.warning("Trading bot is already running")
                return

            self.running = True
            self.start_time = datetime.now()
            
            # Initialize AI model
            await self.ai_model.initialize()
            
            # Start core services
            core_tasks = [
                self._create_task(self._critical_health_check()),
                self._create_task(self._background_services()),
                self._create_task(self._market_data_processor()),
                self._create_task(self._trading_loop())
            ]
            
            await asyncio.gather(*core_tasks)
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            await self.stop()
            raise

    async def _market_data_processor(self):
        """Process market data and update AI model."""
        while self.running:
            try:
                # Fetch latest market data
                market_data = await self.data_fetcher.get_latest_data()
                
                # Process data through AI model
                processed_data = await self.ai_model.process_market_data(market_data)
                
                # Get trading signals
                prediction = await self.ai_model.predict(processed_data)
                
                # Update metrics
                self.metrics['model_latency'].set(prediction['latency'])
                
                # Execute trading strategy based on prediction
                if prediction['confidence'] >= self.config.MIN_CONFIDENCE_THRESHOLD:
                    await self._execute_trade_signal(prediction['signal'], market_data)
                
                await asyncio.sleep(self.config.MARKET_DATA_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in market data processing: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _execute_trade_signal(self, signal: str, market_data: Dict):
        """Execute trades based on AI model signals."""
        try:
            if signal == 'buy':
                order = await self.order_manager.place_buy_order(
                    market_data['symbol'],
                    market_data['price'],
                    self.config.POSITION_SIZE
                )
                if order:
                    logger.info(f"Executed buy order: {order}")
                    
            elif signal == 'sell':
                order = await self.order_manager.place_sell_order(
                    market_data['symbol'],
                    market_data['price'],
                    self.config.POSITION_SIZE
                )
                if order:
                    logger.info(f"Executed sell order: {order}")
                    
        except Exception as e:
            logger.error(f"Error executing trade signal: {e}")

    async def _update_model_performance(self, trade_result: Dict):
        """Update AI model with trade outcomes."""
        try:
            # Convert trade result to model outcome
            if trade_result['profit'] > 0:
                outcome = 1  # Profitable trade
            elif trade_result['profit'] < 0:
                outcome = -1  # Loss
            else:
                outcome = 0  # Break even
                
            # Update model with trade result
            await self.ai_model.update_model(
                trade_result['market_data'],
                outcome
            )
            
            # Update performance metrics
            metrics = self.ai_model.get_performance_metrics()
            self.metrics['model_accuracy'].set(
                metrics['performance_history'].get('accuracy', 0)
            )
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

    async def stop(self):
        """Stop the trading bot and cleanup resources."""
        try:
            if not self.running:
                return
                
            self.running = False
            logger.info("Stopping trading bot...")
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Cleanup AI model
            await self.ai_model.cleanup()
            
            # Cleanup other components
            await self.exchange.close()
            await self.data_storage.close()
            
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
            raise

    def get_status(self) -> Dict:
        """Get current bot status including AI model metrics."""
        return {
            'running': self.running,
            'uptime': self.uptime,
            'latency': self.latency,
            'active_tasks': len(self.tasks),
            'model_metrics': self.ai_model.get_performance_metrics()
        }

def main():
    """Optimized main entry point"""
    try:
        # System initialization
        setup_directories()
        KeyManager().validate_keys()
        
        # Create and run system
        bot = TradingBot()
        ui = TerminalUI(bot)
        
        # Signal handling
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(ui.cleanup()))
        
        loop.run_until_complete(ui.run_ui())
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()