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
        
        # Initialize components with proper dependency injection
        self.exchange = Exchange(config)
        self.data_storage = DataStorage(config)
        self.data_fetcher = DataFetcher(config, self.data_storage)
        self.risk_manager = RiskManager(config)
        self.order_manager = OrderManager(config, self.exchange, self.risk_manager)
        
        # Initialize metrics collection
        self.metrics = {
            'uptime': Gauge('bot_uptime', 'Bot uptime in seconds'),
            'latency': Gauge('bot_latency', 'Current latency in ms'),
            'active_tasks': Gauge('bot_active_tasks', 'Number of active tasks'),
            'memory_usage': Gauge('bot_memory_usage', 'Memory usage in MB')
        }

    async def _critical_health_check(self):
        """Real-time critical system monitoring"""
        while self.running:
            try:
                # Exchange connectivity
                if not await self.exchange.check_connection():
                    logger.warning("Exchange connection lost! Reconnecting...")
                    await self.exchange.reconnect()
                
                # Data pipeline health
                if not await self.data_storage.check_connection():
                    logger.critical("Data storage connection failed!")
                    await self.stop()
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Critical health check failed: {e}")

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
        """Optimized startup sequence"""
        if self.running:
            return

        try:
            self.running = True
            self.start_time = datetime.now()
            logger.info("Starting trading bot...")

            # Ordered startup sequence
            await self.data_storage.initialize()
            await self.exchange.connect()
            
            # Start core services
            core_tasks = [
                self._critical_health_check(),
                self.data_fetcher.start(),
                self.order_manager.start(),
                self._background_services()
            ]
            
            # Create supervised tasks
            for task in core_tasks:
                self.tasks.add(asyncio.create_task(
                    self._supervised_task(task),
                    name=task.__class__.__name__
                ))

            logger.info("Trading bot operational")
        except Exception as e:
            logger.critical(f"Startup failed: {e}")
            await self.stop()

    async def _supervised_task(self, coro):
        """Task wrapper with automatic restart"""
        while self.running:
            try:
                await coro
            except Exception as e:
                logger.error(f"Task failed: {e}, restarting...")
                await asyncio.sleep(1)

    async def stop(self):
        """Atomic shutdown procedure"""
        if not self.running:
            return

        logger.info("Initiating shutdown sequence...")
        self.running = False
        
        # Ordered shutdown
        try:
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close connections
            await self.exchange.close()
            await self.data_storage.close()
            
            # Final report
            logger.info(f"Final uptime: {self.uptime}")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        finally:
            self.tasks.clear()

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