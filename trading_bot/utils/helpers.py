import datetime
import time
import logging

class Helpers:
    @staticmethod
    def timestamp_to_datetime(timestamp):
        """
        Convert a Unix timestamp to a human-readable datetime string.
        Args:
            timestamp (int): Unix timestamp.
        Returns:
            str: Datetime string in the format 'YYYY-MM-DD HH:MM:SS'.
        """
        try:
            dt = datetime.datetime.utcfromtimestamp(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logging.error(f"Error converting timestamp to datetime: {e}")
            return None

    @staticmethod
    def datetime_to_timestamp(date_str, format='%Y-%m-%d %H:%M:%S'):
        """
        Convert a datetime string to a Unix timestamp.
        Args:
            date_str (str): Datetime string in the format 'YYYY-MM-DD HH:MM:SS'.
            format (str): Format of the date string (optional, default is '%Y-%m-%d %H:%M:%S').
        Returns:
            int: Unix timestamp.
        """
        try:
            dt = datetime.datetime.strptime(date_str, format)
            return int(time.mktime(dt.timetuple()))
        except Exception as e:
            logging.error(f"Error converting datetime to timestamp: {e}")
            return None

    @staticmethod
    def format_currency(value, decimals=2):
        """
        Format a numeric value into a currency format with commas and specified decimal places.
        Args:
            value (float or int): Numeric value.
            decimals (int): Number of decimal places (default is 2).
        Returns:
            str: Formatted currency string.
        """
        try:
            return f"{value:,.{decimals}f}"
        except Exception as e:
            logging.error(f"Error formatting currency: {e}")
            return str(value)

    @staticmethod
    def calculate_percentage_change(new_value, old_value):
        """
        Calculate the percentage change between two values.
        Args:
            new_value (float or int): The new value.
            old_value (float or int): The old value.
        Returns:
            float: Percentage change.
        """
        try:
            if old_value == 0:
                return float('inf')  # Avoid division by zero
            change = ((new_value - old_value) / abs(old_value)) * 100
            return round(change, 2)
        except Exception as e:
            logging.error(f"Error calculating percentage change: {e}")
            return None

    @staticmethod
    def safe_divide(numerator, denominator):
        """
        Safely divide two numbers, returning 0 if division by zero occurs.
        Args:
            numerator (float or int): The numerator.
            denominator (float or int): The denominator.
        Returns:
            float: Result of division or 0 if denominator is zero.
        """
        try:
            if denominator == 0:
                return 0
            return numerator / denominator
        except Exception as e:
            logging.error(f"Error in safe division: {e}")
            return 0

    @staticmethod
    def round_to_nearest(value, precision=2):
        """
        Round a number to the nearest specified decimal precision.
        Args:
            value (float or int): The number to round.
            precision (int): Number of decimal places (default is 2).
        Returns:
            float: Rounded value.
        """
        try:
            return round(value, precision)
        except Exception as e:
            logging.error(f"Error rounding value: {e}")
            return value

    @staticmethod
    def to_lowercase(text):
        """
        Convert a string to lowercase.
        Args:
            text (str): The input string.
        Returns:
            str: The string in lowercase.
        """
        try:
            return text.lower()
        except Exception as e:
            logging.error(f"Error converting text to lowercase: {e}")
            return text

    @staticmethod
    def current_timestamp():
        """
        Get the current Unix timestamp.
        Returns:
            int: Current Unix timestamp.
        """
        return int(time.time())

    @staticmethod
    def get_current_utc_datetime():
        """
        Get the current UTC datetime string in the format 'YYYY-MM-DD HH:MM:SS'.
        Returns:
            str: Current UTC datetime string.
        """
        return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')