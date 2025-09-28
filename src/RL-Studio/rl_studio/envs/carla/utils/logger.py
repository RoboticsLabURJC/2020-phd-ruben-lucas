# Code from https://stackoverflow.com/a/8349076
import logging
import sys

from rl_studio.envs.carla.utils.colors import Colors
from rl_studio.envs.carla.utils.constants import ROOT_PATH

from datetime import datetime

class ColorLogger(logging.Formatter):
    """Colored logger with beautified datetime in every message."""

    def __init__(self):
        super().__init__()
        self.err_fmt = "{}%(levelname)s (%(module)s: %(lineno)d): {}{}%(message)s{}".format(
            Colors.FAIL, Colors.ENDC, Colors.CBOLD, Colors.ENDC
        )
        self.info_fmt = "{}%(levelname)s: {}{}%(message)s{}".format(
            Colors.CBLUE2, Colors.ENDC, Colors.CBOLD, Colors.ENDC
        )
        self.wrn_fmt = "{}%(levelname)s: {}{}%(message)s{}".format(
            Colors.CYELLOW, Colors.ENDC, Colors.CBOLD, Colors.ENDC
        )
        self.dbg_fmt = "{}%(levelname)s (%(module)s: %(lineno)d): {}{}%(message)s{}".format(
            Colors.CGREEN2, Colors.ENDC, Colors.CBOLD, Colors.ENDC
        )

    def format(self, record):
        # Apply argument substitution first
        record.message = record.getMessage()

        # Add beautified datetime
        record.asctime = datetime.now().strftime("%A, %B %d, %Y %I:%M:%S %p")

        # Select format based on log level
        if record.levelno == logging.DEBUG:
            log_fmt = f"[{record.asctime}] {self.dbg_fmt}"
        elif record.levelno == logging.INFO:
            log_fmt = f"[{record.asctime}] {self.info_fmt}"
        elif record.levelno == logging.ERROR:
            log_fmt = f"[{record.asctime}] {self.err_fmt}"
        elif record.levelno == logging.WARNING:
            log_fmt = f"[{record.asctime}] {self.wrn_fmt}"
        else:
            log_fmt = f"[{record.asctime}] {self.info_fmt}"

        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class PlainLogger(logging.Formatter):
    """Class for plain white python logs for the application.

    Use if terminal does not support coloring.
    """

    err_fmt = "%(levelname)s (%(module)s: %(lineno)d): %(message)s"
    info_fmt = "%(levelname)s: %(message)s"
    wrn_fmt = "%(levelname)s: %(message)s"
    dbg_fmt = "%(levelname)s (%(module)s: %(lineno)d): %(message)s"

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = PlainLogger.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = PlainLogger.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = PlainLogger.err_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = PlainLogger.wrn_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


def file_handler(logfile):
    with open(ROOT_PATH + '/logs/log.log', 'w') as f:
        f.truncate(0)
    return logging.FileHandler(logfile)


def std_handler():
    return logging.StreamHandler(sys.stdout)


logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorLogger())

logger.addHandler(handler)