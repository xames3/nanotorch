"""\
NanoTorch Logging API
=====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, December 02 2023
Last updated on: Saturday, December 09 2023

Log NanoTorch's messages.

The robust event logging system used by NanoTorch is written in this
module along with an assortment of related functions and classes. The
ability for all the modules to participate in logging is the key benefit
of having a dedicated OR rather customizable logging API. In order to
configure the system's default logging capabilities, it offers the core
abstractions.

The objects from the builtin ``logging`` module are monkey-patched to
achieve this level of modularity. This module is also in responsible
of using colours to represent the severity of the logging levels on the
terminal.

See https://github.com/xames3/nanotorch/ for more help.

:copyright: (c) 2024 Akshay Mestry (XAMES3). All rights reserved.
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import typing as t
from logging.handlers import RotatingFileHandler
from os import path as p

if t.TYPE_CHECKING:
    import types

__all__ = ["create_logger", "get_logger"]

# The ISO8601 standard is used for logging the timestamp of the log
# messages. The logged time is the local time of the user and not UTC.
# For UTC, the user must patch the logger object to use the ``gmtime``.
_iso8601: t.Final[str] = "%Y-%m-%dT%H:%M:%SZ"

# This is the name of the logger used to log the event represented by
# the active ``LogRecord``. Note that this name will always have this
# value, even though it may be emitted by a handler attached to a
# different (ancestor) logger.
_logger_name: t.Final[str] = "nanotorch.main"
_log_file: str = p.join(p.expanduser("~"), ".nanotorch", "session.log")

_logging_level_map: dict[str, int] = {
    "TRACE": 60,
    "FATAL": 50,
    "CRITICAL": 50,
    "ERROR": 40,
    "WARN": 30,
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 00,
}
_possible_truthy_values: frozenset[str | int] = frozenset(
    ["TRUE", "T", "True", "true", "t", 1, "1"]
)

_ansi_escape_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _setup_logs_dir(
    override: bool | str, path: str | None = None
) -> str | None:
    """Create a log directory and log all stdin-stdout I/O to it.

    By default, the logs are maintained in the ``$HOME/.nanotorch``
    directory. If the directory has been purged or does not already
    exist, this function will re-create it.

    If you don't want this behavior, set the ``NANOTORCH_SKIP_LOGGING``
    environment flag to ``TRUE`` to disable logging::

        export NANOTORCH_SKIP_LOGGING=TRUE

    :param override: If set to ``True``, it will override logging.
    :param path: The path to the log file that will be used to establish
                 the logging parent directory, defaults to ``None``.
    :return: Absolute path of the log file, if logging is required.
    """
    if override or override in _possible_truthy_values:
        return None
    if path is None:
        path = _log_file
    if not p.exists(path):
        os.makedirs(p.dirname(path), exist_ok=True)
    return path


def _skip_output_logs(choice: bool) -> bool:
    """Return whether to output logs to a file.

    This function only has effect if the ``NANOTORCH_SKIP_LOGGING``
    environment variable is set to ``TRUE``. It will continue to report
    output to the file if nothing is set.

    :param choice: Boolean flag to skip logs.
    :return: Status whether to log file or not.
    """
    skip = os.getenv("NANOTORCH_SKIP_LOGGING")
    # This ensures that only the True values are considered as valid
    # choices. This is required or else any non-empty string would be
    # considered as True and the loop might accidentally execute.
    if (skip and skip in _possible_truthy_values) or choice:
        return True
    return False


def _select_log_level(level: int | str) -> int:
    """Choose the appropriate logging level.

    The ``NANOTORCH_LOGGING_LEVEL`` environment variable can be used to
    alter the logging level. The verbosity counter will be overridden
    if the variable is set to a proper log level::

        export NANOTORCH_LOGGING_LEVEL=WARNING

    This will use the warning logging level. If nothing is specified,
    the verbosity setting will be used for logging.

    :param level: Verbosity counter value or the implicit logging level.
    :returns: Logging level.

    .. versionchanged:: 1.0.1
        Function returns default logging level of 0, ``debug``.
    """
    env_level = os.getenv("NANOTORCH_LOGGING_LEVEL")
    if env_level:
        return _logging_level_map[env_level]
    if isinstance(level, str):
        return _logging_level_map[level]  # Unnecessary but required :(
    if level in (00, 10, 20, 30, 40, 50, 60):
        return level
    return 00


def _use_color(choice: bool) -> str:
    """Return log format based on the choice.

    If choice is ``True``, colored log format is returned else
    non-colored format is returned.

    :param choice: Boolean value to allow colored logs.
    :returns: Colored or non-colored log format based on choice.
    """
    if choice:
        return (
            "%(gray)s%(asctime)s %(color)s%(levelname)8s%(reset)s "
            "%(gray)s%(stack)s:%(lineno)d%(reset)s : %(message)s"
        )
    return "%(asctime)s %(levelname)8s %(stack)s:%(lineno)d : %(message)s"


class _NanoTorchFormatter(logging.Formatter):
    """NanoTorch's ANSI color scheme formatter.

    This class formats the ``record.pathname`` and ``record.exc_info``
    attributes to generate an uniform and clear log message. The class
    then adds gray hues to the log's metadata and colorizes the levels.

    :param fmt: Format for the log message.
    :param datefmt: Format for the log datetime.
    :var _ansi_attrs: Attributes to be added to the log record.
    :var _ansi_hue_map: Mapping of hues for different logging levels.
    """

    _ansi_attrs: tuple[str, ...] = "color", "gray", "reset"

    # See https://stackoverflow.com/a/14693789/14316408 for the RegEx
    # logic behind the ANSI escape sequence.
    _ansi_hue_map: dict[int, str] = {
        90: "\x1b[38;5;242m",
        60: "\x1b[38;5;128m",
        50: "\x1b[38;5;197m",
        40: "\x1b[38;5;204m",
        30: "\x1b[38;5;215m",
        20: "\x1b[38;5;41m",
        10: "\x1b[38;5;14m",
        00: "\x1b[0m",
    }

    def __init__(self, fmt: str, datefmt: str) -> None:
        """Initialize the formatter with suitable formats."""
        self.fmt = fmt
        self.datefmt = datefmt

    def colorize(self, record: logging.LogRecord) -> None:
        """Add colors to the logging levels by manipulating log records.

        This approach is on the cutting edge because it modifies the
        record object in real time. This has the potential to be a
        disadvantage. We verify if the logging stream is a ``TTY``
        interface or not to avoid memory leaks. If we are certain that
        the stream is a ``TTY``, we alter the object.

        As a result, when writing to a file, this method avoids the
        record from containing unreadable ANSI characters.

        :param record: Logged event's instance.
        """
        # The same could have been done using the ``hasattr()`` too.
        # This ``isatty`` is a special attribute which is injected by
        # the ``nanotorch.logger.StreamHandler()`` class.
        if getattr(record, "isatty", False):
            hue_map = zip(("color", "gray", "reset"), (record.levelno, 90, 0))
            for hue, level in hue_map:
                setattr(record, hue, self._ansi_hue_map[level])
        else:
            for attr in self._ansi_attrs:
                setattr(record, attr, "")

    def decolorize(self, record: logging.LogRecord) -> None:
        """Remove ``color``, ``gray`` and ``reset`` attributes from the
        log record.

        This method is opposite of ``colorize()`` of the same class.
        It prevents the record from writing un-readable ANSI characters
        to a non-TTY interface.

        :param record: Logged event's instance.
        """
        for attr in self._ansi_attrs:
            delattr(record, attr)

    def formatException(
        self,
        ei: tuple[type, BaseException, types.TracebackType | None]
        | tuple[None, ...],
    ) -> str:
        r"""Format exception information as text.

        This implementation does not work directly. The log formatter
        from the standard library is required. The parent class creates
        an output string with ``\n`` which needs to be truncated and
        this method does this well.

        :param ei: Information about the captured exception.
        :returns: Formatted exception string.
        """
        func, lineno = "<module>", 0
        cls_, msg, tbk = ei
        if tbk:
            func, lineno = tbk.tb_frame.f_code.co_name, tbk.tb_lineno
        func = "on" if func in ("<lambda>", "<module>") else f"in {func}() on"
        return f"{cls_.__name__ if cls_ else cls_}: {msg} line {lineno}"

    @staticmethod
    def stack(path: str, func: str) -> str:
        """Format path and function as stack.

        :param path: Path of the module which is logging the event.
        :param func: Callable object's name.
        :returns: Spring-boot style formatted path, well kinda...

        .. note::

            If called from a module, the base path of the module would
            be used else ``shell`` would be returned for the interpreter
            (stdin) based inputs.

        """
        if path == "<stdin>":
            return "shell"  # Should not return this rightaway...
        if os.name == "nt":
            path = p.splitdrive(path)[1]
        # NOTE: This presumes we work through a virtual environment.
        # This is a safe assumption as we peruse through the site-
        # packages. In case, this is not running via the virtualenv, we
        # might get a different result.
        abspath = "site-packages" if "site-packages" in path else os.getcwd()
        path = path.split(abspath)[-1]
        path = path.replace(p.sep, ".")[path[0] != ":" : -3]
        if func not in ("<module>", "<lambda>"):
            path += f".{func}"
        return path

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text.

        If any exception is captured then it is formatted using
        the ``formatException()`` and replaced with the original
        message.

        :param record: Logged event's instance.
        :returns: Captured and formatted output log message.
        """
        # Update the pathname and the invoking function name using the
        # stack. This stack will be set as a record attribute which will
        # allow us to use the %(stack)s placeholder in the log format.
        setattr(record, "stack", self.stack(record.pathname, record.funcName))
        if record.exc_info:
            record.msg = self.formatException(record.exc_info)
            record.exc_info = record.exc_text = None
        self.colorize(record)
        msg = logging.Formatter(self.fmt, self.datefmt).format(record)
        # Escape the ANSI sequence here as this will render the colors
        # on the TTY but won't add them to the non-TTY interfaces, for
        # example, log file.
        record.msg = _ansi_escape_re.sub("", str(record.msg))
        self.decolorize(record)
        return msg


class _NanoTorchStreamHandler(logging.StreamHandler[t.IO[str]]):
    """A StreamHandler derivative which adds an inspection of a TTY
    interface to the stream.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Add hint if the specified stream is a TTY.

        The ``hint`` here, means the boolean specification as this
        attribute helps to identify a stream's interface. This solves a
        major problem when printing un-readable ANSI sequences to a
        non-TTY interface.

        :param record: Logged event's instance.
        :returns: Formatted string for the output stream.
        """
        if hasattr(self.stream, "isatty"):
            try:
                setattr(record, "isatty", self.stream.isatty())
            except ValueError:
                setattr(record, "isatty", False)
        else:
            setattr(record, "isatty", False)
        strict = super().format(record)
        delattr(record, "isatty")
        return strict


def get_logger(module: str) -> logging.Logger:
    """Return logger instance.

    This function is supposed to be used by the modules for logging.
    The logger generated by this function is a child which reports logs
    back to the parent logger defined by the ``create_logger()``.

    The function is most useful for the intermediate modules which want
    to perform logging at the module level as this function returns a
    child logger. Whereas the ``create_logger`` is an application level
    logger.

    :param module: Module to be logged.
    :returns: Logger instance.
    """
    return logging.getLogger(_logger_name).getChild(module)


def create_logger(
    name: str | None = None,
    level: int | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
    color: bool = True,
    filename: str | None = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 10,
    encoding: str | None = None,
    filemode: str = "a",
    skip_logging: bool = False,
    handlers: list[logging.Handler] | None = None,
    stream: t.IO[str] | None = sys.stderr,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure an application level logger.

    This function initializes a logger with default configurations for
    the logging system.

    .. code-block:: python

        from nanotorch.logger import create_logger

        logger = create_logger(
            name='yourapp',  # Logger name
            level=30,  # Logging set to Warning
            color=False,  # Disable colored output
        )

    If any handlers are provided as part of input, the function
    overrides the default behaviour in favour of the provided handler.
    It is a convenience function intended for use by simple applications
    to do one-shot configuration.

    :param name: Name for the logger, defaults to ``None``.
    :param level: Minimum logging level of the event, defaults
                  to ``None``.
    :param fmt: Format for the log message, defaults to ``None``.
    :param datefmt: Format for the log datetime, defaults to ``None``.
    :param color: Boolean option to whether display colored log outputs
                  on the terminal or not, defaults to ``True``.
    :param filename: Log file's absolute path, defaults to ``None``.
    :param max_bytes: Maximum size in bytes after which the rollover
                      should happen, defaults to ``10 MB``.
    :param backup_count: Maximum number of files to archive before
                         discarding, defaults to ``10``.
    :param encoding: Platform-dependent encoding for the file, defaults
                     to ``None``.
    :param filemode: Mode in which the file needs to be opened, defaults
                    to append ``a`` mode.
    :param skip_logging: Boolean option to whether skip the logging
                         process, defaults to ``False``.
    :param handlers: List of various logging handlers to use, defaults
                     to ``None``.
    :param stream: IO stream, defaults to ``sys.stderr``.
    :param capture_warnings: Boolean option to whether capture the
                             warnings while logging, defaults
                             to ``True``.
    :returns: Configured logger instance.
    """
    if name is None:
        name = _logger_name
    logger = logging.getLogger(name)
    if level is None:
        level = logging.INFO
    level = _select_log_level(level)
    logger.setLevel(level)
    if handlers is None:
        handlers = []
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()
    if not logger.handlers:
        if fmt is None:
            fmt = _use_color(color)
        if datefmt is None:
            datefmt = _iso8601
        formatter = _NanoTorchFormatter(fmt, datefmt)
        stream_handler = _NanoTorchStreamHandler(stream)
        handlers.append(stream_handler)
        if not _skip_output_logs(skip_logging):
            filename = _setup_logs_dir(skip_logging, filename)
            if filename:
                file_handler = RotatingFileHandler(
                    filename, filemode, max_bytes, backup_count, encoding
                )
                handlers.append(file_handler)
        for handler in handlers:
            logger.addHandler(handler)
            handler.setFormatter(formatter)
    logging.captureWarnings(capture_warnings)
    return logger
