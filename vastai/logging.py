import os

VAST_LOG_LEVEL = os.environ.get("VAST_LOG_LEVEL")

levels = {"critical": 0, "error": 1, "warning": 2, "info": 3, "debug": 4, "trace": 5}

log_level: int = (
    VAST_LOG_LEVEL
    if isinstance(VAST_LOG_LEVEL, int)
    else levels[VAST_LOG_LEVEL]
    if VAST_LOG_LEVEL in levels
    else 3
)


def log_critical(msg: str):
    if log_level >= 0:
        print(msg)


def log_error(msg: str):
    if log_level >= 1:
        print(msg)


def log_warning(msg: str):
    if log_level >= 2:
        print(msg)


def log_info(msg: str):
    if log_level >= 3:
        print(msg)


def log_debug(msg: str):
    if log_level >= 4:
        print(msg)


def log_trace(msg: str):
    if log_level >= 5:
        print(msg)
