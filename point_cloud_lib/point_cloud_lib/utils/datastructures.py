from dataclasses import dataclass, field, asdict
from typing import List, Dict, OrderedDict, Tuple
from pathlib import Path
from xmlrpc.client import Boolean
from numpy import inf
from collections import namedtuple
import logging

@dataclass
class LoggingConfig():
    #file_log_handler_path : Path = Path("")
    enable_stream_log_handler : bool = True
    enable_file_log_handler : bool = False
    stream_log_handler_level : int = logging.DEBUG
    file_log_handler_level : int = logging.ERROR