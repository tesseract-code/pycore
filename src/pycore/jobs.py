"""
core_types.py
Shared type definitions and enums for the Async Qt system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional
import time

class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()

class ExecutionMode(Enum):
    SYNC = auto()      # QThreadPool (Blocking CPU bound)
    ASYNC = auto()     # QThread + Asyncio (I/O bound)
    AUTO = auto()

@dataclass
class JobMetrics:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    progress: float = 0.0
    start_time: float | None = None
    end_time: float | None = None
    error: Exception | None = None
    result: Any = None
