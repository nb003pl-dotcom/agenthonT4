"""Nested wall-clock cap for optional stages on Linux/macOS main-thread runners."""
from contextlib import contextmanager
import signal
import threading
import time


@contextmanager
def wall_clock_budget(seconds):
    if not hasattr(signal,'setitimer') or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('hard_optional_stage_deadline_unavailable')
    if seconds<=0:raise ValueError('Positive stage budget required')
    previous_handler=signal.getsignal(signal.SIGALRM)
    previous_delay,previous_interval=signal.getitimer(signal.ITIMER_REAL)
    started=time.monotonic()
    def expired(signum,frame):raise TimeoutError('optional_stage_wall_clock_exhausted')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,min(seconds,previous_delay) if previous_delay else seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_delay:
            # Restore remaining whole-task time, never restart its original budget.
            signal.setitimer(signal.ITIMER_REAL,max(.001,previous_delay-(time.monotonic()-started)),previous_interval)
