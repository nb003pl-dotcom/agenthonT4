"""Operational extraction failures eligible for explicit evidence fallback."""
import httpx
from .model import BudgetError,transient_proxy_status


def recoverable_fact_failure(exc):
    if transient_proxy_status(exc) is not None:return True
    if isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError, httpx.TimeoutException)):
        return True
    if isinstance(exc, BudgetError):
        return str(exc) in {'request_budget_reserved_for_prediction',
            'input_token_budget', 'provider_input_budget_exceeded'}
    return type(exc) is RuntimeError and str(exc) in {
        'generation_http_408', 'generation_http_429', 'generation_http_500',
        'generation_http_502', 'generation_http_503', 'generation_http_504'}
