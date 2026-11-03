import asyncio
import httpx
import logging
import math
import random
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


# HTTP statuses that are generally appropriate for retrying.
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    425,  # Too Early
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}


# Transport errors that may be transient.
RETRYABLE_TRANSPORT_ERRORS = (
    # PoolTimeout is deliberately excluded because it usually indicates local connection-pool saturation.
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
)


def _is_retryable(exc: Exception) -> bool:
    """Determines whether an exception represents a retryable failure.

    Retryable failures include selected HTTPX transport errors and
    HTTP status errors with a status code in RETRYABLE_STATUS_CODES.

    Args:
        exc: The exception raised during the operation.

    Returns:
        True if the exception is retryable; otherwise, False.
    """
    if isinstance(exc, RETRYABLE_TRANSPORT_ERRORS):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES

    return False


def _parse_retry_after(exc: Exception) -> float | None:
    """Parses the Retry-After header from an HTTP status error.

    Supports both:
        - Delay in seconds, e.g. Retry-After: 10
        - HTTP date, e.g. Retry-After: Wed, 21 Oct 2026 07:28:00 GMT

    A date in the past produces a delay of zero.

    Args:
        exc: The exception raised during the HTTP operation.

    Returns:
        The requested delay in seconds, or None if the header is
        absent or cannot be parsed.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None

    value = exc.response.headers.get("Retry-After")
    if not value:
        return None

    value = value.strip()

    # Retry-After expressed as a non-negative integer.
    if re.fullmatch(r"[0-9]+", value):
        try:
            return float(value)
        except (ValueError, OverflowError):
            return None

    # Retry-After expressed as an HTTP date.
    try:
        retry_at = parsedate_to_datetime(value)

        if retry_at is None:
            return None

        # Treat a naive date as UTC.
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        return max(0.0, (retry_at - now).total_seconds())

    except (TypeError, ValueError, OverflowError):
        return None


def _calculate_full_jitter_delay(current_cap: float) -> float:
    """Calculates a full-jitter exponential backoff delay.

    The returned delay is uniformly distributed between zero and
    the current exponential backoff cap.

    Args:
        current_cap: The maximum delay window in seconds.

    Returns:
        A randomized delay in seconds.
    """
    return random.uniform(0.0, current_cap)


async def with_retry(
    coro_fn: Callable[[], Awaitable[T]],
    max_retries: int = 2,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    max_retry_after: float = 120.0,
) -> T:
    """Executes an asynchronous operation with bounded retries.

    Retries only exceptions classified as retryable by _is_retryable.
    Uses exponential backoff with full jitter unless the server provides a valid Retry-After header.

    Full-jitter delay formula:
        random.uniform(0, min(max_delay, delay * backoff ** retry_number))

    Retry-After takes precedence over the calculated jitter delay.
    If the requested Retry-After delay exceeds max_retry_after, the operation is not retried.

    The caller is responsible for ensuring that retrying the
    operation is safe. In particular, retrying HTTP POST requests
    may result in duplicate operations if the server processed
    the original request but its response was lost.

    Args:
        coro_fn: A callable that returns an awaitable operation.
        max_retries: Number of retries after the initial attempt.
        delay: Initial exponential backoff window in seconds.
        backoff: Exponential multiplier applied after each retry.
        max_delay: Maximum full-jitter backoff window in seconds.
        max_retry_after: Maximum accepted server-requested delay in seconds.

    Returns:
        The result returned by coro_fn upon successful execution.

    Raises:
        ValueError: If a configuration parameter has an invalid type, is negative where prohibited, or is non-finite.
        Exception: The original operation exception if it is non-retryable or all permitted attempts fail.

    Note:
        asyncio.CancelledError is not caught, so task cancellation
        propagates normally.
    """
    # Validate retry count.
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    # Validate numeric settings and reject bool values.
    numeric_settings = {
        "delay": delay,
        "backoff": backoff,
        "max_delay": max_delay,
        "max_retry_after": max_retry_after,
    }

    for name, value in numeric_settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")

        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False

        if not finite:
            raise ValueError(f"{name} must be a finite number")

    # Normalize numeric settings to floats.
    delay = float(delay)
    backoff = float(backoff)
    max_delay = float(max_delay)
    max_retry_after = float(max_retry_after)

    if delay < 0:
        raise ValueError("delay must be non-negative")

    if backoff < 1:
        raise ValueError("backoff must be greater than or equal to 1")

    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")

    if max_retry_after < 0:
        raise ValueError("max_retry_after must be non-negative")

    # max_retries excludes the initial attempt.
    total_attempts = max_retries + 1

    # The first retry uses the initial backoff window.
    current_cap = min(delay, max_delay)

    for attempt in range(1, total_attempts + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            # Fail immediately for non-retryable exceptions.
            if not _is_retryable(exc):
                logger.warning(
                    "Non-retryable error on attempt %s/%s: %s",
                    attempt,
                    total_attempts,
                    exc,
                )
                raise

            # No retries remain.
            if attempt == total_attempts:
                logger.error(
                    "Operation failed after %s attempt(s): %s",
                    total_attempts,
                    exc,
                )
                raise

            # Respect a valid server-provided Retry-After header.
            retry_after = _parse_retry_after(exc)

            if retry_after is not None:
                if retry_after > max_retry_after:
                    logger.warning(
                        "Retry-After requested %.2fs, exceeding "
                        "the configured maximum of %.2fs. "
                        "Not retrying.",
                        retry_after,
                        max_retry_after,
                    )
                    raise

                wait = retry_after

                logger.warning(
                    "Retryable error on attempt %s/%s: %s. "
                    "Respecting Retry-After: %.2fs",
                    attempt,
                    total_attempts,
                    exc,
                    wait,
                )

            else:
                # Full jitter: random delay between zero and the current exponential backoff cap.
                wait = _calculate_full_jitter_delay(current_cap)

                logger.warning(
                    "Retryable error on attempt %s/%s: %s. "
                    "Retrying in %.2fs (backoff cap: %.2fs)",
                    attempt,
                    total_attempts,
                    exc,
                    wait,
                    current_cap,
                )

            await asyncio.sleep(wait)

            # Increase the backoff window for the next retry.
            current_cap = min(
                max_delay,
                current_cap * backoff,
            )

    # Defensive fallback; normally unreachable.
    raise RuntimeError("Retry loop exited unexpectedly")
