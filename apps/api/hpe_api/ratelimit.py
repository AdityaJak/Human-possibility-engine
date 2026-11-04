"""Token-bucket rate limiting for expensive POST endpoints (per client, per bucket name)."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

from fastapi import HTTPException, Request


class TokenBucketLimiter:
    def __init__(
        self, rate_per_minute: float, burst: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.rate = max(rate_per_minute, 0.0) / 60.0
        self.burst = max(1, int(burst))
        self.clock = clock
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str, bucket: str) -> float:
        """Take one token. Returns 0 on success, otherwise seconds until a token is available."""
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get((key, bucket), (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[(key, bucket)] = (tokens - 1.0, now)
                return 0.0
            self._buckets[(key, bucket)] = (tokens, now)
            if self.rate <= 0:
                return 60.0
            return (1.0 - tokens) / self.rate

    def dependency(self, bucket: str) -> Callable[[Request], None]:
        def _check(request: Request) -> None:
            client = request.client.host if request.client else "unknown"
            wait = self.acquire(client, bucket)
            if wait > 0:
                raise HTTPException(
                    status_code=429,
                    detail=f"rate limit exceeded for {bucket}; retry later",
                    headers={"Retry-After": str(max(1, math.ceil(wait)))},
                )

        return _check
