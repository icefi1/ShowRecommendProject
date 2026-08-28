"""
A shared rate limiter for TMDB requests.

TMDB's published limit is 50 requests per second and 20 connections per IP
(https://developer.themoviedb.org/docs/rate-limiting). The older "40 requests
per 10 seconds" figure was retired in December 2019 and no longer applies.

We run deliberately under both ceilings. The point of a dissertation fetch is
to finish reliably, not to finish fastest, and being throttled mid-run costs
more time than the headroom saves.
"""

import threading
import time

# 60% of TMDB's 50/second. Leaves room for retries, for the burst that happens
# when several workers finish a slow request at once, and for the fact that the
# limit is enforced by a CDN whose accounting we cannot see.
MAX_REQUESTS_PER_SECOND = 30

# TMDB allows 20 connections per IP. Twelve workers keeps a clear margin, and
# past about this many the limiter is the bottleneck anyway - more threads would
# just queue.
WORKERS = 12


class RateLimiter:
    """
    Spaces requests across threads so the process never exceeds a target rate.

    Deliberately not a token bucket. A bucket permits a burst up to its
    capacity, which is exactly the shape that trips a CDN's DDoS heuristics
    after an idle moment. This enforces a hard minimum gap between consecutive
    requests instead, so the request rate is flat rather than spiky.
    """

    def __init__(self, requests_per_second=MAX_REQUESTS_PER_SECOND):
        self.min_gap = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self):
        """Block until this thread's turn, then claim the following slot."""
        with self._lock:
            now = time.monotonic()
            # Claim the later of "right now" and "one gap after the last claim",
            # so slots are handed out in order and never overlap.
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_gap
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)
