"""Bounded parallel maps.

Both variants keep only a fixed number of tasks in flight. That matters here
because every task holds a multi-megabyte buffer: submitting a whole 400 GB
model's worth of chunks up front would queue hundreds of thousands of futures
and let memory grow without limit.

Use `ordered_map` when results must be consumed in input order (the compressor
appends chunks to the archive sequentially) and `unordered_map` when they are
independent (the decompressor writes each chunk straight to its own offset).
"""

from __future__ import annotations

import os
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def default_workers() -> int:
    """Threads to use by default, respecting CPU affinity where reported."""
    n = None
    if hasattr(os, "process_cpu_count"):  # 3.13+, honours affinity and cgroups
        n = os.process_cpu_count()
    return max(1, min(n or os.cpu_count() or 4, 32))


def ordered_map(fn, items, workers: int | None = None, lookahead: int | None = None):
    """Yield fn(item) in input order with at most `lookahead` tasks in flight."""
    workers = workers or default_workers()
    if workers <= 1:
        for item in items:
            yield fn(item)
        return

    lookahead = lookahead or workers * 3
    it = iter(items)
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lmz")
    pending: deque = deque()
    try:
        for _ in range(lookahead):
            try:
                pending.append(ex.submit(fn, next(it)))
            except StopIteration:
                break
        while pending:
            result = pending.popleft().result()
            try:
                pending.append(ex.submit(fn, next(it)))
            except StopIteration:
                pass
            yield result
    finally:
        for fut in pending:
            fut.cancel()
        ex.shutdown(wait=True, cancel_futures=True)


def unordered_map(fn, items, workers: int | None = None, lookahead: int | None = None):
    """Yield fn(item) as results complete, with at most `lookahead` in flight.

    Completions arrive on a queue fed by done-callbacks. The obvious
    alternative, waiting on the pending set with FIRST_COMPLETED, re-installs
    a waiter on every outstanding future for each chunk that finishes; at a
    few dozen in flight that turned into the dominant cost and flattened
    decompression to roughly single-threaded speed.
    """
    workers = workers or default_workers()
    if workers <= 1:
        for item in items:
            yield fn(item)
        return

    lookahead = lookahead or workers * 3
    it = iter(items)
    finished: queue.SimpleQueue = queue.SimpleQueue()
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lmz")
    inflight = 0
    try:
        def submit_one() -> bool:
            nonlocal inflight
            try:
                item = next(it)
            except StopIteration:
                return False
            ex.submit(fn, item).add_done_callback(finished.put)
            inflight += 1
            return True

        for _ in range(lookahead):
            if not submit_one():
                break
        while inflight:
            fut = finished.get()
            inflight -= 1
            result = fut.result()
            submit_one()
            yield result
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
