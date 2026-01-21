import argparse
import asyncio
import contextlib
import os
import random
import signal
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

# Headless-safe plotting
import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt

import numpy as np
from vastai import Serverless


# -----------------------------
# Worker model (as provided)
# -----------------------------
@dataclass
class Worker:
    id: int
    status: str
    cur_load: float
    new_load: float
    cur_load_rolling_avg: float
    cur_perf: float
    perf: float
    measured_perf: float
    dlperf: float
    reliability: float
    reqs_working: int
    disk_usage: float
    loaded_at: float
    started_at: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Worker":
        status = d.get("status") or "UNKNOWN"
        try:
            status = str(status)
        except Exception:
            status = "UNKNOWN"

        return Worker(
            id=int(d.get("id")),
            status=status,
            cur_load=float(d.get("cur_load", 0.0)),
            new_load=float(d.get("new_load", 0.0)),
            cur_load_rolling_avg=float(d.get("cur_load_rolling_avg", 0.0)),
            cur_perf=float(d.get("cur_perf", 0.0)),
            perf=float(d.get("perf", 0.0)),
            measured_perf=float(d.get("measured_perf", 0.0)),
            dlperf=float(d.get("dlperf", 0.0)),
            reliability=float(d.get("reliability", 0.0)),
            reqs_working=int(d.get("reqs_working", 0)),
            disk_usage=float(d.get("disk_usage", 0.0)),
            loaded_at=float(d.get("loaded_at", 0.0)),
            started_at=float(d.get("started_at", 0.0)),
        )


# -----------------------------
# Request tracking
# -----------------------------
STATUS_PENDING = "PENDING"  # task created, not yet entered request
STATUS_IN_FLIGHT = "IN_FLIGHT"  # request issued / awaiting response
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"

STATUS_TO_CODE = {
    STATUS_PENDING: 0,
    STATUS_IN_FLIGHT: 1,
    STATUS_DONE: 2,
    STATUS_FAILED: 3,
}
CODE_TO_STATUS = {v: k for k, v in STATUS_TO_CODE.items()}


@dataclass
class RunState:
    n: int
    statuses: List[str]
    start_ts: List[Optional[float]]
    end_ts: List[Optional[float]]
    errors: List[Optional[str]]
    lock: asyncio.Lock
    t0: float


def _extract_local_path(resp: dict) -> str:
    r = resp.get("response", resp)
    outputs = r.get("output")

    if not isinstance(outputs, list) or len(outputs) == 0:
        err = r.get("error") or r.get("errors") or r.get("detail") or r.get("message") or r
        raise RuntimeError(f"No outputs returned. error={err}")

    first = outputs[0]
    if not isinstance(first, dict) or "local_path" not in first:
        raise RuntimeError(f"Unexpected output entry shape: {first!r}")

    return first["local_path"]


async def generate(endpoint, idx, state, http_port, use_session):
    session = None
    if use_session:
        session = await endpoint.session(cost=100, lifetime=10)
        payload = {
            "input": {
                "modifier": "Text2Image",
                "modifications": {
                    "prompt": f"{random.randint(1, 1000000)} Generate a page from a peanuts comic strip.",
                    "width": 512,
                    "height": 512,
                    "steps": 10,
                    "seed": random.randint(1, 1000000),
                },
                "webhook": {
                    "url": f"http://localhost:{http_port}/session/end",
                    "extra_params": {
                        "session_id": session.session_id,
                        "session_auth": session.auth_data,
                    },
                },
            }
        }
    else:
        payload = {
            "input": {
                "modifier": "Text2Image",
                "modifications": {
                    "prompt": "Generate a page from a peanuts comic strip.",
                    "width": 512,
                    "height": 512,
                    "steps": 10,
                    "seed": random.randint(1, 1000000),
                },
            }
        }

    async with state.lock:
        state.start_ts[idx] = state.t0
        state.statuses[idx] = STATUS_IN_FLIGHT

    try:
        if use_session:
            await session.request("/generate", payload)
            while await session.is_open():
                await asyncio.sleep(0.5)
            await session.close()
        else:
            await endpoint.request("/generate/sync", payload, cost=100)

        t_end = time.monotonic()
        async with state.lock:
            state.end_ts[idx] = t_end
            state.statuses[idx] = STATUS_DONE

        print(f"[{idx}] Request completed (session closed by webhook)")
        return idx

    except asyncio.CancelledError:
        # CancelledError is not an Exception in modern Python
        t_end = time.monotonic()
        async with state.lock:
            state.end_ts[idx] = t_end
            state.statuses[idx] = STATUS_FAILED
            state.errors[idx] = "CancelledError()"

        with contextlib.suppress(Exception):
            if session is not None and hasattr(session, "close"):
                await session.close()

        raise

    except Exception as e:
        t_end = time.monotonic()
        async with state.lock:
            state.end_ts[idx] = t_end
            state.statuses[idx] = STATUS_FAILED
            state.errors[idx] = repr(e)

        print(f"[{idx}] FAILED: {e!r}")
        return None


# -----------------------------
# Samplers (0.1s resolution)
# -----------------------------
async def sample_request_statuses(
    state: RunState,
    done_evt: asyncio.Event,
    interval_s: float,
    req_time: List[float],
    req_counts: Dict[str, List[int]],
    req_matrix_snapshots: List[np.ndarray],
):
    next_tick = time.monotonic()
    while not done_evt.is_set():
        now = time.monotonic()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
            continue
        next_tick += interval_s

        async with state.lock:
            snap = list(state.statuses)

        ctr = Counter(snap)
        req_time.append(now - state.t0)
        for k in (STATUS_PENDING, STATUS_IN_FLIGHT, STATUS_DONE, STATUS_FAILED):
            req_counts[k].append(int(ctr.get(k, 0)))

        req_matrix_snapshots.append(
            np.fromiter((STATUS_TO_CODE[s] for s in snap), dtype=np.uint8, count=state.n)
        )

    # final snapshot
    now = time.monotonic()
    async with state.lock:
        snap = list(state.statuses)
    ctr = Counter(snap)
    req_time.append(now - state.t0)
    for k in (STATUS_PENDING, STATUS_IN_FLIGHT, STATUS_DONE, STATUS_FAILED):
        req_counts[k].append(int(ctr.get(k, 0)))
    req_matrix_snapshots.append(
        np.fromiter((STATUS_TO_CODE[s] for s in snap), dtype=np.uint8, count=state.n)
    )


async def monitor_request_tasks(
    req_tasks: List[asyncio.Task],
    done_evt: asyncio.Event,
    interval_s: float,
    t0: float,
):
    next_tick = time.monotonic()
    dump_counter = 0

    while not done_evt.is_set():
        now = time.monotonic()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
            continue
        next_tick += interval_s

        dump_counter += 1
        elapsed = now - t0

        pending_count = sum(1 for t in req_tasks if not t.done())
        done_count = sum(1 for t in req_tasks if t.done())
        cancelled_count = sum(1 for t in req_tasks if t.cancelled())
        exception_count = sum(
            1
            for t in req_tasks
            if t.done() and not t.cancelled() and t.exception() is not None
        )

        print(
            f"\n[MONITOR {dump_counter}] t={elapsed:.1f}s | Tasks: {len(req_tasks)} total, "
            f"{pending_count} pending, {done_count} done, {cancelled_count} cancelled, {exception_count} with exceptions"
        )

        if pending_count > 0:
            pending_indices = [i for i, t in enumerate(req_tasks) if not t.done()]
            print(
                f"  Pending task indices: {pending_indices[:20]}{' ...' if len(pending_indices) > 20 else ''}"
            )

        if exception_count > 0:
            for i, t in enumerate(req_tasks):
                if t.done() and not t.cancelled():
                    try:
                        exc = t.exception()
                        if exc is not None:
                            print(f"  Task [{i}] has exception: {exc!r}")
                    except Exception as e:
                        print(f"  Task [{i}] error getting exception: {e!r}")


async def sample_workers(
    endpoint,
    done_evt: asyncio.Event,
    interval_s: float,
    w_time: List[float],
    w_counts: List[Dict[str, int]],
    w_totals: List[int],
    w_reqs_working_sum: List[int],
    t0: float,
):
    next_tick = time.monotonic()
    while not done_evt.is_set():
        now = time.monotonic()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
            continue
        next_tick += interval_s

        try:
            workers = endpoint.get_workers()
            if asyncio.iscoroutine(workers):
                workers = await workers
        except Exception:
            workers = []

        ctr: Dict[str, int] = defaultdict(int)
        reqs_sum = 0
        for w in workers:
            try:
                st = getattr(w, "status", None) or "UNKNOWN"
            except Exception:
                st = "UNKNOWN"
            ctr[str(st)] += 1
            try:
                reqs_sum += int(getattr(w, "reqs_working", 0))
            except Exception:
                pass

        w_time.append(now - t0)
        w_counts.append(dict(ctr))
        w_totals.append(len(workers))
        w_reqs_working_sum.append(reqs_sum)

    # final sample
    now = time.monotonic()
    try:
        workers = endpoint.get_workers()
        if asyncio.iscoroutine(workers):
            workers = await workers
    except Exception:
        workers = []
    ctr = defaultdict(int)
    reqs_sum = 0
    for w in workers:
        st = getattr(w, "status", None) or "UNKNOWN"
        ctr[str(st)] += 1
        reqs_sum += int(getattr(w, "reqs_working", 0))
    w_time.append(now - t0)
    w_counts.append(dict(ctr))
    w_totals.append(len(workers))
    w_reqs_working_sum.append(reqs_sum)


# -----------------------------
# Plotting
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _mark_warmup(
    ax,
    warmup_start_s: Optional[float],
    warmup_end_s: Optional[float],
) -> None:
    if warmup_start_s is None:
        return

    ax.axvline(warmup_start_s, linestyle="--", linewidth=1, color="blue", label="warmup start")

    if warmup_end_s is not None and warmup_end_s > warmup_start_s:
        ax.axvline(warmup_end_s, linestyle="--", linewidth=1, color="blue", label="measurement start")
        ax.axvspan(
            warmup_start_s,
            warmup_end_s,
            alpha=0.12,
            facecolor="lightblue",
            label="warmup window",
        )


def _mark_cooldown(
    ax,
    cooldown_start_s: Optional[float],
    cooldown_end_s: Optional[float],
) -> None:
    if cooldown_start_s is None:
        return

    ax.axvline(cooldown_start_s, linestyle="--", linewidth=1, label="cooldown start")

    if cooldown_end_s is not None and cooldown_end_s > cooldown_start_s:
        ax.axvline(cooldown_end_s, linestyle="--", linewidth=1)
        ax.axvspan(
            cooldown_start_s,
            cooldown_end_s,
            alpha=0.12,
            facecolor="0.85",
            label="cooldown window",
        )


def plot_request_counts(
    outdir: str,
    t: List[float],
    counts: Dict[str, List[int]],
    warmup_start_s: Optional[float] = None,
    warmup_end_s: Optional[float] = None,
    cooldown_start_s: Optional[float] = None,
    cooldown_end_s: Optional[float] = None,
) -> str:
    fig, ax = plt.subplots()
    ax.plot(t, counts[STATUS_PENDING], label=STATUS_PENDING)
    ax.plot(t, counts[STATUS_IN_FLIGHT], label=STATUS_IN_FLIGHT)
    ax.plot(t, counts[STATUS_DONE], label=STATUS_DONE)
    ax.plot(t, counts[STATUS_FAILED], label=STATUS_FAILED)

    _mark_warmup(ax, warmup_start_s, warmup_end_s)
    _mark_cooldown(ax, cooldown_start_s, cooldown_end_s)

    ax.set_xlabel("Seconds since start")
    ax.set_ylabel("Number of requests")
    ax.set_title("Request status counts (0.1s sampling)")
    ax.legend()

    path = os.path.join(outdir, "requests_status_counts.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_request_timeline(
    outdir: str,
    snapshots: List[np.ndarray],
    interval_s: float,
    warmup_start_s: Optional[float] = None,
    warmup_end_s: Optional[float] = None,
    cooldown_start_s: Optional[float] = None,
    cooldown_end_s: Optional[float] = None,
) -> str:
    mat = np.stack(snapshots, axis=1)

    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#9e9e9e", "#1f77b4", "#2ca02c", "#d62728"])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xlabel(f"Time samples (each = {interval_s:.1f}s)")
    ax.set_ylabel("Request index")
    ax.set_title("Per-request status timeline (0.1s sampling)")
    cbar = fig.colorbar(ax.images[0], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([CODE_TO_STATUS[i] for i in [0, 1, 2, 3]])

    if warmup_start_s is not None:
        x0 = int(round(warmup_start_s / interval_s))
        ax.axvline(x0, linestyle="--", linewidth=1, color="blue")
        if warmup_end_s is not None and warmup_end_s > warmup_start_s:
            x1 = int(round(warmup_end_s / interval_s))
            ax.axvline(x1, linestyle="--", linewidth=1, color="blue")
            ax.axvspan(x0, x1, alpha=0.10, color="blue")
        ax.text(x0 + 1, -0.5, "warmup", va="bottom", ha="left", color="blue")

    if cooldown_start_s is not None:
        x0 = int(round(cooldown_start_s / interval_s))
        ax.axvline(x0, linestyle="--", linewidth=1, color="k")
        if cooldown_end_s is not None and cooldown_end_s > cooldown_start_s:
            x1 = int(round(cooldown_end_s / interval_s))
            ax.axvline(x1, linestyle="--", linewidth=1, color="k")
            ax.axvspan(x0, x1, alpha=0.10, color="k")
        ax.text(x0 + 1, -0.5, "cooldown", va="bottom", ha="left", color="k")

    path = os.path.join(outdir, "requests_status_timeline.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_completion_distribution(outdir: str, durations_s: np.ndarray) -> Tuple[str, Dict[str, float]]:
    if durations_s.size == 0:
        return "", {}

    p50 = float(np.percentile(durations_s, 50))
    p95 = float(np.percentile(durations_s, 95))
    mean = float(np.mean(durations_s))

    plt.figure()
    plt.hist(durations_s, bins=min(60, max(10, int(np.sqrt(durations_s.size)))))
    plt.axvline(p50, linewidth=2, label=f"p50={p50:.3f}s")
    plt.axvline(p95, linewidth=2, label=f"p95={p95:.3f}s")
    plt.axvline(mean, linewidth=2, label=f"mean={mean:.3f}s")
    plt.xlabel("Completion time (seconds)")
    plt.ylabel("Count")
    plt.title("Completion time distribution")
    plt.legend()
    path = os.path.join(outdir, "completion_time_distribution.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

    return path, {"p50": p50, "p95": p95, "mean": mean}


def plot_worker_counts(
    outdir: str,
    t: List[float],
    samples: List[Dict[str, int]],
    warmup_start_s: Optional[float] = None,
    warmup_end_s: Optional[float] = None,
    cooldown_start_s: Optional[float] = None,
    cooldown_end_s: Optional[float] = None,
) -> str:
    # Official ObservedState enum color mapping
    STATUS_COLORS = {
        "unknown": "#bcbd22",         # olive
        "pending": "#9e9e9e",         # light grey
        "creating": "#17becf",        # cyan
        "created": "#1f8c9f",         # teal
        "loading": "#9467bd",         # purple
        "model_loading": "#b47fdb",   # light purple
        "idle": "#2ca02c",            # green (ready to work)
        "stop_queued": "#ff6b6b",     # light red
        "stopping": "#ff7f0e",        # orange
        "stopped": "#404040",         # dark grey
        "starting": "#ffd700",        # yellow/gold
        "rebooting": "#ff8c00",       # dark orange
        "destroying": "#8b0000",      # dark red
        "unavail": "#7f7f7f",         # medium grey
        "error": "#d62728",           # red
        "offline": "#2f2f2f",         # very dark grey
        "OTHER": "#8c564b",           # brown
    }

    all_statuses = Counter()
    for s in samples:
        all_statuses.update(s)

    top = [k for k, _ in all_statuses.most_common(10)]
    series: Dict[str, List[int]] = {k: [] for k in top}
    series["OTHER"] = []

    for s in samples:
        other = 0
        for k, v in s.items():
            if k in series:
                series[k].append(int(v))
            else:
                other += int(v)
        series["OTHER"].append(other)
        for k in top:
            if len(series[k]) < len(series["OTHER"]):
                series[k].append(0)

    fig, ax = plt.subplots()
    for k, ys in series.items():
        if k == "OTHER" and sum(ys) == 0:
            continue
        # Use status-specific color if available, otherwise use default
        color = STATUS_COLORS.get(k.lower(), STATUS_COLORS.get(k, None))
        ax.plot(t, ys, label=k, color=color, linewidth=2)

    _mark_warmup(ax, warmup_start_s, warmup_end_s)
    _mark_cooldown(ax, cooldown_start_s, cooldown_end_s)

    ax.set_xlabel("Seconds since start")
    ax.set_ylabel("Number of workers")
    ax.set_title("Workers by status (polled at 0.1s)")
    ax.legend()

    path = os.path.join(outdir, "workers_by_status.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_workers_total_and_reqs(
    outdir: str,
    t: List[float],
    totals: List[int],
    reqs_sum: List[int],
    warmup_start_s: Optional[float] = None,
    warmup_end_s: Optional[float] = None,
    cooldown_start_s: Optional[float] = None,
    cooldown_end_s: Optional[float] = None,
) -> Tuple[str, str]:
    fig, ax = plt.subplots()
    ax.plot(t, totals, label="total_workers")

    _mark_warmup(ax, warmup_start_s, warmup_end_s)
    _mark_cooldown(ax, cooldown_start_s, cooldown_end_s)

    ax.set_xlabel("Seconds since start")
    ax.set_ylabel("Workers")
    ax.set_title("Total workers over time")
    ax.legend()
    p1 = os.path.join(outdir, "workers_total.png")
    plt.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(t, reqs_sum, label="sum(reqs_working)")

    _mark_warmup(ax, warmup_start_s, warmup_end_s)
    _mark_cooldown(ax, cooldown_start_s, cooldown_end_s)

    ax.set_xlabel("Seconds since start")
    ax.set_ylabel("Requests working (sum)")
    ax.set_title("Sum of reqs_working over time")
    ax.legend()
    p2 = os.path.join(outdir, "workers_reqs_working_sum.png")
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()

    return p1, p2


# -----------------------------
# Main
# -----------------------------
async def main(
    concurrency: int,
    interval_s: float,
    outdir: str,
    endpoint_name: str,
    warmup_s: float,
    cooldown_s: float,
    http_port: int,
    use_session: bool,
    auto_instance: str,
) -> int:
    _ensure_dir(outdir)
    t0 = time.monotonic()

    state = RunState(
        n=concurrency,
        statuses=[STATUS_PENDING] * concurrency,
        start_ts=[None] * concurrency,
        end_ts=[None] * concurrency,
        errors=[None] * concurrency,
        lock=asyncio.Lock(),
        t0=t0,
    )

    req_time: List[float] = []
    req_counts: Dict[str, List[int]] = {
        STATUS_PENDING: [],
        STATUS_IN_FLIGHT: [],
        STATUS_DONE: [],
        STATUS_FAILED: [],
    }
    req_matrix_snapshots: List[np.ndarray] = []

    w_time: List[float] = []
    w_counts: List[Dict[str, int]] = []
    w_totals: List[int] = []
    w_reqs_working_sum: List[int] = []

    done_evt = asyncio.Event()

    # Ctrl-C should request a graceful stop (not abort plotting)
    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop_evt.set)
    except NotImplementedError:
        pass

    warmup_start_s: Optional[float] = None
    warmup_end_s: Optional[float] = None
    cooldown_start_s: Optional[float] = None
    cooldown_end_s: Optional[float] = None

    results: List[object] = []

    async with Serverless(debug=True, instance=auto_instance) as client:
        endpoint = await client.get_endpoint(name=endpoint_name)

        # Warmup period: observe worker state before measurement
        if warmup_s > 0:
            warmup_start_s = 0.0
            warmup_end_s = float(warmup_s)

            print(f"Starting warmup period of {warmup_s:.1f}s to observe worker state...")

            # Start worker sampling during warmup (no requests yet)
            warmup_done_evt = asyncio.Event()
            warmup_sampler = asyncio.create_task(
                sample_workers(
                    endpoint,
                    warmup_done_evt,
                    interval_s,
                    w_time,
                    w_counts,
                    w_totals,
                    w_reqs_working_sum,
                    t0,
                )
            )

            # Wait for warmup duration
            await asyncio.sleep(warmup_s)

            # Stop warmup sampling
            warmup_done_evt.set()
            await warmup_sampler

            print(f"Warmup complete. Starting measurement phase...")

        req_tasks = [
            asyncio.create_task(generate(endpoint, idx, state, http_port, use_session))
            for idx in range(concurrency)
        ]

        sampler_tasks = [
            asyncio.create_task(
                sample_request_statuses(
                    state, done_evt, interval_s, req_time, req_counts, req_matrix_snapshots
                )
            ),
            asyncio.create_task(
                sample_workers(
                    endpoint,
                    done_evt,
                    interval_s,
                    w_time,
                    w_counts,
                    w_totals,
                    w_reqs_working_sum,
                    t0,
                )
            ),
            asyncio.create_task(
                monitor_request_tasks(
                    req_tasks,
                    done_evt,
                    interval_s * 10,
                    t0,
                )
            ),
        ]

        gather_task = asyncio.gather(*req_tasks, return_exceptions=True)
        stop_task = asyncio.create_task(stop_evt.wait())

        try:
            print(f"Launched {len(req_tasks)} request tasks; waiting (Ctrl-C will stop + still plot)...")

            done, _pending = await asyncio.wait(
                {gather_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stop_task in done:
                interrupt_s = time.monotonic() - t0
                print(f"\n[INTERRUPT] Ctrl-C at t={interrupt_s:.3f}s; cancelling request tasks...")

                for t in req_tasks:
                    t.cancel()

                results = await gather_task
                print(f"Gather completed after interrupt! Got {len(results)} results/entries")
            else:
                results = await gather_task
                print(f"Gather completed! Got {len(results)} results/entries")

                async with state.lock:
                    ends_snapshot = list(state.end_ts)

                non_none_ends = [e for e in ends_snapshot if e is not None]
                t_last_end = max(non_none_ends) if non_none_ends else time.monotonic()

                cooldown_start_s = float(t_last_end - t0)
                cooldown_end_s = cooldown_start_s + float(cooldown_s)
                cooldown_end_abs = t_last_end + float(cooldown_s)

                now = time.monotonic()
                remaining = cooldown_end_abs - now
                if remaining > 0:
                    print(
                        f"All requests completed at t={cooldown_start_s:.3f}s; "
                        f"sampling cooldown for {cooldown_s:.1f}s..."
                    )
                    await asyncio.sleep(remaining)
                else:
                    print(
                        f"All requests completed at t={cooldown_start_s:.3f}s; "
                        f"cooldown window already elapsed by {-remaining:.3f}s; stopping sampling."
                    )

        finally:
            done_evt.set()
            await asyncio.gather(*sampler_tasks, return_exceptions=True)

            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task

            print("samplers finished; exiting Serverless context...")

    async with state.lock:
        statuses_final = list(state.statuses)
        starts = list(state.start_ts)
        ends = list(state.end_ts)
        errs = list(state.errors)

    ok_durations = []
    fail_durations = []
    for i in range(concurrency):
        if starts[i] is None or ends[i] is None:
            continue
        dt = float(ends[i] - starts[i])
        if statuses_final[i] == STATUS_DONE:
            ok_durations.append(dt)
        elif statuses_final[i] == STATUS_FAILED:
            fail_durations.append(dt)

    ok_durations_np = np.array(ok_durations, dtype=float)

    # Plots (will run even if interrupted)
    p_req_counts = plot_request_counts(outdir, req_time, req_counts, warmup_start_s, warmup_end_s, cooldown_start_s, cooldown_end_s)
    p_req_timeline = plot_request_timeline(outdir, req_matrix_snapshots, interval_s, warmup_start_s, warmup_end_s, cooldown_start_s, cooldown_end_s)
    p_workers_by_status = plot_worker_counts(outdir, w_time, w_counts, warmup_start_s, warmup_end_s, cooldown_start_s, cooldown_end_s)
    p_workers_total, p_workers_reqs = plot_workers_total_and_reqs(
        outdir, w_time, w_totals, w_reqs_working_sum, warmup_start_s, warmup_end_s, cooldown_start_s, cooldown_end_s
    )

    p_dist, stats = plot_completion_distribution(outdir, ok_durations_np)

    print("\nAll generations complete (or cancelled):")
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            print(f" - [{i}] EXCEPTION: {r!r}")
        else:
            print(f" - [{i}] {r}")

    print("\nFinal request status counts:")
    print(Counter(statuses_final))

    if warmup_start_s is not None and warmup_end_s is not None:
        print("\nWarmup window (seconds since start):")
        print(f" - start: {warmup_start_s:.3f}s")
        print(f" - end  : {warmup_end_s:.3f}s")

    if cooldown_start_s is not None and cooldown_end_s is not None:
        print("\nCooldown window (seconds since start):")
        print(f" - start: {cooldown_start_s:.3f}s")
        print(f" - end  : {cooldown_end_s:.3f}s")

    if stats:
        print("\nCompletion time stats (successful requests only):")
        print(f" - p50 : {stats['p50']:.4f}s")
        print(f" - p95 : {stats['p95']:.4f}s")
        print(f" - mean: {stats['mean']:.4f}s")
        print(f" - n   : {ok_durations_np.size}")

    if fail_durations:
        print("\nFailures (includes cancellations):")
        failed_idxs = [i for i, s in enumerate(statuses_final) if s == STATUS_FAILED]
        print(f" - n_failed: {len(failed_idxs)}")
        for i in failed_idxs[:10]:
            print(f"   - [{i}] {errs[i]}")

    print("\nSaved plots:")
    print(f" - {p_req_counts}")
    print(f" - {p_req_timeline}")
    print(f" - {p_dist if p_dist else '(no completion distribution; no successful requests)'}")
    print(f" - {p_workers_by_status}")
    print(f" - {p_workers_total}")
    print(f" - {p_workers_reqs}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds (default: 0.1)")
    parser.add_argument("--outdir", type=str, default="./run_metrics", help="Output directory for plots")
    parser.add_argument("--endpoint-name", type=str, default="my-comfy-endpoint")

    # Better CLI behavior than type=bool
    parser.add_argument("--use-session", action="store_true", help="Use session+webhook flow instead of /generate/sync")

    parser.add_argument("--auto-instance", type=str, default="prod")
    parser.add_argument(
        "--warmup",
        type=float,
        default=10.0,
        help="Warmup duration to observe worker state before measurement (seconds)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=60.0,
        help="Cooldown sampling duration after last request returns (seconds)",
    )
    parser.add_argument("--http-port", type=int, default=3001, help="HTTP port for session/end webhook (default: 3001)")
    args = parser.parse_args()

    try:
        raise SystemExit(
            asyncio.run(
                main(
                    args.concurrency,
                    args.interval,
                    args.outdir,
                    args.endpoint_name,
                    args.warmup,
                    args.cooldown,
                    args.http_port,
                    args.use_session,
                    args.auto_instance,
                )
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(130)
