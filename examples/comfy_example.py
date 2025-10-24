import asyncio
from vastai_sdk import Serverless
import os
import paramiko
from scp import SCPClient
from urllib.parse import urlparse
import random
import collections
import time
import uuid
from prometheus_client import start_http_server, Gauge, Histogram

API_KEY = os.environ.get("VAST_API_KEY")

# Generate unique run ID for this session
RUN_ID = time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

# Prometheus metrics
REQUEST_STATUS = Gauge(
    "vast_request_status_current",
    "Current number of requests by status",
    ["status", "run_id"]
)
LATENCY = Histogram(
    "vast_request_latency_seconds",
    "Latency of responses (seconds)",
    ["run_id"]
)

LATENCY_AVG = Gauge(
    "vast_request_latency_avg_seconds",
    "Rolling average latency (seconds) over recent responses",
    ["run_id"]
)


latencies = collections.deque(maxlen=50)


async def status_reporter(responses, latencies, window_size=50):
    """Continuously updates gauges to reflect live status counts."""
    while True:
        status_counts = collections.Counter()

        for r in responses:
            try:
                status = getattr(r, "status", None)
                if callable(status):
                    status = status()
                if status is None:
                    status = "unknown"
                status_counts[status] += 1
            except Exception:
                status_counts["error"] += 1

        # Reset gauges before updating
        for label_tuple in list(REQUEST_STATUS._metrics.keys()):
            label_status, label_run_id = label_tuple
            if label_run_id == RUN_ID:
                REQUEST_STATUS.labels(status=label_status, run_id=RUN_ID).set(0)

        # Update gauges to reflect current counts
        for status, count in status_counts.items():
            REQUEST_STATUS.labels(status=status, run_id=RUN_ID).set(count)

        # Compute rolling latency average
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Record it in Prometheus
        LATENCY_AVG.labels(run_id=RUN_ID).set(avg_latency)

        # Console display
        print("\n=== Live Response Status ===")
        for status, count in status_counts.items():
            print(f"{status:>12}: {count}")
        print(f"Rolling Avg Latency (last {len(latencies)}): {avg_latency:.2f} s")
        print("=============================\n")

        await asyncio.sleep(1)


def latency_callback(response):
    latency = response.get("latency")
    if latency is not None:
        latencies.append(latency)
        LATENCY.labels(run_id=RUN_ID).observe(latency)


async def main():
    client = Serverless(API_KEY, debug=False, instance="alpha")
    endpoint = await client.get_endpoint(name="comfy")

    prompts = [
        "a page from a peanuts comic strip",
    ]

    responses = []

    asyncio.create_task(status_reporter(responses, latencies))

    LOAD_PER_REQUEST = 100
    MAX_LOAD = 2000
    start_time = time.time()
    ramp_time = 30 * 60 # seconds
    while True:
        payload = {
            "input": {
                "modifier": "Text2Image",
                "modifications": {
                    "prompt": random.choice(prompts),
                    "width": 512,
                    "height": 512,
                    "steps": 10,
                    "seed": random.randint(1, 10)
                }
            }
        }
        CUR_LOAD = (max(min(time.time() - start_time, ramp_time), ramp_time * (1/100.0)) / ramp_time) * MAX_LOAD
        #CUR_LOAD = 300
        request = endpoint.request("/generate/sync", payload, cost=LOAD_PER_REQUEST).then(latency_callback)
        responses.append(request)
        await asyncio.sleep(LOAD_PER_REQUEST / CUR_LOAD)


if __name__ == "__main__":
    start_http_server(8000)
    asyncio.run(main())
