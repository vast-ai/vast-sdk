import asyncio
import time
from deploy import app, square


async def main():
    prev_time = time.perf_counter()
    for x in range(1, 10):
        result = await square(x)
        now = time.perf_counter()
        elapsed = now - prev_time
        label = "first response" if x == 1 else "since last"
        print(f"square({x}) = {result}  ({label}: {elapsed:.3f}s)")
        prev_time = now
    await app.client.close()


if __name__ == "__main__":
    asyncio.run(main())
