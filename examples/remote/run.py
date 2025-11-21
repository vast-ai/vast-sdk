import asyncio
import endpoint


if __name__ == "__main__":
    result = asyncio.run(endpoint.remote_func_a(1))

    