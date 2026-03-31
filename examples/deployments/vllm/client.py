import asyncio
from deploy import app, generate


async def main():
    result = await generate("Explain quantum computing in one sentence.")
    print(f"Response: {result}")
    await app.async_client.close()


if __name__ == "__main__":
    asyncio.run(main())
