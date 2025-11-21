import qwendpoint
import asyncio

async def main():
    result = await qwendpoint.llm_chat("What is the square root of one million?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())