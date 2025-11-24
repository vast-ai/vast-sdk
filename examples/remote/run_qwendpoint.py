import qwendpoint
import asyncio

async def main():
    retry = True
    while retry:
        result = await qwendpoint.llm_completions("What is the square root of one million?")
        if result.get("choices"):
            retry = False
            print(result["choices"][0]["text"])

if __name__ == "__main__":
    asyncio.run(main())