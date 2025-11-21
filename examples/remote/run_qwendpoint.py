import qwendpoint
import asyncio

async def main():
    payload = {
        "model":       "Qwen/Qwen3-8B",
        "prompt":      "What is the square root of one million",
        "temperature":  0.7,
        "max_tokens":   500
    }
    result = await qwendpoint.llm_infer(payload)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())