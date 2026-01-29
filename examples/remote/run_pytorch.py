import pytorch_endpoint
import asyncio

async def main():
    result = await pytorch_endpoint.matmul(a=[[2, 0], [0, 2]], b=[[1, 2], [3, 4]])
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
