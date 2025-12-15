import asyncio
from vastai import Serverless
import random

my_tasks = []

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="my-pytorch-endpoint")



        async def train_pytorch():
            print("Training pytorch")
            await asyncio.sleep(10)
            print("Training complete!")
            return
        
        async def status():
            return 

        task_config = TaskConfig(
            name="my-pytorch-task",
            main=train_pytorch
        )

        task_id = endpoint.start_task(task_config)

        my_tasks.append(task_id)

        print(endpoint.get_task)

        # Get the file from the path on the local machine using SCP or SFTP
        # or configure S3 to upload to cloud storage.
        print(response["response"]["result"])

if __name__ == "__main__":
    asyncio.run(main())