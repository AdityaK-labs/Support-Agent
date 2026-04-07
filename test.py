import asyncio
import os
import traceback
from huggingface_hub import AsyncInferenceClient
from inference import call_api_model

async def test():
    client = AsyncInferenceClient(model='Qwen/Qwen2.5-72B-Instruct', token=os.getenv('HF_TOKEN'))
    try:
        await call_api_model({}, client, 'Give me some JSON')
    except Exception as e:
        with open('error.txt', 'w') as f:
            f.write(traceback.format_exc())

asyncio.run(test())
