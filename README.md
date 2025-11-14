# Vast.ai Python SDK
[![PyPI version](https://badge.fury.io/py/vastai-sdk.svg)](https://badge.fury.io/py/vastai-sdk)

The official Vast.ai SDK pip package.

## Install
```bash
pip install vastai-sdk
```
## Examples

NOTE: Ensure your Vast.ai API key is set in your working environment as `VAST_API_KEY`. Alternatively, you may pass the API key in as a parameter to either client.

### Using the VastAI CLI client

1. Create the client
```python
from vastai import VastAI
vastai = VastAI() # or, VastAI("YOUR_API_KEY")
```
2. Run commands
```python
vastai.search_offers()
```
3. Get help
```python
help(v.create_instances)
```

### Using the Serverless client

1. Create the client
```python
from vastai import Serverless
serverless = Serverless() # or, Serverless("YOUR_API_KEY")
```
2. Get an endpoint
```python
endpoint = await serverless.get_endpoint("my-endpoint")
```
3. Make a request
```python
request_body = {
                "input" : {
                    "model": "Qwen/Qwen3-8B",
                    "prompt" : "Who are you?",
                    "max_tokens" : 100,
                    "temperature" : 0.7
                }
            }
response = await serverless.request("/v1/completions", request_body)
```
4. Read the response
```python
text = response["response"]["choices"][0]["text"]
print(text)
```

Find more examples in the `examples` directory
