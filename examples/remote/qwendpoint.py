from vastai.serverless.remote.endpoint import Endpoint, benchmark, remote
import os
import random
import nltk
import requests

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

def completions_benchmark_generator() -> dict:
    prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
    model = os.environ.get("MODEL_NAME")
    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    benchmark_data = {
        "body": {
            "model": model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 500,
        }
    }

    return benchmark_data

@benchmark(
    endpoint_name="qwendpoint",
    generator=completions_benchmark_generator
)
@remote(endpoint_name="qwendpoint")
async def llm_infer(body: dict):
    MODEL_SERVER_URL  = "http://127.0.0.1:18000/v1/completions"

    payload = {
        "model":       body.get("model"),
        "prompt":      body.get("prompt"),
        "temperature": body.get("temperature", 0.7),
        "max_tokens":  body.get("max_tokens", 500),
    }

    try:
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    return resp.json() 

endpoint = Endpoint(
    name="qwendpoint",
    model_backend_load_logs=["Application startup complete."],
    model_log_file="/var/log/portal/vllm.log"
)

endpoint.ready()