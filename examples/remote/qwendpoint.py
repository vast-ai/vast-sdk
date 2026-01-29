from vastai.serverless.remote.deployment import Deployment, benchmark, remote
import os
import random
import nltk
import requests

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

sample_prompts = [
    "Explain this concept as if I’m five years old.",
    "Summarize this text in one paragraph.",
    "Give me pros and cons of this idea.",
    "Rewrite this in a more formal tone.",
    "Generate three creative story ideas.",
    "Explain the steps to solve this problem.",
    "Suggest improvements to this paragraph.",
    "Translate this text into Spanish.",
    "Write a short poem about the ocean.",
    "Create a bullet-point outline for this topic.",
    "Give me 10 title ideas for a blog post.",
    "Suggest debugging steps for this code.",
    "Explain the meaning of this error message.",
    "Simplify this technical explanation.",
    "Brainstorm new product feature ideas.",
    "Generate realistic test data in JSON format.",
    "Turn this list into a well-structured table.",
    "Provide a counter-argument to this claim.",
    "Rewrite this text to be more concise.",
    "Create a step-by-step learning plan."
]


benchmark_dataset = [
    {
        "prompt" : prompt
    } for prompt in sample_prompts
]

@benchmark(
    endpoint_name="qwendpoint",
    dataset=benchmark_dataset
)
@remote(endpoint_name="qwendpoint")
async def llm_completions(prompt: str, max_tokens: int = 1024):
    MODEL_SERVER_URL  = "http://127.0.0.1:18000/v1/completions"

    payload = {
        "model":       "Qwen/Qwen3-8B",
        "prompt":      prompt,
        "temperature": 0.7,
        "max_tokens":  max_tokens
    }

    try:
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    return resp.json() 

@remote(endpoint_name="qwendpoint")
async def llm_chat(prompt: str, max_tokens: int = 1024):
    MODEL_SERVER_URL  = "http://127.0.0.1:18000/v1/chat/completions"
    system_prompt = (
        "You are Qwen.\n"
        "You are to only speak in English.\n"
    )

    user_prompt = prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    return resp.json() 


deployment = Deployment(
    name="qwendpoint",
    image_name="vastai/vllm:@vastai-automatic-tag",
    max_workers=5,
    env_vars={
        "PORTAL_CONFIG" : "\"localhost:18000:18000:/docs:vLLM API|localhost:28265:28265:/:Ray Dashboard\"",
        "MODEL_NAME" : "Qwen/Qwen3-8B",
        "VLLM_MODEL" : "Qwen/Qwen3-8B",
        "VLLM_ARGS" : "\"--max-model-len 35840 --gpu-memory-utilization 0.80 --reasoning-parser deepseek_r1 --download-dir /workspace/models --host 127.0.0.1 --port 18000 --enable-auto-tool-choice --tool-call-parser hermes\"",
        "RAY_ARGS" : "--head",
        "USE_ALL_GPUS" : "true",
        "HF_TOKEN": "${HF_TOKEN:-1}",
        "MODEL_LOG" : "/var/log/portal/vllm.log",
        "MODEL_SERVER_URL" : "http://127.0.0.1:18000",
        "MODEL_HEALTH_ENDPOINT" : "${MODEL_SERVER_URL}/health"
    },
    model_backend_load_logs=["Application startup complete."],
    model_backend_error_logs=["RuntimeError: Engine"],
    model_log_file="/var/log/portal/vllm.log"
)
deployment.on_start("entrypoint.sh &")

deployment.ready()