from vastai.serverless.remote.endpoint import Endpoint, benchmark, remote
import random
import requests
import sys

benchmark_prompts = [
    "Cartoon hoodie hero; orc, anime cat, bunny; black goo; buff; vector on white.",
    "Cozy farming-game scene with fine details.",
    "2D vector child with soccer ball; airbrush chrome; swagger; antique copper.",
    "Realistic futuristic downtown of low buildings at sunset.",
    "Perfect wave front view; sunny seascape; ultra-detailed water; artful feel.",
    "Clear cup with ice, fruit, mint; creamy swirls; fluid-sim CGI; warm glow.",
    "Male biker with backpack on motorcycle; oilpunk; award-worthy magazine cover.",
    "Collage for textile; surreal cartoon cat in cap/jeans before poster; crisp.",
    "Medieval village inside glass sphere; volumetric light; macro focus.",
    "Iron Man with glowing axe; mecha sci-fi; jungle scene; dynamic light.",
    "Pope Francis DJ in leather jacket, mixing on giant console; dramatic.",
]


def parse_request(json_msg):
    return {"input" : json_msg}

benchmark_dataset = [
    {
        "prompt" : prompt
    } for prompt in benchmark_prompts
]

@benchmark(
    endpoint_name="comfy_endpoint",
    dataset=benchmark_dataset
)
@remote(endpoint_name="comfy_endpoint")
async def generate_image(prompt: str):
    import base64
    MODEL_SERVER_URL  = "http://127.0.0.1:18288/generate/sync"

    payload = {
        "input": {
            "modifier": "Text2Image",
            "modifications": {
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "steps": 10,
                "seed": random.randint(1, 1000)
            }
        }
    }

    try:
        # Get the file from the path on the local machine using SCP or SFTP
        # or configure S3 to upload to cloud storage.

        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        # Extract the local path to the generated image
        local_path = body["output"][0]["local_path"]
        # Read file and base64-encode it
        with open(local_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        return encoded_image
    except Exception as e:
        return {"error": str(e)}


endpoint = Endpoint(
    name="comfy_endpoint",
    image_name="vastai/comfy:@vastai-automatic-tag",
    max_workers=5,
    env_vars={
        "SERVERLESS" : "true",
        "COMFYUI_VERSION" : "latest",
        "VLLM_MODEL" : "Qwen/Qwen3-8B",
        "BACKEND" : "comfyui-json",
        "COMFYUI_ARGS" : "\"--disable-auto-launch --port 18188 --disable-xformers\"",
        "BENCHMARK_TEST_WIDTH" : "512",
        "BENCHMARK_TEST_HEIGHT" : "512",
        "BENCHMARK_TEST_STEPS" : "20",
        "PROVISIONING_SCRIPT" : "https://raw.githubusercontent.com/vast-ai/base-image/refs/heads/main/derivatives/pytorch/derivatives/comfyui/provisioning_scripts/serverless/starter-template.sh",
        "HF_TOKEN": "${HF_TOKEN:-1}",
        "MODEL_LOG" : "/var/log/portal/comfyui.log",
        "COMFYUI_API_BASE" : "http://localhost:18188",
        "MODEL_HEALTH_ENDPOINT" : "${MODEL_SERVER_URL}/health"
    },
    model_backend_load_logs=["To see the GUI go to: http://127.0.0.1:18188"],
    model_backend_error_logs= ["MetadataIncompleteBuffer", "Value not in list: ", "[ERROR] Provisioning Script failed"],
    model_log_file="/var/log/portal/comfyui.log"
)
endpoint.on_start("entrypoint.sh &")

endpoint.ready()