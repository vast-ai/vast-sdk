from vastai.serverless.server.lib.data_types import (
    EndpointHandler,
    ApiPayload,
    JsonDataException,
    WorkerConfig,
    HandlerConfig,
    LogAction,
)
import os
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Type, Dict, Any, Optional
from aiohttp import web, ClientResponse
import nltk
import logging

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()
log = logging.getLogger(__name__)

"""
Generic dataclass accepts any dictionary in input.
"""


@dataclass
class vLLMPayload(ApiPayload, ABC):
    input: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "vLLMPayload":
        return cls(input=data["input"])

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "vLLMPayload":
        errors = {}

        # Validate required parameters
        required_params = ["input"]
        for param in required_params:
            if param not in json_msg:
                errors[param] = "missing parameter"

        if errors:
            raise JsonDataException(errors)

        try:
            # Create clean data dict and delegate to from_dict
            clean_data = {"input": json_msg["input"]}

            return cls.from_dict(clean_data)

        except (json.JSONDecodeError, JsonDataException) as e:
            errors["parameters"] = str(e)
            raise JsonDataException(errors)

    @classmethod
    @abstractmethod
    def for_test(cls) -> "vLLMPayload":
        pass

    def generate_payload_json(self) -> Dict[str, Any]:
        return self.input

    def count_workload(self) -> int:
        return self.input.get("max_tokens", 0)


@dataclass
class GenericHandler(EndpointHandler[vLLMPayload], ABC):

    @property
    @abstractmethod
    def endpoint(self) -> str:
        pass

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return os.environ.get("MODEL_HEALTH_ENDPOINT")

    @classmethod
    def payload_cls(cls) -> Type[vLLMPayload]:
        return vLLMPayload

    @abstractmethod
    def make_benchmark_payload(self) -> vLLMPayload:
        pass

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        match model_response.status:
            case 200:
                # Check if the response is actually streaming based on response headers/content-type
                is_streaming_response = (
                    model_response.content_type == "text/event-stream"
                    or model_response.content_type == "application/x-ndjson"
                    or model_response.headers.get("Transfer-Encoding") == "chunked"
                    or "stream" in model_response.content_type.lower()
                )

                if is_streaming_response:
                    log.debug("Detected streaming response...")
                    res = web.StreamResponse()
                    res.content_type = model_response.content_type
                    await res.prepare(client_request)
                    async for chunk in model_response.content:
                        await res.write(chunk)
                    await res.write_eof()
                    log.debug("Done streaming response")
                    return res
                else:
                    log.debug("Detected non-streaming response...")
                    content = await model_response.read()
                    
                    # IMPORTANT: Preserve or set correct content type
                    content_type = model_response.content_type
                    if not content_type or content_type == "application/octet-stream":
                        # Default to JSON for vLLM responses
                        content_type = "application/json"
                    
                    return web.Response(
                        body=content,
                        status=200,
                        content_type=content_type,  # Use the determined content type
                    )
            case code:
                log.debug(f"SENDING RESPONSE: ERROR: status code {code}")
                return web.Response(status=code)


@dataclass
class CompletionsData(vLLMPayload):
    @classmethod
    def for_test(cls) -> "CompletionsData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        model = os.environ.get("MODEL_NAME")
        if not model:
            raise ValueError("MODEL_NAME environment variable not set")

        test_input = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 500,
        }
        return cls(input=test_input)


@dataclass
class CompletionsHandler(GenericHandler):
    @property
    def endpoint(self) -> str:
        return "/v1/completions"

    @classmethod
    def payload_cls(cls) -> Type[CompletionsData]:
        return CompletionsData

    def make_benchmark_payload(self) -> CompletionsData:
        return CompletionsData.for_test()


@dataclass
class ChatCompletionsData(vLLMPayload):
    """Chat completions-specific data implementation"""

    @classmethod
    def for_test(cls) -> "ChatCompletionsData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        model = os.environ.get("MODEL_NAME")
        if not model:
            raise ValueError("MODEL_NAME environment variable not set")

        # Chat completions use messages format instead of prompt
        test_input = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
        }
        return cls(input=test_input)


@dataclass
class ChatCompletionsHandler(GenericHandler):
    @property
    def endpoint(self) -> str:
        return "/v1/chat/completions"

    @classmethod
    def payload_cls(cls) -> Type[ChatCompletionsData]:
        return ChatCompletionsData

    def make_benchmark_payload(self) -> ChatCompletionsData:
        return ChatCompletionsData.for_test()


# vLLM-specific log messages
MODEL_SERVER_START_LOG_MSG = [
    "Application startup complete.",  # vLLM
    "llama runner started",  # Ollama
    '"message":"Connected","target":"text_generation_router"',  # TGI
    '"message":"Connected","target":"text_generation_router::server"',  # TGI
]

MODEL_SERVER_ERROR_LOG_MSGS = [
    "INFO exited: vllm",  # vLLM
    "RuntimeError: Engine",  # vLLM
    "Error: pull model manifest:",  # Ollama
    "stalled; retrying",  # Ollama
    "Error: WebserverFailed",  # TGI
    "Error: DownloadError",  # TGI
    "Error: ShardCannotStart",  # TGI
]


def create_vllm_config(
    model_server_url: Optional[str] = None,
    model_server_port: Optional[int] = None,
    model_log_file: Optional[str] = None,
    healthcheck_endpoint: Optional[str] = None,
    allow_parallel_requests: bool = True,
    benchmark_runs: int = 3,
    benchmark_words: int = 256,
) -> WorkerConfig:
    """
    Create a preconfigured WorkerConfig for vLLM.
    
    Args:
        model_server_url: URL of the model server (defaults to MODEL_SERVER_URL env var or http://127.0.0.1)
        model_server_port: Port of the model server (defaults to 18000 or extracted from MODEL_SERVER_URL)
        model_log_file: Path to model log file (defaults to MODEL_LOG env var or /var/log/portal/vllm.log)
        healthcheck_endpoint: Health check endpoint (defaults to MODEL_HEALTH_ENDPOINT env var or /health)
        allow_parallel_requests: Whether to allow parallel requests
        benchmark_runs: Number of benchmark runs
        benchmark_words: Number of words in benchmark prompts
    
    Returns:
        WorkerConfig configured for vLLM
    """
    # Get configuration from environment or use defaults
    if model_server_url is None:
        model_server_url = os.environ.get('MODEL_SERVER_URL', 'http://127.0.0.1')
    
    # Extract port from URL if not provided
    if model_server_port is None:
        if ':' in model_server_url.split('//')[1]:
            model_server_port = int(model_server_url.split(':')[-1].rstrip('/'))
            model_server_url = ':'.join(model_server_url.split(':')[:-1])
        else:
            model_server_port = 18000
    
    if model_log_file is None:
        model_log_file = os.environ.get('MODEL_LOG', '/var/log/portal/vllm.log')
    
    if healthcheck_endpoint is None:
        healthcheck_endpoint = os.environ.get('MODEL_HEALTH_ENDPOINT', '/health')
    
    # Create log actions
    log_actions = [
        *[(LogAction.ModelLoaded, msg) for msg in MODEL_SERVER_START_LOG_MSG],
        (LogAction.Info, '"message":"Download'),
        *[(LogAction.ModelError, msg) for msg in MODEL_SERVER_ERROR_LOG_MSGS],
    ]
    
    # Create handlers
    completions_handler = HandlerConfig(
        route="/v1/completions",
        healthcheck=healthcheck_endpoint,
        payload_class=CompletionsData,
        benchmark_data=[],  # Will use for_test() method
    )
    
    chat_completions_handler = HandlerConfig(
        route="/v1/chat/completions",
        healthcheck=healthcheck_endpoint,
        payload_class=ChatCompletionsData,
        benchmark_data=[],  # Will use for_test() method
    )
    
    return WorkerConfig(
        model_server_port=model_server_port,
        model_log_file=model_log_file,
        model_server_url=model_server_url,
        handlers=[completions_handler, chat_completions_handler],
        allow_parallel_requests=allow_parallel_requests,
        benchmark_route="/v1/completions",
        log_actions=log_actions,
    )


# Exported vLLM config - callable to create with defaults
vLLM = create_vllm_config