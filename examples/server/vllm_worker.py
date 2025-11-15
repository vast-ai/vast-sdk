
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Type, Dict, Any
from aiohttp import web, ClientResponse
import json
import logging
import nltk
import random
import os

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig
from vastai.serverless.server.lib.data_types import (
    EndpointHandler,
    ApiPayload,
    JsonDataException,
    WorkerConfig,
    HandlerConfig
)

# vLLM model configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18000
MODEL_LOG_FILE             = '/var/log/portal/vllm.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# vLLM-specific log messages
MODEL_LOAD_LOG_MSG = [
    "Application startup complete.",
]

MODEL_ERROR_LOG_MSGS = [
    "INFO exited: vllm",
    "RuntimeError: Engine",
]

MODEL_INFO_LOG_MSGS = [
    '"message":"Download'
]

log = logging.getLogger(__name__)

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

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



# Create handlers
completions_handler = HandlerConfig(
    route="/v1/completions",
    healthcheck=MODEL_HEALTHCHECK_ENDPOINT,
    payload_class=CompletionsData,
    benchmark_data=[],
)

chat_completions_handler = HandlerConfig(
    route="/v1/chat/completions",
    healthcheck=MODEL_HEALTHCHECK_ENDPOINT,
    payload_class=ChatCompletionsData,
    benchmark_data=[],
)

# Create log actions
log_action_config = LogActionConfig(
    on_load=MODEL_LOAD_LOG_MSG,
    on_error=MODEL_ERROR_LOG_MSGS,
    on_info=MODEL_INFO_LOG_MSGS
)

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    handlers=[completions_handler, chat_completions_handler],
    allow_parallel_requests=True,
    benchmark_route="/v1/completions",
    log_action_config=log_action_config
)

Worker(worker_config).run()