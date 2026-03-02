"""Unit tests for vastai.serverless.server.worker components.

Tests LogActionConfig, WorkerConfig, HandlerConfig, BenchmarkConfig,
and EndpointHandlerFactory. Does not start the full Worker server.
"""
import pytest

from vastai.serverless.server.lib.data_types import LogAction
from vastai.serverless.server.worker import (
    LogActionConfig,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    EndpointHandlerFactory,
)

class TestLogActionConfig:
    """Verify LogActionConfig builds log_actions correctly."""

    def test_log_actions_empty_when_no_config(self) -> None:
        """
        Verifies that log_actions returns empty list when no actions configured.

        This test verifies by:
        1. Creating LogActionConfig with default (empty) lists
        2. Asserting log_actions is empty

        Assumptions:
        - Default LogActionConfig has empty on_load, on_error, on_info
        """
        config = LogActionConfig()
        assert config.log_actions == []

    def test_log_actions_includes_on_load_messages(self) -> None:
        """
        Verifies that on_load messages are mapped to LogAction.ModelLoaded.

        This test verifies by:
        1. Creating LogActionConfig with on_load messages
        2. Asserting log_actions contains (ModelLoaded, msg) for each

        Assumptions:
        - log_actions property builds list from on_load, on_error, on_info
        """
        config = LogActionConfig(on_load=["Model loaded", "Ready"])
        actions = config.log_actions
        assert (LogAction.ModelLoaded, "Model loaded") in actions
        assert (LogAction.ModelLoaded, "Ready") in actions
        assert len(actions) == 2

    def test_log_actions_includes_on_error_messages(self) -> None:
        """
        Verifies that on_error messages are mapped to LogAction.ModelError.

        This test verifies by:
        1. Creating LogActionConfig with on_error messages
        2. Asserting log_actions contains (ModelError, msg) for each

        Assumptions:
        - log_actions property builds list from on_load, on_error, on_info
        """
        config = LogActionConfig(on_error=["Error occurred"])
        actions = config.log_actions
        assert (LogAction.ModelError, "Error occurred") in actions

    def test_log_actions_includes_on_info_messages(self) -> None:
        """
        Verifies that on_info messages are mapped to LogAction.Info.

        This test verifies by:
        1. Creating LogActionConfig with on_info messages
        2. Asserting log_actions contains (Info, msg) for each

        Assumptions:
        - log_actions property builds list from on_load, on_error, on_info
        """
        config = LogActionConfig(on_info=["Info message"])
        actions = config.log_actions
        assert (LogAction.Info, "Info message") in actions

    def test_log_actions_combines_all_action_types(self) -> None:
        """
        Verifies that log_actions combines on_load, on_error, on_info in order.

        This test verifies by:
        1. Creating LogActionConfig with all three list types populated
        2. Asserting log_actions contains correct tuples in expected order

        Assumptions:
        - extend order: on_load first, then on_error, then on_info
        """
        config = LogActionConfig(
            on_load=["load1"],
            on_error=["err1"],
            on_info=["info1"],
        )
        actions = config.log_actions
        assert actions == [
            (LogAction.ModelLoaded, "load1"),
            (LogAction.ModelError, "err1"),
            (LogAction.Info, "info1"),
        ]


class TestEndpointHandlerFactory:
    """Verify EndpointHandlerFactory creates handlers from WorkerConfig."""

    def test_factory_with_empty_handlers_creates_default_route(
        self, minimal_worker_config: WorkerConfig
    ) -> None:
        """
        Verifies that empty handlers list creates a default handler at /.

        This test verifies by:
        1. Creating EndpointHandlerFactory with config that has no handlers
        2. Asserting get_handler("/") returns a handler
        3. Asserting handler.endpoint == "/"

        Assumptions:
        - minimal_worker_config fixture provides config with empty handlers
        """
        factory = EndpointHandlerFactory(minimal_worker_config)
        handler = factory.get_handler("/")
        assert handler is not None
        assert handler.endpoint == "/"

    def test_get_handler_returns_none_for_unknown_route(
        self, minimal_worker_config: WorkerConfig
    ) -> None:
        """
        Verifies that get_handler returns None for unregistered route.

        This test verifies by:
        1. Creating factory with minimal config
        2. Calling get_handler with unknown route
        3. Asserting result is None

        Assumptions:
        - get_handler uses dict.get, returns None for missing key
        """
        factory = EndpointHandlerFactory(minimal_worker_config)
        assert factory.get_handler("/unknown") is None

    def test_get_all_handlers_returns_copy(self, minimal_worker_config: WorkerConfig) -> None:
        """
        Verifies that get_all_handlers returns a copy of handlers dict.

        This test verifies by:
        1. Creating factory and getting handlers
        2. Mutating the returned dict
        3. Asserting factory's internal state is unchanged

        Assumptions:
        - get_all_handlers returns .copy()
        """
        factory = EndpointHandlerFactory(minimal_worker_config)
        handlers = factory.get_all_handlers()
        handlers["/"] = None
        assert factory.get_handler("/") is not None

    def test_has_handlers_true_when_handlers_exist(
        self, minimal_worker_config: WorkerConfig
    ) -> None:
        """
        Verifies that has_handlers returns True when handlers are registered.

        This test verifies by:
        1. Creating factory with default handler
        2. Asserting has_handlers() is True

        Assumptions:
        - minimal_worker_config creates default / handler
        """
        factory = EndpointHandlerFactory(minimal_worker_config)
        assert factory.has_handlers() is True

    def test_model_server_base_url_formats_correctly(
        self, minimal_worker_config: WorkerConfig
    ) -> None:
        """
        Verifies that model_server_base_url returns url:port format.

        This test verifies by:
        1. Creating factory with url and port
        2. Asserting model_server_base_url == "http://localhost:8000"

        Assumptions:
        - minimal_worker_config has url and port set
        """
        factory = EndpointHandlerFactory(minimal_worker_config)
        assert factory.model_server_base_url == "http://localhost:8000"

    def test_get_benchmark_handler_raises_when_none_has_benchmark(
        self, minimal_worker_config: WorkerConfig
    ) -> None:
        """
        Verifies that get_benchmark_handler raises when no handler has benchmark.

        This test verifies by:
        1. Creating factory with empty handlers (default handler has no BenchmarkConfig)
        2. Calling get_benchmark_handler
        3. Asserting Exception is raised with "Missing EndpointHandler"

        Assumptions:
        - minimal_worker_config creates default handler with no BenchmarkConfig
        """
        factory = EndpointHandlerFactory(minimal_worker_config)
        with pytest.raises(Exception, match="Missing EndpointHandler with BenchmarkConfig"):
            factory.get_benchmark_handler()

    def test_factory_with_handler_and_benchmark_config_creates_handler(
        self, worker_config_with_handler
    ) -> None:
        """
        Verifies that HandlerConfig with BenchmarkConfig creates benchmark handler.

        This test verifies by:
        1. Using worker_config_with_handler fixture to build config with /predict handler
        2. Creating EndpointHandlerFactory
        3. Asserting get_benchmark_handler returns the handler
        4. Asserting get_handler returns handler for the route

        Assumptions:
        - worker_config_with_handler creates valid config with BenchmarkConfig
        """
        config = worker_config_with_handler(route="/predict", dataset=[{"input": "test"}])
        factory = EndpointHandlerFactory(config)
        benchmark_handler = factory.get_benchmark_handler()
        assert benchmark_handler is not None
        assert benchmark_handler.endpoint == "/predict"
        assert benchmark_handler.has_benchmark is True
        assert factory.get_handler("/predict") is benchmark_handler

    def test_get_benchmark_handler_raises_when_multiple_have_benchmark(
        self, worker_config_with_handler
    ) -> None:
        """
        Verifies that get_benchmark_handler raises when multiple handlers have BenchmarkConfig.

        This test verifies by:
        1. Using worker_config_with_handler with extra_handlers to add second benchmark
        2. Creating EndpointHandlerFactory
        3. Calling get_benchmark_handler
        4. Asserting Exception is raised with "Cannot define BenchmarkConfig"

        Assumptions:
        - Exactly one handler may have BenchmarkConfig
        """
        config = worker_config_with_handler(
            route="/a",
            dataset=[{"x": 1}],
            extra_handlers=[
                HandlerConfig(
                    route="/b",
                    benchmark_config=BenchmarkConfig(dataset=[{"x": 2}]),
                ),
            ],
        )
        factory = EndpointHandlerFactory(config)
        with pytest.raises(Exception, match="Cannot define BenchmarkConfig for more than one"):
            factory.get_benchmark_handler()

    def test_factory_with_explicit_handler_config_creates_handler(
        self, worker_config_with_handler
    ) -> None:
        """
        Verifies that explicit HandlerConfig creates handler at specified route.

        This test verifies by:
        1. Using worker_config_with_handler for /v1/chat route
        2. Creating EndpointHandlerFactory
        3. Asserting get_handler("/v1/chat") returns handler with that endpoint

        Assumptions:
        - worker_config_with_handler creates config with handlers list
        """
        config = worker_config_with_handler(
            route="/v1/chat", dataset=[{"messages": []}]
        )
        factory = EndpointHandlerFactory(config)
        handler = factory.get_handler("/v1/chat")
        assert handler is not None
        assert handler.endpoint == "/v1/chat"


class TestWorkerConfigAndDataclasses:
    """Verify WorkerConfig, HandlerConfig, BenchmarkConfig construction."""

    def test_worker_config_defaults(self) -> None:
        """
        Verifies that WorkerConfig has expected default values.

        This test verifies by:
        1. Creating WorkerConfig with no args
        2. Asserting key defaults

        Assumptions:
        - Defaults match dataclass field defaults
        """
        config = WorkerConfig()
        assert config.model_server_url is None
        assert config.model_server_port is None
        assert config.handlers == []
        assert config.max_sessions == 10

    def test_handler_config_with_benchmark(self) -> None:
        """
        Verifies that HandlerConfig accepts benchmark_config.

        This test verifies by:
        1. Creating HandlerConfig with BenchmarkConfig
        2. Asserting benchmark_config is set

        Assumptions:
        - benchmark_config is optional
        """
        bc = BenchmarkConfig(dataset=[{"a": 1}], runs=4)
        hc = HandlerConfig(route="/", benchmark_config=bc)
        assert hc.benchmark_config is bc
        assert hc.benchmark_config.runs == 4

    def test_benchmark_config_with_generator(self) -> None:
        """
        Verifies that BenchmarkConfig accepts generator callable.

        This test verifies by:
        1. Creating BenchmarkConfig with generator
        2. Asserting generator is set

        Assumptions:
        - generator is optional, alternative to dataset
        """
        def gen() -> dict:
            return {"sample": 1}

        config = BenchmarkConfig(generator=gen)
        assert config.generator is gen
        assert config.generator() == {"sample": 1}