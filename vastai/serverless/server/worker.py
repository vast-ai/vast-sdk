from vastai.serverless.server.lib import data_types, backend, server
from dataclasses import dataclass
from aiohttp import web, ClientResponse


class Worker:
    """
    This class provides a simple to use abstraction over the pyworker backend.
    All custom implementations of pyworker can be created by configuring a Worker object.
    The pyworker starts by calling Worker.run()
    """

    def __init__(self, config: data_types.WorkerConfig):
        handler_factory = data_types.GenericEndpointFactory(config)
        
        # Get all endpoint handlers
        handlers = handler_factory.get_all_handlers()
        benchmark_handler = handler_factory.get_benchmark_handler()
        
        # Create backend
        self.backend = backend.Backend(
            model_server_url=f"{config.model_server_url}:{config.model_server_port}",
            model_log_file=config.model_log_file,
            allow_parallel_requests=config.allow_parallel_requests,
            benchmark_handler=benchmark_handler,
            log_actions=config.log_actions
        )
        
        self.routes = []
        for route_path, handler in handlers.items():
            self.routes.append(
                web.post(route_path, self.backend.create_handler(handler))
            )
        


    def run(self):
        server.start_server(self.backend, self.routes)