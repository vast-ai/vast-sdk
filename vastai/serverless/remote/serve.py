from base import Config, Deployment_
from ..server.worker import Worker
from typing import (
    ParamSpec,
    Type,
    Any,
    Callable,
    Awaitable,
    TypeVar,
    AsyncContextManager,
)
import asyncio

T = TypeVar("T")


class Deployment(Deployment_["Deployment"], AsyncContextManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts: dict[Type, Any] = {}
        self.context_managers: dict[Type, AsyncContextManager] = {}
        self.remote_funcs: dict[tuple[str], Callable[..., Awaitable]] = {}

    def context(
        self, *args, **kwargs
    ) -> Callable[
        [Type[AsyncContextManager[T]]], Type[AsyncContextManager[T]]
    ]:  # registers context_class as deployment context
        def decorator(
            context_class: Type[AsyncContextManager[T]],
        ) -> Type[AsyncContextManager[T]]:
            self.context_managers[context_class] = context_class(*args, **kwargs)
            return context_class

        return decorator

    async def __aenter__(self):
        self.contexts = dict(
            zip(
                self.context_managers.keys(),
                await asyncio.gather(
                    *(
                        manager.__aenter__()
                        for manager in self.context_managers.values()
                    )
                ),
            )
        )

    async def __aexit__(self, exc_type, exc, tb):
        await asyncio.gather(
            *(
                self.context_managers[cls_].__aexit__(exc_type, exc, tb)
                for cls_ in self.contexts.keys()  # only run aexits for contexts we've aentered
            )
        )
        self.contexts = {}  # invalidate contexts

    # Worker should register post handlers for all remote_funcs in our map, /remote/a/b/... should server remote_funcs[(a,b,...)]
    # Worker should start sending loading status updates to autoscaler, start Deployment.__aenter__, and only report ready/start serving remote funcs once __aenter__ completes
    # i.e., __aenter__ replaces the log-based ready detection
    # Worker should guarantee that __aexit__ should run on termination, with the exceptions passed through if terminated due to exception
    def into_worker(self) -> Worker:
        raise NotImplementedError("not implemented")

    P = ParamSpec("P")

    def remote(self, f: Callable[P, Awaitable[Any]]) -> Callable[P, Awaitable[Any]]:
        pass
