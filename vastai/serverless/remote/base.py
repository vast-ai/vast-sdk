from abc import ABC, abstractmethod
from typing import (
    Callable,
    Any,
    Awaitable,
    ParamSpec,
    Optional,
    TypeVar,
    Type,
    AsyncContextManager,
    Self,
)
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    pip_installs: list[str]
    apt_gets: list[str]
    envs: list[list[str]] | list[tuple[str, ...]]  # format [(KEY,VALUE)]
    runs: (
        list[str | list[str]] | list[str | tuple[str, ...]]
    )  # tuples get turned into list after JSON serialize->deserialize round trip


T = TypeVar("T")


class Deployment_(ABC):
    by_name: dict[str, Self] = {}

    def __init__(
        self,
        name: Optional[str] = None,
        tag: str = "default",
        version_label: Optional[str] = None,
    ):
        self.name = name
        self.tag = tag
        self.version_label: str | None = version_label
        self.root_module: str | None = None

    def context(
        self, *args, **kwargs
    ) -> Callable[
        [Type[AsyncContextManager[T]]], Type[AsyncContextManager[T]]
    ]:  # registers context_class as deployment context
        return lambda c: c  # no-op when not in serve mode

    def get_context(self, context_class: Type[AsyncContextManager[T]]) -> T:
        raise NotImplementedError(
            f"{context_class}"
        )  # should not be run when not in serve modej

    @property
    def name(self):
        return self.name_

    @name.setter
    def name(self: Self, name: str | None):
        self.name_ = name
        if isinstance(name, str):
            self.__class__.by_name[name] = self

    @classmethod
    def lookup(cls, name: str) -> Self | None:
        return cls.by_name.get(name)

    def relativize(self, f: Callable[..., Any]) -> tuple[str]:
        mod = f.__module__
        name = f.__name__
        file = f.__globals__.get("__file__")
        # assume single file deployment for now.
        if self.root_module is None:
            self.root_module = mod
            if self.name_ is None:
                self.name = mod
            if file is None:
                raise Exception(
                    f"Cowardly refusing to deploy function {name} defined in your interactive Python session! We need it to be defined in a .py file to deploy from."
                )
            self.file = file
        if f.__module__ != self.root_module:
            raise Exception(
                f"Using a single-file deployment object to deploy remote functions from multiple .py files! Remote function {name} is defined in {file} but we're already trying to deploy from {self.file}."
            )
        return (name,)

    P = ParamSpec("P")

    # TODO: add type checked version of remote
    @abstractmethod
    def remote(
        self: Self,
        f: Callable[P, Awaitable[Any]] | None = None,
        *,
        benchmark_dataset: list[dict] | None = None,
        benchmark_generator: Callable[[], dict] | None = None,
        benchmark_runs: int = 10,
    ) -> (
        Callable[P, Awaitable[Any]]
        | Callable[[Callable[P, Awaitable[Any]]], Callable[P, Awaitable[Any]]]
    ): ...

    async def ensure_ready(self):  # NO-OP in serve mode
        pass
